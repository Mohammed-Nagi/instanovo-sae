"""instanovo_io.py — the single integration layer between the SAE pipeline and
the InstaNovo repository.

Every repo-facing call lives here, so there is exactly ONE place to keep in
sync with InstaNovo's Python API. This exists because the pipeline scripts were
originally written against an assumed API that did not match the real repo;
this module is the corrected, repo-grounded version.

All function bodies below are grounded in the InstaNovo-internal source:
  - transformer/model.py   InstaNovo.load / from_pretrained / forward / encoder
  - transformer/data.py    TransformerDataProcessor
  - transformer/train.py   the canonical loss computation
  - utils/data_handler.py  SpectrumDataFrame.load / to_dataset
  - common/dataset.py      DataProcessor.collate_fn
  - utils/residues.py      ResidueSet (PAD/SOS/EOS, decode)

"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Repo-facing imports — these are the module paths.
from instanovo.transformer.model import InstaNovo            # transformer/model.py:44
from instanovo.transformer.data import TransformerDataProcessor  # transformer/data.py:21
from instanovo.utils.data_handler import SpectrumDataFrame   # utils/data_handler.py:50

# Canonical legacy-PTM -> UNIMOD remapping (e.g. "M(+15.99)" -> "M[UNIMOD:35]").
# The nine-species benchmark writes mods in (+mass) notation that the bare
# ResidueSet vocabulary does not contain; this table is how InstaNovo maps them.
# Guarded because the constant's location could shift across versions.
try:
    from instanovo.constants import LEGACY_PTM_TO_UNIMOD
except Exception:  # pragma: no cover
    LEGACY_PTM_TO_UNIMOD: dict[str, str] = {}

LOG = logging.getLogger("instanovo_io")

# PAD index is 0 in the residue vocabulary; train.py uses CrossEntropyLoss(ignore_index=0).
PAD_INDEX = 0


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_instanovo(
    source: str | Path,
    device: str = "cuda",
    by_id: bool = False,
) -> tuple[InstaNovo, Any, Any]:
    """Load InstaNovo and return (model, config, residue_set).

    IMPORTANT: both InstaNovo.load(path) and InstaNovo.from_pretrained(id)
    return a (model, config) TUPLE — not a bare model. The residue set lives
    on the model as `model.residue_set`; do not construct ResidueSet() yourself
    (its constructor requires a residue_masses dict).

    Grounded in transformer/model.py:
        load(cls, path, ...) -> tuple[InstaNovo, DictConfig]          (line 137)
        from_pretrained(cls, model_id, ...) -> tuple[InstaNovo, ...]  (line 201)
        residue_set property                                          (line 114)

    Args:
        source:  local checkpoint path (by_id=False) or a model id like
                 "instanovo-v1.1.0" (by_id=True).
        device:  torch device string.
        by_id:   use from_pretrained() instead of load().
    """
    if by_id:
        model, config = InstaNovo.from_pretrained(str(source))
    else:
        model, config = InstaNovo.load(str(source))
    model.eval().to(device)

    # Disable the encoder's nested-tensor fast path. In eval mode with a padding
    # mask, nn.TransformerEncoder otherwise packs the batch into a NestedTensor, so
    # every encoder layer outputs a NestedTensor with padding stripped — it has no
    # .shape and breaks the SAE hook's [B, seq, D] per-token assumption (capture in
    # extract, substitution in evaluate). The fast path is a pure optimisation, so
    # disabling it leaves the real-token activations numerically identical.
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        for attr in ("enable_nested_tensor", "use_nested_tensor"):
            if hasattr(encoder, attr):
                setattr(encoder, attr, False)

    # Make the residue set able to encode dataset peptide notations. The
    # nine-species benchmark writes mods as "(+15.99)" etc., which the bare
    # ResidueSet vocabulary does not contain, so encoding a modified peptide
    # would raise KeyError. InstaNovo's own inference path does the same step
    # (common/predictor.py:107: residue_set.update_remapping(...)). We apply the
    # legacy (+mass) table plus any remapping carried in the checkpoint config.
    if hasattr(model.residue_set, "update_remapping"):
        remapping: dict[str, str] = dict(LEGACY_PTM_TO_UNIMOD)
        try:
            cfg_remap = config.get("residue_remapping", None) if hasattr(config, "get") else None
            if cfg_remap:
                remapping.update(dict(cfg_remap))
        except Exception:  # pragma: no cover - config shape varies across versions
            pass
        if remapping:
            model.residue_set.update_remapping(remapping)

    return model, config, model.residue_set


def get_encoder_layer(model: InstaNovo, layer_idx: int) -> torch.nn.Module:
    """Return the nn.TransformerEncoderLayer to hook for the SAE at this depth.

    model.encoder is an nn.TransformerEncoder (transformer/model.py:41), whose
    .layers is the ModuleList of encoder blocks. Index 0..8 for the 9-layer
    encoder. This is the substitution point for Phase 7/8 ablation hooks and
    the capture point for activation extraction.
    """
    return model.encoder.layers[layer_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_spectrum_dataframe(
    source: str | Path,
    annotated: bool = True,
    shuffle: bool = False,
) -> SpectrumDataFrame:
    """Load spectra into a SpectrumDataFrame.

    Supports a parquet file, a glob of parquet files, or an mzML/MGF/etc.
    source (SpectrumDataFrame.load dispatches on extension). For the merged
    nine-species parquet produced by run_pipeline.sh, pass that file path.

    Grounded in utils/data_handler.py:
        load(cls, source, ..., is_annotated=False, shuffle=False, ...)  (line 1166)

    The returned object supports BOTH:
      - __getitem__(i)  -> dict for the spectrum at global index i
        (used to build per-spectrum metadata at fixed chunk boundaries)
      - to_dataset()    -> HuggingFace Dataset (fed to the processor + DataLoader)
    """
    return SpectrumDataFrame.load(
        str(source),
        is_annotated=annotated,
        shuffle=shuffle,
    )


def make_dataloader(
    sdf: SpectrumDataFrame,
    residue_set: Any,
    batch_size: int = 64,
    num_workers: int = 4,
    n_peaks: int = 200,
    annotated: bool = True,
    in_memory: bool = True,
    metadata_columns: list[str] | None = None,
) -> DataLoader:
    """Build a DataLoader over a SpectrumDataFrame using the repo's own
    TransformerDataProcessor.

    Mirrors the canonical flow in common/predictor.py:
        dataset   = sdf.to_dataset(in_memory=True)              (predictor.py:208)
        processor = TransformerDataProcessor(residue_set, ...)  (predictor.py:62 / data.py:21)
        processed = processor.process_dataset(dataset)          (dataset.py:80)
        loader    = DataLoader(processed, collate_fn=processor.collate_fn)  (predictor.py:343)

    Batches are dicts with the keys InstaNovo.forward expects:
        spectra, precursors, peptides, spectra_mask, peptides_mask
    (see transformer/train.py:149-155).

    NOTE: TransformerDataProcessor reverses the peptide by default
    (reverse_peptide=True, data.py:36) because InstaNovo decodes C->N. The
    forward/CE alignment below handles this transparently; it only matters if
    you map decoder positions back to residues by hand.

    shuffle is forced False so chunk boundaries are deterministic and align
    with SpectrumDataFrame index order.
    """
    dataset = sdf.to_dataset(in_memory=in_memory)
    processor = TransformerDataProcessor(
        residue_set,
        n_peaks=n_peaks,
        annotated=annotated,
        metadata_columns=metadata_columns,
    )
    processed = processor.process_dataset(dataset)
    return DataLoader(
        processed,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=processor.collate_fn,
        shuffle=False,
        pin_memory=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass + loss (the computation from transformer/train.py)
# ─────────────────────────────────────────────────────────────────────────────
def model_forward_logits(
    model: InstaNovo,
    batch: dict,
    device: str,
) -> torch.Tensor:
    """Run the InstaNovo forward pass and return logits [B, T+1, vocab].

    Mirrors transformer/train.py:149-155 exactly:
        preds = model(x=spectra, p=precursors, y=peptides,
                      x_mask=spectra_mask, y_mask=peptides_mask)

    The keyword names are x/p/y/x_mask/y_mask — NOT spectra/precursors/...
    With add_bos=True (the default), the output has T+1 token positions.
    """
    return model(
        x=batch["spectra"].to(device),
        p=batch["precursors"].to(device),
        y=batch["peptides"].to(device),
        x_mask=batch["spectra_mask"].to(device),
        y_mask=batch["peptides_mask"].to(device),
    )


def per_token_ce_and_top1(
    logits: torch.Tensor,
    peptides: torch.Tensor,
    pad_index: int = PAD_INDEX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token cross-entropy and top-1 predictions, aligned exactly as
    train.py computes the loss.

    train.py:157-159:
        preds = preds[:, :-1].reshape(-1, vocab)
        loss  = CrossEntropyLoss(ignore_index=0)(preds, peptides.flatten())

    So we drop the trailing position from the logits (the model emits T+1
    positions because add_bos=True) and align the first T against the T target
    residues. PAD (index 0) is ignored.

    Returns (ce[B,T], top1[B,T], targets[B,T], valid_mask[B,T]) where ce is
    per-token cross-entropy (0 where PAD), valid_mask marks non-PAD targets.
    """
    targets = peptides.to(logits.device)
    B, Tp1, V = logits.shape
    preds = logits[:, :-1, :]                       # [B, T, V] — drop trailing position
    T = preds.shape[1]
    if targets.shape[1] != T:                       # defensive length alignment
        T = min(T, targets.shape[1])
        preds = preds[:, :T, :]
        targets = targets[:, :T]

    ce_flat = F.cross_entropy(
        preds.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=pad_index,
        reduction="none",
    )
    ce = ce_flat.reshape(targets.shape)             # [B, T]; entries at PAD are 0
    top1 = preds.argmax(dim=-1)                     # [B, T]
    valid = targets != pad_index                    # [B, T]
    # Zero the CE at PAD positions (cross_entropy with ignore_index leaves them
    # at 0 already, but make it explicit for safe summation downstream).
    ce = ce * valid
    return ce, top1, targets, valid


def sequence_ce_and_correct(
    model: InstaNovo,
    batch: dict,
    device: str,
    pad_index: int = PAD_INDEX,
) -> tuple[float, int, int]:
    """Convenience: forward a batch and return
    (summed per-token CE over valid tokens, #correct top-1, #valid tokens).

    This is the building block for Phase 7 (loss recovered) and Phase 8
    (causal ablation) once a SAE-substitution hook is installed on
    get_encoder_layer(model, target_layer). Install the hook, call this,
    remove the hook; the difference vs the un-hooked baseline is the effect.
    """
    with torch.no_grad():
        logits = model_forward_logits(model, batch, device)
        ce, top1, targets, valid = per_token_ce_and_top1(logits, batch["peptides"], pad_index)
        ce_sum = float(ce.sum().item())
        correct = int(((top1 == targets) & valid).sum().item())
        n_valid = int(valid.sum().item())
    return ce_sum, correct, n_valid
