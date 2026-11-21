"""instanovo_io.py — the single integration layer between the SAE pipeline and
the InstaNovo repository.

Every repo-facing call lives here, so there is exactly ONE place to keep in
sync with InstaNovo's Python API. The rest of the pipeline imports this module
and never touches `instanovo.*` directly, which is what lets the pipeline live
outside the InstaNovo repo and depend on it as an ordinary package.

Only six upstream symbols are used, all imported below:
    InstaNovo, TransformerDataProcessor, SpectrumDataFrame, BeamSearchDecoder,
    ANNOTATED_COLUMN, LEGACY_PTM_TO_UNIMOD

All function bodies below are grounded in the InstaNovo source:
  - transformer/model.py   InstaNovo.load / from_pretrained / forward / encoder
  - transformer/data.py    TransformerDataProcessor
  - transformer/train.py   the canonical loss computation
  - utils/data_handler.py  SpectrumDataFrame.load / to_dataset
  - common/dataset.py      DataProcessor.collate_fn
  - utils/residues.py      ResidueSet (PAD/SOS/EOS, decode)
  - inference/beam_search.py  BeamSearchDecoder (the inspector's prediction)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# isort: off
# The InstaNovo boundary. These are the only upstream symbols this repo imports,
# here and nowhere else; the trailing comments give the source location upstream.
# Kept grouped and unsorted on purpose, so the boundary stays legible.
from instanovo.transformer.model import InstaNovo            # transformer/model.py:44
from instanovo.transformer.data import TransformerDataProcessor  # transformer/data.py:21
from instanovo.utils.data_handler import SpectrumDataFrame   # utils/data_handler.py:50
from instanovo.inference import BeamSearchDecoder           # inference/beam_search.py:17
from instanovo.constants import ANNOTATED_COLUMN            # constants.py:70, processor's peptide column
# isort: on

# Canonical legacy-PTM -> UNIMOD remapping (e.g. "M(+15.99)" -> "M[UNIMOD:35]").
# The nine-species benchmark writes mods in (+mass) notation that the bare
# ResidueSet vocabulary does not contain; this table is how InstaNovo maps them.
# Guarded because the constant's location could shift across versions.
try:
    from instanovo.constants import LEGACY_PTM_TO_UNIMOD
except Exception:  # pragma: no cover  # noqa: BLE001
    # Broad on purpose: the constant's location may shift across versions, and
    # any failure to reach it must still leave this module importable.
    LEGACY_PTM_TO_UNIMOD: dict[str, str] = {}

LOG = logging.getLogger("instanovo_io")

# PAD index is 0 in the residue vocabulary; train.py uses CrossEntropyLoss(ignore_index=0).
PAD_INDEX = 0

# Peaks retained per spectrum by TransformerDataProcessor. Extraction and the
# Phase 7/8 re-run loader MUST agree on this: it determines how many peak tokens
# each spectrum contributes, so a mismatch silently misaligns per-spectrum CE
# against the per-spectrum concept prevalence computed from the chunks.
DEFAULT_N_PEAKS = 200


# --- Model loading ------------------------------------------------------------

def _disable_nested_tensor_fastpath(model: InstaNovo) -> None:
    """Turn off the encoder's nested-tensor fast path.

    In eval mode with a padding mask, nn.TransformerEncoder otherwise packs the
    batch into a NestedTensor, so every encoder layer outputs a NestedTensor with
    padding stripped — it has no .shape and breaks the SAE hook's [B, seq, D]
    per-token assumption (capture in extract, substitution in evaluate). The fast
    path is a pure optimisation, so disabling it leaves the real-token
    activations numerically identical.
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for attr in ("enable_nested_tensor", "use_nested_tensor"):
        if hasattr(encoder, attr):
            setattr(encoder, attr, False)


def _apply_residue_remapping(model: InstaNovo, config: Any) -> None:
    """Teach the residue set the dataset's modification notation.

    The nine-species benchmark writes mods as "(+15.99)" etc., which the bare
    ResidueSet vocabulary does not contain, so encoding a modified peptide would
    raise KeyError. InstaNovo's own inference path does the same step
    (common/predictor.py:107: residue_set.update_remapping(...)). The legacy
    (+mass) table is applied plus any remapping carried in the checkpoint config.
    """
    if not hasattr(model.residue_set, "update_remapping"):
        return

    remapping: dict[str, str] = dict(LEGACY_PTM_TO_UNIMOD)
    try:
        cfg_remap = config.get("residue_remapping", None) if hasattr(config, "get") else None
        if cfg_remap:
            remapping.update(dict(cfg_remap))
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        # Config shape varies across versions; the built-in table is a usable
        # fallback, so log the reason rather than failing the load.
        LOG.debug("Could not read residue_remapping from config: %s", exc)

    if remapping:
        model.residue_set.update_remapping(remapping)


def looks_like_model_id(source: str | Path) -> bool:
    """Whether `source` is a pretrained model id rather than a checkpoint path.

    Mirrors InstaNovo.from_pretrained's own rule (transformer/model.py): anything
    containing a path separator or ending in .ckpt is a local file; everything
    else (e.g. "instanovo-v1.1.0") is an id to resolve and download.
    """
    s = str(source)
    return not (s.endswith(".ckpt") or "/" in s or "\\" in s)


def load_instanovo(
    source: str | Path,
    device: str = "cuda",
    by_id: bool | None = None,
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
        source:  local checkpoint path, or a model id like "instanovo-v1.1.0"
                 which from_pretrained resolves against InstaNovo's models.json
                 and downloads into its cache on first use.
        device:  torch device string.
        by_id:   force the id (True) or path (False) route. The default, None,
                 detects it from `source` the same way InstaNovo itself does,
                 so callers can accept either form without branching.
    """
    if by_id is None:
        by_id = looks_like_model_id(source)

    if by_id:
        LOG.info("Resolving pretrained InstaNovo model id %r", str(source))
        model, config = InstaNovo.from_pretrained(str(source))
    else:
        model, config = InstaNovo.load(str(source))
    model.eval().to(device)

    _disable_nested_tensor_fastpath(model)
    _apply_residue_remapping(model, config)

    return model, config, model.residue_set


def uses_flash_attention(model: InstaNovo) -> bool:
    """Whether this checkpoint bypasses the standard encoder stack.

    Flash attention does not run nn.TransformerEncoder's layer modules, so the
    SAE hooks would never fire: extraction would capture nothing, and Phase 7/8
    substitution would silently be a no-op (loss_recovered ~1, all delta-CE ~0)
    rather than an error. Both extract.py and evaluate.py check this before use.
    """
    return bool(getattr(model, "use_flash_attention", False))


def get_encoder_layer(model: InstaNovo, layer_idx: int) -> torch.nn.Module:
    """Return the nn.TransformerEncoderLayer to hook for the SAE at this depth.

    model.encoder is an nn.TransformerEncoder (transformer/model.py:41), whose
    .layers is the ModuleList of encoder blocks. Index 0..8 for the 9-layer
    encoder. This is the substitution point for Phase 7/8 ablation hooks and
    the capture point for activation extraction.
    """
    return model.encoder.layers[layer_idx]


# --- Data loading -------------------------------------------------------------

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


def make_processor(
    residue_set: Any,
    n_peaks: int = DEFAULT_N_PEAKS,
    annotated: bool = True,
    metadata_columns: list[str] | None = None,
    use_spectrum_utils: bool = True,
) -> TransformerDataProcessor:
    """The TransformerDataProcessor every consumer in this repo preprocesses with.

    One constructor for extraction's DataLoader and the inspector's single
    spectra, so both see the same peaks. use_spectrum_utils=True is the
    processor's default, what extraction ran with, and the standard input for
    the v1.1.0 checkpoint (InstaNovo 1.1.0 always preprocessed with
    spectrum_utils; only 1.2.x's predict.py passes False). The two paths keep
    different peaks: spectrum_utils' min_intensity is relative to the base peak
    and its output is m/z-sorted, the fallback's is absolute and top-k ordered.
    """
    return TransformerDataProcessor(
        residue_set,
        n_peaks=n_peaks,
        annotated=annotated,
        metadata_columns=metadata_columns,
        use_spectrum_utils=use_spectrum_utils,
    )


def preprocess_spectrum(
    processor: TransformerDataProcessor,
    mz_array,
    intensity_array,
    precursor_mz: float,
    precursor_charge: int,
    peptide: str | None = None,
) -> dict:
    """Preprocess ONE raw spectrum into a model-ready batch of size 1.

    The same two steps the DataLoader runs per row and per batch
    (process_row, then collate_fn), so a spectrum preprocessed here gets
    exactly the peaks and padding layout extraction gave it. `peptide` is the
    sequence to teacher-force (InstaNovo notation, e.g. "PEPM(+15.99)K" or
    "PEPM[UNIMOD:35]K"); it requires an annotated processor and is stored
    under the processor's annotated column, as SpectrumDataFrame does with
    `modified_sequence`.
    """
    row = {
        "mz_array": list(map(float, mz_array)),
        "intensity_array": list(map(float, intensity_array)),
        "precursor_mz": float(precursor_mz),
        "precursor_charge": int(precursor_charge),
    }
    if peptide is not None:
        if not processor.annotated:
            raise ValueError("preprocess_spectrum: a peptide needs an annotated processor")
        row[ANNOTATED_COLUMN] = peptide
    elif processor.annotated:
        raise ValueError("preprocess_spectrum: an annotated processor needs a peptide")
    processed = processor.process_row(row)
    # The DataLoader path hands collate_fn 0-d tensors in the dataset's dtypes
    # (precursor_mz float64, precursor_charge int64), and collate_fn computes
    # the precursor mass before casting to float32; python floats would make
    # that arithmetic float32 and shift the decoder input slightly.
    processed["precursor_mz"] = torch.tensor(row["precursor_mz"], dtype=torch.float64)
    processed["precursor_charge"] = torch.tensor(row["precursor_charge"], dtype=torch.int64)
    return processor.collate_fn([processed])


def residue_delta_mass(residue_set: Any, token: str) -> float | None:
    """Modification mass carried by a vocabulary token, or None if it has none.

    "M[UNIMOD:35]" -> mass(token) - mass("M"); an N-terminal token such as
    "[UNIMOD:1]" carries its whole mass; a plain residue returns None.
    """
    if "[" not in token and "(" not in token:
        return None
    base = token[0] if token[0].isalpha() else None
    mass = float(residue_set.get_mass(token))
    return mass - float(residue_set.get_mass(base)) if base else mass


def beam_search(
    model: InstaNovo,
    batch: dict,
    device: str,
    num_beams: int,
    max_length: int,
) -> dict:
    """InstaNovo's beam search on a preprocessed batch.

    Mirrors transformer/predict.py: BeamSearchDecoder(model, float_dtype=float64)
    and decode(spectra, precursors, beam_size, max_length, return_beam). Returns
    the decoder's dict: "predictions" (residue token lists, N->C, empty when no
    beam completes), "prediction_log_probability" and
    "prediction_token_log_probabilities" (N->C), one entry per spectrum.
    """
    decoder = BeamSearchDecoder(model, float_dtype=torch.float64)
    with torch.inference_mode():
        return decoder.decode(
            spectra=batch["spectra"].to(device),
            precursors=batch["precursors"].to(device),
            beam_size=num_beams,
            max_length=max_length,
            return_beam=num_beams > 1,
        )


def make_dataloader(
    sdf: SpectrumDataFrame,
    residue_set: Any,
    batch_size: int = 64,
    num_workers: int = 4,
    n_peaks: int = DEFAULT_N_PEAKS,
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

    n_peaks caps the peaks kept per spectrum and therefore the number of peak
    tokens each spectrum contributes. Every consumer in the pipeline must use
    the SAME value: extraction's token count is what the annotation labels and
    SAE activations are indexed against, and Phase 7/8 compare per-spectrum CE
    from a re-run loader against per-spectrum prevalence from those chunks.

    NOTE: TransformerDataProcessor reverses the peptide by default
    (reverse_peptide=True, data.py:36) because InstaNovo decodes C->N. The
    forward/CE alignment below handles this transparently; it only matters if
    you map decoder positions back to residues by hand.

    shuffle is forced False so chunk boundaries are deterministic and align
    with SpectrumDataFrame index order.
    """
    dataset = sdf.to_dataset(in_memory=in_memory)
    processor = make_processor(
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


# --- Forward pass + loss (the computation from transformer/train.py) ----------

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

    non_blocking=True is safe on every transfer below: make_dataloader always
    builds its DataLoader with pin_memory=True, so these host->device copies
    can run asynchronously instead of blocking the calling thread.
    """
    return model(
        x=batch["spectra"].to(device, non_blocking=True),
        p=batch["precursors"].to(device, non_blocking=True),
        y=batch["peptides"].to(device, non_blocking=True),
        x_mask=batch["spectra_mask"].to(device, non_blocking=True),
        y_mask=batch["peptides_mask"].to(device, non_blocking=True),
    )


def encoder_forward(model: InstaNovo, batch: dict, device: str) -> torch.Tensor:
    """Run the encoder alone (peak embedding, latent prepend, encoder stack, and
    the precursor prepend after it) and return its output, firing any hooks on
    model.encoder.layers. Used when there is no peptide to teacher-force.
    Calling model.encoder directly would skip the embedding and latent token."""
    out, _mask = model._encoder(
        x=batch["spectra"].to(device),
        p=batch["precursors"].to(device),
        x_mask=batch["spectra_mask"].to(device),
    )
    return out


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
    vocab_size = logits.shape[-1]
    preds = logits[:, :-1, :]                       # [B, T, V] — drop trailing position

    # Defensive length alignment: truncate both sides to the shorter length so a
    # collator that pads targets differently degrades to a shorter aligned
    # prefix rather than a silent off-by-one against the residues.
    T = preds.shape[1]
    if targets.shape[1] != T:
        T = min(T, targets.shape[1])
        preds = preds[:, :T, :]
        targets = targets[:, :T]

    ce_flat = F.cross_entropy(
        preds.reshape(-1, vocab_size),
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
