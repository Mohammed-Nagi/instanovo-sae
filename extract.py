"""extract.py -- multi-layer activation extraction for InstaNovo (schema v4).

Runs InstaNovo once over a spectrum dataset and caches the encoder activations
needed to train and evaluate sparse autoencoders, so the expensive forward pass
is paid once and reused across every layer, seed and evaluation.

Design:
  - One forward pass captures all target encoder layers at once, via hooks on
    model.encoder.layers[N]. In each spectrum the latent summary token is
    position 0 and the peak tokens follow.
  - Activations are written one file per layer (acts_L{layer}_{chunk}.pt) and
    metadata once per chunk (meta_{chunk}.pt). A per-layer SAE training run
    therefore reads only its own layer's bytes, and the annotator reads
    metadata only -- never activations.
  - Per-spectrum metadata (mask, ProForma, bare peptide, modifications,
    processed m/z + intensity, precursor info, and cached baseline top-1 / CE)
    is the single source of truth for the rest of the pipeline, so no consumer
    re-derives these fields.
  - Streaming iterator with chunk-level resume: extract_all() walks the dataset
    once and writes every chunk; iterating yields each ChunkMeta as it lands.
  - manifest.json records the run config and the meta + per-layer paths for
    every chunk, so the train / annotate / evaluate consumers can plan I/O.

Output layout under --output-dir:
    manifest.json              # global config + chunk index (meta + per-layer paths)
    chunks/
        meta_00000.pt          # ChunkMeta: token maps, masks, peptide/proforma,
        meta_00001.pt          #   modifications, m/z + intensity, precursor, baseline
        ...
        acts_L2_00000.pt       # one [total_tokens, d_model] tensor per (layer, chunk)
        acts_L4_00000.pt       #   so a per-layer SAE training run reads only its
        acts_L6_00000.pt       #   own layer's bytes, and the annotator reads no
        acts_L8_00000.pt       #   activations at all (meta only).
        acts_L2_00001.pt
        ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.hooks import RemovableHandle

# InstaNovo bindings go through the single integration layer (instanovo_io),
# which is the one place that talks to the real repo API. Run from the repo
# (or with the repo on PYTHONPATH) so both `instanovo` and `instanovo_io` import.
import instanovo_io
from instanovo.transformer.model import InstaNovo

SCHEMA_VERSION = 4  # v4: per-layer activation files + separate meta file (was v3 bundled)
LOG = logging.getLogger("extract")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class ExtractionConfig:
    """All knobs for an extraction run. Persisted alongside the chunks in the manifest."""

    model_path: Path
    dataset_path: Path
    output_dir: Path

    target_layers: tuple[int, ...] = (2, 4, 6, 8)
    chunk_size: int = 1024              # number of spectra written per .pt file
    batch_size: int = 32                # forward-pass batch size
    num_workers: int = 4                # DataLoader workers
    device: str = "cuda"
    dtype: torch.dtype = torch.float32  # activation storage dtype
    save_baseline: bool = True          # cache top-1 + per-position CE
    resume: bool = True                 # skip chunks that already exist
    max_spectra: int | None = None      # debug / smoke-test limit

    def as_jsonable(self) -> dict:
        """JSON-safe representation for the manifest."""
        out = dataclasses.asdict(self)
        out["model_path"] = str(self.model_path)
        out["dataset_path"] = str(self.dataset_path)
        out["output_dir"] = str(self.output_dir)
        out["target_layers"] = list(self.target_layers)
        out["dtype"] = str(self.dtype).replace("torch.", "")
        return out


# -----------------------------------------------------------------------------
# Chunk metadata schema (activations live in separate per-layer files)
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class ChunkMeta:
    """Per-chunk metadata -- everything except the layer activations themselves.

    Activations live in separate per-layer files (see chunk_acts_path) so a
    per-layer SAE training run reads only its own layer's bytes and the
    annotator reads no activations at all. The token-level fields here
    (token_to_*) have length total_tokens and align row-for-row with every
    acts_L{layer} tensor for this chunk. Spectrum-level fields have length
    n_spectra. The latent summary token occupies position 0 of each spectrum.
    """

    schema_version: int
    chunk_idx: int
    n_spectra: int
    total_tokens: int
    target_layers: list[int]             # layers with an activation file for this chunk

    # Per-spectrum metadata (single source of truth for downstream consumers).
    spectrum_ids: list[str]
    peptides: list[str]
    proforma_strings: list[str]
    modifications: list[list[dict]]      # per-spectrum: [{position, mod_name, unimod_id}]
    precursor_charges: torch.Tensor      # [n_spectra]  int
    precursor_mzs: torch.Tensor          # [n_spectra]  float
    mz_arrays: list[torch.Tensor]        # per-spectrum [n_peaks_i]  float (processed)
    intensity_arrays: list[torch.Tensor] # per-spectrum [n_peaks_i]  float (processed)
    spectra_mask: torch.Tensor           # [n_spectra, max_seq_len]  bool, True-where-valid, latent at idx 0

    # Token provenance (align with each per-layer activation tensor).
    token_to_spectrum: torch.Tensor      # [total_tokens]  long, row in this chunk
    token_to_position: torch.Tensor      # [total_tokens]  long, 0 = latent, 1.. = peaks

    # Cached baseline predictions; None if save_baseline=False at extraction time.
    baseline_top1: torch.Tensor | None             # [n_spectra, decoder_seq_len]
    baseline_ce: torch.Tensor | None               # [n_spectra, decoder_seq_len]
    baseline_decoder_mask: torch.Tensor | None     # [n_spectra, decoder_seq_len]

    def save(self, path: Path) -> None:
        torch.save(dataclasses.asdict(self), path)

    @classmethod
    def load(cls, path: Path) -> ChunkMeta:
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


# -----------------------------------------------------------------------------
# On-disk layout helpers (the single definition of the v4 file naming, imported
# by the train / annotate / evaluate consumers so the contract lives in one place)
# -----------------------------------------------------------------------------
def chunk_meta_path(chunks_dir: Path, chunk_idx: int) -> Path:
    return chunks_dir / f"meta_{chunk_idx:05d}.pt"


def chunk_acts_path(chunks_dir: Path, chunk_idx: int, layer: int) -> Path:
    return chunks_dir / f"acts_L{layer}_{chunk_idx:05d}.pt"


def save_layer_activations(
    acts: torch.Tensor, path: Path, chunk_idx: int, layer: int
) -> None:
    """Write one layer's [total_tokens, d_model] activation tensor, wrapped with
    a tiny header so consumers can validate chunk/layer/token-count alignment."""
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "chunk_idx": chunk_idx,
            "layer": layer,
            "total_tokens": int(acts.size(0)),
            "activations": acts,
        },
        path,
    )


def load_layer_activations(path: Path) -> torch.Tensor:
    """Read one layer's activation tensor written by save_layer_activations."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["activations"]


# -----------------------------------------------------------------------------
# Forward-hook capture for multi-layer activations in one pass
# -----------------------------------------------------------------------------
class MultiLayerCapture:
    """Register forward hooks on multiple encoder layers and capture their outputs.

    One instance is reused across the whole extraction run; hooks stay registered
    for the lifetime of the context manager. Each forward pass overwrites the
    captured tensors, so callers must consume them between batches.
    """

    def __init__(self, model: InstaNovo, target_layers: tuple[int, ...]):
        self.model = model
        self.target_layers = tuple(sorted(target_layers))
        self._captured: dict[int, torch.Tensor] = {}
        self._handles: list[RemovableHandle] = []

    def __enter__(self) -> MultiLayerCapture:
        for layer_idx in self.target_layers:
            layer_module = self.model.encoder.layers[layer_idx]
            self._handles.append(
                layer_module.register_forward_hook(self._make_hook(layer_idx))
            )
        LOG.debug("Registered hooks on layers %s", self.target_layers)
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._captured.clear()

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            # PyTorch transformer encoder layers may return either a tensor or a
            # tuple depending on version/implementation. The first element is the
            # post-LayerNorm hidden state we want.
            act = output[0] if isinstance(output, tuple) else output
            self._captured[layer_idx] = act.detach()
        return hook

    def get_captured(self) -> dict[int, torch.Tensor]:
        """Return the most recent forward pass's activations, one tensor per layer."""
        if set(self._captured) != set(self.target_layers):
            missing = set(self.target_layers) - set(self._captured)
            raise RuntimeError(f"Hooks did not fire for layers: {missing}")
        return self._captured


# -----------------------------------------------------------------------------
# Per-batch processing helpers
# -----------------------------------------------------------------------------
def build_valid_mask(spectra_mask_padding: torch.Tensor) -> torch.Tensor:
    """Convert InstaNovo's True-where-padding attention mask into a usable
    True-where-valid mask of length seq_len+1, with a True column prepended
    for the latent summary token at position 0.

    InstaNovo encoder output shape is [B, n_peaks+1, D] -- the latent token
    sits at index 0, followed by n_peaks peak tokens. This helper standardises
    the mask shape and polarity for the rest of the pipeline.
    """
    batch_size = spectra_mask_padding.size(0)
    latent_col = torch.ones(batch_size, 1, dtype=torch.bool,
                            device=spectra_mask_padding.device)
    peak_valid = ~spectra_mask_padding
    return torch.cat([latent_col, peak_valid], dim=1)


def flatten_per_token(
    layer_acts: dict[int, torch.Tensor],
    valid_mask: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Flatten each layer's [B, S, D] activations to [total_tokens, D] using the
    valid mask, and build the token->spectrum and token->position index vectors
    so every row can be traced back to its source spectrum and position.
    """
    batch_size, seq_len = valid_mask.shape

    # Index vectors before masking; broadcast to match valid_mask shape.
    spectrum_idx = torch.arange(batch_size, device=valid_mask.device)
    spectrum_idx = spectrum_idx.unsqueeze(1).expand(batch_size, seq_len)
    position_idx = torch.arange(seq_len, device=valid_mask.device)
    position_idx = position_idx.unsqueeze(0).expand(batch_size, seq_len)

    token_to_spectrum = spectrum_idx[valid_mask].cpu().to(torch.long)
    token_to_position = position_idx[valid_mask].cpu().to(torch.long)

    flat = {}
    for layer_idx, act in layer_acts.items():
        # Defensive size match: if the layer's S dimension differs from the
        # mask (shouldn't happen but worth guarding), clamp before indexing.
        if act.size(1) != seq_len:
            min_s = min(act.size(1), seq_len)
            mask_use = valid_mask[:, :min_s]
            act_use = act[:, :min_s]
        else:
            mask_use, act_use = valid_mask, act
        flat[layer_idx] = act_use[mask_use].to(dtype=dtype, device="cpu")

    return flat, token_to_spectrum, token_to_position


def compute_baseline(
    model: InstaNovo,
    batch: dict,
    device: str,
) -> dict[str, torch.Tensor]:
    """Compute per-position top-1 amino-acid prediction and cross-entropy.

    Cached at extraction time so a later causal-ablation pass only needs to run
    the ablated forward, not the baseline. This full forward also fires the
    encoder hooks, but the captures are owned by the hook-driven extraction in
    the same pass, not read here.
    """
    with torch.no_grad():
        # Full InstaNovo forward (kwargs x/p/y/x_mask/y_mask); logits have T+1
        # positions because add_bos=True. The CE / top-1 alignment with the
        # peptide targets (drop last position, ignore PAD) lives in instanovo_io,
        # which is the single place that mirrors InstaNovo's training loss.
        logits = instanovo_io.model_forward_logits(model, batch, device)
        ce, top1, targets, valid = instanovo_io.per_token_ce_and_top1(
            logits, batch["peptides"], pad_index=instanovo_io.PAD_INDEX,
        )

    return {
        "top1": top1.cpu(),
        "ce": ce.cpu(),
        "decoder_mask": valid.cpu(),
    }


def parse_proforma_modifications(proforma: str) -> list[dict]:
    """Extract modifications from a ProForma string in square-bracket form.

    Expects the bracket notation that _to_proforma produces (e.g. "M[+15.99]"),
    so it is run on the converted string, not the dataset's raw "(+15.99)" form.
    Returns one {position, mod_name, unimod_id} dict per modification; positions
    are 1-indexed into the bare peptide (0 means an N-terminal modification).
    mod_name is the raw bracket content and unimod_id is parsed from a
    "UNIMOD:<n>" tag when present, else -1 (delta-mass mods carry no UNIMOD id).
    """
    import re

    modifications: list[dict] = []
    bare_position = 0
    pattern = re.compile(r"\[([^\]]+)\]")

    # Walk the string character by character so bare-sequence positions are
    # tracked correctly even with multiple modifications and N/C-terminal tags.
    i = 0
    while i < len(proforma):
        ch = proforma[i]
        if ch == "[":
            match = pattern.match(proforma, i)
            if match:
                tag = match.group(1)
                unimod_match = re.search(r"UNIMOD:(\d+)", tag, re.IGNORECASE)
                modifications.append({
                    "position": bare_position,
                    "mod_name": tag,
                    "unimod_id": int(unimod_match.group(1)) if unimod_match else -1,
                })
                i = match.end()
                continue
        # ProForma residue characters are A-Z; brackets are skipped above.
        if ch.isalpha():
            bare_position += 1
        i += 1

    return modifications


def _bare_sequence(proforma: str) -> str:
    """Bare residue sequence from a ProForma string: drop bracketed modification
    tags and keep only residue letters (used for cleavage-site geometry, which
    indexes the unmodified sequence)."""
    import re

    no_mods = re.sub(r"\[[^\]]*\]", "", proforma)
    return "".join(c for c in no_mods if c.isalpha())


def _to_proforma(modified_sequence: str) -> str:
    """Convert InstaNovo delta-mass modification notation to ProForma 2.0.

    The ms_ninespecies_benchmark writes modifications as parenthesised mass
    deltas on the preceding residue, e.g. 'YGPHTM(+15.99)AGDDPTK' (oxidation)
    and 'DTFNTSSTSN(+.98)STSSSSSNSK' (deamidation). ProForma -- which
    spectrum_utils parses in the annotator -- uses square brackets and requires
    a leading zero on bare-dot masses, e.g. 'YGPHTM[+15.99]AGDDPTK' and
    'DTFNTSSTSN[+0.98]STSSSSSNSK'. Anything already in bracket form is left as is.
    """
    import re

    def repl(m: "re.Match") -> str:
        sign, num = m.group(1), m.group(2)
        if num.startswith("."):
            num = "0" + num
        return f"[{sign}{num}]"

    return re.sub(r"\(([+-])(\d*\.?\d+)\)", repl, modified_sequence)


# -----------------------------------------------------------------------------
# Main extractor
# -----------------------------------------------------------------------------
class ActivationExtractor:
    """Drives a full extraction run. Iterable for streaming consumption,
    or call extract_all() to write every chunk to disk.
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.chunks_dir = self.output_dir / "chunks"
        self.manifest_path = self.output_dir / "manifest.json"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        # Chunk boundaries align with spectrum indices only if every non-final
        # chunk holds exactly chunk_size spectra, which requires this.
        if config.chunk_size % config.batch_size != 0:
            raise ValueError(
                f"chunk_size ({config.chunk_size}) must be a multiple of "
                f"batch_size ({config.batch_size}) so chunk boundaries align "
                "with dataset indices used for metadata lookup."
            )
        if config.max_spectra is not None:
            if config.max_spectra <= 0:
                raise ValueError("max_spectra must be positive when set.")
            if config.max_spectra % config.batch_size != 0:
                raise ValueError(
                    f"max_spectra ({config.max_spectra}) must be a multiple of "
                    f"batch_size ({config.batch_size}); extraction processes "
                    "whole DataLoader batches."
                )

        LOG.info("Loading model from %s", config.model_path)
        # InstaNovo.load returns (model, config); residue_set lives on the model.
        self.model, self._model_config, self.residue_set = instanovo_io.load_instanovo(
            config.model_path, device=config.device,
        )

        # Hook-based capture reads model.encoder.layers[N] outputs; the
        # flash-attention path does not run that standard encoder stack, so the
        # hooks would never fire. Require a standard-attention checkpoint.
        if getattr(self.model, "use_flash_attention", False):
            raise RuntimeError(
                "Activation extraction hooks the standard nn.TransformerEncoder "
                "layers, but this checkpoint uses flash attention. Load a "
                "non-flash InstaNovo checkpoint for SAE extraction."
            )

        LOG.info("Loading dataset from %s", config.dataset_path)
        # SpectrumDataFrame supports both index access (for per-spectrum
        # metadata at fixed chunk boundaries) and to_dataset() (for the loader).
        self.dataset = instanovo_io.load_spectrum_dataframe(
            config.dataset_path, annotated=True,
        )
        self.loader = instanovo_io.make_dataloader(
            self.dataset,
            self.residue_set,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            metadata_columns=[
                "spectrum_id",
                "sequence",
                "modified_sequence",
                "precursor_mz",
                "precursor_charge",
            ],
        )

    # ─── chunk files & resume ────────────────────────────────────────────────
    def _chunk_files(self, chunk_idx: int) -> list[Path]:
        """All files that make up a complete chunk: the meta file plus one
        activation file per target layer."""
        files = [chunk_meta_path(self.chunks_dir, chunk_idx)]
        files += [
            chunk_acts_path(self.chunks_dir, chunk_idx, L)
            for L in self.config.target_layers
        ]
        return files

    def _is_chunk_done(self, chunk_idx: int) -> bool:
        return self.config.resume and all(p.exists() for p in self._chunk_files(chunk_idx))

    # ─── per-batch processing ───────────────────────────────────────────────
    def _process_batch(
        self,
        batch: dict,
        capture: MultiLayerCapture,
    ) -> dict:
        """Run one forward pass (captures + optional baseline) and return
        per-token flattened activations plus the per-spectrum PROCESSED peaks
        that align one-to-one with the peak tokens.
        """
        device = self.config.device

        # A forward pass through InstaNovo fires the encoder-layer hooks. With
        # save_baseline we run the full model (encoder + decoder) and keep the
        # logits-derived top-1/CE; otherwise we run the encoder only.
        if self.config.save_baseline:
            baseline = compute_baseline(self.model, batch, device)
        else:
            with torch.no_grad():
                # InstaNovo._encoder embeds the peaks, prepends the latent token
                # and runs self.encoder (firing the layer hooks). Calling
                # self.model.encoder directly would bypass the peak embedding and
                # latent prepend and feed raw [B, P, 2] into the transformer.
                self.model._encoder(
                    x=batch["spectra"].to(device),
                    p=batch["precursors"].to(device),
                    x_mask=batch["spectra_mask"].to(device),
                )
            baseline = None
        captured = capture.get_captured()

        spectra_mask_padding = batch["spectra_mask"]  # True-where-padding, [B, n_peaks]
        valid_mask = build_valid_mask(spectra_mask_padding.to(device))

        flat_acts, token_to_spectrum, token_to_position = flatten_per_token(
            captured, valid_mask, dtype=self.config.dtype,
        )

        # Per-spectrum PROCESSED peaks, taken from the model's own input tensor so
        # they align exactly with the peak tokens. batch["spectra"] is [B, P, 2]
        # with column 0 = m/z and column 1 = intensity, already filtered, capped
        # at n_peaks, precursor-removed and rescaled by the data processor. Peak
        # token (j+1) of a spectrum is its j-th valid (non-padded) row here.
        spectra = batch["spectra"]                     # CPU, [B, P, 2]
        peak_valid = ~spectra_mask_padding             # CPU, [B, P] True-where-valid
        mz_per_spectrum, intensity_per_spectrum = [], []
        for b in range(spectra.size(0)):
            pv = peak_valid[b]
            mz_per_spectrum.append(spectra[b, pv, 0].clone().float())
            intensity_per_spectrum.append(spectra[b, pv, 1].clone().float())

        # Consistency guard: the captured activations must have one row per token.
        # This catches a hook capturing the wrong sequence length (e.g. if it
        # ever saw the post-precursor-prepend tensor of length n_peaks+2 instead
        # of the latent+peaks tensor of length n_peaks+1), which would silently
        # desync activations from token_to_position downstream.
        n_tokens = int(token_to_spectrum.size(0))
        for layer_idx, act in flat_acts.items():
            if act.size(0) != n_tokens:
                raise RuntimeError(
                    f"Activation/token desync at layer {layer_idx}: "
                    f"{act.size(0)} activation rows vs {n_tokens} tokens. The "
                    "captured encoder sequence length disagrees with the spectra "
                    "mask (expected latent + n_peaks)."
                )

        return {
            "flat_acts": flat_acts,
            "token_to_spectrum": token_to_spectrum,
            "token_to_position": token_to_position,
            "valid_mask": valid_mask.cpu(),
            "baseline": baseline,
            "mz_per_spectrum": mz_per_spectrum,
            "intensity_per_spectrum": intensity_per_spectrum,
            "batch_metadata": {
                key: batch[key]
                for key in ("spectrum_id", "sequence", "modified_sequence", "precursor_mz", "precursor_charge")
                if key in batch
            },
        }

    def _build_chunk(
        self,
        chunk_idx: int,
        spectrum_start: int,
        accumulated: list[dict],
    ) -> tuple[ChunkMeta, dict[int, torch.Tensor]]:
        """Combine per-batch accumulator outputs into a ChunkMeta plus the
        per-layer activation tensors (written to separate files by _save_chunk).
        """
        # Per-layer concatenation along the token dimension.
        layer_acts = {L: torch.cat([a["flat_acts"][L] for a in accumulated], dim=0)
                      for L in self.config.target_layers}

        token_to_spectrum_parts = []
        token_to_position_parts = []
        running_spectrum_offset = 0
        for a in accumulated:
            # Re-base spectrum indices so they're global within the chunk.
            token_to_spectrum_parts.append(a["token_to_spectrum"] + running_spectrum_offset)
            token_to_position_parts.append(a["token_to_position"])
            running_spectrum_offset += a["valid_mask"].size(0)

        token_to_spectrum = torch.cat(token_to_spectrum_parts, dim=0)
        token_to_position = torch.cat(token_to_position_parts, dim=0)

        # Spectra masks vary in S+1 length across batches; pad to a common
        # length for storage.
        max_seq_len = max(a["valid_mask"].size(1) for a in accumulated)
        padded_masks = []
        for a in accumulated:
            m = a["valid_mask"]
            if m.size(1) < max_seq_len:
                pad = torch.zeros(m.size(0), max_seq_len - m.size(1), dtype=torch.bool)
                m = torch.cat([m, pad], dim=1)
            padded_masks.append(m)
        spectra_mask = torch.cat(padded_masks, dim=0)

        # Baseline: concatenate along spectrum dim if present in every batch.
        # Dynamically pad the decoder length (dim=1) across batches since
        # InstaNovo's DataLoader collator pads to the batch's max target length.
        if self.config.save_baseline:
            max_dec_len = max(a["baseline"]["top1"].size(1) for a in accumulated)

            def pad_dec(tensors, pad_value):
                padded = []
                for t in tensors:
                    if t.size(1) < max_dec_len:
                        pad_shape = (t.size(0), max_dec_len - t.size(1))
                        pad_tensor = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
                        t = torch.cat([t, pad_tensor], dim=1)
                    padded.append(t)
                return torch.cat(padded, dim=0)

            baseline_top1 = pad_dec([a["baseline"]["top1"] for a in accumulated], 0)
            baseline_ce = pad_dec([a["baseline"]["ce"] for a in accumulated], 0.0)
            baseline_decoder_mask = pad_dec([a["baseline"]["decoder_mask"] for a in accumulated], False)
        else:
            baseline_top1 = baseline_ce = baseline_decoder_mask = None

        # Per-spectrum PROCESSED peaks, gathered from the batches in order so
        # they align one-to-one with the peak tokens (NOT re-read from the raw
        # dataset, which would give a different peak set and break the join).
        mz_arrays = [m for a in accumulated for m in a["mz_per_spectrum"]]
        intensity_arrays = [it for a in accumulated for it in a["intensity_per_spectrum"]]

        # Textual + precursor metadata from the dataset for this chunk's range.
        n_spectra = spectra_mask.size(0)
        meta_fields = self._collect_metadata(spectrum_start, n_spectra, accumulated)

        total_tokens = int(token_to_spectrum.size(0))

        # One processed-peak array per spectrum; every layer aligns with tokens.
        assert len(mz_arrays) == n_spectra, (
            f"peak-array count {len(mz_arrays)} != n_spectra {n_spectra}"
        )
        for L, act in layer_acts.items():
            assert act.size(0) == total_tokens, (
                f"layer {L} has {act.size(0)} rows != total_tokens {total_tokens}"
            )

        meta = ChunkMeta(
            schema_version=SCHEMA_VERSION,
            chunk_idx=chunk_idx,
            n_spectra=n_spectra,
            total_tokens=total_tokens,
            target_layers=list(self.config.target_layers),
            spectrum_ids=meta_fields["spectrum_ids"],
            peptides=meta_fields["peptides"],
            proforma_strings=meta_fields["proforma_strings"],
            modifications=meta_fields["modifications"],
            precursor_charges=meta_fields["precursor_charges"],
            precursor_mzs=meta_fields["precursor_mzs"],
            mz_arrays=mz_arrays,
            intensity_arrays=intensity_arrays,
            spectra_mask=spectra_mask,
            token_to_spectrum=token_to_spectrum,
            token_to_position=token_to_position,
            baseline_top1=baseline_top1,
            baseline_ce=baseline_ce,
            baseline_decoder_mask=baseline_decoder_mask,
        )
        return meta, layer_acts

    def _save_chunk(
        self,
        chunk_idx: int,
        meta: ChunkMeta,
        layer_acts: dict[int, torch.Tensor],
    ) -> None:
        """Write one activation file per layer, then the meta file.

        _is_chunk_done requires the meta and every per-layer file to exist, so an
        interrupted write is re-run rather than treated as complete. Writing the
        meta last reinforces this: it is the final file to appear, so a chunk
        with a meta file always has all its activations too.
        """
        for L, act in layer_acts.items():
            save_layer_activations(
                act, chunk_acts_path(self.chunks_dir, chunk_idx, L), chunk_idx, L,
            )
        meta.save(chunk_meta_path(self.chunks_dir, chunk_idx))

    def _collect_metadata(
        self,
        start: int,
        n_spectra: int,
        accumulated: list[dict] | None = None,
    ) -> dict:
        """Pull textual + precursor metadata for this chunk's spectra from the
        dataset by global index.

        ms_ninespecies_benchmark schema (verified against the dataset card):
          sequence           bare peptide, e.g. "YGPHTMAGDDPTK"
          modified_sequence  label with parenthesised mass deltas,
                             e.g. "YGPHTM(+15.99)AGDDPTK"
          precursor_mz, precursor_charge, mz_array, intensity_array

        Peaks are NOT read here -- they come from the processed batch tensors so
        they align with the tokens. Spectrum order is deterministic (DataLoader
        shuffle=False, process_dataset preserves row order), so global index i
        is the i-th spectrum the loader yielded; this requires chunk_size to be a
        multiple of batch_size (enforced in __init__).
        """
        end = start + n_spectra
        dataset_len = len(self.dataset)
        if start < 0 or end > dataset_len:
            raise IndexError(
                f"Chunk metadata range [{start}, {end}) is outside dataset length "
                f"{dataset_len}. This indicates resume/chunk accounting drift."
            )

        spectrum_ids, peptides, proforma_strings = [], [], []
        modifications = []
        precursor_charges, precursor_mzs = [], []
        observed: dict[str, list] = {}
        if accumulated:
            for key in ("spectrum_id", "sequence", "modified_sequence", "precursor_mz", "precursor_charge"):
                vals = []
                for a in accumulated:
                    batch_meta = a.get("batch_metadata", {})
                    if key not in batch_meta:
                        vals = []
                        break
                    vals.extend(batch_meta[key])
                if vals:
                    if all(v is None for v in vals):
                        continue
                    if any(v is None for v in vals):
                        raise RuntimeError(
                            f"Loader metadata field {key!r} is missing for some "
                            "spectra. Cannot verify chunk alignment."
                        )
                    if len(vals) != n_spectra:
                        raise RuntimeError(
                            f"Loader metadata field {key!r} has {len(vals)} values for "
                            f"{n_spectra} spectra. Cannot verify chunk alignment."
                        )
                    observed[key] = vals

        for local_idx, i in enumerate(range(start, end)):
            entry = self.dataset[i]
            spectrum_id = str(entry.get("spectrum_id", f"row_{i}"))
            if "spectrum_id" in observed and str(observed["spectrum_id"][local_idx]) != spectrum_id:
                raise RuntimeError(
                    f"DataLoader/dataset order mismatch at global row {i}: "
                    f"loader spectrum_id={observed['spectrum_id'][local_idx]!r}, "
                    f"dataset spectrum_id={spectrum_id!r}."
                )
            spectrum_ids.append(spectrum_id)

            # `sequence` is the bare peptide; `modified_sequence` carries the
            # mods. Fall back to other common names, then to deriving the bare
            # sequence from the (ProForma-normalised) label.
            bare = entry.get("sequence") or entry.get("peptide") or ""
            modseq = entry.get("modified_sequence") or entry.get("proforma") or bare
            if "sequence" in observed and str(observed["sequence"][local_idx]) != str(bare):
                raise RuntimeError(
                    f"DataLoader/dataset sequence mismatch at global row {i}: "
                    f"loader={observed['sequence'][local_idx]!r}, dataset={bare!r}."
                )
            if "modified_sequence" in observed and str(observed["modified_sequence"][local_idx]) != str(modseq):
                raise RuntimeError(
                    f"DataLoader/dataset modified_sequence mismatch at global row {i}: "
                    f"loader={observed['modified_sequence'][local_idx]!r}, dataset={modseq!r}."
                )
            proforma = _to_proforma(str(modseq))
            bare = str(bare) if bare else _bare_sequence(proforma)

            peptides.append(bare)
            proforma_strings.append(proforma)
            modifications.append(parse_proforma_modifications(proforma))

            charge = int(entry.get("precursor_charge", 0))
            mz = float(entry.get("precursor_mz", 0.0))
            if "precursor_charge" in observed and int(observed["precursor_charge"][local_idx]) != charge:
                raise RuntimeError(
                    f"DataLoader/dataset precursor_charge mismatch at global row {i}: "
                    f"loader={observed['precursor_charge'][local_idx]!r}, dataset={charge!r}."
                )
            if "precursor_mz" in observed and abs(float(observed["precursor_mz"][local_idx]) - mz) > 1e-5:
                raise RuntimeError(
                    f"DataLoader/dataset precursor_mz mismatch at global row {i}: "
                    f"loader={observed['precursor_mz'][local_idx]!r}, dataset={mz!r}."
                )
            precursor_charges.append(charge)
            precursor_mzs.append(mz)

        return {
            "spectrum_ids": spectrum_ids,
            "peptides": peptides,
            "proforma_strings": proforma_strings,
            "modifications": modifications,
            "precursor_charges": torch.tensor(precursor_charges, dtype=torch.long),
            "precursor_mzs": torch.tensor(precursor_mzs, dtype=torch.float32),
        }

    # ─── resume helper ───────────────────────────────────────────────────────
    def _drain_spectra(self, loader_iter, n_spectra: int, chunk_idx: int) -> int:
        """Advance loader_iter by exactly n_spectra without processing them.

        The count comes from the saved ChunkMeta, so final short chunks and
        max_spectra smoke runs resume exactly instead of assuming chunk_size.
        """
        drained = 0
        while drained < n_spectra:
            try:
                batch = next(loader_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"DataLoader exhausted while draining completed chunk {chunk_idx}: "
                    f"drained {drained}/{n_spectra} spectra. Existing chunk files "
                    "do not match this dataset/configuration."
                ) from exc

            bsz = int(batch["spectra"].size(0))
            if drained + bsz > n_spectra:
                raise RuntimeError(
                    f"Completed chunk {chunk_idx} has n_spectra={n_spectra}, but "
                    f"draining the next DataLoader batch of {bsz} would overshoot "
                    f"from {drained}. Existing chunk files do not align with the "
                    "current DataLoader batch boundaries."
                )
            drained += bsz
        return drained

    # ─── public entry points ────────────────────────────────────────────────
    def __iter__(self) -> Iterator[ChunkMeta]:
        """Yield each chunk's ChunkMeta one at a time, writing the meta file and
        per-layer activation files to disk as a side effect.

        Uses an explicit iterator (loader_iter) rather than a for-loop so that
        skipped chunks can drain the DataLoader by the exact number of batches
        they would have consumed. The for-loop `continue` that the old code used
        left the iterator pointing at the start of the skipped chunk's spectra,
        causing every chunk that followed a resumed skip to receive the wrong
        spectra's activations paired with the correct chunk's metadata -- a silent
        misalignment that corrupted all downstream training and evaluation.
        """
        with MultiLayerCapture(self.model, self.config.target_layers) as capture:
            chunk_idx = 0
            accumulated: list[dict] = []
            spectra_in_chunk = 0
            total_seen = 0

            t0 = time.time()
            loader_iter = iter(self.loader)

            while True:
                # ── Resume: skip completed chunks, draining the loader ──────
                # Check before pulling the next batch so we never process a
                # batch that belongs to a skipped chunk.
                if self._is_chunk_done(chunk_idx) and spectra_in_chunk == 0:
                    done_meta = ChunkMeta.load(chunk_meta_path(self.chunks_dir, chunk_idx))
                    LOG.info(
                        "Chunk %d already exists -- draining %d spectra and skipping",
                        chunk_idx, done_meta.n_spectra,
                    )
                    drained = self._drain_spectra(loader_iter, done_meta.n_spectra, chunk_idx)
                    total_seen += drained
                    chunk_idx += 1
                    if self.config.max_spectra and total_seen >= self.config.max_spectra:
                        break
                    continue

                # ── Pull the next batch ─────────────────────────────────────
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break

                bsz = batch["spectra"].size(0)

                processed = self._process_batch(batch, capture)
                accumulated.append(processed)
                spectra_in_chunk += bsz
                total_seen += bsz

                if spectra_in_chunk >= self.config.chunk_size:
                    chunk_start = total_seen - spectra_in_chunk
                    meta, layer_acts = self._build_chunk(chunk_idx, chunk_start, accumulated)
                    self._save_chunk(chunk_idx, meta, layer_acts)
                    LOG.info(
                        "Wrote chunk %d (n_spectra=%d, total_tokens=%d, layers=%s, elapsed=%.1fs)",
                        chunk_idx, meta.n_spectra, meta.total_tokens,
                        meta.target_layers, time.time() - t0,
                    )
                    yield meta
                    chunk_idx += 1
                    accumulated.clear()
                    spectra_in_chunk = 0

                if self.config.max_spectra and total_seen >= self.config.max_spectra:
                    break

            # Flush partial final chunk.
            if accumulated:
                chunk_start = total_seen - spectra_in_chunk
                meta, layer_acts = self._build_chunk(chunk_idx, chunk_start, accumulated)
                self._save_chunk(chunk_idx, meta, layer_acts)
                LOG.info("Wrote final chunk %d (n_spectra=%d)", chunk_idx, meta.n_spectra)
                yield meta

    def extract_all(self) -> None:
        """Materialise every chunk to disk. Updates the manifest at the end."""
        n_chunks = 0
        n_spectra = 0
        n_tokens = 0
        for meta in self:
            n_chunks += 1
            n_spectra += meta.n_spectra
            n_tokens += meta.total_tokens
        self._write_manifest(n_chunks, n_spectra, n_tokens)
        LOG.info(
            "Extraction complete: %d chunks, %d spectra, %d tokens",
            n_chunks, n_spectra, n_tokens,
        )

    def _write_manifest(self, n_chunks: int, n_spectra: int, n_tokens: int) -> None:
        def rel(p: Path) -> str:
            return str(p.relative_to(self.output_dir))

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.as_jsonable(),
            "n_chunks": n_chunks,
            "n_spectra": n_spectra,
            "n_tokens": n_tokens,
            "target_layers": list(self.config.target_layers),
            "chunks": [
                {
                    "idx": i,
                    "meta": rel(chunk_meta_path(self.chunks_dir, i)),
                    "activations": {
                        str(L): rel(chunk_acts_path(self.chunks_dir, i, L))
                        for L in self.config.target_layers
                    },
                }
                for i in range(n_chunks)
            ],
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2))
        LOG.info("Wrote manifest to %s", self.manifest_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help="Storage dtype for activations (bfloat16 ~halves disk).")
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip caching baseline top-1 and CE.")
    p.add_argument("--no-resume", action="store_true",
                   help="Force rebuild even if chunks already exist.")
    p.add_argument("--max-spectra", type=int, default=None,
                   help="Cap total spectra processed (for smoke tests).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    config = ExtractionConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        target_layers=tuple(sorted(args.layers)),
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        dtype=dtype_map[args.dtype],
        save_baseline=not args.no_baseline,
        resume=not args.no_resume,
        max_spectra=args.max_spectra,
    )

    extractor = ActivationExtractor(config)
    extractor.extract_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
