"""Multi-layer activation extraction for InstaNovo.

Runs InstaNovo once over a spectrum dataset and caches the encoder activations
needed to train and evaluate sparse autoencoders, so the expensive forward pass
is paid once and reused across every layer, seed and evaluation.

Design
    Hooks on model.encoder.layers[N] capture every target layer in a single
    forward pass. Within a spectrum the latent summary token is position 0 and
    the peak tokens follow.

    Output is split by layer: activations go to one file per (layer, chunk) and
    metadata to one file per chunk, so a per-layer SAE training run reads only
    its own layer's bytes and the annotator reads no activations at all.

    ChunkMeta is the single source of truth for per-spectrum fields (mask,
    ProForma, bare peptide, modifications, processed m/z + intensity, precursor
    info, cached baseline top-1 / CE); no consumer re-derives them.

Resume
    Chunks are written whole and skipped if already present, so an interrupted
    run continues where it stopped. Skipped chunks still advance the DataLoader
    by their exact spectrum count -- otherwise every later chunk would pair the
    wrong spectra's activations with the right chunk's metadata.

    Reuse is only safe while the config that produced those chunks still holds,
    so extract_config.json records the content-critical fields and a resume that
    disagrees with it is refused.

Output layout under --output-dir
    manifest.json              run config + chunk index (meta and per-layer paths)
    extract_config.json        content-critical config, checked on resume
    chunks/
        meta_00000.pt          ChunkMeta: token maps, masks, peptide/proforma,
                               modifications, peaks, precursor, baseline
        acts_L2_00000.pt       one [total_tokens, d_model] tensor per (layer, chunk)
        acts_L4_00000.pt
        ...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.hooks import RemovableHandle

# InstaNovo bindings go through the single integration layer (instanovo_io),
# which is the one place that talks to the real repo API -- including the
# InstaNovo type used in annotations below. Run from the repo (or with the repo
# on PYTHONPATH) so both `instanovo` and `instanovo_io` import.
import instanovo_io
from schema import EXTRACT_SCHEMA_VERSION

LOG = logging.getLogger("extract")


def atomic_torch_save(obj, path: Path) -> None:
    """torch.save to a temp file in the same directory, then rename into place.

    Every resume decision in this pipeline is an existence check -- extract's
    _is_chunk_done, annotate's out_path.exists(), the shell's sentinel tests --
    so a file that exists is assumed complete. A process killed mid-write
    (OOM-kill, cluster preemption, Ctrl-C) would otherwise leave a truncated
    file that the next run silently accepts as a finished chunk. os.replace is
    atomic within a filesystem, so the final path only ever names a whole file.

    Small enough to duplicate in annotate.py rather than share, for the same
    reason _load_layer_activations is duplicated in train.py: the annotator must
    not have to import this model-dependent module.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

# Fields the DataLoader carries alongside each batch so loader order can be
# checked against dataset order (see _cross_check_row).
#
# How many survive depends on the InstaNovo version: the published collate_fn
# passes metadata through untouched, while variants that tensorise it for
# multi-GPU broadcast drop the string columns. Each field is therefore used
# only if present, _gather_loader_metadata raises if none are, and the surviving
# set is logged once per run so the level of protection is visible.
METADATA_COLUMNS = (
    "spectrum_id",
    "sequence",
    "modified_sequence",
    "precursor_mz",
    "precursor_charge",
)

# Tolerance for comparing a loader-reported precursor m/z against the dataset's.
MZ_MATCH_TOLERANCE = 1e-5

# Config fields a resumed run must match, because each changes chunk CONTENT:
# mixing them yields a silently inconsistent dataset (a different n_peaks gives
# some chunks 201 tokens per spectrum and others 101, joined as if uniform).
# batch_size and chunk_size are excluded -- they only move where boundaries
# fall, and resuming with a smaller batch after an OOM is legitimate.
RESUME_CRITICAL_FIELDS = (
    "model_path",
    "dataset_path",
    "n_peaks",
    "dtype",
    "save_baseline",
)

# Holds RESUME_CRITICAL_FIELDS, written at start-up. manifest.json cannot serve
# this purpose: it appears only once a run completes, and an interrupted run is
# exactly the case that needs checking.
RESUME_FINGERPRINT_NAME = "extract_config.json"


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
    # Peaks kept per spectrum. Recorded here so it reaches the manifest: it sets
    # how many peak tokens each spectrum contributes, so every later consumer
    # (annotation labels, SAE activations, the Phase 7/8 re-run loader) must use
    # the same value or the per-token join silently misaligns.
    n_peaks: int = instanovo_io.DEFAULT_N_PEAKS
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

    def resume_fingerprint(self) -> dict:
        """The content-critical subset of the config a resumed run must match."""
        full = self.as_jsonable()
        return {key: full[key] for key in RESUME_CRITICAL_FIELDS}

    def validate(self) -> None:
        """Reject configurations that would break chunk/dataset index alignment.

        Chunk boundaries align with dataset indices only if every non-final
        chunk holds exactly chunk_size spectra, which requires chunk_size to be
        a whole number of batches. _collect_metadata looks spectra up by global
        index, so drift here silently pairs the wrong metadata with a chunk.
        """
        if self.chunk_size % self.batch_size != 0:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be a multiple of "
                f"batch_size ({self.batch_size}) so chunk boundaries align "
                "with dataset indices used for metadata lookup."
            )
        if self.max_spectra is None:
            return
        if self.max_spectra <= 0:
            raise ValueError("max_spectra must be positive when set.")
        if self.max_spectra % self.batch_size != 0:
            raise ValueError(
                f"max_spectra ({self.max_spectra}) must be a multiple of "
                f"batch_size ({self.batch_size}); extraction processes "
                "whole DataLoader batches."
            )


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
        # Shallow field dict rather than dataclasses.asdict: asdict deep-copies
        # every value, duplicating this chunk's tensors in RAM while the batch
        # activations are still live, to write a byte-identical file.
        atomic_torch_save(dict(self.__dict__), path)

    @classmethod
    def load(cls, path: Path) -> ChunkMeta:
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


# --- On-disk layout -----------------------------------------------------------
# The single definition of the file naming, imported by the train / annotate /
# evaluate consumers so the contract lives in one place.

def chunk_meta_path(chunks_dir: Path, chunk_idx: int) -> Path:
    return chunks_dir / f"meta_{chunk_idx:05d}.pt"


def chunk_acts_path(chunks_dir: Path, chunk_idx: int, layer: int) -> Path:
    return chunks_dir / f"acts_L{layer}_{chunk_idx:05d}.pt"


def save_layer_activations(
    acts: torch.Tensor, path: Path, chunk_idx: int, layer: int
) -> None:
    """Write one layer's [total_tokens, d_model] activation tensor, wrapped with
    a tiny header so consumers can validate chunk/layer/token-count alignment."""
    atomic_torch_save(
        {
            "schema_version": EXTRACT_SCHEMA_VERSION,
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


class MultiLayerCapture:
    """Register forward hooks on multiple encoder layers and capture their outputs.

    One instance is reused across the whole extraction run; hooks stay registered
    for the lifetime of the context manager. Each forward pass overwrites the
    captured tensors, so callers must consume them between batches.
    """

    def __init__(self, model: instanovo_io.InstaNovo, target_layers: tuple[int, ...]):
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

    def __exit__(self, *_exc) -> None:
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

    def clear(self) -> None:
        """Drop the previous batch's captures.

        Called before each forward so a hook that fails to fire raises in
        get_captured() rather than returning stale activations, which would be
        paired with this batch's tokens.
        """
        self._captured.clear()

    def get_captured(self) -> dict[int, torch.Tensor]:
        """Return the most recent forward pass's activations, one tensor per layer."""
        if set(self._captured) != set(self.target_layers):
            missing = set(self.target_layers) - set(self._captured)
            raise RuntimeError(f"Hooks did not fire for layers: {missing}")
        return self._captured


# --- Per-batch tensor helpers -------------------------------------------------

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
        # The capture must have one position per mask column (latent + peaks).
        # A disagreement means the hooked tensor no longer has the token layout
        # the pipeline assumes -- e.g. a capture point seeing the precursor
        # prepended (n_peaks+2). Truncating to the shorter length would keep the
        # row count intact and hide the one-position shift, so this must be a
        # hard error rather than a clamp.
        if act.size(1) != seq_len:
            raise RuntimeError(
                f"Encoder sequence length {act.size(1)} at layer {layer_idx} "
                f"disagrees with the spectra mask ({seq_len} = latent + n_peaks). "
                "The capture point no longer matches the expected token layout."
            )
        flat[layer_idx] = act[valid_mask].to(dtype=dtype, device="cpu")

    return flat, token_to_spectrum, token_to_position


def pad_and_concat(tensors: list[torch.Tensor], pad_value, target_len: int) -> torch.Tensor:
    """Right-pad each [N, L_i] tensor along dim 1 to target_len, then concatenate
    along dim 0. Used for masks and baselines, whose dim-1 length varies across
    batches because the collator pads to each batch's own maximum.
    """
    padded = []
    for t in tensors:
        if t.size(1) < target_len:
            pad_shape = (t.size(0), target_len - t.size(1))
            pad = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


def compute_baseline(
    model: instanovo_io.InstaNovo,
    batch: dict,
    device: str,
) -> dict[str, torch.Tensor]:
    """Compute per-position top-1 amino-acid prediction and cross-entropy.

    Cached at extraction time so a later causal-ablation pass only needs to run
    the ablated forward, not the baseline. This full forward also fires the
    encoder hooks, but the captures are owned by the hook-driven extraction in
    the same pass, not read here.

    Logits have T+1 positions because add_bos=True. The CE / top-1 alignment
    with the peptide targets (drop last position, ignore PAD) lives in
    instanovo_io, the single place that mirrors InstaNovo's training loss.
    """
    with torch.inference_mode():
        logits = instanovo_io.model_forward_logits(model, batch, device)
        ce, top1, _targets, valid = instanovo_io.per_token_ce_and_top1(
            logits, batch["peptides"], pad_index=instanovo_io.PAD_INDEX,
        )

    return {
        "top1": top1.cpu(),
        "ce": ce.cpu(),
        "decoder_mask": valid.cpu(),
    }


# --- ProForma helpers ---------------------------------------------------------

_BRACKET_TAG = re.compile(r"\[([^\]]+)\]")
_UNIMOD_TAG = re.compile(r"UNIMOD:(\d+)", re.IGNORECASE)
_DELTA_MASS_PAREN = re.compile(r"\(([+-])(\d*\.?\d+)\)")


def parse_proforma_modifications(proforma: str) -> list[dict]:
    """Extract modifications from a ProForma string in square-bracket form.

    Expects the bracket notation that _to_proforma produces (e.g. "M[+15.99]"),
    so it is run on the converted string, not the dataset's raw "(+15.99)" form.
    Returns one {position, mod_name, unimod_id} dict per modification; positions
    are 1-indexed into the bare peptide (0 means an N-terminal modification).
    mod_name is the raw bracket content and unimod_id is parsed from a
    "UNIMOD:<n>" tag when present, else -1 (delta-mass mods carry no UNIMOD id).
    """
    modifications: list[dict] = []
    bare_position = 0

    # Walk the string character by character so bare-sequence positions are
    # tracked correctly even with multiple modifications and N/C-terminal tags.
    i = 0
    while i < len(proforma):
        ch = proforma[i]
        if ch == "[":
            match = _BRACKET_TAG.match(proforma, i)
            if match:
                tag = match.group(1)
                unimod_match = _UNIMOD_TAG.search(tag)
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
    no_mods = _BRACKET_TAG.sub("", proforma)
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
    def repl(m: re.Match) -> str:
        sign, num = m.group(1), m.group(2)
        if num.startswith("."):
            num = "0" + num
        return f"[{sign}{num}]"

    return _DELTA_MASS_PAREN.sub(repl, modified_sequence)


# --- Loader/dataset cross-checking -------------------------------------------

def _gather_loader_metadata(accumulated: list[dict], n_spectra: int) -> dict[str, list]:
    """Collect the metadata columns the DataLoader reported for this chunk.

    Returns only fields present and populated for every spectrum -- the ones
    _cross_check_row can meaningfully compare. A field the collator dropped is
    skipped; one that is only partially present means the batches do not line
    up and is an error.

    Raises if nothing survived, since the guard would then report success while
    verifying nothing, which is worse than having no guard at all.
    """
    observed: dict[str, list] = {}
    for key in METADATA_COLUMNS:
        values = []
        for a in accumulated:
            batch_meta = a.get("batch_metadata", {})
            if key not in batch_meta:
                values = []
                break
            values.extend(batch_meta[key])

        if not values or all(v is None for v in values):
            continue
        if any(v is None for v in values):
            raise RuntimeError(
                f"Loader metadata field {key!r} is missing for some "
                "spectra. Cannot verify chunk alignment."
            )
        if len(values) != n_spectra:
            raise RuntimeError(
                f"Loader metadata field {key!r} has {len(values)} values for "
                f"{n_spectra} spectra. Cannot verify chunk alignment."
            )
        observed[key] = values

    if not observed:
        raise RuntimeError(
            "The DataLoader reported none of the metadata columns "
            f"{list(METADATA_COLUMNS)}, so loader/dataset ordering cannot be "
            "verified and activations could be paired with the wrong peptide. "
            "Check that the dataset carries these columns, and that this "
            "InstaNovo version's collate_fn does not drop the metadata it "
            "cannot convert to a tensor."
        )

    return observed


# Extra diagnosis appended to specific mismatch messages. A spectrum_id
# disagreement is the signature of loader/dataset ordering drift, the root cause
# worth naming; other fields disagree for more varied reasons. Reachable only
# while the collator preserves string metadata (see METADATA_COLUMNS).
_MISMATCH_HINT = {
    "spectrum_id": " The loader is not yielding spectra in dataset order.",
}


def _cross_check_row(
    observed: dict[str, list],
    local_idx: int,
    global_row: int,
    dataset_values: dict,
) -> None:
    """Raise if the loader's metadata for a spectrum disagrees with the dataset's.

    Guards the assumption that the DataLoader yields spectra in dataset order,
    which is what makes index-based metadata lookup valid. A mismatch means
    activations would be paired with another spectrum's peptide.
    """
    for key, dataset_value in dataset_values.items():
        if key not in observed:
            continue
        loader_value = observed[key][local_idx]

        if key == "precursor_mz":
            matches = abs(float(loader_value) - float(dataset_value)) <= MZ_MATCH_TOLERANCE
        elif key == "precursor_charge":
            matches = int(loader_value) == int(dataset_value)
        else:
            matches = str(loader_value) == str(dataset_value)

        if not matches:
            raise RuntimeError(
                f"DataLoader/dataset {key} mismatch at global row {global_row}: "
                f"loader={loader_value!r}, dataset={dataset_value!r}."
                + _MISMATCH_HINT.get(key, "")
            )


def _read_dataset_row(entry: dict, global_row: int) -> tuple[dict, dict]:
    """Normalise one dataset row, returning (checkable, derived).

    `sequence` is the bare peptide and `modified_sequence` carries the mods;
    both fall back to other common column names.

    The dicts are separate because the cross-check must compare RAW column
    values (what the loader also saw) while ChunkMeta stores derived ones.
    `peptide` in particular falls back to deriving the bare sequence from the
    ProForma label when `sequence` is empty; checking against that derived value
    would weaken the guard, since the fallback could produce a matching string
    and mask a real loader/dataset disagreement.
    """
    raw_bare = entry.get("sequence") or entry.get("peptide") or ""
    modseq = entry.get("modified_sequence") or entry.get("proforma") or raw_bare
    proforma = _to_proforma(str(modseq))

    checkable = {
        "spectrum_id": str(entry.get("spectrum_id", f"row_{global_row}")),
        "sequence": str(raw_bare),
        "modified_sequence": str(modseq),
        # `or` rather than a .get default: the column may be present but null,
        # in which case .get returns None and int()/float() would raise.
        "precursor_charge": int(entry.get("precursor_charge") or 0),
        "precursor_mz": float(entry.get("precursor_mz") or 0.0),
    }
    derived = {
        "peptide": str(raw_bare) if raw_bare else _bare_sequence(proforma),
        "proforma": proforma,
    }
    return checkable, derived


class ActivationExtractor:
    """Drives a full extraction run. Iterable for streaming consumption,
    or call extract_all() to write every chunk to disk.
    """

    def __init__(self, config: ExtractionConfig):
        config.validate()
        self.config = config
        self.output_dir = config.output_dir
        self.chunks_dir = self.output_dir / "chunks"
        self.manifest_path = self.output_dir / "manifest.json"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        # One-shot log of which fields the loader/dataset cross-check can use.
        self._logged_cross_check_fields = False

        self._check_resume_fingerprint()
        self._load_model()
        self._load_data()

    def _check_resume_fingerprint(self) -> None:
        """Reject a resume whose config would produce chunks incompatible with
        those already on disk, then record this run's config for the next one.

        Chunk reuse is gated on file existence alone (_is_chunk_done), so
        without this a resume under a different n_peaks / dtype / dataset
        silently mixes incompatible chunks -- the failure every downstream
        schema check guards against, but which extraction could not previously
        detect in its own artefacts.
        """
        path = self.output_dir / RESUME_FINGERPRINT_NAME
        current = self.config.resume_fingerprint()

        # No fingerprint beside existing chunks means they predate this check:
        # nothing to compare, so proceed and record the config for next time.
        if path.exists() and self.config.resume:
            previous = json.loads(path.read_text())
            differing = {
                key: (previous.get(key), value)
                for key, value in current.items()
                if previous.get(key) != value
            }
            if differing and any(self.chunks_dir.glob("meta_*.pt")):
                detail = "; ".join(
                    f"{key}: existing={old!r}, requested={new!r}"
                    for key, (old, new) in sorted(differing.items())
                )
                raise RuntimeError(
                    f"Existing chunks in {self.chunks_dir} were extracted with a "
                    f"different configuration ({detail}). Resuming would mix "
                    "incompatible chunks. Re-run with --no-resume to rebuild, or "
                    "point --output-dir at a fresh directory."
                )

        path.write_text(json.dumps(current, indent=2, sort_keys=True))

    def _validate_resumed_chunk(self, meta: ChunkMeta, chunk_idx: int) -> None:
        """Reject a cached chunk this build cannot safely reuse."""
        if meta.schema_version != EXTRACT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Chunk {chunk_idx} has extract schema {meta.schema_version}, but "
                f"this build writes {EXTRACT_SCHEMA_VERSION}. Re-run with --no-resume."
            )
        missing = sorted(set(self.config.target_layers) - set(meta.target_layers))
        if missing:
            raise RuntimeError(
                f"Chunk {chunk_idx} was extracted for layers "
                f"{sorted(meta.target_layers)}, which does not cover requested "
                f"layers {missing}. Re-run with --no-resume."
            )

    def _load_model(self) -> None:
        """Load InstaNovo and reject checkpoints the capture hooks cannot read."""
        config = self.config
        LOG.info("Loading model from %s", config.model_path)
        # InstaNovo.load returns (model, config); residue_set lives on the model.
        self.model, _config, self.residue_set = instanovo_io.load_instanovo(
            config.model_path, device=config.device,
        )

        # Hook-based capture reads model.encoder.layers[N] outputs; the
        # flash-attention path does not run that standard encoder stack, so the
        # hooks would never fire. Require a standard-attention checkpoint.
        if instanovo_io.uses_flash_attention(self.model):
            raise RuntimeError(
                "Activation extraction hooks the standard nn.TransformerEncoder "
                "layers, but this checkpoint uses flash attention. Load a "
                "non-flash InstaNovo checkpoint for SAE extraction."
            )

    def _load_data(self) -> None:
        """Open the dataset and build the DataLoader used for the forward pass."""
        config = self.config
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
            n_peaks=config.n_peaks,
            metadata_columns=list(METADATA_COLUMNS),
        )

    # --- chunk files & resume -------------------------------------------------

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

    # --- per-batch processing -------------------------------------------------

    def _run_forward(self, batch: dict) -> dict | None:
        """Run the forward pass that fires the capture hooks.

        With save_baseline we run the full model (encoder + decoder) and keep the
        logits-derived top-1/CE; otherwise the encoder alone is enough.
        """
        if self.config.save_baseline:
            return compute_baseline(self.model, batch, self.config.device)

        device = self.config.device
        with torch.inference_mode():
            # _encoder embeds the peaks, prepends the latent token and runs
            # self.encoder (firing the hooks). Calling self.model.encoder
            # directly would skip the embedding and latent prepend, feeding raw
            # [B, P, 2] into the transformer. non_blocking is safe because
            # make_dataloader sets pin_memory=True.
            self.model._encoder(
                x=batch["spectra"].to(device, non_blocking=True),
                p=batch["precursors"].to(device, non_blocking=True),
                x_mask=batch["spectra_mask"].to(device, non_blocking=True),
            )
        return None

    @staticmethod
    def _split_processed_peaks(
        batch: dict, spectra_mask_padding: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per-spectrum processed m/z and intensity arrays, taken from the model's
        own input tensor so they align exactly with the peak tokens.

        batch["spectra"] is [B, P, 2] with column 0 = m/z and column 1 =
        intensity, already filtered, capped at n_peaks, precursor-removed and
        rescaled by the data processor. Peak token (j+1) of a spectrum is its
        j-th valid (non-padded) row here.
        """
        spectra = batch["spectra"]          # CPU, [B, P, 2]
        peak_valid = ~spectra_mask_padding  # CPU, [B, P] True-where-valid

        mz_per_spectrum, intensity_per_spectrum = [], []
        for b in range(spectra.size(0)):
            pv = peak_valid[b]
            mz_per_spectrum.append(spectra[b, pv, 0].clone().float())
            intensity_per_spectrum.append(spectra[b, pv, 1].clone().float())
        return mz_per_spectrum, intensity_per_spectrum

    @staticmethod
    def _assert_token_alignment(flat_acts: dict[int, torch.Tensor], n_tokens: int) -> None:
        """Verify every layer contributed exactly one activation row per token.

        flatten_per_token already rejects a sequence-length mismatch; this is
        the final row-count invariant, catching anything that slips past it
        before the rows reach disk as a silent activation/token misalignment.
        """
        for layer_idx, act in flat_acts.items():
            if act.size(0) != n_tokens:
                raise RuntimeError(
                    f"Activation/token desync at layer {layer_idx}: "
                    f"{act.size(0)} activation rows vs {n_tokens} tokens. The "
                    "captured encoder sequence length disagrees with the spectra "
                    "mask (expected latent + n_peaks)."
                )

    def _process_batch(self, batch: dict, capture: MultiLayerCapture) -> dict:
        """Run one forward pass (captures + optional baseline) and return
        per-token flattened activations plus the per-spectrum PROCESSED peaks
        that align one-to-one with the peak tokens.
        """
        capture.clear()
        baseline = self._run_forward(batch)
        captured = capture.get_captured()

        spectra_mask_padding = batch["spectra_mask"]  # True-where-padding, [B, n_peaks]
        valid_mask = build_valid_mask(spectra_mask_padding.to(self.config.device, non_blocking=True))

        flat_acts, token_to_spectrum, token_to_position = flatten_per_token(
            captured, valid_mask, dtype=self.config.dtype,
        )

        mz_per_spectrum, intensity_per_spectrum = self._split_processed_peaks(
            batch, spectra_mask_padding,
        )
        self._assert_token_alignment(flat_acts, int(token_to_spectrum.size(0)))

        return {
            "flat_acts": flat_acts,
            "token_to_spectrum": token_to_spectrum,
            "token_to_position": token_to_position,
            "valid_mask": valid_mask.cpu(),
            "baseline": baseline,
            "mz_per_spectrum": mz_per_spectrum,
            "intensity_per_spectrum": intensity_per_spectrum,
            "batch_metadata": {
                key: batch[key] for key in METADATA_COLUMNS if key in batch
            },
        }

    # --- chunk assembly -------------------------------------------------------

    @staticmethod
    def _concat_token_maps(accumulated: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate the per-batch token maps, re-basing spectrum indices so
        they are global within the chunk rather than local to their batch."""
        spectrum_parts, position_parts = [], []
        offset = 0
        for a in accumulated:
            spectrum_parts.append(a["token_to_spectrum"] + offset)
            position_parts.append(a["token_to_position"])
            offset += a["valid_mask"].size(0)
        return torch.cat(spectrum_parts, dim=0), torch.cat(position_parts, dim=0)

    @staticmethod
    def _concat_spectra_masks(accumulated: list[dict]) -> torch.Tensor:
        """Concatenate the per-batch validity masks, padding to a common S+1
        length since it varies across batches."""
        masks = [a["valid_mask"] for a in accumulated]
        max_seq_len = max(m.size(1) for m in masks)
        return pad_and_concat(masks, False, max_seq_len)

    @staticmethod
    def _concat_baselines(
        accumulated: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate cached baselines along the spectrum dimension, padding the
        decoder length since InstaNovo's collator pads to each batch's own
        maximum target length."""
        max_dec_len = max(a["baseline"]["top1"].size(1) for a in accumulated)
        fields = {"top1": 0, "ce": 0.0, "decoder_mask": False}
        return tuple(
            pad_and_concat([a["baseline"][name] for a in accumulated], pad, max_dec_len)
            for name, pad in fields.items()
        )

    def _build_chunk(
        self,
        chunk_idx: int,
        spectrum_start: int,
        accumulated: list[dict],
    ) -> tuple[ChunkMeta, dict[int, torch.Tensor]]:
        """Combine per-batch accumulator outputs into a ChunkMeta plus the
        per-layer activation tensors (written to separate files by _save_chunk).

        Consumes `accumulated`'s activation slices, so the caller must not reuse
        them afterwards -- _flush_chunk is the only caller and clears the list
        immediately after.
        """
        # Concatenate one layer at a time, dropping each batch's slice as it is
        # consumed. These are the largest objects in the run, so holding every
        # batch's slices alongside all four concatenated tensors would double
        # peak memory; popping caps the overhead at one layer's worth.
        layer_acts = {}
        for L in self.config.target_layers:
            layer_acts[L] = torch.cat([a["flat_acts"].pop(L) for a in accumulated], dim=0)

        token_to_spectrum, token_to_position = self._concat_token_maps(accumulated)
        spectra_mask = self._concat_spectra_masks(accumulated)

        if self.config.save_baseline:
            baseline_top1, baseline_ce, baseline_decoder_mask = self._concat_baselines(
                accumulated,
            )
        else:
            baseline_top1 = baseline_ce = baseline_decoder_mask = None

        # Gathered from the batches in order so they align one-to-one with the
        # peak tokens (NOT re-read from the raw dataset, which would give a
        # different peak set and break the join).
        mz_arrays = [m for a in accumulated for m in a["mz_per_spectrum"]]
        intensity_arrays = [it for a in accumulated for it in a["intensity_per_spectrum"]]

        n_spectra = spectra_mask.size(0)
        total_tokens = int(token_to_spectrum.size(0))
        meta_fields = self._collect_metadata(spectrum_start, n_spectra, accumulated)

        assert len(mz_arrays) == n_spectra, (
            f"peak-array count {len(mz_arrays)} != n_spectra {n_spectra}"
        )
        for L, act in layer_acts.items():
            assert act.size(0) == total_tokens, (
                f"layer {L} has {act.size(0)} rows != total_tokens {total_tokens}"
            )

        meta = ChunkMeta(
            schema_version=EXTRACT_SCHEMA_VERSION,
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
        accumulated: list[dict],
    ) -> dict:
        """Pull textual + precursor metadata for this chunk's spectra from the
        dataset by global index.

        Expected columns: `sequence` (bare peptide, "YGPHTMAGDDPTK"),
        `modified_sequence` (parenthesised mass deltas, "YGPHTM(+15.99)AGDDPTK"),
        `precursor_mz` and `precursor_charge`.

        Peaks are NOT read here -- they come from the processed batch tensors so
        they align with the tokens. Lookup by index is valid because spectrum
        order is deterministic (DataLoader shuffle=False, process_dataset
        preserves row order), so global index i is the i-th spectrum the loader
        yielded; this needs chunk_size to be a multiple of batch_size, which
        ExtractionConfig.validate enforces.
        """
        end = start + n_spectra
        dataset_len = len(self.dataset)
        if start < 0 or end > dataset_len:
            raise IndexError(
                f"Chunk metadata range [{start}, {end}) is outside dataset length "
                f"{dataset_len}. This indicates resume/chunk accounting drift."
            )

        observed = _gather_loader_metadata(accumulated, n_spectra)
        if observed and not self._logged_cross_check_fields:
            LOG.info(
                "Verifying loader/dataset alignment on: %s (of requested %s)",
                ", ".join(sorted(observed)), ", ".join(METADATA_COLUMNS),
            )
            self._logged_cross_check_fields = True

        spectrum_ids, peptides, proforma_strings = [], [], []
        modifications, precursor_charges, precursor_mzs = [], [], []

        for local_idx, global_row in enumerate(range(start, end)):
            checkable, derived = _read_dataset_row(self.dataset[global_row], global_row)
            _cross_check_row(observed, local_idx, global_row, checkable)

            spectrum_ids.append(checkable["spectrum_id"])
            peptides.append(derived["peptide"])
            proforma_strings.append(derived["proforma"])
            modifications.append(parse_proforma_modifications(derived["proforma"]))
            precursor_charges.append(checkable["precursor_charge"])
            precursor_mzs.append(checkable["precursor_mz"])

        return {
            "spectrum_ids": spectrum_ids,
            "peptides": peptides,
            "proforma_strings": proforma_strings,
            "modifications": modifications,
            "precursor_charges": torch.tensor(precursor_charges, dtype=torch.long),
            "precursor_mzs": torch.tensor(precursor_mzs, dtype=torch.float32),
        }

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

    def _flush_chunk(
        self, chunk_idx: int, chunk_start: int, accumulated: list[dict]
    ) -> ChunkMeta:
        """Build, write and return one chunk from the accumulated batches."""
        meta, layer_acts = self._build_chunk(chunk_idx, chunk_start, accumulated)
        self._save_chunk(chunk_idx, meta, layer_acts)
        return meta

    # --- public entry points --------------------------------------------------

    def __iter__(self) -> Iterator[ChunkMeta]:
        """Yield every chunk's ChunkMeta in order, including chunks skipped on
        resume -- their meta is read back from disk and yielded without
        rewriting any files. extract_all()'s counts, and therefore the manifest,
        depend on seeing every chunk here rather than only the new ones.

        The explicit loader_iter, rather than a for-loop, is what lets a skipped
        chunk drain the DataLoader by the exact number of batches it would have
        consumed. A for-loop `continue` would leave the iterator at the start of
        the skipped chunk's spectra, so every chunk after a resumed skip would
        pair the wrong spectra's activations with the right metadata.
        """
        with MultiLayerCapture(self.model, self.config.target_layers) as capture:
            chunk_idx = 0
            accumulated: list[dict] = []
            spectra_in_chunk = 0
            total_seen = 0

            t0 = time.time()
            loader_iter = iter(self.loader)

            while True:
                # Resume check comes before pulling the next batch, so we never
                # process a batch that belongs to an already-completed chunk.
                if self._is_chunk_done(chunk_idx) and spectra_in_chunk == 0:
                    done_meta = ChunkMeta.load(chunk_meta_path(self.chunks_dir, chunk_idx))
                    self._validate_resumed_chunk(done_meta, chunk_idx)
                    LOG.info(
                        "Chunk %d already exists -- draining %d spectra and skipping",
                        chunk_idx, done_meta.n_spectra,
                    )
                    total_seen += self._drain_spectra(
                        loader_iter, done_meta.n_spectra, chunk_idx,
                    )
                    yield done_meta
                    chunk_idx += 1
                    if self.config.max_spectra and total_seen >= self.config.max_spectra:
                        break
                    continue

                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break

                bsz = batch["spectra"].size(0)
                accumulated.append(self._process_batch(batch, capture))
                spectra_in_chunk += bsz
                total_seen += bsz

                if spectra_in_chunk >= self.config.chunk_size:
                    meta = self._flush_chunk(
                        chunk_idx, total_seen - spectra_in_chunk, accumulated,
                    )
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

            if accumulated:
                meta = self._flush_chunk(
                    chunk_idx, total_seen - spectra_in_chunk, accumulated,
                )
                LOG.info("Wrote final chunk %d (n_spectra=%d)", chunk_idx, meta.n_spectra)
                yield meta

    def extract_all(self) -> None:
        """Materialise every chunk to disk. Updates the manifest at the end."""
        n_chunks = n_spectra = n_tokens = 0
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
            "schema_version": EXTRACT_SCHEMA_VERSION,
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


# --- CLI ----------------------------------------------------------------------

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-peaks", type=int, default=instanovo_io.DEFAULT_N_PEAKS,
                   help="Peaks kept per spectrum. Recorded in the manifest; "
                        "evaluate.py reads it from there so Phase 7/8 rebuild "
                        "the same spectra.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", default="float32",
                   choices=list(DTYPE_MAP),
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
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    config = ExtractionConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        target_layers=tuple(sorted(args.layers)),
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_peaks=args.n_peaks,
        device=args.device,
        dtype=DTYPE_MAP[args.dtype],
        save_baseline=not args.no_baseline,
        resume=not args.no_resume,
        max_spectra=args.max_spectra,
    )

    ActivationExtractor(config).extract_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
