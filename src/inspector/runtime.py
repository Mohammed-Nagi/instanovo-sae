"""runtime.py -- explain(spectrum): what InstaNovo's encoder extracted from one spectrum.

Scope: the encoder. The output says which SAE features fire on each peak at
layers 2/4/6/8, a summary of what the feature dictionary knows about them, how
well each SAE reconstructs this spectrum, and (on request) what removing a
feature does to the log-probability of each predicted residue. It does not
explain how the decoder chose each residue.

Pipeline for one raw spectrum (m/z, intensity, precursor m/z, charge):
  1. Preprocess with InstaNovo's own TransformerDataProcessor via instanovo_io,
     configured as extraction was (n_peaks=200, use_spectrum_utils=True by
     default), so peak j is encoder token j+1 and the latent token is 0.
  2. Beam search (beam count and max_length from the InstaNovo config unless
     given).
  3. Teacher-forced pass on the predicted peptide with hooks on the encoder
     layers: per-residue log p (= -CE, the computation of Phase 7 and of the
     baseline cached at extraction) and the layer activations. The encoder does
     not see the peptide, so the activations are the encoder's for this spectrum.
  4. Each layer through its SAE (JumpReLU inference): top-k features per token,
     L0, relative reconstruction error, and the layer's FVE on this spectrum.
  5. Peaks annotated against the *predicted* peptide with annotate.py.
  6. A dictionary summary for every feature that reaches a top-k list (full
     rows: ``feature_row``, checked against the dictionary file's hash).

Exact error-preserving ablation (V1-4b): subtract a_{t,f} * W_dec[f] from the
output of encoder layer L at the chosen tokens and rerun, teacher-forced on the
same peptide. The model runs on the real activation, so the SAE error term is
kept; only the SAE-captured component of the feature is removed. Scopes:
"local" = one token, "global" = every token where the feature fires.
``ablation_context`` holds what one spectrum's ablations need (optionally
teacher-forcing a given target peptide instead of the prediction),
``ablate_token`` removes every active feature at one token, one at a time and
then jointly, ``ablate`` takes any features in either scope, and ``ablate_set``
/ ``ablate_sets`` remove arbitrary sets of (feature, token) pairs jointly.
``explain(latent_ablation=True)`` adds ``ablate_token`` at the latent token for
every layer.

Dependencies on the pipeline: instanovo_io, annotate, schema, the dictionary,
and the SAE checkpoints (read by train.load_sae_from_checkpoint, the checkpoint
format's own loader). Nothing here touches the network: the checkpoint must be
a local file (no model-id download path) and modified residues are given to
spectrum_utils as delta masses, never UNIMOD tags.

CLI (from the repo root)::

    PYTHONPATH=src uv run python -m inspector.runtime time \\
        --parquet <raw spectra parquet> --rows 4506 4507 4508 [--ablation]
    PYTHONPATH=src uv run python -m inspector.runtime compare-preprocessing \\
        --parquet <raw spectra parquet> --rows 4506 ...
    PYTHONPATH=src uv run python -m inspector.runtime accuracy \\
        --parquet <HF test parquet> --n 200 --seed 0
    PYTHONPATH=src uv run python -m inspector.runtime audit-cases \\
        --parquet <HF test parquet> --n 200 --seed 0    # V1-4c part A, no ablation
    PYTHONPATH=src uv run python -m inspector.runtime audit-part-b \\
        --parquet <HF test parquet>                     # V1-4c part B, the registered tests

V1-5 (outputs in analysis/v1_5/; the demo set from `python -m analysis --demo-set`)::

    PYTHONPATH=src uv run python -m inspector.runtime demo-explain --parquet <HF test parquet>
    PYTHONPATH=src uv run python -m inspector.runtime screen-sample --parquet <HF test parquet>
    PYTHONPATH=src uv run python -m inspector.runtime screen-ablate --parquet <HF test parquet>
    PYTHONPATH=src uv run python -m inspector.runtime screen-verify --parquet <HF test parquet>

--checkpoint defaults to checkpoints/instanovo_v1.1.0.ckpt and --outputs-root
to sae_pipeline_outputs/, both under the repo root.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

import annotate
import instanovo_io
from inspector.dictionary import SUMMARY_KEYS, FeatureDictionary
from schema import EXPLAIN_SCHEMA_VERSION
from train import load_sae_from_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
# The InstaNovo v1.1.0 release asset (1,136,412,059 bytes, the size GitHub
# lists; no published digest, so the sha256 is our own, pinned in the tests).
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "instanovo_v1.1.0.ckpt"
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "sae_pipeline_outputs"
DEFAULT_LAYERS = (2, 4, 6, 8)
DEFAULT_TOP_K = 8
MAX_TOP_K = 32
DEFAULT_NUM_BEAMS = 5       # InstaNovo's inference default (configs/inference/default.yaml)
SIGNIFICANT_DIGITS = 6      # floats in the output are rounded to this many digits

SCOPE_NOTE = (
    "Encoder only: these are the features InstaNovo's encoder extracted from the "
    "spectrum, not an account of how the decoder chose each residue. Peak "
    "annotations are against the predicted peptide, not a ground truth."
)
FIT_NOTE = (
    "fve_centred = 1 - SSE / sum((x - mean)^2), the Phase 1-2 definition, with the "
    "mean taken over this spectrum's tokens (not the corpus), so it is not directly "
    "comparable to the corpus FVE in the eval report."
)
MAX_ABLATION_BATCH = 32     # ablations per forward pass (plus one baseline copy)
SPECTRUM_SUMMARY_KEYS = ("n_tokens_active", "max_activation", "fires_on_latent")
LATENT_LABEL = "latent token only"
LOCAL_NOTE = "local: the feature removed at this one token only"
GLOBAL_NOTE = "global: the feature removed at every token where it fires (latent and peaks)"
ABLATION_NOTE = (
    "Single-feature effect: what removing this feature alone does. It understates the "
    "importance of features whose role others share (redundancy); a small value means "
    "'removing this alone changes little', not 'unimportant'. Only the SAE-captured "
    "component is removed (shrinkage can leave part of a concept in the kept error term), "
    "so effects are lower bounds for the concept."
)
JOINT_NOTE = (
    "joint = what the SAE-explained content at this token contributes; a joint effect far "
    "from the sum of singles indicates redundancy or interaction"
)
SET_NOTE = (
    "set = the named (feature, token) pairs removed together, each at its own token only "
    "(local), at one layer; the SAE error is kept everywhere and every other token is "
    "untouched; a set effect far from the sum of its singles indicates redundancy or interaction"
)


def ablation_provenance(target_kind: str) -> str:
    """Provenance of an ablation result; the teacher-forcing target is part of it."""
    if target_kind == "prediction":
        target = "the predicted peptide"
    elif target_kind == "given":
        target = "the given target peptide"
    else:
        raise ValueError(f"target kind must be 'prediction' or 'given', got {target_kind!r}")
    return (f"exact ablation, error-preserving (SAE error kept), teacher-forced on {target}, "
            f"n = 1 spectrum, no CI")

_UNIMOD_RE = re.compile(r"UNIMOD:(\d+)")


# --- Input --------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Spectrum:
    """One raw spectrum, before any preprocessing."""

    mz: tuple[float, ...]
    intensity: tuple[float, ...]
    precursor_mz: float
    precursor_charge: int
    spectrum_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "mz", tuple(float(v) for v in self.mz))
        object.__setattr__(self, "intensity", tuple(float(v) for v in self.intensity))
        if len(self.mz) != len(self.intensity):
            raise ValueError(f"{len(self.mz)} m/z values but {len(self.intensity)} intensities")
        if not self.mz:
            raise ValueError("empty spectrum")
        if int(self.precursor_charge) < 1:
            raise ValueError(f"precursor charge must be >= 1, got {self.precursor_charge}")
        if not math.isfinite(float(self.precursor_mz)) or float(self.precursor_mz) <= 0:
            raise ValueError(f"invalid precursor m/z {self.precursor_mz}")


_SPECTRUM_COLUMNS = ["mz_array", "intensity_array", "precursor_mz", "precursor_charge"]


def parquet_num_rows(parquet_path: Path) -> int:
    return pq.ParquetFile(parquet_path).metadata.num_rows


def read_raw_records(parquet_path: Path, rows: list[int], columns: list[str]) -> list[dict]:
    """The given rows of a parquet as dicts, in the order asked, reading each
    row group that holds any of them once."""
    pf = pq.ParquetFile(parquet_path)
    starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows
                              for i in range(pf.metadata.num_row_groups)])
    by_group: dict[int, list[int]] = {}
    for row in rows:
        if not 0 <= row < starts[-1]:
            raise IndexError(f"row {row} outside {parquet_path} ({starts[-1]} rows)")
        by_group.setdefault(int(np.searchsorted(starts, row, side="right") - 1), []).append(row)
    found: dict[int, dict] = {}
    for group, members in by_group.items():
        table = pf.read_row_group(group, columns=columns)
        for row in members:
            found[row] = table.slice(row - int(starts[group]), 1).to_pylist()[0]
    return [found[row] for row in rows]


def read_raw_spectra(parquet_path: Path, rows: list[int]) -> list[Spectrum]:
    """Read rows of a raw-spectra parquet (InstaNovo columns mz_array,
    intensity_array, precursor_mz, precursor_charge), touching only the row
    groups that hold them. spectrum_id is "<file stem>:<row>"."""
    return [Spectrum(rec["mz_array"], rec["intensity_array"], rec["precursor_mz"],
                     rec["precursor_charge"], spectrum_id=f"{Path(parquet_path).stem}:{row}")
            for row, rec in zip(rows, read_raw_records(parquet_path, rows, _SPECTRUM_COLUMNS),
                                strict=True)]


# --- Output -------------------------------------------------------------------

@dataclasses.dataclass
class Explanation:
    """explain()'s result: plain JSON-native fields (dict keys are strings)."""

    schema_version: int
    scope_note: str
    provenance: dict
    input: dict
    peaks: list
    prediction: dict | None
    layers: dict
    features: dict
    ablation: dict | None
    timings_s: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> Explanation:
        fields = {f.name for f in dataclasses.fields(cls)}
        if set(data) != fields:
            raise ValueError(
                f"explanation keys differ from schema {EXPLAIN_SCHEMA_VERSION}: "
                f"missing {sorted(fields - set(data))}, unexpected {sorted(set(data) - fields)}"
            )
        if data["schema_version"] != EXPLAIN_SCHEMA_VERSION:
            raise ValueError(
                f"explanation schema {data['schema_version']!r}, this tool reads {EXPLAIN_SCHEMA_VERSION}"
            )
        _validate(data)
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> Explanation:
        return cls.from_dict(json.loads(text))


_TOKEN_KEYS = {"position", "kind", "l0", "recon_rel_error", "top"}
_PEAK_KEYS = {"index", "mz", "intensity", "annotation"}
_FEATURE_KEYS = set(SUMMARY_KEYS) | set(SPECTRUM_SUMMARY_KEYS)
_ABLATION_BLOCK_KEYS = {"label", "scope", "position", "note", "layers"}


def _validate(data: dict) -> None:
    """Structural checks beyond the top-level keys; raises ValueError."""
    n_peaks = data["input"]["n_processed_peaks"]
    if len(data["peaks"]) != n_peaks:
        raise ValueError(f"{len(data['peaks'])} peaks listed, input says {n_peaks}")
    for peak in data["peaks"]:
        if set(peak) != _PEAK_KEYS:
            raise ValueError(f"peak entry keys {sorted(peak)}")
    ablation = data["ablation"]
    ablated: dict[str, set[str]] = {}
    if ablation is not None and set(ablation) != {"skipped"}:
        if set(ablation) != _ABLATION_BLOCK_KEYS or set(ablation["layers"]) != set(data["layers"]):
            raise ValueError(f"ablation block keys {sorted(ablation)} malformed")
        for layer, rep in ablation["layers"].items():
            if set(rep) != set(AblationReport.KEYS):
                raise ValueError(f"ablation layer {layer} keys {sorted(rep)}")
            ablated[layer] = {str(r["feature"]) for r in rep["results"]}
    for layer, block in data["layers"].items():
        if not layer.isdigit() or set(block) != {"fit", "tokens"}:
            raise ValueError(f"layer block {layer!r} malformed")
        if len(block["tokens"]) != n_peaks + 1:
            raise ValueError(f"layer {layer}: {len(block['tokens'])} tokens for {n_peaks} peaks")
        for tok in block["tokens"]:
            if set(tok) != _TOKEN_KEYS:
                raise ValueError(f"layer {layer} token keys {sorted(tok)}")
        listed = {str(f["feature"]) for tok in block["tokens"] for f in tok["top"]}
        features = data["features"].get(layer, {})
        if listed | ablated.get(layer, set()) != set(features):
            raise ValueError(f"layer {layer}: features block does not match the top-k lists "
                             f"and the ablated features")
        for f, summary in features.items():
            if set(summary) != _FEATURE_KEYS:
                raise ValueError(f"layer {layer} feature {f}: summary keys {sorted(summary)}")


def _round_sig(x: float) -> float:
    return float(f"{x:.{SIGNIFICANT_DIGITS}g}")


def _finalise(obj: Any, path: str = "$") -> Any:
    """Round floats to SIGNIFICANT_DIGITS and convert numpy/torch scalars and
    tuples to JSON-native types. A non-finite float is a bug upstream, so it
    raises here rather than surfacing later as invalid JSON."""
    if isinstance(obj, dict):
        return {str(k): _finalise(v, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finalise(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            raise ValueError(f"non-finite value {x} at {path}")
        return _round_sig(x)
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"not JSON-native at {path}: {type(obj).__name__}")


# --- Pure helpers (unit-tested without the model) -------------------------------

def fve(x: torch.Tensor, x_hat: torch.Tensor) -> tuple[float, float]:
    """(centred, uncentred) fraction of variance explained over the rows of x.

    The Phase 1+2 definition (evaluate._ReconSparsityAccumulator):
    centred = 1 - SSE / (sum x^2 - sum_d (sum_t x_d)^2 / n), uncentred = 1 - SSE / sum x^2.
    """
    x64, xh64 = x.double(), x_hat.double()
    sse = float(((x64 - xh64) ** 2).sum())
    sq_total = float((x64 ** 2).sum())
    mean_energy = float((x64.sum(dim=0) ** 2).sum()) / max(x64.size(0), 1)
    return (1.0 - sse / max(sq_total - mean_energy, 1e-12),
            1.0 - sse / max(sq_total, 1e-12))


def decoder_to_residue_order(values: list[float], n_residues: int) -> tuple[list[float], float]:
    """Map per-position decoder values to (per residue N->C, EOS value).

    InstaNovo decodes C->N: decoder targets are the reversed residues then EOS,
    so decoder position t < n holds residue n-1-t and position n holds EOS.
    """
    if len(values) != n_residues + 1:
        raise ValueError(f"{len(values)} decoder positions for {n_residues} residues + EOS")
    return list(values[:n_residues][::-1]), values[n_residues]


def peptide_for_annotation(tokens: list[str], residue_set: Any) -> tuple[str, str, list[dict]]:
    """InstaNovo residue tokens -> (bare peptide, ProForma, modifications) for annotate.

    Modifications are written as exact delta masses from the residue set
    (spectrum_utils resolves UNIMOD tags over the network). An N-terminal
    modification takes ProForma's hyphen, "[+42.010565]-PEPTIDE": spectrum_utils
    rejects the hyphen-less form extraction's _to_proforma writes. Each
    modification dict is {position (1-indexed, 0 = N-terminal), mod_name, unimod_id}.
    """
    bare, proforma, mods = "", "", []
    for i, tok in enumerate(tokens):
        letter = tok[0] if tok[0].isalpha() else None
        delta = instanovo_io.residue_delta_mass(residue_set, tok)
        if letter is None and i != 0:
            raise ValueError(f"N-terminal token {tok!r} at position {i} of {tokens}")
        if letter is not None:
            bare += letter
        if delta is None:
            proforma += tok
            continue
        name = f"{delta:+.6f}"
        uid = _UNIMOD_RE.search(tok)
        mods.append({"position": len(bare) if letter else 0, "mod_name": name,
                     "unimod_id": int(uid.group(1)) if uid else -1})
        proforma += f"{letter}[{name}]" if letter else f"[{name}]-"
    return bare, proforma, mods


def peak_label(ann: dict) -> str:
    """Short human label for an annotation dict: y7+, b3++-H2O, I(K), m2:5+, precursor, noise."""
    t, idx, z, nl = ann["ion_type"], ann["ion_index"], int(ann["charge"]), ann["neutral_loss"]
    if t == "noise":
        return "noise"
    if t == "I":
        base = f"I({idx})"
    elif t == "internal":
        base = f"m{idx[0]}:{idx[1]}"
    elif t == "precursor":
        base = "precursor"
    else:
        base = f"{t}{idx}"
    return base + "+" * max(z, 0) + (f"-{nl}" if nl else "")


def top_k_features(features: torch.Tensor, k: int) -> list[list[tuple[int, float]]]:
    """Per row, up to k (feature, activation) pairs with activation > 0, strongest first."""
    k = min(k, features.size(1))
    vals, idx = features.topk(k, dim=1)
    out = []
    for v_row, i_row in zip(vals.tolist(), idx.tolist(), strict=True):
        out.append([(i, v) for i, v in zip(i_row, v_row, strict=True) if v > 0])
    return out


def ablation_positions(features: torch.Tensor, feature: int, scope: str,
                       position: int | None = None) -> list[int]:
    """Tokens an ablation of `feature` touches: local = [position] (the feature
    must be active there), global = every token where it is active. Raises
    rather than returning a no-op ablation."""
    if scope == "local":
        if position is None:
            raise ValueError("local ablation needs a token position")
        if not 0 <= position < features.size(0):
            raise ValueError(f"position {position} outside 0..{features.size(0) - 1}")
        if float(features[position, feature]) <= 0:
            raise ValueError(f"feature {feature} is not active at token {position}")
        return [int(position)]
    if scope == "global":
        if position is not None:
            raise ValueError("global ablation takes no position")
        active = (features[:, feature] > 0).nonzero().flatten().tolist()
        if not active:
            raise ValueError(f"feature {feature} is not active at any token")
        return active
    raise ValueError(f"scope must be 'local' or 'global', got {scope!r}")


def ablation_delta(features: torch.Tensor, w_dec: torch.Tensor,
                   items: list[tuple[int, list[int]]]) -> torch.Tensor:
    """[n_tokens, d_model] change to the layer output that removes each
    (feature, positions) item's SAE component: -a[t, f] * W_dec[f] at those tokens.
    The model then runs on x + delta, so the SAE error x - x_hat is unchanged."""
    delta = torch.zeros(features.size(0), w_dec.size(1), dtype=w_dec.dtype)
    for f, pos in items:
        a = features[pos, f].to(w_dec.dtype).unsqueeze(1)          # [len(pos), 1]
        delta[pos] = delta[pos] - a * w_dec[f]
    return delta


def joint_delta(features: torch.Tensor, w_dec: torch.Tensor, position: int) -> torch.Tensor:
    """Remove every active feature at one token: delta[t] = -(x_hat[t] - b_dec).
    b_dec, the SAE's 'nothing active' state, stays, as in single ablations."""
    active = (features[position] > 0).nonzero().flatten().tolist()
    return ablation_delta(features, w_dec, [(f, [position]) for f in active])


def _normalise_pairs(pairs) -> list[tuple[int, int]]:
    out = [(int(f), int(t)) for f, t in pairs]
    if not out:
        raise ValueError("an ablation set needs at least one (feature, token) pair")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate (feature, token) pairs in {out}")
    return out


def set_delta(features: torch.Tensor, w_dec: torch.Tensor, pairs) -> torch.Tensor:
    """Remove an arbitrary set of (feature, token) pairs together: -a[t, f] * W_dec[f]
    at each pair's own token only. Every pair must be active at its token; the set
    must be non-empty and hold no pair twice. All features active at one token
    give joint_delta at that token."""
    pairs = _normalise_pairs(pairs)
    return ablation_delta(features, w_dec,
                          [(f, ablation_positions(features, f, "local", t)) for f, t in pairs])


@contextlib.contextmanager
def _capture_encoder_layers(model: Any, layers: tuple[int, ...],
                            deltas: dict[int, torch.Tensor] | None = None,
                            capture_after_delta: bool = False):
    """Collect the outputs of the given encoder layers during a forward pass.

    The same capture point as extract.MultiLayerCapture (the layer module's
    output, first element if a tuple); kept local because the tool does not
    depend on extract.py. tests/test_inspector_runtime.py checks the captured
    activations against the stored chunks.

    deltas (ablation): {layer: [batch, tokens, d_model]} added to that layer's
    output, which the rest of the model then sees. None (the default) leaves
    the forward pass untouched. capture_after_delta records x + delta instead of x.
    """
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if deltas is None or layer not in deltas:
                captured[layer] = out.detach()
                return None
            new = out + deltas[layer].to(device=out.device, dtype=out.dtype)
            captured[layer] = (new if capture_after_delta else out).detach()
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return hook

    handles = [instanovo_io.get_encoder_layer(model, L).register_forward_hook(make_hook(L))
               for L in layers]
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _sha256_prefix(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


@dataclasses.dataclass
class ForwardResult:
    peaks_mz: torch.Tensor                  # [n_peaks] processed m/z (token j+1)
    peaks_intensity: torch.Tensor           # [n_peaks] processed intensity
    activations: dict[int, torch.Tensor]    # layer -> [1 + n_peaks, d_model] fp32
    ce: torch.Tensor | None                 # [n_residues + 1] decoder-order CE, or None


@dataclasses.dataclass
class AblationContext:
    """What one spectrum's ablations need, built by Inspector.ablation_context."""

    spectrum_id: str | None
    batch: dict                              # teacher-forced batch (size 1) on the target
    target: dict                             # {"kind": "prediction"|"given", "peptide", "tokens"}
    features: dict[int, torch.Tensor]        # layer -> [n_tokens, d_dict] JumpReLU features
    baseline_log_prob: torch.Tensor          # [n_residues + 1] decoder order, on the target

    @property
    def n_residues(self) -> int:
        return len(self.target["tokens"])


@dataclasses.dataclass
class AblationReport:
    """One ablate / ablate_token call: per-feature results plus, for a token,
    the joint ablation of every active feature there. Per-residue lists follow
    the target's residues N->C; EOS is separate; totals include EOS."""

    layer: int
    scope: str
    position: int | None                     # ablate_token's token, else None
    label: str | None                        # "latent token only" / "peak j only"
    target: dict
    provenance: str
    scope_note: str
    caveat: str
    results: list                            # per feature, see Inspector._result
    joint: dict | None                       # ablate_token only
    joint_delta_total: float | None
    sum_of_singles: float | None
    joint_note: str | None

    KEYS = ("layer", "scope", "position", "label", "target", "provenance", "scope_note", "caveat",
            "results", "joint", "joint_delta_total", "sum_of_singles", "joint_note")

    def to_dict(self, include_summaries: bool = True) -> dict:
        out = {k: getattr(self, k) for k in self.KEYS}
        if not include_summaries:
            out["results"] = [{k: v for k, v in r.items() if k != "summary"} for r in self.results]
        return _finalise(out)


# --- The inspector -------------------------------------------------------------

class Inspector:
    """InstaNovo, the layer SAEs and the feature dictionaries, loaded once."""

    def __init__(
        self,
        instanovo_checkpoint: Path = DEFAULT_CHECKPOINT,
        outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        seed: int = 0,
        device: str = "cpu",
        top_k: int = DEFAULT_TOP_K,
        num_beams: int = DEFAULT_NUM_BEAMS,
        max_length: int | None = None,
        use_spectrum_utils: bool = True,
    ):
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be in [1, {MAX_TOP_K}], got {top_k}")
        ckpt = Path(instanovo_checkpoint)
        if not ckpt.is_file():
            # A local file only: the model-id route can download.
            raise FileNotFoundError(f"InstaNovo checkpoint not found: {ckpt}")
        t0 = time.perf_counter()
        self.layers = tuple(sorted(layers))
        self.device = device
        self.top_k = top_k
        self.num_beams = num_beams
        self.use_spectrum_utils = use_spectrum_utils

        self.model, self.config, self.residue_set = instanovo_io.load_instanovo(
            ckpt, device=device, by_id=False,
        )
        if instanovo_io.uses_flash_attention(self.model):
            raise RuntimeError("flash-attention checkpoints bypass the encoder layers the SAEs read")
        if max_length is None:
            if self.config.get("max_length") is None:
                raise ValueError("the InstaNovo config has no max_length; pass max_length")
            max_length = int(self.config["max_length"])
        self.max_length = max_length
        # Extraction's n_peaks: the SAEs were trained on this many peak tokens.
        self.n_peaks = instanovo_io.DEFAULT_N_PEAKS
        self.processor = instanovo_io.make_processor(
            self.residue_set, n_peaks=self.n_peaks, annotated=False,
            use_spectrum_utils=use_spectrum_utils,
        )
        self.tf_processor = instanovo_io.make_processor(
            self.residue_set, n_peaks=self.n_peaks, annotated=True,
            use_spectrum_utils=use_spectrum_utils,
        )

        root = Path(outputs_root)
        self.outputs_root = root
        self.saes, sae_provenance = {}, {}
        for L in self.layers:
            path = root / "sae" / f"layer_{L}" / f"seed_{seed}" / "checkpoint.pt"
            sae = load_sae_from_checkpoint(path, device=device)
            if sae.d_model != int(self.config["dim_model"]):
                raise ValueError(f"layer {L} SAE d_model {sae.d_model} != InstaNovo {self.config['dim_model']}")
            self.saes[L] = sae
            sae_provenance[str(L)] = {"path": str(path), "sha256_16": _sha256_prefix(path),
                                      "d_dict": sae.d_dict}
        self.dictionaries = {L: FeatureDictionary.load(root, L) for L in self.layers}

        # Annotation as the pipeline ran it (AnnotationConfig defaults equal the
        # recorded run config; tested). The two directories are the batch
        # runner's and are not used by annotate_spectrum.
        self.annotation_config = annotate.AnnotationConfig(extract_dir=Path(), output_dir=Path())
        self.registry = annotate.ConceptRegistry.build()

        self.provenance = {
            "instanovo_checkpoint": {"path": str(ckpt), "bytes": ckpt.stat().st_size},
            "instanovo_package": importlib.metadata.version("instanovo"),
            "sae": sae_provenance,
            "sae_seed": seed,
            "dictionary": {str(L): {"path": str(d.path), "sha256_16": d.sha256_16,
                                    "schema": d.metadata.get("schema_version"),
                                    "blinded_run": d.metadata.get("blinded_run"),
                                    "causal_provenance": d.causal_provenance()}
                           for L, d in self.dictionaries.items()},
            "preprocessing": {"processor": "TransformerDataProcessor", "n_peaks": self.n_peaks,
                              "use_spectrum_utils": use_spectrum_utils,
                              "min_mz": self.processor.min_mz, "max_mz": self.processor.max_mz,
                              "min_intensity": self.processor.min_intensity,
                              "remove_precursor_tol": self.processor.remove_precursor_tol},
            "decoding": {"decoder": "BeamSearchDecoder", "num_beams": num_beams,
                         "max_length": max_length, "float_dtype": "float64"},
            "annotation": {k: v for k, v in self.annotation_config.as_jsonable().items()
                           if k in ("ion_types", "enable_internal", "fragment_tol_mass", "fragment_tol_mode")},
            "top_k": top_k,
            "device": device,
        }
        self.load_seconds = time.perf_counter() - t0
        self._json_size_bound: int | None = None

    # -- steps ---------------------------------------------------------------

    def preprocess(self, spectrum: Spectrum, peptide: str | None = None,
                   use_spectrum_utils: bool | None = None) -> dict:
        if use_spectrum_utils is None or use_spectrum_utils == self.use_spectrum_utils:
            processor = self.tf_processor if peptide is not None else self.processor
        else:
            processor = instanovo_io.make_processor(
                self.residue_set, n_peaks=self.n_peaks, annotated=peptide is not None,
                use_spectrum_utils=use_spectrum_utils,
            )
        batch = instanovo_io.preprocess_spectrum(
            processor, spectrum.mz, spectrum.intensity, spectrum.precursor_mz,
            spectrum.precursor_charge, peptide=peptide,
        )
        spectra = batch["spectra"][0]
        if spectra.shape == (1, 2) and spectra[0, 0] == 0:
            # TransformerDataProcessor's stand-in for a spectrum with no peak left.
            raise ValueError("no peak survives preprocessing (m/z range, precursor removal, intensity)")
        return batch

    def predict(self, spectrum: Spectrum, use_spectrum_utils: bool | None = None,
                decoder: str = "beam") -> dict:
        """Decode: {"tokens", "log_prob", "token_log_probs"}; tokens empty if no beam completed.

        decoder "beam" is what explain() uses; "greedy" (InstaNovo 1.1.0's
        default before refinement) is for the accuracy comparison only.
        """
        batch = self.preprocess(spectrum, use_spectrum_utils=use_spectrum_utils)
        if decoder == "beam":
            out = instanovo_io.beam_search(self.model, batch, self.device, self.num_beams, self.max_length)
        elif decoder == "greedy":
            out = instanovo_io.greedy_search(self.model, batch, self.device, self.max_length)
        else:
            raise ValueError(f"decoder must be 'beam' or 'greedy', got {decoder!r}")
        tokens = list(out["predictions"][0])
        return {"tokens": tokens,
                "log_prob": float(out["prediction_log_probability"][0]) if tokens else None,
                "token_log_probs": [float(v) for v in out["prediction_token_log_probabilities"][0]]}

    def forward(self, spectrum: Spectrum, peptide: str | None) -> ForwardResult:
        """One pass with the encoder layers hooked: teacher-forced on `peptide`
        (InstaNovo notation) if given, else the encoder alone."""
        batch = self.preprocess(spectrum, peptide=peptide)
        ce = None
        with torch.inference_mode(), _capture_encoder_layers(self.model, self.layers) as captured:
            if peptide is None:
                instanovo_io.encoder_forward(self.model, batch, self.device)
            else:
                logits = instanovo_io.model_forward_logits(self.model, batch, self.device)
                ce_b, _top1, _targets, valid = instanovo_io.per_token_ce_and_top1(
                    logits, batch["peptides"], pad_index=instanovo_io.PAD_INDEX,
                )
                ce = ce_b[0][valid[0]].float().cpu()
        missing = set(self.layers) - set(captured)
        if missing:
            raise RuntimeError(f"encoder hooks did not fire for layers {sorted(missing)}")
        n_peaks = batch["spectra"].size(1)
        acts = {}
        for L in self.layers:
            a = captured[L][0]
            if a.size(0) != n_peaks + 1:
                raise RuntimeError(f"layer {L}: {a.size(0)} encoder positions for {n_peaks} peaks + latent")
            acts[L] = a.float().cpu()
        return ForwardResult(batch["spectra"][0, :, 0].clone(), batch["spectra"][0, :, 1].clone(), acts, ce)

    def _annotate(self, tokens: list[str], spectrum: Spectrum, fwd: ForwardResult) -> tuple[list[dict], dict]:
        bare, proforma, mods = peptide_for_annotation(tokens, self.residue_set)
        anns, tok = annotate.annotate_spectrum(
            peptide=bare, proforma=proforma, modifications=mods,
            mz_array=fwd.peaks_mz, intensity_array=fwd.peaks_intensity,
            precursor_mz=spectrum.precursor_mz, precursor_charge=spectrum.precursor_charge,
            config=self.annotation_config, registry=self.registry, strict=True,
        )
        names = self.registry.names
        spectrum_level = set(annotate.compute_spectrum_concepts(mods, spectrum.precursor_charge, bare))
        labels = tok.labels
        peak_annotations = []
        for j, ann in enumerate(anns):
            idx = ann["ion_index"]
            peak_annotations.append({
                "ion_type": ann["ion_type"],
                "ion_index": list(idx) if isinstance(idx, tuple) else idx,
                "charge": int(ann["charge"]),
                "neutral_loss": ann["neutral_loss"],
                "label": peak_label(ann),
                "concepts": [names[c] for c in labels[j + 1].nonzero().flatten().tolist()
                             if names[c] not in spectrum_level],
            })
        summary = {
            "bare_peptide": bare,
            "annotation_proforma": proforma,
            "spectrum_concepts": [n for n in names if n in spectrum_level and labels[0, names.index(n)]],
            "latent_token_concepts": [names[c] for c in labels[0].nonzero().flatten().tolist()
                                      if names[c] not in spectrum_level],
        }
        return peak_annotations, summary

    # -- explain -------------------------------------------------------------

    def _feature_summary(self, layer: int, feature: int, feats: torch.Tensor) -> dict:
        col = feats[:, feature]
        return {**self.dictionaries[layer].summary(feature),
                "n_tokens_active": int((col > 0).sum()), "max_activation": float(col.max()),
                "fires_on_latent": bool(col[0] > 0)}

    def explain(self, spectrum: Spectrum, latent_ablation: bool = False) -> Explanation:
        """What the encoder extracted from one spectrum (see the module docstring).

        latent_ablation: also run ablate_token at the latent token for every
        layer (every active feature there, one at a time and jointly; several
        seconds per layer), stored under `ablation`. Off by default, so
        explain() costs what it did in V1-4a.
        """
        timings: dict[str, float] = {}

        def lap(name, t):
            timings[name] = time.perf_counter() - t
            return time.perf_counter()

        t = time.perf_counter()
        pred = self.predict(spectrum)
        t = lap("preprocess_and_beam_search", t)
        tokens = pred["tokens"]
        peptide = "".join(tokens) if tokens else None
        if tokens and self.residue_set.tokenize(peptide) != tokens:
            raise RuntimeError(f"predicted tokens {tokens} do not re-tokenize from {peptide!r}")

        fwd = self.forward(spectrum, peptide)
        t = lap("teacher_forced_pass", t)

        layers_out, features_out, layer_feats = {}, {}, {}
        for L in self.layers:
            x = fwd.activations[L]
            with torch.inference_mode():
                out = self.saes[L].forward_inference(x.to(self.device))
            feats, x_hat = out["features"].cpu(), out["x_hat"].cpu()
            layer_feats[L] = feats
            l0 = (feats > 0).sum(dim=1).tolist()
            rel = (((x - x_hat) ** 2).sum(dim=1) / (x ** 2).sum(dim=1).clamp_min(1e-12)).tolist()
            tops = top_k_features(feats, self.top_k)
            centred, uncentred = fve(x, x_hat)
            layers_out[str(L)] = {
                "fit": {"fve_centred": centred, "fve_uncentred": uncentred,
                        "n_tokens": x.size(0), "l0_mean": float(np.mean(l0)), "note": FIT_NOTE},
                "tokens": [{"position": p, "kind": "latent" if p == 0 else "peak", "l0": l0[p],
                            "recon_rel_error": rel[p],
                            "top": [{"feature": f, "activation": a} for f, a in tops[p]]}
                           for p in range(x.size(0))],
            }
            distinct = sorted({f for row in tops for f, _ in row})
            features_out[str(L)] = {str(f): self._feature_summary(L, f, feats) for f in distinct}
        t = lap("sae_and_dictionary", t)

        prediction, peak_annotations = None, None
        if tokens:
            peak_annotations, summary = self._annotate(tokens, spectrum, fwd)
            logp = (-fwd.ce).tolist()
            per_residue, eos = decoder_to_residue_order(logp, len(tokens))
            beam_lp = pred["token_log_probs"]
            mass_mz = self.residue_set.get_sequence_mass(tokens, spectrum.precursor_charge)
            prediction = {
                "peptide": peptide,
                "tokens": tokens,
                **summary,
                "beam_log_prob": pred["log_prob"],
                "sequence_log_prob_tf": float(sum(logp)),
                "eos_log_prob_tf": eos,
                "precursor_mz_error_ppm": (mass_mz - spectrum.precursor_mz) / spectrum.precursor_mz * 1e6,
                "residues": [{"token": tok, "log_prob_tf": lp,
                              "log_prob_beam": beam_lp[i] if i < len(beam_lp) else None}
                             for i, (tok, lp) in enumerate(zip(tokens, per_residue, strict=True))],
            }
        t = lap("annotation", t)

        ablation = None
        if latent_ablation and not tokens:
            ablation = {"skipped": "no beam completed"}
        elif latent_ablation:
            ctx, _logp = self._build_context(spectrum, tokens, target=None)
            reports = {}
            for L in self.layers:
                rep = self.ablate_token(ctx, L, 0, with_summaries=False)
                reports[str(L)] = rep.to_dict(include_summaries=False)
                for r in rep.results:
                    features_out[str(L)].setdefault(
                        str(r["feature"]), self._feature_summary(L, r["feature"], layer_feats[L]))
            ablation = {"label": LATENT_LABEL, "scope": "local", "position": 0,
                        "note": ABLATION_NOTE, "layers": reports}
            t = lap("latent_ablation", t)

        peaks = []
        for j in range(fwd.peaks_mz.numel()):
            peaks.append({"index": j, "mz": float(fwd.peaks_mz[j]), "intensity": float(fwd.peaks_intensity[j]),
                          "annotation": peak_annotations[j] if peak_annotations else None})

        payload = {
            "schema_version": EXPLAIN_SCHEMA_VERSION,
            "scope_note": SCOPE_NOTE,
            "provenance": self.provenance,
            "input": {"spectrum_id": spectrum.spectrum_id, "precursor_mz": spectrum.precursor_mz,
                      "precursor_charge": spectrum.precursor_charge, "n_raw_peaks": len(spectrum.mz),
                      "n_processed_peaks": len(peaks)},
            "peaks": peaks,
            "prediction": prediction,
            "layers": layers_out,
            "features": features_out,
            "ablation": ablation,
            "timings_s": timings,
        }
        lap("assemble", t)
        payload["timings_s"] = dict(timings)
        return Explanation.from_dict(_finalise(payload))

    # -- exact ablation ------------------------------------------------------

    def _tf_log_probs(self, batch: dict, deltas: dict[int, torch.Tensor] | None = None,
                      capture: bool = False, capture_after_delta: bool = False):
        """Teacher-forced pass over every row of `batch` (all rows the same
        spectrum and peptide): log p per decoder position [B, n+1], the argmax
        per position [B, n+1], and the hooked layer outputs if `capture`."""
        with torch.inference_mode(), _capture_encoder_layers(
                self.model, self.layers, deltas, capture_after_delta) as captured:
            logits = instanovo_io.model_forward_logits(self.model, batch, self.device)
            ce, top1, _targets, valid = instanovo_io.per_token_ce_and_top1(
                logits, batch["peptides"], pad_index=instanovo_io.PAD_INDEX)
        cols = valid[0]
        if not bool((valid == cols).all()):
            raise RuntimeError("rows of an ablation batch have different peptide lengths")
        acts = {L: captured[L].float().cpu() for L in self.layers} if capture else None
        return (-ce[:, cols]).float().cpu(), top1[:, cols].cpu(), acts

    def _build_context(self, spectrum: Spectrum, pred_tokens: list[str],
                       target: str | None) -> tuple[AblationContext, torch.Tensor]:
        """(context, log p of the prediction in decoder order). The SAE features
        come from the teacher-forced pass on the prediction; with a target, the
        target pass must see the same encoder activations (it does not see the
        peptide), else this raises."""
        peptide = "".join(pred_tokens)
        batch_pred = self.preprocess(spectrum, peptide=peptide)
        logp_pred, _top1, acts = self._tf_log_probs(batch_pred, capture=True)
        features = {}
        with torch.inference_mode():
            for L in self.layers:
                features[L] = self.saes[L].forward_inference(acts[L][0].to(self.device))["features"].cpu()
        if target is None:
            tgt = {"kind": "prediction", "peptide": peptide, "tokens": list(pred_tokens)}
            batch, baseline = batch_pred, logp_pred[0]
        else:
            rs = self.residue_set
            t_tokens = [rs.residue_remapping.get(t, t) for t in rs.tokenize(target)]
            if not t_tokens:
                raise ValueError(f"target peptide {target!r} has no residues")
            t_peptide = "".join(t_tokens)
            batch = self.preprocess(spectrum, peptide=t_peptide)
            logp_t, _t, acts_t = self._tf_log_probs(batch, capture=True)
            for L in self.layers:
                if not torch.allclose(acts_t[L], acts[L], rtol=0, atol=1e-5):
                    raise RuntimeError(f"layer {L}: encoder activations differ between the "
                                       f"prediction and target passes")
            tgt = {"kind": "given", "peptide": t_peptide, "tokens": t_tokens}
            baseline = logp_t[0]
        return AblationContext(spectrum.spectrum_id, batch, tgt, features, baseline), logp_pred[0]

    def ablation_context(self, spectrum: Spectrum, explanation: Explanation,
                         target: str | None = None) -> AblationContext:
        """Everything the ablations of one spectrum need, built once (~0.1-0.3 s).

        The spectrum must be the one `explanation` was computed from: the
        teacher-forced log p of its prediction is recomputed and must equal the
        explanation's to 1e-5. `target` (dataset or InstaNovo notation) is
        teacher-forced instead of the prediction, e.g. the true peptide; every
        result then records it and scores its residues.
        """
        pred = explanation.prediction
        if pred is None:
            raise ValueError("the explanation has no prediction (no beam completed); nothing to teacher-force")
        inp = explanation.input
        if (_round_sig(spectrum.precursor_mz) != inp["precursor_mz"]
                or spectrum.precursor_charge != inp["precursor_charge"]
                or len(spectrum.mz) != inp["n_raw_peaks"]):
            raise ValueError("the spectrum does not match the explanation's input")
        ctx, logp_pred = self._build_context(spectrum, pred["tokens"], target)
        per_residue, eos = decoder_to_residue_order(logp_pred.tolist(), len(pred["tokens"]))
        expected = [r["log_prob_tf"] for r in pred["residues"]] + [pred["eos_log_prob_tf"]]
        gap = max(abs(a - b) - 1e-5 * abs(b) for a, b in zip(per_residue + [eos], expected, strict=True))
        if gap > 1e-5:
            raise ValueError("the spectrum does not reproduce the explanation's log p "
                             "(wrong spectrum/explanation pair?)")
        return ctx

    def _run_ablations(self, ctx: AblationContext, layer: int,
                       deltas: list[torch.Tensor]) -> list[tuple[list[float], list[bool]]]:
        """Per delta ([n_tokens, d_model] for `layer`): (delta log p per decoder
        position, argmax changed per position). Up to MAX_ABLATION_BATCH deltas
        per pass; copy 0 of each pass is the unablated baseline, so every delta
        is taken against a baseline from the same batch."""
        out = []
        for s in range(0, len(deltas), MAX_ABLATION_BATCH):
            chunk = deltas[s:s + MAX_ABLATION_BATCH]
            stacked = torch.stack([torch.zeros_like(chunk[0]), *chunk])
            logp, top1, _ = self._tf_log_probs(_repeat_batch(ctx.batch, stacked.size(0)),
                                               deltas={layer: stacked})
            for i in range(1, stacked.size(0)):
                out.append(((logp[i] - logp[0]).tolist(), (top1[i] != top1[0]).tolist()))
        return out

    def _result(self, ctx: AblationContext, feature: int, positions: list[int], activation: float,
                delta: list[float], changed: list[bool], summary: dict | None) -> dict:
        per_residue, eos = decoder_to_residue_order(delta, ctx.n_residues)
        changed_res, changed_eos = decoder_to_residue_order(changed, ctx.n_residues)
        out = {"feature": int(feature), "positions": positions, "activation": float(activation),
               "delta_log_prob": per_residue, "delta_eos": eos, "delta_total": float(sum(delta)),
               "argmax_changed": changed_res, "eos_argmax_changed": bool(changed_eos)}
        if summary is not None:
            out["summary"] = summary
        return out

    def _report(self, ctx: AblationContext, layer: int, scope: str, results: list,
                position: int | None = None, label: str | None = None,
                joint: dict | None = None) -> AblationReport:
        return AblationReport(
            layer=layer, scope=scope, position=position, label=label, target=ctx.target,
            provenance=ablation_provenance(ctx.target["kind"]),
            scope_note=LOCAL_NOTE if scope == "local" else GLOBAL_NOTE, caveat=ABLATION_NOTE,
            results=results, joint=joint,
            joint_delta_total=None if joint is None else joint["delta_total"],
            sum_of_singles=None if joint is None else float(sum(r["delta_total"] for r in results)),
            joint_note=None if joint is None else JOINT_NOTE,
        )

    def ablate(self, ctx: AblationContext, layer: int, features: list[int], scope: str,
               position: int | None = None, with_summaries: bool = True) -> AblationReport:
        """Exact ablation of each feature on its own, in input order.

        scope "local": at token `position` only (each feature must be active
        there); "global": at every token where the feature fires. A result's
        `activation` is the feature's activation at the token (local) or its
        largest over the ablated tokens (global).
        """
        feats = ctx.features[layer]
        w_dec = self.saes[layer].W_dec.detach().cpu()
        items = [(int(f), ablation_positions(feats, int(f), scope, position)) for f in features]
        runs = self._run_ablations(ctx, layer, [ablation_delta(feats, w_dec, [it]) for it in items])
        results = [self._result(ctx, f, pos, float(feats[pos, f].max()), d, c,
                                self._feature_summary(layer, f, feats) if with_summaries else None)
                   for (f, pos), (d, c) in zip(items, runs, strict=True)]
        return self._report(ctx, layer, scope, results)

    def ablate_token(self, ctx: AblationContext, layer: int, position: int,
                     with_summaries: bool = True) -> AblationReport:
        """Local ablation of every active feature at one token (position 0 =
        latent token, j + 1 = peak j), sorted by |delta_total|, plus one joint
        ablation of all of them together (b_dec stays; see joint_delta)."""
        feats = ctx.features[layer]
        if not 0 <= position < feats.size(0):
            raise ValueError(f"position {position} outside 0..{feats.size(0) - 1}")
        row = feats[position]
        active = sorted((row > 0).nonzero().flatten().tolist(), key=lambda f: -float(row[f]))
        label = LATENT_LABEL if position == 0 else f"peak {position - 1} only"
        if not active:
            return self._report(ctx, layer, "local", [], position, label)
        w_dec = self.saes[layer].W_dec.detach().cpu()
        deltas = [ablation_delta(feats, w_dec, [(f, [position])]) for f in active]
        runs = self._run_ablations(ctx, layer, deltas + [joint_delta(feats, w_dec, position)])
        results = [self._result(ctx, f, [position], float(row[f]), d, c,
                                self._feature_summary(layer, f, feats) if with_summaries else None)
                   for f, (d, c) in zip(active, runs[:-1], strict=True)]
        results.sort(key=lambda r: -abs(r["delta_total"]))
        joint = self._result(ctx, -1, [position], 0.0, *runs[-1], None)
        joint = {"positions": [position], "n_features": len(active),
                 **{k: joint[k] for k in ("delta_log_prob", "delta_eos", "delta_total",
                                          "argmax_changed", "eos_argmax_changed")}}
        return self._report(ctx, layer, "local", results, position, label, joint)

    def ablate_sets(self, ctx: AblationContext, layer: int, sets: list,
                    with_singles: bool = False, with_summaries: bool = False) -> list[AblationReport]:
        """Joint local ablation of each set of (feature, token) pairs at one layer
        (set_delta), all sets in as few batched passes as possible (up to
        MAX_ABLATION_BATCH per pass, each pass against its own in-batch
        baseline). A report's `joint` is the set's effect; with_singles also
        ablates each pair alone, in the same passes, into `results` (input order),
        so `sum_of_singles` compares with the set. Without it, results is empty
        and sum_of_singles None."""
        feats = ctx.features[layer]
        w_dec = self.saes[layer].W_dec.detach().cpu()
        norm = [_normalise_pairs(s) for s in sets]
        deltas = [set_delta(feats, w_dec, s) for s in norm]
        if with_singles:
            deltas += [ablation_delta(feats, w_dec, [(f, [t])]) for s in norm for f, t in s]
        runs = self._run_ablations(ctx, layer, deltas) if deltas else []
        singles = iter(runs[len(norm):])
        reports = []
        for s, (d, c) in zip(norm, runs[:len(norm)], strict=True):
            r = self._result(ctx, -1, sorted({t for _f, t in s}), 0.0, d, c, None)
            joint = {"pairs": [[f, t] for f, t in s], "positions": r["positions"], "n_features": len(s),
                     "activation_sum": float(sum(float(feats[t, f]) for f, t in s)),
                     **{k: r[k] for k in ("delta_log_prob", "delta_eos", "delta_total",
                                          "argmax_changed", "eos_argmax_changed")}}
            results = []
            if with_singles:
                for f, t in s:
                    ds, cs = next(singles)
                    results.append(self._result(ctx, f, [t], float(feats[t, f]), ds, cs,
                                                self._feature_summary(layer, f, feats) if with_summaries else None))
            reports.append(AblationReport(
                layer=layer, scope="local", position=None, label=f"set of {len(s)} (feature, token) pairs",
                target=ctx.target, provenance=ablation_provenance(ctx.target["kind"]),
                scope_note=SET_NOTE, caveat=ABLATION_NOTE, results=results, joint=joint,
                joint_delta_total=joint["delta_total"],
                sum_of_singles=float(sum(x["delta_total"] for x in results)) if with_singles else None,
                joint_note=SET_NOTE))
        return reports

    def ablate_set(self, ctx: AblationContext, layer: int, pairs,
                   with_singles: bool = False, with_summaries: bool = False) -> AblationReport:
        """One set of (feature, token) pairs removed jointly (see ablate_sets)."""
        return self.ablate_sets(ctx, layer, [pairs], with_singles, with_summaries)[0]

    # -- size bound ----------------------------------------------------------

    @property
    def json_size_bound(self) -> int:
        """Upper bound on len(explanation.to_json()) without latent ablation."""
        if self._json_size_bound is None:
            self._json_size_bound = self.size_bound()
        return self._json_size_bound

    def size_bound(self, latent_counts: dict[int, int] | None = None) -> int:
        """Upper bound on len(explanation.to_json()) for this configuration.

        Built from worst-case entries: n_peaks peaks with the longest label and
        every concept; max_length residues; per layer, (n_peaks + 1) tokens with
        k features each, plus one feature summary per distinct feature, at most
        min((n_peaks + 1) * k, live features), each a worst-case summary. 64 KB
        covers provenance, input, prediction scalars, timings and notes.
        latent_counts ({layer: active features at the latent token}) adds the
        latent ablation block: one worst-case result per feature, the joint
        result and 4 KB of notes per layer, plus that many more summaries (the
        number at the latent token is not capped by design, so it is a parameter).
        """
        w = -1.23457e-05                                  # longest 6-digit float repr

        def size(o) -> int:
            return len(json.dumps(o)) + 2                 # + ", " separator

        peak = {"index": 999, "mz": w, "intensity": w, "annotation": {
            "ion_type": "precursor", "ion_index": [999, 999], "charge": 99,
            "neutral_loss": "H2O+NH3", "label": "m999:999" + "+" * 99 + "-H2O+NH3",
            "concepts": list(self.registry.names)}}
        residue = {"token": "x" * 16, "log_prob_tf": w, "log_prob_beam": w}
        token = {"position": 999, "kind": "latent", "l0": 99999, "recon_rel_error": w,
                 "top": [{"feature": 99999, "activation": w}] * self.top_k}
        longest_concept = max(
            [len(n) for n in self.registry.names]
            + [len(str(d.lookup(f).get("concept_1") or "")) for d in self.dictionaries.values() for f in d])
        summary = {**{k: w for k in SUMMARY_KEYS}, **{k: w for k in SPECTRUM_SUMMARY_KEYS},
                   "in_dictionary": False, "has_description": False,
                   "has_significant_association": False, "description_blinded": False,
                   "description_beats_baseline": False, "fires_on_latent": False,
                   "top_concept_meets_display_rule": False,
                   "top_concept": "x" * longest_concept, "reliable_concept": "x" * longest_concept,
                   "n_tokens_active": 999}
        summary_size = size(summary) + 8                  # + '"12345": '
        result = {"feature": 99999, "positions": [999], "activation": w,
                  "delta_log_prob": [w] * self.max_length, "delta_eos": w, "delta_total": w,
                  "argmax_changed": [False] * self.max_length, "eos_argmax_changed": False}
        n_tokens = self.n_peaks + 1
        bound = 64_000 + self.n_peaks * size(peak) + self.max_length * (size(residue) + 2 * 18)
        for L, d in self.dictionaries.items():
            bound += n_tokens * size(token) + min(n_tokens * self.top_k, len(d)) * summary_size
            if latent_counts is not None:
                n = latent_counts.get(L, 0)
                bound += 4_000 + (n + 1) * size(result) + n * summary_size
        return int(bound)


def _repeat_batch(batch: dict, n: int) -> dict:
    """A size-1 model batch repeated n times along the batch dimension."""
    return {k: (v.repeat(n, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def feature_row(explanation: Explanation, dictionaries: dict[int, FeatureDictionary],
                layer: int, feature: int) -> dict | None:
    """The full dictionary row behind a feature summary in `explanation`.

    Raises if the loaded dictionary is not the file the explanation was built
    from (its sha256 differs): the row could then disagree with the summary.
    """
    expected = explanation.provenance["dictionary"][str(layer)]["sha256_16"]
    d = dictionaries[layer]
    if d.sha256_16 != expected:
        raise ValueError(f"layer {layer}: dictionary {d.path} (sha256 {d.sha256_16}) is not the one "
                         f"this explanation was built from ({expected})")
    return d.lookup(feature)


_DEFAULT: Inspector | None = None


def explain(spectrum: Spectrum, latent_ablation: bool = False, inspector: Inspector | None = None,
            **inspector_kwargs) -> Explanation:
    """explain() with a cached default Inspector, built from inspector_kwargs on first use."""
    global _DEFAULT
    if inspector is None:
        if _DEFAULT is None:
            _DEFAULT = Inspector(**inspector_kwargs)
        inspector = _DEFAULT
    return inspector.explain(spectrum, latent_ablation=latent_ablation)


# --- CLI ----------------------------------------------------------------------

def _cmd_time(args, inspector: Inspector) -> None:
    print(f"cold load: {inspector.load_seconds:.2f} s")
    print(f"json size bound: {inspector.json_size_bound:,} bytes")
    for spec in read_raw_spectra(args.parquet, args.rows):
        t = time.perf_counter()
        exp = inspector.explain(spec)
        total = time.perf_counter() - t
        t = time.perf_counter()
        text = exp.to_json()
        ser = time.perf_counter() - t
        stages = ", ".join(f"{k} {v:.3f}" for k, v in exp.timings_s.items())
        pred = exp.prediction
        gap = (max(abs(r["log_prob_tf"] - r["log_prob_beam"]) for r in pred["residues"]
                   if r["log_prob_beam"] is not None) if pred else float("nan"))
        print(f"{spec.spectrum_id}: total {total:.3f} s ({stages}, serialise {ser:.3f}); "
              f"{exp.input['n_processed_peaks']} peaks; prediction "
              f"{pred['peptide'] if pred else None}; json {len(text):,} bytes; "
              f"max |logp_tf - logp_beam| {gap:.2e}")
        if args.ablation and pred:
            _time_ablation(inspector, spec, exp)


def _time_ablation(inspector: Inspector, spec: Spectrum, exp: Explanation) -> None:
    """V1-4b timings: one ablation per scope, per-click times of the per-peak
    view (largest-L0 and median-L0 peak) and the latent panel per layer, and
    the explanation's size with latent ablation."""
    t0 = time.perf_counter()
    ctx = inspector.ablation_context(spec, exp)
    print(f"  ablation context {time.perf_counter() - t0:.3f} s")
    for L in inspector.layers:
        feats = ctx.features[L]
        l0 = (feats > 0).sum(dim=1)
        peak_l0 = l0[1:]
        big = int(peak_l0.argmax()) + 1
        med = int(peak_l0.argsort()[len(peak_l0) // 2]) + 1
        f_loc = int(feats[med].argmax())
        t0 = time.perf_counter()
        inspector.ablate(ctx, L, [f_loc], "local", med, with_summaries=False)
        t_local = time.perf_counter() - t0
        t0 = time.perf_counter()
        inspector.ablate(ctx, L, [f_loc], "global", with_summaries=False)
        t_global = time.perf_counter() - t0
        line = [(f"  L{L}: one ablation local {t_local:.3f} s, global {t_global:.3f} s "
                 f"(F{f_loc}, {int((feats[:, f_loc] > 0).sum())} tokens)")]
        for name, pos in (("peak max-L0", big), ("peak median-L0", med), ("latent", 0)):
            t0 = time.perf_counter()
            rep = inspector.ablate_token(ctx, L, pos)
            t_click = time.perf_counter() - t0
            joint = rep.joint_delta_total
            extra = (f" [joint {joint:+.3f}, sum of singles {rep.sum_of_singles:+.3f}]"
                     if joint is not None else "")
            line.append(f"{name} (token {pos}, L0 {int(l0[pos])}) {t_click:.2f} s{extra}")
        print("; ".join(line))
    t0 = time.perf_counter()
    exp2 = inspector.explain(spec, latent_ablation=True)
    t_exp2 = time.perf_counter() - t0
    text2 = exp2.to_json()
    counts = {int(L): len(r["results"]) for L, r in exp2.ablation["layers"].items()}
    print(f"  explain(latent_ablation=True) {t_exp2:.2f} s (latent ablation "
          f"{exp2.timings_s['latent_ablation']:.2f} s); json {len(text2):,} bytes; latent L0 {counts}; "
          f"bound {inspector.size_bound(counts):,}")


def _cmd_compare(args, inspector: Inspector) -> None:
    n_diff = 0
    spectra = read_raw_spectra(args.parquet, args.rows)
    for spec in spectra:
        a = inspector.predict(spec, use_spectrum_utils=True)["tokens"]
        b = inspector.predict(spec, use_spectrum_utils=False)["tokens"]
        differ = a != b
        n_diff += differ
        print(f"{spec.spectrum_id}: {'DIFFER' if differ else 'same  '} "
              f"True={''.join(a)!r} False={''.join(b)!r}")
    print(f"{n_diff} of {len(spectra)} predictions differ between use_spectrum_utils True and False")


# --- Accuracy check --------------------------------------------------------------

def normalise_peptide(tokens: list[str]) -> list[str]:
    """Tokens compared at peptide level: I and L are one residue (isobaric)."""
    return ["L" if t == "I" else t for t in tokens]


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def accuracy_check(inspector: Inspector, parquet_path: Path, rows: list[int],
                   decoders: tuple[str, ...] = ("beam", "greedy"), precursor_ppm: float = 20.0) -> dict:
    """Peptide-level accuracy of InstaNovo's prediction against modified_sequence.

    exact: predicted tokens equal the true ones after mapping the dataset's
    notation to InstaNovo's ("M(+15.99)" -> "M[UNIMOD:35]") and I -> L.
    instanovo_recall: InstaNovo's own peptide recall (Metrics.compute_precision_recall,
    mass-matched residues; I/L and other near-isobaric pairs count as equal).
    meets_precursor: min |ppm| over isotope offsets 0..1 <= precursor_ppm (20,
    InstaNovo's filter_precursor_ppm), via Metrics.matches_precursor.
    """
    rs = inspector.residue_set
    metrics = instanovo_io.instanovo_metrics(rs)
    records = read_raw_records(parquet_path, rows, _SPECTRUM_COLUMNS + ["modified_sequence"])
    truths = [[rs.residue_remapping.get(t, t) for t in rs.tokenize(r["modified_sequence"])]
              for r in records]
    modified = [any("[" in t for t in tr) for tr in truths]
    report = {"n": len(rows), "parquet": str(parquet_path), "precursor_ppm": precursor_ppm,
              "n_modified_truth": int(sum(modified)), "decoders": {}}
    for dec in decoders:
        preds, seconds, exact, meets = [], [], [], []
        for row, rec, truth in zip(rows, records, truths, strict=True):
            spec = Spectrum(rec["mz_array"], rec["intensity_array"], rec["precursor_mz"],
                            rec["precursor_charge"], spectrum_id=str(row))
            t0 = time.perf_counter()
            tokens = inspector.predict(spec, decoder=dec)["tokens"]
            seconds.append(time.perf_counter() - t0)
            preds.append(tokens)
            exact.append(bool(tokens) and normalise_peptide(tokens) == normalise_peptide(truth))
            if tokens:
                _ok, ppm = metrics.matches_precursor(tokens, rec["precursor_mz"], rec["precursor_charge"])
                meets.append(min(abs(p) for p in ppm) <= precursor_ppm)
            else:
                meets.append(False)
        _aa_prec, aa_recall, pep_recall, _pep_prec = metrics.compute_precision_recall(truths, preds)
        n, k = len(rows), int(sum(exact))
        meet_idx = [i for i, m in enumerate(meets) if m]
        report["decoders"][dec] = {
            "exact": k, "exact_rate": k / n, "exact_ci95": wilson_interval(k, n),
            "instanovo_peptide_recall": pep_recall, "instanovo_aa_recall": aa_recall,
            "n_empty": sum(not p for p in preds),
            "meets_precursor": len(meet_idx),
            "exact_among_meeting": sum(exact[i] for i in meet_idx),
            "exact_unmodified": sum(e for e, m in zip(exact, modified, strict=True) if not m),
            "exact_modified": sum(e for e, m in zip(exact, modified, strict=True) if m),
            "seconds_median": float(np.median(seconds)), "seconds_total": float(sum(seconds)),
            "examples_wrong": [("".join(truths[i]), "".join(preds[i]))
                               for i in range(n) if not exact[i]][:10],
        }
    return report


def _cmd_accuracy(args, inspector: Inspector) -> None:
    if args.rows:
        rows = sorted(args.rows)
    else:
        rows = sorted(np.random.default_rng(args.seed).choice(
            parquet_num_rows(args.parquet), size=args.n, replace=False).tolist())
    rep = accuracy_check(inspector, args.parquet, rows, decoders=tuple(args.decoders))
    n = rep["n"]
    print(f"{n} spectra from {Path(args.parquet).name} (seed {args.seed}; rows {rows[0]}..{rows[-1]}); "
          f"{rep['n_modified_truth']} with a modified true peptide; num_beams {inspector.num_beams}, "
          f"max_length {inspector.max_length}")
    for dec, r in rep["decoders"].items():
        lo, hi = r["exact_ci95"]
        print(f"[{dec}] exact (I=L) {r['exact']}/{n} = {r['exact_rate']:.3f} [95% CI {lo:.3f}, {hi:.3f}]; "
              f"InstaNovo peptide recall {r['instanovo_peptide_recall']:.3f}, aa recall "
              f"{r['instanovo_aa_recall']:.3f}; meets precursor (<= {rep['precursor_ppm']:g} ppm, "
              f"isotope 0-1) {r['meets_precursor']}/{n}, exact among those {r['exact_among_meeting']}; "
              f"exact unmodified {r['exact_unmodified']}/{n - rep['n_modified_truth']}, modified "
              f"{r['exact_modified']}/{rep['n_modified_truth']}; empty {r['n_empty']}; "
              f"median {r['seconds_median']:.2f} s/spectrum")
        for truth, pred in r["examples_wrong"][:5]:
            print(f"    truth {truth}  pred {pred}")


# --- Audit case selection (V1-4c part A) -----------------------------------------
#
# A fixed rule picks InstaNovo errors for the audit case study, before any
# ablation: wrong predictions (I=L) that meet the precursor, ranked by mean
# per-residue log p, each annotated against both the predicted and the true
# peptide to count distinguishing peaks. The pure helpers are unit-tested.

FRAGMENT_ION_TYPES = ("b", "y", "I", "internal")   # annotator ion types that are fragments
BACKBONE_ION_TYPES = ("b", "y")
EVIDENCE_AMBIGUOUS = "ambiguous"
EVIDENCE_FOR_TRUE = "evidence for true"
EVIDENCE_FOR_PRED = "evidence favours prediction"
V1_4A_BEAM_EXACT, V1_4A_BEAM_MEETS = 123, 167        # the V1-4a record (n 200, seed 0)
AUDIT_SELECTION_RULE = (
    "candidates: beam prediction wrong (I=L, modified_sequence) and meeting the precursor "
    "(min |ppm| over isotopes 0-1 <= 20); ranked by mean teacher-forced log p over the "
    "predicted residues (EOS excluded), highest first, ties by row; the top n_top are "
    "selected, then the next-ranked 'evidence for true' candidates until min_evidence "
    "selected cases are 'evidence for true'. Distinguishing peak: a processed peak labelled "
    "a fragment (b, y, immonium, internal) under one peptide and not under the other "
    "(annotate_spectrum, strict, 20 ppm)."
)


def differing_region(a: list[str], b: list[str]) -> tuple[int, int, int]:
    """(start, end_a, end_b): the regions a[start:end_a] and b[start:end_b] lie
    between the longest common prefix and the longest common suffix (I = L),
    the suffix not overlapping the prefix. The regions may differ in length and
    one may be empty. Raises for sequences equal up to I/L."""
    na, nb = normalise_peptide(a), normalise_peptide(b)
    if na == nb:
        raise ValueError(f"no differing region: {a} and {b} are equal up to I/L")
    n = min(len(na), len(nb))
    start = 0
    while start < n and na[start] == nb[start]:
        start += 1
    suffix = 0
    while suffix < n - start and na[-1 - suffix] == nb[-1 - suffix]:
        suffix += 1
    return start, len(a) - suffix, len(b) - suffix


def _base_residue(token: str) -> str:
    return token[0] if token[0].isalpha() else ""


def mismatch_type(a: list[str], b: list[str], region: tuple[int, int, int]) -> str:
    """Class of the difference in the region, checked in order: 'length change'
    (region lengths differ); 'modification' (every differing position has the
    same base residue); 'swap' (two or more tokens, same multiset);
    'substitution' (every differing position has a different base residue);
    'other' (a mix of modification-only and residue changes). I = L."""
    start, end_a, end_b = region
    ra, rb = normalise_peptide(a)[start:end_a], normalise_peptide(b)[start:end_b]
    if len(ra) != len(rb):
        return "length change"
    diff = [i for i in range(len(ra)) if ra[i] != rb[i]]
    if not diff:
        raise ValueError(f"region {region} of {a} / {b} holds no difference")
    same_base = [_base_residue(ra[i]) == _base_residue(rb[i]) for i in diff]
    if all(same_base):
        return "modification"
    if len(ra) >= 2 and sorted(ra) == sorted(rb):
        return "swap"
    if not any(same_base):
        return "substitution"
    return "other"


def distinguishing_peaks(ann_pred: list[dict], ann_true: list[dict]) -> dict:
    """Peaks labelled a fragment (FRAGMENT_ION_TYPES) under one peptide and not
    under the other (noise or precursor). One label per peak, as the annotator
    keeps it. Indices are peak indices (encoder token = index + 1)."""
    if len(ann_pred) != len(ann_true):
        raise ValueError(f"{len(ann_pred)} vs {len(ann_true)} peak annotations")

    def frag(ann):
        return ann["ion_type"] in FRAGMENT_ION_TYPES

    pred_only = [j for j, (p, t) in enumerate(zip(ann_pred, ann_true)) if frag(p) and not frag(t)]
    true_only = [j for j, (p, t) in enumerate(zip(ann_pred, ann_true)) if frag(t) and not frag(p)]
    return {"pred_only": pred_only, "true_only": true_only,
            "n_pred_only": len(pred_only), "n_true_only": len(true_only),
            "n_pred_only_by": sum(ann_pred[j]["ion_type"] in BACKBONE_ION_TYPES for j in pred_only),
            "n_true_only_by": sum(ann_true[j]["ion_type"] in BACKBONE_ION_TYPES for j in true_only)}


def evidence_class(n_pred_only: int, n_true_only: int) -> str:
    """'ambiguous' (no distinguishing peak), 'evidence for true' (at least one
    peak is a fragment only under the true peptide), else 'evidence favours
    prediction' (distinguishing peaks only under the prediction)."""
    if n_true_only > 0:
        return EVIDENCE_FOR_TRUE
    if n_pred_only > 0:
        return EVIDENCE_FOR_PRED
    return EVIDENCE_AMBIGUOUS


EVIDENCE_BY_COLUMNS = ("rank", "row", "selected", "n_pred_only", "n_pred_only_by", "n_true_only",
                       "n_true_only_by", "evidence_class", "evidence_class_by", "changed")


def backbone_evidence_classes(candidates: list[dict]) -> list[dict]:
    """V1-4c amendment 1: the evidence class recomputed on distinguishing b/y
    peaks only (`n_pred_only_by`, `n_true_only_by`), same rule otherwise.
    Internal-fragment matches are at chance level and immonium labels are
    peptide-independent (Finding 9), so only b/y peaks are kept. Uses the
    stored counts: a peak counts when it is b/y under one peptide and noise or
    precursor under the other; a peak that is b/y under one and internal or
    immonium under the other is not counted. Raises if the stored counts do not
    reproduce the stored class. Rows are candidates.csv rows (strings or numbers)."""
    out = []
    for c in candidates:
        n = {k: int(c[k]) for k in ("n_pred_only", "n_pred_only_by", "n_true_only", "n_true_only_by")}
        stored = evidence_class(n["n_pred_only"], n["n_true_only"])
        if stored != c["evidence_class"]:
            raise ValueError(f"row {c['row']}: stored class {c['evidence_class']!r} is not reproduced "
                             f"by its counts ({stored!r})")
        by = evidence_class(n["n_pred_only_by"], n["n_true_only_by"])
        selected = c["selected"] if isinstance(c["selected"], bool) else str(c["selected"]).lower() == "true"
        out.append({"rank": int(c["rank"]), "row": int(c["row"]), "selected": selected, **n,
                    "evidence_class": stored, "evidence_class_by": by, "changed": by != stored})
    return out


def mean_residue_log_prob(ce: list[float], n_residues: int) -> float:
    """Mean log p over the residues from decoder-order CE (residues then EOS); EOS excluded."""
    per_residue, _eos = decoder_to_residue_order([-float(v) for v in ce], n_residues)
    return float(np.mean(per_residue))


def select_audit_cases(candidates: list[dict], n_top: int = 5,
                       min_evidence: int = 2) -> tuple[list[dict], dict]:
    """Rank candidates (dicts with row, mean_residue_log_prob, evidence_class)
    by mean residue log p, highest first (ties by row); select the top n_top,
    then the next-ranked 'evidence for true' candidates until min_evidence
    selected cases are 'evidence for true'. Returns (ranked copies with rank,
    selected, selection_reason; a summary with any shortfall)."""
    ranked = sorted((dict(c) for c in candidates),
                    key=lambda c: (-c["mean_residue_log_prob"], c["row"]))
    n_for_true = 0
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
        c["selected"] = i < n_top
        c["selection_reason"] = f"top {n_top}" if i < n_top else ""
        n_for_true += c["selected"] and c["evidence_class"] == EVIDENCE_FOR_TRUE
    for c in ranked[n_top:]:
        if n_for_true >= min_evidence:
            break
        if c["evidence_class"] == EVIDENCE_FOR_TRUE:
            c["selected"], c["selection_reason"] = True, f"added: {EVIDENCE_FOR_TRUE}"
            n_for_true += 1
    summary = {"n_candidates": len(ranked), "n_top": n_top, "min_evidence": min_evidence,
               "n_selected": sum(c["selected"] for c in ranked),
               "n_selected_evidence_for_true": n_for_true,
               "shortfall": max(0, min_evidence - n_for_true)}
    return ranked, summary


def audit_candidates(inspector: Inspector, parquet_path: Path, rows: list[int],
                     precursor_ppm: float = 20.0) -> tuple[list[dict], list[dict]]:
    """(predictions, candidates). Beam prediction for every row (Inspector
    settings), exact and meets-precursor as in accuracy_check. Each wrong
    prediction meeting the precursor becomes a candidate: teacher-forced mean
    residue log p, the differing region, mismatch type, region mass difference,
    and distinguishing peaks from annotating the processed peaks against both
    peptides. Candidates carry the annotations under '_ann_pred' / '_ann_true'."""
    rs = inspector.residue_set
    metrics = instanovo_io.instanovo_metrics(rs)
    records = read_raw_records(parquet_path, rows, _SPECTRUM_COLUMNS + ["modified_sequence"])
    predictions, candidates = [], []
    for row, rec in zip(rows, records, strict=True):
        truth = [rs.residue_remapping.get(t, t) for t in rs.tokenize(rec["modified_sequence"])]
        spec = Spectrum(rec["mz_array"], rec["intensity_array"], rec["precursor_mz"],
                        rec["precursor_charge"], spectrum_id=f"{Path(parquet_path).stem}:{row}")
        pred = inspector.predict(spec)
        tokens = pred["tokens"]
        exact = bool(tokens) and normalise_peptide(tokens) == normalise_peptide(truth)
        min_ppm = None
        if tokens:
            _ok, ppm = metrics.matches_precursor(tokens, rec["precursor_mz"], rec["precursor_charge"])
            min_ppm = float(min(abs(p) for p in ppm))
        meets = min_ppm is not None and min_ppm <= precursor_ppm
        p = {"row": row, "truth": "".join(truth), "prediction": "".join(tokens), "exact": exact,
             "meets_precursor": meets, "min_abs_ppm": min_ppm, "beam_log_prob": pred["log_prob"],
             "n_residues": len(tokens), "modified_truth": any("[" in t for t in truth)}
        predictions.append(p)
        if exact or not meets:
            continue
        fwd = inspector.forward(spec, "".join(tokens))
        ann_pred, _ = inspector._annotate(tokens, spec, fwd)
        ann_true, _ = inspector._annotate(truth, spec, fwd)
        region = differing_region(tokens, truth)
        start, end_p, end_t = region
        dist = distinguishing_peaks(ann_pred, ann_true)
        candidates.append({
            **p, "pred_tokens": tokens, "truth_tokens": truth,
            "mean_residue_log_prob": mean_residue_log_prob(fwd.ce.tolist(), len(tokens)),
            "region_start": start, "region_end_pred": end_p, "region_end_true": end_t,
            "pred_region": "".join(tokens[start:end_p]), "true_region": "".join(truth[start:end_t]),
            "mismatch_type": mismatch_type(tokens, truth, region),
            "region_mass_delta_da": (sum(float(rs.get_mass(t)) for t in tokens[start:end_p])
                                     - sum(float(rs.get_mass(t)) for t in truth[start:end_t])),
            **{k: dist[k] for k in ("n_pred_only", "n_true_only", "n_pred_only_by", "n_true_only_by")},
            "pred_only_peaks": dist["pred_only"], "true_only_peaks": dist["true_only"],
            "evidence_class": evidence_class(dist["n_pred_only"], dist["n_true_only"]),
            "n_processed_peaks": len(ann_pred), "_ann_pred": ann_pred, "_ann_true": ann_true,
        })
    return predictions, candidates


def _cleavage_site(ann: dict, n_residues: int) -> int | None:
    """Backbone cleavage site k (between residues k-1 and k, 0-indexed) of a b/y annotation."""
    if ann["ion_type"] == "b":
        return int(ann["ion_index"])
    if ann["ion_type"] == "y":
        return n_residues - int(ann["ion_index"])
    return None


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def case_view(inspector: Inspector, exp: Explanation, cand: dict) -> str:
    """Markdown report of one audit case from explain() output and the two
    annotations: what the hypothesis is written from. No ablation."""
    pred, truth = cand["pred_tokens"], cand["truth_tokens"]
    start, end_p, end_t = cand["region_start"], cand["region_end_pred"], cand["region_end_true"]
    ann_p, ann_t = cand["_ann_pred"], cand["_ann_true"]

    def marked(tokens, end):
        return "".join(tokens[:start]) + " [" + "".join(tokens[start:end]) + "] " + "".join(tokens[end:])

    out = [f"# Row {cand['row']} — {cand['evidence_class']}", "",
           f"- prediction: `{marked(pred, end_p)}`",
           f"- true:       `{marked(truth, end_t)}`",
           f"- mismatch: {cand['mismatch_type']}; region mass Δ (pred − true) "
           f"{cand['region_mass_delta_da']:+.4f} Da; min |ppm| {cand['min_abs_ppm']:.2f}; "
           f"mean residue log p {cand['mean_residue_log_prob']:.4f}; rank {cand.get('rank')}",
           f"- distinguishing peaks: pred-only {cand['n_pred_only']} (b/y {cand['n_pred_only_by']}) "
           f"{cand['pred_only_peaks']}; true-only {cand['n_true_only']} (b/y {cand['n_true_only_by']}) "
           f"{cand['true_only_peaks']}",
           f"- precursor charge {exp.input['precursor_charge']}, {exp.input['n_processed_peaks']} "
           f"processed peaks (peak j = encoder token j + 1)", "",
           "## Predicted residues (teacher-forced log p)", "",
           "| i | token | log p | in region |", "|---|---|---|---|"]
    for i, r in enumerate(exp.prediction["residues"]):
        out.append(f"| {i} | {r['token']} | {r['log_prob_tf']:.4f} | {'*' if start <= i < end_p else ''} |")
    out.append(f"| EOS | | {exp.prediction['eos_log_prob_tf']:.4f} | |")

    near = set(cand["pred_only_peaks"]) | set(cand["true_only_peaks"])
    for j, (a, b) in enumerate(zip(ann_p, ann_t)):
        kp, kt = _cleavage_site(a, len(pred)), _cleavage_site(b, len(truth))
        if (kp is not None and start <= kp <= end_p) or (kt is not None and start <= kt <= end_t):
            near.add(j)
    intensity_rank = {j: r + 1 for r, j in enumerate(
        sorted(range(len(exp.peaks)), key=lambda j: -exp.peaks[j]["intensity"]))}
    out += ["", "## Peaks near the region", "",
            "Distinguishing peaks plus peaks whose b/y cleavage site lies in or at the edges of "
            "the region under either peptide.", "",
            "| peak | m/z | intensity (rank) | under pred | under true | distinguishing |",
            "|---|---|---|---|---|---|"]
    for j in sorted(near):
        pk = exp.peaks[j]
        tag = ("pred-only" if j in cand["pred_only_peaks"] else
               "true-only" if j in cand["true_only_peaks"] else "")
        out.append(f"| {j} | {pk['mz']:.4f} | {pk['intensity']:.4g} ({intensity_rank[j]}) | "
                   f"{peak_label(ann_p[j])} | {peak_label(ann_t[j])} | {tag} |")

    used: dict[int, set[int]] = {L: set() for L in inspector.layers}

    def feature_table(position: int) -> list[str]:
        rows = ["| layer | feature | act | top concept | F1dom | sig | ΔCE (Phase 8) | lat. frac | desc |",
                "|---|---|---|---|---|---|---|---|---|"]
        for L in inspector.layers:
            for item in exp.layers[str(L)]["tokens"][position]["top"]:
                f = item["feature"]
                s = exp.features[str(L)][str(f)]
                used[L].add(f)
                desc = ("–" if not s["has_description"] else
                        ("blind, beats" if s["description_beats_baseline"] else "blind, not beats")
                        if s["description_blinded"] else "unblinded")
                rows.append(f"| L{L} | F{f} | {item['activation']:.3f} | {_fmt(s['top_concept'])} | "
                            f"{_fmt(s['top_concept_f1_dom'])} | {_fmt(s['has_significant_association'])} | "
                            f"{_fmt(s['causal_delta_ce'], 4) if s['causal_delta_ce'] is not None else 'untested'} | "
                            f"{_fmt(s['latent_token_fraction'])} | {desc} |")
        return rows

    out += ["", "## Latent token (token 0), top features per layer", ""] + feature_table(0)
    for j in sorted(near):
        out += ["", f"## Peak {j} (token {j + 1}): m/z {exp.peaks[j]['mz']:.4f}, "
                    f"pred {peak_label(ann_p[j])}, true {peak_label(ann_t[j])}", ""] + feature_table(j + 1)

    out += ["", "## Descriptions of the features above", "",
            "Layer 8 has no descriptions yet (V1-2c). Only blinded descriptions are in the dictionary.", ""]
    for L in inspector.layers:
        for f in sorted(used[L]):
            row = feature_row(exp, inspector.dictionaries, L, f)
            if row and row.get("description"):
                out.append(f"- **L{L} F{f}** (r {_fmt(row.get('description_r'))}, beats baseline "
                           f"{_fmt(row.get('description_beats_baseline'))}): {row['description']}")
    return "\n".join(out) + "\n"


# --- V1-4c descriptive checks (no ablation) ----------------------------------------

POOL_SIZE = 10               # control pool: nearest-rank non-named active features
POOL_MIN_RATIO = 0.5         # pool members need activation >= 0.5 x the named one (judgement call)
POOL_MIN_SIZE = 3            # fewer qualifying members -> the item's control is underpowered


def control_pool(row_acts: dict[int, float], named: set[int], feature: int,
                 size: int = POOL_SIZE, min_ratio: float = POOL_MIN_RATIO) -> list[int]:
    """Activation-matched control pool for one named (feature, token): the
    `size` non-named active features at the token nearest in activation rank to
    `feature`, among those with activation >= min_ratio x its activation (ties
    in rank distance go to the higher activation)."""
    order = sorted(row_acts, key=lambda f: -row_acts[f])
    rank = {f: i for i, f in enumerate(order)}
    floor = min_ratio * row_acts[feature]
    cands = [f for f in order if f not in named and row_acts[f] >= floor]
    return sorted(cands, key=lambda f: (abs(rank[f] - rank[feature]), rank[f]))[:size]


def n_distinct_control_sets(pools: list[tuple[int, list[int]]], cap: int = 10_000) -> int:
    """Distinct control sets: one member per named item from its pool, no
    (feature, token) pair twice. pools = [(token, pool)]. Counted up to `cap`."""
    seen: set[frozenset] = set()

    def walk(i, chosen):
        if len(seen) >= cap:
            return
        if i == len(pools):
            seen.add(frozenset(chosen))
            return
        tok, pool = pools[i]
        for f in pool:
            if (f, tok) not in chosen:
                walk(i + 1, chosen | {(f, tok)})

    walk(0, frozenset())
    return len(seen)


# peptide_decoys (V1-4c check (c)'s decoy rule) lives in annotate, beside the
# label audit that also uses it; re-exported here for the audit checks.
peptide_decoys = annotate.peptide_decoys


def _is_fragment(ann: dict) -> bool:
    return ann["ion_type"] in FRAGMENT_ION_TYPES


def audit_descriptive_checks(inspector: Inspector, parquet_path: Path, candidates_csv: Path,
                             hypotheses: dict, isotope_rows: list[int], decoy_rows: list[int],
                             n_decoys: int = 20, seed: int = 0) -> dict:
    """V1-4c checks before part B, none of which ablates:
    isotope: each row re-annotated under the prediction with max_isotope=1;
      true-only peaks (fragment under the true peptide, not under the
      prediction at max_isotope 0) that become a fragment of the prediction
      (necessarily an isotope-1 match), plus pred-only vs true-only b/y counts
      and the precursor ppm at isotope offsets 0 and 1.
    decoys: peaks labelled internal (and any fragment) under n same-composition
      decoys of the prediction, against the prediction and the true peptide.
    pools: per hypothesis item, L0 at the token, the named feature's activation
      rank, the control pool (control_pool) and whether it is underpowered;
      per hypothesis, the number of distinct control sets."""
    import csv
    rs = inspector.residue_set
    metrics = instanovo_io.instanovo_metrics(rs)
    cands = {int(r["row"]): r for r in csv.DictReader(open(candidates_csv, encoding="utf-8"))}
    cfg = inspector.annotation_config

    def load(row):
        rec = read_raw_records(parquet_path, [row], _SPECTRUM_COLUMNS + ["modified_sequence"])[0]
        spec = Spectrum(rec["mz_array"], rec["intensity_array"], rec["precursor_mz"],
                        rec["precursor_charge"], spectrum_id=f"{Path(parquet_path).stem}:{row}")
        pred = rs.tokenize(cands[row]["prediction"])
        truth = [rs.residue_remapping.get(t, t) for t in rs.tokenize(rec["modified_sequence"])]
        if "".join(truth) != cands[row]["truth"]:
            raise RuntimeError(f"row {row}: true peptide differs from candidates.csv")
        return rec, spec, pred, truth, inspector.forward(spec, "".join(pred))

    isotope = []
    for row in isotope_rows:
        rec, spec, pred, truth, fwd = load(row)
        ann_p, _ = inspector._annotate(pred, spec, fwd)
        ann_t, _ = inspector._annotate(truth, spec, fwd)
        _bare, proforma, _mods = peptide_for_annotation(pred, rs)
        ann_p1 = annotate.run_spectrum_utils(
            proforma=proforma, mz_array=fwd.peaks_mz, intensity_array=fwd.peaks_intensity,
            precursor_mz=spec.precursor_mz, precursor_charge=spec.precursor_charge,
            ion_types=cfg.ion_types, fragment_tol_mass=cfg.fragment_tol_mass,
            fragment_tol_mode=cfg.fragment_tol_mode, enable_internal=cfg.enable_internal,
            strict=True, max_isotope=1)
        d = distinguishing_peaks(ann_p, ann_t)
        became = [j for j in d["true_only"] if _is_fragment(ann_p1[j])]
        _ok, ppm = metrics.matches_precursor(pred, rec["precursor_mz"], rec["precursor_charge"])
        isotope.append({
            "row": row, "rank": int(cands[row]["rank"]), "prediction": "".join(pred), "truth": "".join(truth),
            "precursor_ppm_isotope_0": float(ppm[0]), "precursor_ppm_isotope_1": float(ppm[1]),
            "n_pred_only": d["n_pred_only"], "n_pred_only_by": d["n_pred_only_by"],
            "n_true_only": d["n_true_only"], "n_true_only_by": d["n_true_only_by"],
            "true_only_become_pred_isotope": len(became),
            "true_only_become_pred_isotope_by": sum(ann_t[j]["ion_type"] in BACKBONE_ION_TYPES for j in became),
            "became": [{"peak": j, "mz": float(fwd.peaks_mz[j]), "true_label": peak_label(ann_t[j]),
                        "pred_isotope_label": peak_label(ann_p1[j])} for j in became],
        })

    decoys = []
    for row in decoy_rows:
        rec, spec, pred, truth, fwd = load(row)
        rng = np.random.default_rng([seed, row])
        real = {}
        for name, toks in (("prediction", pred), ("truth", truth)):
            ann, _ = inspector._annotate(toks, spec, fwd)
            real[name] = {"internal": sum(a["ion_type"] == "internal" for a in ann),
                          "fragment": sum(_is_fragment(a) for a in ann)}
        per = []
        for dtok in peptide_decoys(pred, n_decoys, rng, exclude=[pred, truth]):
            ann, _ = inspector._annotate(dtok, spec, fwd)
            per.append({"decoy": "".join(dtok), "internal": sum(a["ion_type"] == "internal" for a in ann),
                        "fragment": sum(_is_fragment(a) for a in ann)})
        internal = [p["internal"] for p in per]
        frag = [p["fragment"] for p in per]
        decoys.append({"row": row, "n_peaks": len(fwd.peaks_mz), "real": real, "n_decoys": len(per),
                       "decoy_internal_mean": float(np.mean(internal)), "decoy_internal_min": min(internal),
                       "decoy_internal_max": max(internal), "decoy_fragment_mean": float(np.mean(frag)),
                       "decoy_fragment_min": min(frag), "decoy_fragment_max": max(frag), "decoys": per})

    pools = []
    for h in hypotheses["hypotheses"]:
        rec, spec, pred, truth, fwd = load(h["row"])
        L = h["layer"]
        with torch.inference_mode():
            feats = inspector.saes[L].forward_inference(fwd.activations[L].to(inspector.device))["features"].cpu()
        named_by_token: dict[int, set[int]] = {}
        for it in h["features"]:
            named_by_token.setdefault(it["token"], set()).add(it["feature"])
        items, token_pools = [], []
        for it in h["features"]:
            row_t = feats[it["token"]]
            acts = {int(f): float(row_t[f]) for f in (row_t > 0).nonzero().flatten().tolist()}
            pool = control_pool(acts, named_by_token[it["token"]], it["feature"])
            order = sorted(acts, key=lambda f: -acts[f])
            items.append({"feature": it["feature"], "token": it["token"], "l0": len(acts),
                          "activation": acts[it["feature"]], "rank": order.index(it["feature"]) + 1,
                          "pool": pool, "pool_activations": [acts[f] for f in pool],
                          "underpowered": len(pool) < POOL_MIN_SIZE})
            token_pools.append((it["token"], pool))
        pools.append({"id": h["id"], "row": h["row"], "layer": L, "items": items,
                      "n_distinct_control_sets": n_distinct_control_sets(token_pools)})
    return {"isotope": isotope, "decoys": decoys, "pools": pools,
            "settings": {"n_decoys": n_decoys, "seed": seed, "pool_size": POOL_SIZE,
                         "pool_min_ratio": POOL_MIN_RATIO, "pool_min_size": POOL_MIN_SIZE}}


def _cmd_audit_checks(args, inspector: Inspector) -> int:
    """V1-4c descriptive checks (audit_descriptive_checks) -> descriptive_checks.json."""
    out = Path(args.out_dir)
    hyp = json.loads((out / "hypotheses.json").read_text(encoding="utf-8"))
    rep = audit_descriptive_checks(inspector, args.parquet, out / "candidates.csv", hyp,
                                   isotope_rows=args.isotope_rows, decoy_rows=args.decoy_rows)
    (out / "descriptive_checks.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for r in rep["isotope"]:
        print(f"isotope row {r['row']} (rank {r['rank']}): ppm iso0 {r['precursor_ppm_isotope_0']:+.1f}, "
              f"iso1 {r['precursor_ppm_isotope_1']:+.1f}; pred-only {r['n_pred_only']} (b/y {r['n_pred_only_by']}), "
              f"true-only {r['n_true_only']} (b/y {r['n_true_only_by']}); true-only -> pred isotope "
              f"{r['true_only_become_pred_isotope']} (b/y {r['true_only_become_pred_isotope_by']})")
    for d in rep["decoys"]:
        print(f"decoys row {d['row']} ({d['n_peaks']} peaks): real internal {d['real']}, decoy internal "
              f"mean {d['decoy_internal_mean']:.1f} [{d['decoy_internal_min']}, {d['decoy_internal_max']}], "
              f"decoy fragment mean {d['decoy_fragment_mean']:.1f} [{d['decoy_fragment_min']}, {d['decoy_fragment_max']}]")
    for p in rep["pools"]:
        items = "; ".join(f"F{i['feature']}@{i['token']} L0 {i['l0']} rank {i['rank']} act {i['activation']:.2f} "
                          f"pool {len(i['pool'])}{' UNDERPOWERED' if i['underpowered'] else ''}" for i in p["items"])
        print(f"pools {p['id']} row {p['row']}: distinct sets {p['n_distinct_control_sets']}; {items}")
    return 0


# Modified residues whose elemental composition equals another residue's:
# deamidated N is D, deamidated Q is E (same formula, same mass).
NOTATION_EQUIVALENT = {"N[UNIMOD:7]": "D", "Q[UNIMOD:7]": "E"}


def notation_equivalent(a: list[str], b: list[str]) -> bool:
    """Whether two token lists are the same molecule written differently: equal
    after N[UNIMOD:7] -> D, Q[UNIMOD:7] -> E and I -> L. Swaps of residues with
    the same combined composition (e.g. DQ vs EN) are not notation-equivalent."""
    def canon(tokens):
        return normalise_peptide([NOTATION_EQUIVALENT.get(t, t) for t in tokens])
    return canon(a) == canon(b)


def audit_notation(inspector: Inspector, predictions_csv: Path, candidates_csv: Path) -> dict:
    """Finding 8(b): for every prediction of the V1-4c run, whether it is exact
    (I = L), notation-equivalent to the true peptide, and counted correct by
    InstaNovo's mass-based peptide metric (Metrics.compute_precision_recall on
    the single pair). No model call beyond loading the residue set."""
    import csv
    rs = inspector.residue_set
    metrics = instanovo_io.instanovo_metrics(rs)
    ranks = {int(r["row"]): int(r["rank"]) for r in csv.DictReader(open(candidates_csv, encoding="utf-8"))}
    rows = []
    for p in csv.DictReader(open(predictions_csv, encoding="utf-8")):
        truth, pred = rs.tokenize(p["truth"]), rs.tokenize(p["prediction"])
        exact = bool(pred) and normalise_peptide(pred) == normalise_peptide(truth)
        if exact != (p["exact"] == "True"):
            raise RuntimeError(f"row {p['row']}: exact {exact} disagrees with predictions.csv")
        _ap, _ar, pep_recall, _pp = metrics.compute_precision_recall([truth], [pred])
        rows.append({"row": int(p["row"]), "candidate_rank": ranks.get(int(p["row"])),
                     "truth": p["truth"], "prediction": p["prediction"], "exact": exact,
                     "meets_precursor": p["meets_precursor"] == "True",
                     "notation_equivalent": bool(pred) and notation_equivalent(pred, truth),
                     "instanovo_mass_match": pep_recall == 1.0})
    mass_not_exact = [r for r in rows if r["instanovo_mass_match"] and not r["exact"]]
    return {"n": len(rows), "n_exact": sum(r["exact"] for r in rows),
            "n_instanovo_mass_match": sum(r["instanovo_mass_match"] for r in rows),
            "n_notation_equivalent_not_exact": sum(r["notation_equivalent"] and not r["exact"] for r in rows),
            "notation_equivalent_candidates": sorted(r["candidate_rank"] for r in rows
                                                     if r["notation_equivalent"] and r["candidate_rank"]),
            "mass_match_not_exact": mass_not_exact, "rows": rows}


def _cmd_audit_notation(args, inspector: Inspector) -> int:
    """Finding 8(b) -> notation_equivalent.json beside the V1-4c outputs."""
    out = Path(args.out_dir)
    rep = audit_notation(inspector, out / "predictions.csv", out / "candidates.csv")
    (out / "notation_equivalent.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"{rep['n']} predictions: exact {rep['n_exact']}, InstaNovo mass match "
          f"{rep['n_instanovo_mass_match']}, notation-equivalent but not exact "
          f"{rep['n_notation_equivalent_not_exact']} (candidate ranks {rep['notation_equivalent_candidates']})")
    for r in rep["mass_match_not_exact"]:
        print(f"  mass match, not exact: row {r['row']} (rank {r['candidate_rank']}), notation-equivalent "
              f"{r['notation_equivalent']}: {r['prediction']} / {r['truth']}")
    return 0


def _cmd_audit_evidence_by(args) -> int:
    """V1-4c amendment 1 -> evidence_classes_by.csv beside candidates.csv (read-only on it)."""
    import csv
    out = Path(args.out_dir)
    with open(out / "candidates.csv", newline="", encoding="utf-8") as f:
        rows = backbone_evidence_classes(list(csv.DictReader(f)))
    _write_csv(out / "evidence_classes_by.csv", rows, EVIDENCE_BY_COLUMNS)
    count = {k: {cls: sum(r[k] == cls for r in rows)
                 for cls in (EVIDENCE_FOR_TRUE, EVIDENCE_AMBIGUOUS, EVIDENCE_FOR_PRED)}
             for k in ("evidence_class", "evidence_class_by")}
    print(f"{len(rows)} candidates; classes all fragments {count['evidence_class']}, "
          f"b/y only {count['evidence_class_by']}; {sum(r['changed'] for r in rows)} change")
    for r in rows:
        if r["changed"]:
            print(f"  rank {r['rank']} row {r['row']}{' (selected)' if r['selected'] else ''}: "
                  f"{r['evidence_class']} -> {r['evidence_class_by']}")
    return 0


# --- V1-4c part B: the pre-registered ablation tests ------------------------------------

HYPOTHESES_SHA256 = "2177039b13515fdf46d8f07de6d1c3542d92f7829c8b4ee73cca9120963e84a9"
SAVED_EXPLANATION_SCHEMA = 2       # analysis/v1_4c/explanations/*.json (read as raw JSON)
N_CONTROL_DRAWS = 20
MIN_CONTROL_SETS = 10              # fewer possible distinct control sets -> undetermined
TIE_TOLERANCE = 1e-6               # |E_control - E| at or below this is a tie (Amendment 2)
MAX_DRAW_ATTEMPTS = 10_000         # rejection sampling per draw (Amendment 2)
CONTROL_SEED = 0                   # registered control: default_rng([0, row])
SECONDARY_SEED = 1                 # Amendment 2 peak-matched control: default_rng([1, row])
SECONDARY_CONTROL_IDS = ("H7",)
EXPLORATORY_IDS = ("H7",)
EXPLORATORY_TOP = 10               # features listed per sign in the exploratory sweep
UNDETERMINED_BY_CONSTRUCTION = "undetermined by construction (registered)"
PART_B_PROVENANCE = ("exact ablation, error-preserving (SAE error kept), local at the named tokens, "
                     "one layer (L8), joint removal of the named set; contexts ctx_p (teacher-forced "
                     "on the prediction) and ctx_t (teacher-forced on T = the prediction outside the "
                     "region plus the true region with I mapped to L); n = 1 spectrum per hypothesis, "
                     "no CI; every test spectrum was SAE training data (Finding 7)")


def saved_prediction(path: Path) -> dict:
    """The prediction of a saved V1-4c explanation, read as raw JSON (schema 2
    on disk; Explanation.from_json rejects it by design): tokens, per-residue
    teacher-forced log p and EOS log p."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SAVED_EXPLANATION_SCHEMA:
        raise ValueError(f"{path}: schema_version {data.get('schema_version')!r}, expected "
                         f"{SAVED_EXPLANATION_SCHEMA}")
    pred = data.get("prediction")
    if pred is None:
        raise ValueError(f"{path}: no prediction")
    return {"tokens": list(pred["tokens"]), "log_prob_tf": [float(r["log_prob_tf"]) for r in pred["residues"]],
            "eos_log_prob_tf": float(pred["eos_log_prob_tf"])}


def audit_target_tokens(pred: list[str], truth: list[str]) -> tuple[list[str], tuple[int, int], tuple[int, int]]:
    """Part B's ctx_t target T = pred[:start] + L(true region) + pred[end_pred:]
    (I mapped to L), with the region spans in the prediction and in T."""
    start, end_p, end_t = differing_region(pred, truth)
    region = normalise_peptide(truth[start:end_t])
    target = list(pred[:start]) + region + list(pred[end_p:])
    return target, (start, end_p), (start, start + len(region))


def region_sum(values: list[float], span: tuple[int, int]) -> float:
    return float(sum(values[span[0]:span[1]]))


def audit_effect(delta_p: list[float], delta_t: list[float], span_p: tuple[int, int],
                 span_t: tuple[int, int]) -> float:
    """E = M(intact) - M(removed) from per-residue deltas (residue order) in the
    two contexts: -(sum of the pred-region deltas in ctx_p - sum of the T-region
    deltas in ctx_t). E > 0: the removed set pushed the model towards its answer.
    Given the baselines instead of deltas, -audit_effect is M(intact)."""
    return -(region_sum(delta_p, span_p) - region_sum(delta_t, span_t))


def draw_controls(pools: list[list], n: int, rng: np.random.Generator,
                  max_attempts: int = MAX_DRAW_ATTEMPTS) -> list[list]:
    """n control draws, each one member per pool, drawn uniformly from the full
    pool in pool order; a draw holding a member twice is redrawn whole
    (rejection sampling, Amendment 2). Raises after max_attempts for one draw,
    or for an empty pool. Members must be hashable."""
    if any(len(p) == 0 for p in pools):
        raise ValueError("a control pool is empty")
    draws = []
    for _ in range(n):
        for _attempt in range(max_attempts):
            choice = [p[int(rng.integers(len(p)))] for p in pools]
            if len(set(choice)) == len(choice):
                draws.append(choice)
                break
        else:
            raise RuntimeError(f"no valid control draw in {max_attempts} attempts")
    return draws


def _signed(sign: str) -> int:
    if sign not in ("+", "-"):
        raise ValueError(f"predicted sign must be '+' or '-', got {sign!r}")
    return 1 if sign == "+" else -1


def control_percentile(e: float, controls: list[float], sign: str, tol: float = TIE_TOLERANCE) -> float:
    """(#controls below the hypothesis's E + 1/2 #ties) / #controls, in the
    predicted direction (signs flipped for a predicted -)."""
    s = _signed(sign)
    below = sum(s * c < s * e and abs(c - e) > tol for c in controls)
    ties = sum(abs(c - e) <= tol for c in controls)
    return (below + 0.5 * ties) / len(controls)


def audit_outcome(e: float, controls: list[float], sign: str, n_possible_sets: int,
                  tol: float = TIE_TOLERANCE) -> str:
    """The pre-registered outcome: undetermined (< MIN_CONTROL_SETS possible
    distinct control sets), supported (predicted sign and beyond every control
    in that direction), weak (predicted sign only), not supported (wrong sign)."""
    if n_possible_sets < MIN_CONTROL_SETS:
        return "undetermined"
    s = _signed(sign)
    if s * e <= 0:
        return "not supported"
    if all(s * c < s * e and abs(c - e) > tol for c in controls):
        return "supported"
    return "weak"


def intensity_ranks(intensities: list[float]) -> dict[int, int]:
    """Peak index -> intensity rank (1 = most intense; ties by peak index), as in case_view."""
    return {j: r + 1 for r, j in enumerate(sorted(range(len(intensities)), key=lambda j: -intensities[j]))}


def peak_matched_pool(named_peak: int, eligible: list[int], rank: dict[int, int],
                      size: int = POOL_SIZE) -> list[int]:
    """Amendment 2: the `size` eligible peaks nearest in intensity rank to the
    named peak (ties in rank distance to the higher intensity). The caller
    filters eligibility (non-distinguishing b/y, not named, some active feature)."""
    cands = [j for j in eligible if j != named_peak]
    return sorted(cands, key=lambda j: (abs(rank[j] - rank[named_peak]), rank[j]))[:size]


def _abs_rank(values: dict, key) -> int:
    """1 + the number of entries with a strictly larger |value|."""
    v = abs(values[key])
    return 1 + sum(abs(x) > v for x in values.values())


def _effects_at_token(inspector: Inspector, ctx_p: AblationContext, ctx_t: AblationContext,
                      layer: int, token: int, span_p, span_t) -> dict:
    """ablate_token at one token in both contexts: per active feature, its
    effect on M (E_f), activation and ranks; plus the joint (reference scale)."""
    rep_p = inspector.ablate_token(ctx_p, layer, token, with_summaries=True)
    rep_t = inspector.ablate_token(ctx_t, layer, token, with_summaries=False)
    by_t = {r["feature"]: r for r in rep_t.results}
    feats = ctx_p.features[layer][token]
    acts = {r["feature"]: r["activation"] for r in rep_p.results}
    act_order = sorted(acts, key=lambda f: -acts[f])
    effects = {r["feature"]: audit_effect(r["delta_log_prob"], by_t[r["feature"]]["delta_log_prob"],
                                          span_p, span_t) for r in rep_p.results}
    rows = []
    for r in rep_p.results:
        f = r["feature"]
        rows.append({"feature": f, "activation": acts[f], "activation_rank": act_order.index(f) + 1,
                     "effect_on_M": effects[f], "abs_effect_rank": _abs_rank(effects, f),
                     "delta_total_p": r["delta_total"], "delta_total_t": by_t[f]["delta_total"],
                     "summary": r["summary"]})
    rows.sort(key=lambda x: x["abs_effect_rank"])
    joint = None
    if rep_p.joint is not None:
        joint = {"n_features": rep_p.joint["n_features"],
                 "effect_on_M": audit_effect(rep_p.joint["delta_log_prob"], rep_t.joint["delta_log_prob"],
                                             span_p, span_t),
                 "delta_total_p": rep_p.joint["delta_total"], "delta_total_t": rep_t.joint["delta_total"],
                 "sum_of_singles_effect_on_M": float(sum(effects.values())),
                 "delta_log_prob_p": rep_p.joint["delta_log_prob"], "delta_log_prob_t": rep_t.joint["delta_log_prob"]}
    if len(rows) != int((feats > 0).sum()):
        raise RuntimeError(f"token {token}: sweep covers {len(rows)} features, "
                           f"{int((feats > 0).sum())} are active")
    return {"token": token, "l0": len(rows), "features": rows, "joint": joint}


def _set_effects(reports_p: list[AblationReport], reports_t: list[AblationReport], span_p, span_t) -> list[dict]:
    out = []
    for rp, rt in zip(reports_p, reports_t, strict=True):
        out.append({"pairs": rp.joint["pairs"], "activation_sum": rp.joint["activation_sum"],
                    "E": audit_effect(rp.joint["delta_log_prob"], rt.joint["delta_log_prob"], span_p, span_t),
                    "delta_log_prob_p": rp.joint["delta_log_prob"], "delta_eos_p": rp.joint["delta_eos"],
                    "delta_total_p": rp.joint["delta_total"],
                    "delta_log_prob_t": rt.joint["delta_log_prob"], "delta_eos_t": rt.joint["delta_eos"],
                    "delta_total_t": rt.joint["delta_total"]})
    return out


def _control_block(sets_eff: list[dict], e: float, m0: float, named_act: float, sign: str) -> dict:
    es = [c["E"] for c in sets_eff]
    return {"draws": [{"pairs": c["pairs"], "E": c["E"], "E_over_M": c["E"] / m0,
                       "activation_ratio": c["activation_sum"] / named_act} for c in sets_eff],
            "n_distinct_draws": len({frozenset(map(tuple, c["pairs"])) for c in sets_eff}),
            "percentile": control_percentile(e, es, sign)}


def _part_b_case(inspector: Inspector, parquet_path: Path, out: Path, h: dict, cand: dict,
                 saved_pools: dict, class_by: str) -> dict:
    t_start = time.perf_counter()
    rs = inspector.residue_set
    row, L, sign = int(h["row"]), int(h["layer"]), h["predicted_sign"]
    spec = read_raw_spectra(parquet_path, [row])[0]
    rec = read_raw_records(parquet_path, [row], ["modified_sequence"])[0]
    pred = rs.tokenize(cand["prediction"])
    truth = [rs.residue_remapping.get(t, t) for t in rs.tokenize(rec["modified_sequence"])]
    if "".join(truth) != cand["truth"]:
        raise RuntimeError(f"row {row}: true peptide differs from candidates.csv")

    saved = saved_prediction(out / "explanations" / f"row_{row}.json")
    exp = inspector.explain(spec)
    if exp.prediction is None or exp.prediction["tokens"] != saved["tokens"] or saved["tokens"] != pred:
        raise RuntimeError(f"row {row}: fresh explain() predicted "
                           f"{exp.prediction and exp.prediction['peptide']}, saved {''.join(saved['tokens'])}")
    fresh = [r["log_prob_tf"] for r in exp.prediction["residues"]] + [exp.prediction["eos_log_prob_tf"]]
    gap = max(abs(a - b) for a, b in zip(fresh, saved["log_prob_tf"] + [saved["eos_log_prob_tf"]], strict=True))
    if gap > 1e-5:
        raise RuntimeError(f"row {row}: fresh log p differs from the saved explanation by {gap:.2e}")

    target, span_p, span_t = audit_target_tokens(pred, truth)
    if (span_p[0] != int(cand["region_start"]) or "".join(pred[span_p[0]:span_p[1]]) != cand["pred_region"]
            or "".join(truth[span_p[0]:span_p[0] + span_t[1] - span_t[0]]) != cand["true_region"]):
        raise RuntimeError(f"row {row}: region differs from candidates.csv")
    if [rs.residue_remapping.get(t, t) for t in rs.tokenize("".join(target))] != target:
        raise RuntimeError(f"row {row}: target T does not round-trip through the tokenizer")
    ctx_p = inspector.ablation_context(spec, exp)
    ctx_t = inspector.ablation_context(spec, exp, target="".join(target))
    if ctx_t.target["tokens"] != target or not torch.equal(ctx_p.features[L], ctx_t.features[L]):
        raise RuntimeError(f"row {row}: ctx_t does not hold T or the same features")
    feats = ctx_p.features[L]
    base_p, eos_p = decoder_to_residue_order(ctx_p.baseline_log_prob.tolist(), ctx_p.n_residues)
    base_t, eos_t = decoder_to_residue_order(ctx_t.baseline_log_prob.tolist(), ctx_t.n_residues)
    m0 = -audit_effect(base_p, base_t, span_p, span_t)

    # Registered control pools, recomputed; must equal descriptive_checks.json's.
    items = [(int(it["feature"]), int(it["token"])) for it in h["features"]]
    named_by_token: dict[int, set[int]] = {}
    for f, t in items:
        named_by_token.setdefault(t, set()).add(f)
    pools, pool_info = [], []
    for (f, t), sv in zip(items, saved_pools["items"], strict=True):
        acts = {int(g): float(feats[t, g]) for g in (feats[t] > 0).nonzero().flatten().tolist()}
        pool = control_pool(acts, named_by_token[t], f)
        if (sv["feature"], sv["token"]) != (f, t) or sv["pool"] != pool:
            raise RuntimeError(f"{h['id']} F{f}@{t}: pool {pool} differs from descriptive_checks.json {sv['pool']}")
        pools.append((t, pool))
        pool_info.append({"feature": f, "token": t, "activation": acts[f], "pool": pool,
                          "underpowered": len(pool) < POOL_MIN_SIZE})
    n_sets = n_distinct_control_sets(pools)
    if n_sets != saved_pools["n_distinct_control_sets"]:
        raise RuntimeError(f"{h['id']}: {n_sets} control sets, descriptive_checks.json "
                           f"{saved_pools['n_distinct_control_sets']}")
    decidable = n_sets >= MIN_CONTROL_SETS
    draws = (draw_controls([[(g, t) for g in pool] for t, pool in pools], N_CONTROL_DRAWS,
                           np.random.default_rng([CONTROL_SEED, row])) if decidable else [])

    # The named set, each item alone, and the registered controls: one batch per context.
    sets = [items] + [[p] for p in items] + draws
    eff = _set_effects(inspector.ablate_sets(ctx_p, L, sets), inspector.ablate_sets(ctx_t, L, sets), span_p, span_t)
    hyp, singles, ctrl = eff[0], eff[1:1 + len(items)], eff[1 + len(items):]
    e = hyp["E"]
    named_act = hyp["activation_sum"]
    result = {
        "id": h["id"], "row": row, "layer": L, "predicted_sign": sign,
        "class_registered": h["class"], "class_amendment_1": class_by,
        "prediction": "".join(pred), "truth": "".join(truth), "target_T": "".join(target),
        "span_pred": list(span_p), "span_T": list(span_t),
        "pred_region": "".join(pred[span_p[0]:span_p[1]]), "T_region": "".join(target[span_t[0]:span_t[1]]),
        "M_intact": m0, "region_log_p_pred": region_sum(base_p, span_p), "region_log_p_T": region_sum(base_t, span_t),
        "eos_log_p_pred": eos_p, "eos_log_p_T": eos_t,
        "E": e, "E_over_M": e / m0 if m0 != 0 else None, "M_intact_nonpositive": m0 <= 0,
        "set": hyp,
        "singles": [{"feature": p[0], "token": p[1], "activation": s["activation_sum"], "E": s["E"],
                     "E_over_M": s["E"] / m0, "delta_total_p": s["delta_total_p"],
                     "delta_total_t": s["delta_total_t"]} for p, s in zip(items, singles, strict=True)],
        "pools": pool_info, "n_possible_control_sets": n_sets,
        "underpowered_items": [f"F{p['feature']}@{p['token']}" for p in pool_info if p["underpowered"]],
    }
    if decidable:
        result["control"] = _control_block(ctrl, e, m0, named_act, sign)
        result["outcome"] = audit_outcome(e, [c["E"] for c in ctrl], sign, n_sets)
    else:
        result["control"] = None
        result["outcome"] = UNDETERMINED_BY_CONSTRUCTION

    # Step (c) and the joint reference scale: every active feature at each named token.
    sweeps: dict[int, dict] = {}
    for t in dict.fromkeys(t for _f, t in items):
        sweeps[t] = _effects_at_token(inspector, ctx_p, ctx_t, L, t, span_p, span_t)
    result["step_c"] = [{"feature": f, "token": t, "l0": sweeps[t]["l0"],
                         **{k: next(x for x in sweeps[t]["features"] if x["feature"] == f)[k]
                            for k in ("activation", "activation_rank", "effect_on_M", "abs_effect_rank")}}
                        for f, t in items]
    result["joint_reference"] = [{"token": t, **sweeps[t]["joint"]} for t in sweeps]

    if h["id"] in SECONDARY_CONTROL_IDS or h["id"] in EXPLORATORY_IDS:
        fwd = inspector.forward(spec, "".join(pred))
        ann_p, _ = inspector._annotate(pred, spec, fwd)
        ann_t, _ = inspector._annotate(truth, spec, fwd)
        d = distinguishing_peaks(ann_p, ann_t)
        if d["true_only"] != [int(x) for x in cand["true_only_peaks"].split()]:
            raise RuntimeError(f"row {row}: true-only peaks differ from candidates.csv")
        by_true_only = [j for j in d["true_only"] if ann_t[j]["ion_type"] in BACKBONE_ION_TYPES]
        if len(by_true_only) != int(cand["n_true_only_by"]):
            raise RuntimeError(f"row {row}: {len(by_true_only)} true-only b/y peaks, candidates.csv "
                               f"{cand['n_true_only_by']}")
        labels = {j: {"pred": peak_label(ann_p[j]), "true": peak_label(ann_t[j])} for j in range(len(ann_p))}

    if h["id"] in SECONDARY_CONTROL_IDS:
        rank = intensity_ranks([pk["intensity"] for pk in exp.peaks])
        named_peaks = {t - 1 for _f, t in items}
        eligible = [j for j in range(len(ann_p))
                    if ann_p[j]["ion_type"] in BACKBONE_ION_TYPES and ann_t[j]["ion_type"] in BACKBONE_ION_TYPES
                    and j not in named_peaks and bool((feats[j + 1] > 0).any())]
        sec_pools = [peak_matched_pool(t - 1, eligible, rank) for _f, t in items]
        peak_draws = draw_controls(sec_pools, N_CONTROL_DRAWS, np.random.default_rng([SECONDARY_SEED, row]))
        top = {j: int(feats[j + 1].argmax()) for pool in sec_pools for j in pool}
        sec_sets = [[(top[j], j + 1) for j in dr] for dr in peak_draws]
        sec = _set_effects(inspector.ablate_sets(ctx_p, L, [items] + sec_sets),
                           inspector.ablate_sets(ctx_t, L, [items] + sec_sets), span_p, span_t)
        block = _control_block(sec[1:], sec[0]["E"], m0, named_act, sign)
        block.update({
            "rule": "Amendment 2 (secondary, does not change the outcome)",
            "E_in_batch": sec[0]["E"], "E_batch_difference": sec[0]["E"] - e,
            "n_possible_sets": n_distinct_control_sets([(0, p) for p in sec_pools]),
            "n_eligible_peaks": len(eligible),
            "pools": [{"feature": f, "token": t, "named_peak": t - 1, "named_intensity_rank": rank[t - 1],
                       "pool_peaks": p, "pool_intensity_ranks": [rank[j] for j in p],
                       "pool_labels": [labels[j]["pred"] for j in p]}
                      for (f, t), p in zip(items, sec_pools, strict=True)],
            "named_feature_is_top_on_pool_peak": sorted({f"F{top[j]}@{j + 1}" for p in sec_pools for j in p
                                                         if top[j] in {f for f, _t in items}}),
        })
        result["secondary_control"] = block

    if h["id"] in EXPLORATORY_IDS:
        for j in by_true_only:
            if j + 1 not in sweeps:
                sweeps[j + 1] = _effects_at_token(inspector, ctx_p, ctx_t, L, j + 1, span_p, span_t)
        pairs = [{"token": j + 1, "peak": j, "label_true": labels[j]["true"], **x}
                 for j in by_true_only for x in sweeps[j + 1]["features"]]
        pos = sorted(pairs, key=lambda x: -x["effect_on_M"])[:EXPLORATORY_TOP]
        neg = sorted(pairs, key=lambda x: x["effect_on_M"])[:EXPLORATORY_TOP]
        result["exploratory"] = {
            "label": "exploratory (not pre-registered)", "true_only_by_peaks": by_true_only,
            "peak_labels": {str(j): labels[j]["true"] for j in by_true_only},
            "n_pairs": len(pairs), "top_positive": pos, "top_negative": neg,
            "joint_reference": [{"token": j + 1, **sweeps[j + 1]["joint"]} for j in by_true_only],
        }

    result["sweeps"] = {str(t): s for t, s in sweeps.items()}
    result["seconds"] = time.perf_counter() - t_start
    return result


def audit_part_b(inspector: Inspector, parquet_path: Path, out_dir: Path) -> dict:
    """V1-4c part B: the pre-registered tests, run as registered (STATUS, V1-4c
    part B protocol, Amendments 1 and 2). Stops if hypotheses.json is not the
    registered file, a fresh explain() does not reproduce a saved prediction,
    or a recomputed control pool differs from descriptive_checks.json."""
    import csv
    out = Path(out_dir)
    sha = _sha256_prefix(out / "hypotheses.json", 64)
    if sha != HYPOTHESES_SHA256:
        raise RuntimeError(f"hypotheses.json sha256 {sha} is not the registered {HYPOTHESES_SHA256}")
    hyp = json.loads((out / "hypotheses.json").read_text(encoding="utf-8"))
    checks = json.loads((out / "descriptive_checks.json").read_text(encoding="utf-8"))
    saved_pools = {p["id"]: p for p in checks["pools"]}
    with open(out / "candidates.csv", newline="", encoding="utf-8") as f:
        cands = {int(r["row"]): r for r in csv.DictReader(f)}
    with open(out / "evidence_classes_by.csv", newline="", encoding="utf-8") as f:
        class_by = {int(r["row"]): r["evidence_class_by"] for r in csv.DictReader(f)}
    t0 = time.perf_counter()
    cases = [_part_b_case(inspector, parquet_path, out, h, cands[int(h["row"])], saved_pools[h["id"]],
                          class_by[int(h["row"])]) for h in hyp["hypotheses"]]
    decidable = [c for c in cases if c["control"] is not None]
    return {
        "provenance_note": PART_B_PROVENANCE, "hypotheses_sha256": sha,
        "settings": {"n_control_draws": N_CONTROL_DRAWS, "min_control_sets": MIN_CONTROL_SETS,
                     "tie_tolerance": TIE_TOLERANCE, "max_draw_attempts": MAX_DRAW_ATTEMPTS,
                     "control_seed": CONTROL_SEED, "secondary_seed": SECONDARY_SEED,
                     "pool_size": POOL_SIZE, "pool_min_ratio": POOL_MIN_RATIO, "pool_min_size": POOL_MIN_SIZE,
                     "exploratory_top": EXPLORATORY_TOP},
        "n_decidable": len(decidable),
        "expected_false_supported": len(decidable) / (N_CONTROL_DRAWS + 1),
        "seconds": time.perf_counter() - t0,
        "inspector_provenance": inspector.provenance,
        "cases": cases,
    }


_PART_B_SUMMARY_COLUMNS = ("id", "row", "class_registered", "class_amendment_1", "predicted_sign",
                           "pred_region", "T_region", "M_intact", "E", "E_over_M", "delta_total_p",
                           "delta_total_t", "n_possible_control_sets", "n_distinct_draws", "percentile",
                           "secondary_percentile", "underpowered_items", "outcome")
_PART_B_CONTROL_COLUMNS = ("id", "row", "control", "draw", "pairs", "E", "E_over_M", "activation_ratio")
_PART_B_SWEEP_COLUMNS = ("id", "row", "sweep", "token", "peak", "named", "feature", "activation",
                         "activation_rank", "l0", "effect_on_M", "abs_effect_rank", "delta_total_p",
                         "delta_total_t", "top_concept", "top_concept_f1_dom", "top_concept_chance_fraction",
                         "reliable_concept", "reliable_concept_f1_dom", "reliable_concept_chance_fraction",
                         "causal_delta_ce")


def _part_b_tables(rep: dict, reliability: dict) -> tuple[list, list, list]:
    summary, controls, sweep = [], [], []
    for c in rep["cases"]:
        ctrl, sec = c["control"], c.get("secondary_control")
        summary.append({
            "id": c["id"], "row": c["row"], "class_registered": c["class_registered"],
            "class_amendment_1": c["class_amendment_1"], "predicted_sign": c["predicted_sign"],
            "pred_region": c["pred_region"], "T_region": c["T_region"], "M_intact": c["M_intact"],
            "E": c["E"], "E_over_M": c["E_over_M"], "delta_total_p": c["set"]["delta_total_p"],
            "delta_total_t": c["set"]["delta_total_t"], "n_possible_control_sets": c["n_possible_control_sets"],
            "n_distinct_draws": ctrl["n_distinct_draws"] if ctrl else "",
            "percentile": ctrl["percentile"] if ctrl else "", "secondary_percentile": sec["percentile"] if sec else "",
            "underpowered_items": c["underpowered_items"], "outcome": c["outcome"]})
        for name, block in (("registered", ctrl), ("secondary", sec)):
            for i, d in enumerate(block["draws"] if block else []):
                controls.append({"id": c["id"], "row": c["row"], "control": name, "draw": i,
                                 "pairs": [f"F{f}@{t}" for f, t in d["pairs"]], "E": d["E"],
                                 "E_over_M": d["E_over_M"], "activation_ratio": d["activation_ratio"]})
        named = {(f, t) for f, t in c["set"]["pairs"]}
        named_tokens = {t for _f, t in named}
        for t_str, s in c["sweeps"].items():
            t = int(t_str)
            for x in s["features"]:
                sm = x["summary"]
                rc = sm.get("reliable_concept")
                sweep.append({
                    "id": c["id"], "row": c["row"], "sweep": "step_c" if t in named_tokens else "exploratory",
                    "token": t, "peak": t - 1, "named": (x["feature"], t) in named, "feature": x["feature"],
                    "activation": x["activation"], "activation_rank": x["activation_rank"], "l0": s["l0"],
                    "effect_on_M": x["effect_on_M"], "abs_effect_rank": x["abs_effect_rank"],
                    "delta_total_p": x["delta_total_p"], "delta_total_t": x["delta_total_t"],
                    "top_concept": sm.get("top_concept"), "top_concept_f1_dom": sm.get("top_concept_f1_dom"),
                    "top_concept_chance_fraction": sm.get("top_concept_chance_fraction"),
                    "reliable_concept": rc, "reliable_concept_f1_dom": sm.get("reliable_concept_f1_dom"),
                    "reliable_concept_chance_fraction": reliability[rc]["chance_fraction"] if rc else None,
                    "causal_delta_ce": sm.get("causal_delta_ce")})
    return summary, controls, sweep


def _cmd_audit_part_b(args, inspector: Inspector) -> int:
    """V1-4c part B -> part_b.json and three CSVs beside the part A outputs."""
    out = Path(args.out_dir)
    rep = audit_part_b(inspector, args.parquet, out)
    (out / "part_b.json").write_text(json.dumps(_finalise(rep), indent=1), encoding="utf-8")
    reliability = inspector.dictionaries[8].concept_reliability()
    summary, controls, sweep = _part_b_tables(rep, reliability)
    _write_csv(out / "part_b_summary.csv", summary, _PART_B_SUMMARY_COLUMNS)
    _write_csv(out / "part_b_controls.csv", controls, _PART_B_CONTROL_COLUMNS)
    _write_csv(out / "part_b_token_sweep.csv", sweep, _PART_B_SWEEP_COLUMNS)
    print(f"part B: {len(rep['cases'])} hypotheses, {rep['n_decidable']} decidable, "
          f"{rep['seconds']:.0f} s; outputs in {out}")
    print("id | row | class (A1) | sign | M intact | E | E/M | pct | sec pct | outcome")
    for s in summary:
        em = "" if s["E_over_M"] is None else f"{s['E_over_M']:+.4f}"
        print(f"{s['id']} | {s['row']} | {s['class_amendment_1']} | {s['predicted_sign']} | "
              f"{s['M_intact']:+.4f} | {s['E']:+.4f} | {em} | {s['percentile']} | "
              f"{s['secondary_percentile']} | {s['outcome']}")
    for c in rep["cases"]:
        steps = "; ".join(f"F{x['feature']}@{x['token']} |E| rank {x['abs_effect_rank']}/{x['l0']} "
                          f"(act rank {x['activation_rank']})" for x in c["step_c"])
        print(f"  {c['id']} step (c): {steps}")
    for name in ("part_b.json", "part_b_summary.csv", "part_b_controls.csv", "part_b_token_sweep.csv"):
        print(f"  sha256 {name}: {_sha256_prefix(out / name, 64)}")
    return 0


_PREDICTION_COLUMNS = ("row", "truth", "prediction", "exact", "meets_precursor", "min_abs_ppm",
                       "beam_log_prob", "n_residues", "modified_truth")
_CANDIDATE_COLUMNS = ("rank", "row", "mean_residue_log_prob", "prediction", "truth", "mismatch_type",
                      "region_start", "pred_region", "true_region", "region_mass_delta_da",
                      "min_abs_ppm", "evidence_class", "n_pred_only", "n_pred_only_by",
                      "n_true_only", "n_true_only_by", "pred_only_peaks", "true_only_peaks",
                      "n_processed_peaks", "selected", "selection_reason")


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([" ".join(map(str, r[c])) if isinstance(r[c], list) else r[c] for c in columns])


def _cmd_audit_cases(args, inspector: Inspector) -> int:
    """V1-4c part A: regenerate the V1-4a beam predictions, select audit cases
    by the fixed rule, explain() the selected ones (no ablation) and write case
    views. Never writes hypotheses.json."""
    default_sample = not args.rows and args.n == 200 and args.seed == 0
    rows = sorted(args.rows) if args.rows else sorted(np.random.default_rng(args.seed).choice(
        parquet_num_rows(args.parquet), size=args.n, replace=False).tolist())
    out = Path(args.out_dir)
    (out / "explanations").mkdir(parents=True, exist_ok=True)
    (out / "case_views").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    predictions, candidates = audit_candidates(inspector, args.parquet, rows)
    t_pred = time.perf_counter() - t0
    _write_csv(out / "predictions.csv", predictions, _PREDICTION_COLUMNS)
    n_exact = sum(p["exact"] for p in predictions)
    n_meets = sum(p["meets_precursor"] for p in predictions)
    print(f"{len(rows)} rows; exact {n_exact}, meets precursor {n_meets}, exact among meeting "
          f"{sum(p['exact'] and p['meets_precursor'] for p in predictions)}; "
          f"{len(candidates)} candidates; {t_pred:.0f} s")
    if default_sample and (n_exact, n_meets) != (V1_4A_BEAM_EXACT, V1_4A_BEAM_MEETS):
        print(f"REPRODUCTION FAILED: V1-4a recorded exact {V1_4A_BEAM_EXACT}, meets {V1_4A_BEAM_MEETS}; "
              f"got {n_exact}, {n_meets}. V1-4a saved no per-row predictions, so the differing rows "
              f"cannot be named; every wrong or non-meeting row follows.")
        for p in predictions:
            if not p["exact"] or not p["meets_precursor"]:
                print(f"  row {p['row']}: exact {p['exact']}, meets {p['meets_precursor']}, "
                      f"min |ppm| {p['min_abs_ppm']}, pred {p['prediction']}, truth {p['truth']}")
        return 1

    ranked, summary = select_audit_cases(candidates, n_top=args.n_top, min_evidence=args.min_evidence)
    _write_csv(out / "candidates.csv", ranked, _CANDIDATE_COLUMNS)
    selected = [c for c in ranked if c["selected"]]
    for c in selected:
        spec = read_raw_spectra(args.parquet, [c["row"]])[0]
        exp = inspector.explain(spec)
        if exp.prediction is None or exp.prediction["tokens"] != c["pred_tokens"]:
            raise RuntimeError(f"row {c['row']}: explain() predicted {exp.prediction and exp.prediction['peptide']}, "
                               f"the audit run {c['prediction']}")
        (out / "explanations" / f"row_{c['row']}.json").write_text(exp.to_json(), encoding="utf-8")
        (out / "case_views" / f"row_{c['row']}.md").write_text(case_view(inspector, exp, c), encoding="utf-8")
    selection = {
        "rule": AUDIT_SELECTION_RULE, "parquet": Path(args.parquet).name, "seed": args.seed,
        "n": len(rows), "rows": rows, "reproduces_v1_4a": default_sample,
        "counts": {"exact": n_exact, "meets_precursor": n_meets, "candidates": len(candidates)},
        "summary": summary, "selected_rows": [c["row"] for c in selected],
        "sha256": {name: _sha256_prefix(out / name, 64) for name in ("predictions.csv", "candidates.csv")},
        "provenance": inspector.provenance,
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")

    print(f"selected {summary['n_selected']} ({summary['n_selected_evidence_for_true']} "
          f"'{EVIDENCE_FOR_TRUE}', shortfall {summary['shortfall']}); outputs in {out}")
    print("rank | row | mean logp | type | dmass (Da) | class | pred-only (b/y) | true-only (b/y) | sel | pred | true")
    for c in ranked:
        print(f"{c['rank']} | {c['row']} | {c['mean_residue_log_prob']:.3f} | {c['mismatch_type']} | "
              f"{c['region_mass_delta_da']:+.3f} | {c['evidence_class']} | {c['n_pred_only']} "
              f"({c['n_pred_only_by']}) | {c['n_true_only']} ({c['n_true_only_by']}) | "
              f"{'Y' if c['selected'] else ''} | {c['prediction']} [{c['pred_region']}] | "
              f"{c['truth']} [{c['true_region']}]")
    return 0


# --- V1-5: demo explanations, screen sample, exhaustive local ablation ------------------
#
# Registered in STATUS (V1-5 pre-registration): the demo set comes from
# `analysis --demo-set`; the screen sample is the longest prefix of a seeded
# random order of the 2,048 local test rows whose predicted exhaustive-ablation
# cost fits the budget under a frozen cost model; screen-ablate runs every
# single-feature local ablation (plus the per-token joint) at every token with
# L0 > 0, one parquet per (row, layer), resumable.

V1_5_DIR = DEFAULT_OUTPUTS_ROOT / "analysis" / "v1_5"
HF_TEST_OFFSET = 527_974                 # merged row of HF test row 0 (train 499,402 + validation 28,572)
CHUNK_SIZE = 1024
SCREEN_CHUNKS = (520, 572)
SCREEN_SEED = 0
SCREEN_BUDGET_HOURS = 9.0
PEAK_INTENSITY_RTOL = 1e-6              # V1-5 deviation: float32 rounding in the peak check
COST_INTERCEPT_S, COST_SLOPE_S = 0.006, 0.00068   # frozen c(n) = 0.006 + 0.00068 n s per ablated row
SCREEN_RULE = (
    "population: the 2,048 HF test rows in chunks 520/572, ascending; order: "
    "np.random.default_rng(0).permutation; walk the order, skipping (and counting) rows whose beam "
    "does not complete; the sample is the longest prefix whose predicted local-ablation cost is <= the "
    "budget (9 h). Cost per spectrum = sum over layers 2/4/6/8 of (active pairs + tokens with L0 > 0) "
    "x c(n), c(n) = 0.006 + 0.00068 n s per ablated row (batched 32 + 1), n = processed peaks; pairs "
    "exact from explain(). Frozen, never recalibrated; measured timing is reported only.")
FINDING7_SCREEN_NOTE = "every sample spectrum was SAE training data (Finding 7)"
SCREEN_PROVENANCE = ("exact ablation, error-preserving (SAE error kept), local (one token), teacher-forced "
                     "on the predicted peptide, n = 1 spectrum per row, no CI; every active (feature, token) "
                     "pair at every token with L0 > 0 (latent included) plus the per-token joint")


def hf_test_row_location(row: int) -> tuple[int, int]:
    """(chunk, index within the chunk) of an HF test row in the merged corpus."""
    merged = HF_TEST_OFFSET + int(row)
    return merged // CHUNK_SIZE, merged % CHUNK_SIZE


def screen_population(chunks: tuple[int, ...] = SCREEN_CHUNKS) -> list[int]:
    """HF test rows held by the given local chunks, ascending."""
    rows = []
    for c in chunks:
        rows += [c * CHUNK_SIZE + i - HF_TEST_OFFSET for i in range(CHUNK_SIZE)]
    if min(rows) < 0:
        raise ValueError(f"chunks {chunks} reach before the test split")
    return sorted(rows)


def screen_order(population: list[int], seed: int = SCREEN_SEED) -> list[int]:
    """The registered walk order: default_rng(seed).permutation of the ascending rows."""
    return [int(r) for r in np.random.default_rng(seed).permutation(np.asarray(sorted(population)))]


def active_features(features: torch.Tensor) -> list[int]:
    """Features active (> 0) at any token, ascending."""
    return sorted((features > 0).any(dim=0).nonzero().flatten().tolist())


def demo_feature_union(per_row: list[dict[int, torch.Tensor]]) -> dict[int, list[int]]:
    """Per layer, the union over spectra of the features active at any token."""
    out: dict[int, set[int]] = {}
    for feats in per_row:
        for L, f in feats.items():
            out.setdefault(int(L), set()).update(active_features(f))
    return {L: sorted(v) for L, v in sorted(out.items())}


def cost_per_ablated_row(n_peaks: int) -> float:
    return COST_INTERCEPT_S + COST_SLOPE_S * n_peaks


def ablation_cost(explanation: dict) -> dict:
    """Predicted exhaustive local-ablation cost of one explained spectrum (an
    Explanation dict): per layer the active pairs (sum of token L0), the tokens
    with L0 > 0 (one joint each) and their ablated rows, and the total seconds."""
    n = int(explanation["input"]["n_processed_peaks"])
    c = cost_per_ablated_row(n)
    layers, total = {}, 0.0
    for L, block in explanation["layers"].items():
        l0 = [int(t["l0"]) for t in block["tokens"]]
        pairs, tokens = sum(l0), sum(v > 0 for v in l0)
        layers[str(L)] = {"pairs": pairs, "tokens": tokens, "rows": pairs + tokens}
        total += (pairs + tokens) * c
    return {"n_peaks": n, "c_n": c, "layers": layers, "seconds": total}


def random_sample(order: list[int], cost_of, budget_s: float) -> dict:
    """Walk `order`; cost_of(row) gives the predicted seconds, or None when the
    beam does not complete (skipped, counted). The sample is the longest prefix
    (over completed rows) whose total stays <= budget_s; the walk stops at the
    first row that would exceed it. cost_of is called lazily, in order."""
    sample, skipped, total, walked = [], [], 0.0, 0
    first_excluded = None
    for row in order:
        walked += 1
        cost = cost_of(row)
        if cost is None:
            skipped.append(int(row))
            continue
        if total + cost > budget_s:
            first_excluded = {"row": int(row), "seconds": float(cost)}
            break
        sample.append({"row": int(row), "seconds": float(cost)})
        total += cost
    return {"sample": sample, "skipped": skipped, "n_walked": walked,
            "first_excluded": first_excluded, "total_seconds": total}


def exhaustive_local(inspector: Inspector, ctx: AblationContext, layer: int) -> list[AblationReport]:
    """ablate_token at every token with L0 > 0 (latent included), in token order:
    every active feature alone, then all of them jointly."""
    feats = ctx.features[layer]
    tokens = (feats > 0).any(dim=1).nonzero().flatten().tolist()
    return [inspector.ablate_token(ctx, layer, int(t), with_summaries=False) for t in tokens]


_SCREEN_COLUMNS = ("row", "layer", "token", "peak", "kind", "feature", "activation", "l0", "delta_total",
                   "delta_eos", "delta_log_prob", "argmax_changed", "eos_argmax_changed")


def screen_ablation_rows(row: int, layer: int, reports: list[AblationReport]) -> list[dict]:
    """Flat rows for one (row, layer): each single ablation (kind 'single') and
    each token's joint (kind 'joint', feature -1, activation = sum of the token's
    activations)."""
    out = []
    for rep in reports:
        t = int(rep.position)
        l0 = len(rep.results)
        for r in rep.results:
            out.append({"row": row, "layer": layer, "token": t, "peak": t - 1, "kind": "single",
                        "feature": int(r["feature"]), "activation": float(r["activation"]), "l0": l0,
                        "delta_total": float(r["delta_total"]), "delta_eos": float(r["delta_eos"]),
                        "delta_log_prob": [float(v) for v in r["delta_log_prob"]],
                        "argmax_changed": [bool(v) for v in r["argmax_changed"]],
                        "eos_argmax_changed": bool(r["eos_argmax_changed"])})
        if rep.joint is not None:
            j = rep.joint
            out.append({"row": row, "layer": layer, "token": t, "peak": t - 1, "kind": "joint", "feature": -1,
                        "activation": float(sum(r["activation"] for r in rep.results)), "l0": l0,
                        "delta_total": float(j["delta_total"]), "delta_eos": float(j["delta_eos"]),
                        "delta_log_prob": [float(v) for v in j["delta_log_prob"]],
                        "argmax_changed": [bool(v) for v in j["argmax_changed"]],
                        "eos_argmax_changed": bool(j["eos_argmax_changed"])})
    return out


def _screen_schema():
    import pyarrow as pa
    return pa.schema([("row", pa.int64()), ("layer", pa.int64()), ("token", pa.int64()), ("peak", pa.int64()),
                      ("kind", pa.string()), ("feature", pa.int64()), ("activation", pa.float64()),
                      ("l0", pa.int64()), ("delta_total", pa.float64()), ("delta_eos", pa.float64()),
                      ("delta_log_prob", pa.list_(pa.float64())), ("argmax_changed", pa.list_(pa.bool_())),
                      ("eos_argmax_changed", pa.bool_())])


def screen_ablation_path(v15_dir: Path, row: int, layer: int) -> Path:
    return Path(v15_dir) / "screen" / "ablation" / f"row_{row}_L{layer}.parquet"


def _write_parquet_atomic(rows: list[dict], path: Path, metadata: dict[str, str]) -> None:
    import os
    import pyarrow as pa
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_screen_schema()).replace_schema_metadata(metadata)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def explanation_matches(exp: Explanation, saved: dict, tol: float = 1e-5) -> str | None:
    """None if a fresh explanation reproduces a saved one (prediction tokens,
    teacher-forced log p to tol, every token's L0 at every layer), else why not."""
    p, s = exp.prediction, saved.get("prediction")
    if (p is None) != (s is None):
        return "beam completion differs"
    if p is not None:
        if p["tokens"] != s["tokens"]:
            return f"prediction {p['peptide']} != saved {s['peptide']}"
        a = [r["log_prob_tf"] for r in p["residues"]] + [p["eos_log_prob_tf"]]
        b = [r["log_prob_tf"] for r in s["residues"]] + [s["eos_log_prob_tf"]]
        gap = max(abs(x - y) for x, y in zip(a, b, strict=True))
        if gap > tol:
            return f"teacher-forced log p differs by {gap:.2e}"
    for L, block in exp.layers.items():
        if [t["l0"] for t in block["tokens"]] != [t["l0"] for t in saved["layers"][L]["tokens"]]:
            return f"layer {L}: token L0 differ"
    return None


def _write_json_atomic(obj: Any, path: Path) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def demo_explain(inspector: Inspector, parquet_path: Path, v15_dir: Path,
                 v1_4c_dir: Path = DEFAULT_OUTPUTS_ROOT / "analysis" / "v1_4c", log=print) -> dict:
    """Step 2a: a fresh explain() per demo row; H1/H5/H7 must reproduce the saved
    V1-4c prediction and log p, and the local rows (groups B/C) the stored
    activations (V1-4a tolerance). Writes explanations/row_{r}.json and
    demo_features.json (per layer, the union of features active at any token)."""
    v15 = Path(v15_dir)
    demo = json.loads((v15 / "demo_set.json").read_text(encoding="utf-8"))
    root = inspector.outputs_root
    per_row, rows_out, stored_cache = [], [], {}
    for d in demo["rows"]:
        row = int(d["row"])
        spec = read_raw_spectra(parquet_path, [row])[0]
        t0 = time.perf_counter()
        exp = inspector.explain(spec)
        if exp.prediction is None:
            raise RuntimeError(f"demo row {row}: no beam completed")
        check = {"prediction": exp.prediction["peptide"]}
        if d["group"] == "A":
            saved = saved_prediction(Path(v1_4c_dir) / "explanations" / f"row_{row}.json")
            fresh = [r["log_prob_tf"] for r in exp.prediction["residues"]] + [exp.prediction["eos_log_prob_tf"]]
            if exp.prediction["tokens"] != saved["tokens"]:
                raise RuntimeError(f"row {row}: fresh prediction {exp.prediction['peptide']} != saved "
                                   f"{''.join(saved['tokens'])}")
            gap = max(abs(a - b) for a, b in zip(fresh, saved["log_prob_tf"] + [saved["eos_log_prob_tf"]],
                                                 strict=True))
            if gap > 1e-5:
                raise RuntimeError(f"row {row}: fresh log p differs from V1-4c by {gap:.2e}")
            check["v1_4c_max_abs_log_p_gap"] = gap
        else:
            chunk, s = hf_test_row_location(row)
            fwd = inspector.forward(spec, None)
            if chunk not in stored_cache:
                stored_cache[chunk] = torch.load(root / "extract" / "chunks" / f"meta_{chunk:05d}.pt",
                                                 map_location="cpu", weights_only=False)
            meta = stored_cache[chunk]
            if meta["spectrum_ids"][s] != str(HF_TEST_OFFSET + row):
                raise RuntimeError(f"row {row}: chunk {chunk} index {s} holds spectrum {meta['spectrum_ids'][s]}")
            # V1-5 deviation (STATUS): m/z exact, intensities within 1e-6 relative
            # (float32 rounding; row 58193 differs by up to 5.96e-08 absolute).
            stored_int = meta["intensity_arrays"][s]
            if not torch.equal(fwd.peaks_mz, meta["mz_arrays"][s]):
                raise RuntimeError(f"row {row}: processed m/z differ from chunk {chunk}")
            if fwd.peaks_intensity.shape != stored_int.shape:
                raise RuntimeError(f"row {row}: processed peak count differs from chunk {chunk}")
            rel = ((fwd.peaks_intensity.double() - stored_int.double()).abs()
                   / stored_int.double().abs().clamp_min(1e-30))
            if float(rel.max()) > PEAK_INTENSITY_RTOL:
                raise RuntimeError(f"row {row}: processed intensities differ from chunk {chunk} by up to "
                                   f"{float(rel.max()):.3e} relative (> {PEAK_INTENSITY_RTOL})")
            check["peak_intensity_max_rel_diff"] = float(rel.max())
            check["peak_intensity_max_abs_diff"] = float((fwd.peaks_intensity.double() - stored_int.double()).abs().max())
            check["n_peaks_intensity_differ"] = int((fwd.peaks_intensity != stored_int).sum())
            rows_t = (meta["token_to_spectrum"] == s).nonzero().flatten()
            worst = {}
            for L in inspector.layers:
                stored = torch.load(root / "extract" / "chunks" / f"acts_L{L}_{chunk:05d}.pt", map_location="cpu",
                                    weights_only=False)["activations"][rows_t].float()
                ours = fwd.activations[L]
                rms = float(stored.pow(2).mean().sqrt())
                torch.testing.assert_close(ours, stored, rtol=2 ** -7, atol=2 ** -7 * rms)
                worst[str(L)] = float(((ours - stored).abs() / (stored.abs() + rms)).max())
            check["stored_activation_max_rel_err"] = worst
        ctx = inspector.ablation_context(spec, exp)
        per_row.append(ctx.features)
        _write_json_atomic(json.loads(exp.to_json()), v15 / "explanations" / f"row_{row}.json")
        n_active = {str(L): len(active_features(ctx.features[L])) for L in inspector.layers}
        rows_out.append({"row": row, "group": d["group"], "n_peaks": exp.input["n_processed_peaks"],
                         "n_active_features": n_active, **check})
        log(f"[{_stamp()}] row {row} ({d['group']}): {exp.prediction['peptide']}, "
            f"{exp.input['n_processed_peaks']} peaks, active {n_active}, {time.perf_counter() - t0:.1f} s")
    union = demo_feature_union(per_row)
    out = {"rule": "per layer, the union over the demo spectra of the features active (> 0) at any token "
                   "(not only the top 8), from a fresh explain() + ablation_context on the prediction",
           "demo_set_sha256_16": _sha256_prefix(v15 / "demo_set.json"),
           "rows": rows_out, "n_features": {str(L): len(v) for L, v in union.items()},
           "features": {str(L): v for L, v in union.items()}, "provenance": inspector.provenance}
    _write_json_atomic(_finalise(out), v15 / "demo_features.json")
    log(f"demo_features.json: {out['n_features']}")
    return out


def screen_sample(inspector: Inspector, parquet_path: Path, v15_dir: Path, budget_hours: float,
                  force: bool = False, log=print) -> dict:
    """Step 2b: apply the registered random-sample rule; write
    screen/screen_sample.json and screen/explanations/; time one batch."""
    v15 = Path(v15_dir)
    out_path = v15 / "screen" / "screen_sample.json"
    if out_path.exists() and not force:
        raise RuntimeError(f"{out_path} exists; the sample is deterministic, pass --force to recompute it")
    population = screen_population()
    if max(population) >= parquet_num_rows(parquet_path):
        raise ValueError(f"{parquet_path} has fewer rows than the screen population needs")
    order = screen_order(population)
    rs = inspector.residue_set
    info, explanations = {}, {}
    t_walk = time.perf_counter()

    def cost_of(row: int):
        t0 = time.perf_counter()
        spec = read_raw_spectra(parquet_path, [row])[0]
        exp = inspector.explain(spec)
        rec = read_raw_records(parquet_path, [row], ["modified_sequence"])[0]
        truth = [rs.residue_remapping.get(t, t) for t in rs.tokenize(rec["modified_sequence"])]
        if exp.prediction is None:
            info[row] = {"row": row, "beam_completed": False, "label": "".join(truth)}
            log(f"[{_stamp()}] row {row}: no beam completed (skipped)")
            return None
        d = json.loads(exp.to_json())
        cost = ablation_cost(d)
        pred = exp.prediction["tokens"]
        chunk, s = hf_test_row_location(row)
        info[row] = {"row": row, "chunk": chunk, "spectrum_index": s, "spectrum_id": str(HF_TEST_OFFSET + row),
                     "n_peaks": cost["n_peaks"], "c_n": cost["c_n"], "layers": cost["layers"],
                     "predicted_seconds": cost["seconds"], "prediction": exp.prediction["peptide"],
                     "label": "".join(truth), "exact": normalise_peptide(pred) == normalise_peptide(truth),
                     "explain_seconds": time.perf_counter() - t0}
        explanations[row] = d
        log(f"[{_stamp()}] row {row}: {cost['n_peaks']} peaks, pairs "
            f"{ {L: v['pairs'] for L, v in cost['layers'].items()} }, predicted {cost['seconds'] / 60:.1f} min")
        return cost["seconds"]

    res = random_sample(order, cost_of, budget_hours * 3600.0)
    sample_rows = [x["row"] for x in res["sample"]]
    if not sample_rows:
        raise RuntimeError("the screen sample is empty")
    for row in sample_rows:
        _write_json_atomic(explanations[row], v15 / "screen" / "explanations" / f"row_{row}.json")

    # one-batch calibration (reported only): MAX_ABLATION_BATCH single deltas of the
    # first sample row at layer 2, after one untimed warm-up pass
    first = sample_rows[0]
    spec = read_raw_spectra(parquet_path, [first])[0]
    exp = Explanation.from_dict(explanations[first])
    ctx = inspector.ablation_context(spec, exp)
    L = inspector.layers[0]
    feats = ctx.features[L]
    w_dec = inspector.saes[L].W_dec.detach().cpu()
    pairs = [(int(f), int(t)) for t in range(feats.size(0)) for f in (feats[t] > 0).nonzero().flatten().tolist()]
    deltas = [ablation_delta(feats, w_dec, [(f, [t])]) for f, t in pairs[:MAX_ABLATION_BATCH]]
    inspector._run_ablations(ctx, L, deltas)
    t0 = time.perf_counter()
    inspector._run_ablations(ctx, L, deltas)
    t_batch = time.perf_counter() - t0
    n_peaks = info[first]["n_peaks"]
    timing = {"row": first, "layer": L, "n_peaks": n_peaks, "n_deltas": len(deltas),
              "batch_seconds": t_batch, "measured_per_row_s": t_batch / len(deltas),
              "model_per_row_s": cost_per_ablated_row(n_peaks),
              "ratio_measured_to_model": (t_batch / len(deltas)) / cost_per_ablated_row(n_peaks),
              "note": "one pass of 32 single ablations + the in-batch baseline after one untimed warm-up "
                      "pass; reported only, the cost model is frozen"}

    demo_path = v15 / "demo_set.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8")) if demo_path.is_file() else None
    demo_rows = {int(d["row"]): d["group"] for d in demo["rows"]} if demo else {}
    overlap = [{"row": r, "group": demo_rows[r]} for r in sample_rows if r in demo_rows]
    out = {
        "rule": SCREEN_RULE, "seed": SCREEN_SEED, "chunks": list(SCREEN_CHUNKS), "budget_hours": budget_hours,
        "cost_model": {"intercept_s": COST_INTERCEPT_S, "slope_s": COST_SLOPE_S, "batch": MAX_ABLATION_BATCH},
        "population_size": len(population), "order_head": order[:res["n_walked"]],
        "n_walked": res["n_walked"], "n_sample": len(sample_rows), "sample_rows": sample_rows,
        "rows": [info[r] for r in sample_rows],
        "skipped_no_beam": [info[r] for r in res["skipped"]],
        "first_excluded": None if res["first_excluded"] is None else {
            **res["first_excluded"], "n_peaks": info[res["first_excluded"]["row"]]["n_peaks"]},
        "total_predicted_seconds": res["total_seconds"], "total_predicted_hours": res["total_seconds"] / 3600,
        "n_exact": sum(info[r]["exact"] for r in sample_rows),
        "overlap_with_demo_set": overlap, "demo_set_sha256_16": _sha256_prefix(demo_path) if demo else None,
        "measured_one_batch": timing, "walk_seconds": time.perf_counter() - t_walk,
        "finding_7_note": FINDING7_SCREEN_NOTE, "provenance": inspector.provenance,
    }
    _write_json_atomic(_finalise(out), out_path)
    return out


def screen_ablate(inspector: Inspector, parquet_path: Path, v15_dir: Path, budget_hours: float,
                  force: bool = False, log=print, read_spectrum=None) -> dict:
    """Step 4: exhaustive local ablation of every screen-sample row at every
    layer. Each (row, layer) is one parquet written atomically; existing files
    are skipped (resume). Refuses if the predicted total exceeds the budget
    unless force."""
    from schema import SCREEN_ABLATION_SCHEMA_VERSION
    v15 = Path(v15_dir)
    sample = json.loads((v15 / "screen" / "screen_sample.json").read_text(encoding="utf-8"))
    if sample["total_predicted_hours"] > budget_hours and not force:
        raise RuntimeError(f"predicted {sample['total_predicted_hours']:.2f} h exceeds the budget of "
                           f"{budget_hours} h; pass --force to run anyway")
    read_spectrum = read_spectrum or (lambda row: read_raw_spectra(parquet_path, [row])[0])
    for stale in (v15 / "screen" / "ablation").glob("*.tmp"):
        stale.unlink()
    sample_sha = _sha256_prefix(v15 / "screen" / "screen_sample.json")
    t_start, predicted_done, done, skipped = time.perf_counter(), 0.0, [], []
    total_pred = sample["total_predicted_seconds"]
    n_files = len(sample["rows"]) * len(inspector.layers)

    def layer_pred(info: dict, L: int) -> float:
        return info["layers"][str(L)]["rows"] * info["c_n"]

    remaining_pred = sum(layer_pred(info, L) for info in sample["rows"] for L in inspector.layers
                         if not screen_ablation_path(v15, int(info["row"]), L).is_file())
    n_existing = sum(screen_ablation_path(v15, int(info["row"]), L).is_file()
                     for info in sample["rows"] for L in inspector.layers)
    pred_this_run, files_done = 0.0, n_existing
    log(f"[{_stamp()}] screen-ablate start: {n_existing}/{n_files} (row, layer) files already written; "
        f"predicted remaining work {remaining_pred / 3600:.2f} h (cost model)")
    for i, info in enumerate(sample["rows"], 1):
        row = int(info["row"])
        todo = [L for L in inspector.layers if not screen_ablation_path(v15, row, L).is_file()]
        if not todo:
            skipped.append(row)
            predicted_done += info["predicted_seconds"]
            continue
        t0 = time.perf_counter()
        spec = read_spectrum(row)
        exp = inspector.explain(spec)
        saved = json.loads((v15 / "screen" / "explanations" / f"row_{row}.json").read_text(encoding="utf-8"))
        why = explanation_matches(exp, saved)
        if why is not None:
            raise RuntimeError(f"row {row}: fresh explain() does not match step 2 ({why})")
        ctx = inspector.ablation_context(spec, exp)
        for L in todo:
            t_layer = time.perf_counter()
            reports = exhaustive_local(inspector, ctx, L)
            rows = screen_ablation_rows(row, L, reports)
            n_single = sum(r["kind"] == "single" for r in rows)
            expected = info["layers"][str(L)]
            if n_single != expected["pairs"] or len(reports) != expected["tokens"]:
                raise RuntimeError(f"row {row} L{L}: {n_single} pairs / {len(reports)} tokens, step 2 recorded "
                                   f"{expected['pairs']} / {expected['tokens']}")
            seconds = time.perf_counter() - t_layer
            meta = {"schema_version": str(SCREEN_ABLATION_SCHEMA_VERSION), "row": str(row), "layer": str(L),
                    "spectrum_id": info["spectrum_id"], "prediction": exp.prediction["peptide"],
                    "target": json.dumps(ctx.target), "provenance": SCREEN_PROVENANCE,
                    "finding_7_note": FINDING7_SCREEN_NOTE, "caveat": ABLATION_NOTE,
                    "sae_sha256_16": inspector.provenance["sae"][str(L)]["sha256_16"],
                    "screen_sample_sha256_16": sample_sha, "seconds": f"{seconds:.3f}",
                    "predicted_seconds": f"{expected['rows'] * info['c_n']:.3f}"}
            _write_parquet_atomic(rows, screen_ablation_path(v15, row, L), meta)
            files_done += 1
            pred_this_run += layer_pred(info, L)
            elapsed = time.perf_counter() - t_start
            left = max(remaining_pred - pred_this_run, 0.0) * elapsed / pred_this_run
            finish = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + left))
            log(f"[{_stamp()}] row {row} L{L}: {n_single} pairs + {len(reports)} joints in {seconds:.0f} s "
                f"(predicted {expected['rows'] * info['c_n']:.0f} s); files {files_done}/{n_files}; "
                f"elapsed {elapsed / 3600:.2f} h; projected finish {finish} (measured/model "
                f"{elapsed / pred_this_run:.2f})")
        predicted_done += info["predicted_seconds"]
        done.append(row)
        elapsed = time.perf_counter() - t_start
        log(f"[{_stamp()}] row {row} done ({i}/{len(sample['rows'])}) in {(time.perf_counter() - t0) / 60:.1f} min; "
            f"elapsed {elapsed / 3600:.2f} h; predicted work done {predicted_done / total_pred:.0%}")
    return {"done": done, "skipped_existing": skipped, "seconds": time.perf_counter() - t_start}


def _cmd_demo_explain(args, inspector: Inspector) -> int:
    demo_explain(inspector, args.parquet, args.v1_5_dir)
    return 0


# --- V1-5 verifications 2 and 3 (screen-verify) -------------------------------------------

VERIFY_N_PAIRS = 3
VERIFY_ABLATION_TOL = 1e-6


def pick_verify_pairs(files: list[Path], n: int = VERIFY_N_PAIRS,
                      seed: int = SCREEN_SEED) -> tuple[list[tuple[Path, int]], int]:
    """n distinct single-ablation rows drawn with default_rng(seed) from all
    'single' rows of the screen files (sorted-file then row order), and the
    pool size."""
    pool = []
    for f in sorted(files):
        kinds = pq.read_table(f, columns=["kind"]).column("kind").to_pylist()
        pool += [(f, i) for i, k in enumerate(kinds) if k == "single"]
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} single rows to draw {n} from")
    return [pool[int(i)] for i in np.random.default_rng(seed).choice(len(pool), size=n, replace=False)], len(pool)


def compare_ablation_row(stored: dict, result: dict, tol: float = VERIFY_ABLATION_TOL) -> dict:
    """Recomputed single ablation vs its stored screen row: max |Δ| over
    delta_total, delta_eos and every per-residue delta; activation |Δ|; flags."""
    if len(result["delta_log_prob"]) != len(stored["delta_log_prob"]):
        raise ValueError("per-residue delta length differs from the stored row")
    d = max([abs(result["delta_total"] - stored["delta_total"]), abs(result["delta_eos"] - stored["delta_eos"])]
            + [abs(a - b) for a, b in zip(result["delta_log_prob"], stored["delta_log_prob"])])
    act = abs(float(result["activation"]) - stored["activation"])
    flags = ([bool(v) for v in result["argmax_changed"]] == list(stored["argmax_changed"])
             and bool(result["eos_argmax_changed"]) == bool(stored["eos_argmax_changed"]))
    return {"row": stored["row"], "layer": stored["layer"], "token": stored["token"], "feature": stored["feature"],
            "stored_delta_total": stored["delta_total"], "max_abs_delta_diff": d, "activation_diff": act,
            "flags_equal": flags, "pass": d <= tol and act <= tol and flags}


def fit_comparison(explanations: dict[int, dict], fit_rows: dict[int, list[dict]], ref) -> dict:
    """explain()'s fve_centred vs the reference's per-spectrum value (fit_{L}
    rows: chunk, spectrum_index, spectrum_id, fve_centred) per layer, matched by
    HF row location; spectrum ids are checked. ref: a FitReference."""
    out = {}
    for L, rows in sorted(fit_rows.items()):
        by = {(r["chunk"], r["spectrum_index"]): r for r in rows}
        per = []
        for row, e in sorted(explanations.items()):
            r = by.get(hf_test_row_location(row))
            if r is None:
                raise ValueError(f"row {row}: not in the layer {L} fit reference")
            if str(r["spectrum_id"]) != str(row + HF_TEST_OFFSET) or not str(e["input"]["spectrum_id"]).endswith(f":{row}"):
                raise ValueError(f"row {row}: spectrum ids {r['spectrum_id']!r} / {e['input']['spectrum_id']!r} "
                                 f"do not identify HF test row {row}")
            a, b = float(e["layers"][str(L)]["fit"]["fve_centred"]), float(r["fve_centred"])
            per.append({"row": row, "explain": a, "reference": b, "delta": a - b,
                        "percentile_shift": ref.percentile(L, a) - ref.percentile(L, b)})
        worst = max(per, key=lambda x: abs(x["delta"]))
        out[str(L)] = {"n": len(per), "max_abs_delta": abs(worst["delta"]), "at_row": worst["row"],
                       "mean_delta": float(np.mean([x["delta"] for x in per])),
                       "max_abs_percentile_shift": max(abs(x["percentile_shift"]) for x in per), "per_row": per}
    return out


def local_explanations(v15_dir: Path) -> dict[int, dict]:
    """Saved demo and screen explanations of rows in the local chunks 520/572."""
    out = {}
    for d in (Path(v15_dir) / "explanations", Path(v15_dir) / "screen" / "explanations"):
        for f in sorted(d.glob("row_*.json")):
            row = int(f.stem.split("_")[1])
            if hf_test_row_location(row)[0] in SCREEN_CHUNKS:
                out[row] = json.loads(f.read_text(encoding="utf-8"))
    return out


def screen_verify(inspector, parquet_path: Path, v15_dir: Path = V1_5_DIR, log=print,
                  read_spectrum=None) -> dict:
    """V1-5 verification 3 (VERIFY_N_PAIRS random screen pairs recomputed with
    `ablate`, scope local, must match to 1e-6) and verification 2 (explain()'s
    fve_centred vs the fit reference for the local demo and screen spectra,
    max |Δ| reported). Writes screen/screen_verify.json; raises after writing
    if a pair fails."""
    from inspector.dictionary import FitReference
    v15 = Path(v15_dir)
    read_spectrum = read_spectrum or (lambda row: read_raw_spectra(parquet_path, [row])[0])
    files = sorted((v15 / "screen" / "ablation").glob("row_*_L*.parquet"))
    sample = json.loads((v15 / "screen" / "screen_sample.json").read_text(encoding="utf-8"))
    n_expected = len(sample["sample_rows"]) * len(inspector.layers)
    if len(files) != n_expected:
        raise RuntimeError(f"{len(files)} screen ablation files, expected {n_expected}; finish screen-ablate first")
    picks, pool = pick_verify_pairs(files)
    pairs = []
    for f, i in picks:
        stored = pq.read_table(f).slice(i, 1).to_pylist()[0]
        spec = read_spectrum(stored["row"])
        ctx = inspector.ablation_context(spec, inspector.explain(spec))
        res = inspector.ablate(ctx, stored["layer"], [stored["feature"]], "local", position=stored["token"],
                               with_summaries=False).results[0]
        rec = compare_ablation_row(stored, res)
        log(f"verification 3: row {rec['row']} L{rec['layer']} token {rec['token']} F{rec['feature']}: "
            f"max |delta diff| {rec['max_abs_delta_diff']:.2e}, activation diff {rec['activation_diff']:.2e}, "
            f"flags equal {rec['flags_equal']} -> {'pass' if rec['pass'] else 'FAIL'}")
        pairs.append(rec)
    root = inspector.outputs_root
    fit_rows = {L: pq.read_table(Path(root) / "dictionary" / "examples" / f"fit_{L}.parquet").to_pylist()
                for L in inspector.layers}
    fit = fit_comparison(local_explanations(v15), fit_rows, FitReference.load(root))
    for L, v in fit.items():
        log(f"verification 2: L{L}: n {v['n']}, max |delta| {v['max_abs_delta']:.2e} (row {v['at_row']}), "
            f"mean delta {v['mean_delta']:+.2e}, max percentile shift {v['max_abs_percentile_shift']:.2f}")
    out = {"verification_3": {"n_pairs": len(pairs), "pool_single_rows": pool, "seed": SCREEN_SEED,
                              "tolerance": VERIFY_ABLATION_TOL, "pairs": pairs,
                              "all_pass": all(p["pass"] for p in pairs)},
           "verification_2": {"layers": fit, "note": "explain() JSON rounds fve_centred to 6 decimals; "
                              "the reference is from the stored bf16 extraction, explain() from a fresh "
                              "float32 forward pass"}}
    _write_json_atomic(out, v15 / "screen" / "screen_verify.json")
    if not out["verification_3"]["all_pass"]:
        raise RuntimeError("verification 3 failed: a recomputed pair differs from its stored row beyond 1e-6")
    return out


def _cmd_screen_verify(args, inspector: Inspector) -> int:
    screen_verify(inspector, args.parquet, args.v1_5_dir)
    print(f"screen-verify: all pairs pass -> {args.v1_5_dir / 'screen' / 'screen_verify.json'}")
    return 0


def _cmd_screen_sample(args, inspector: Inspector) -> int:
    out = screen_sample(inspector, args.parquet, args.v1_5_dir, args.budget_hours, force=args.force)
    t = out["measured_one_batch"]
    print(f"screen sample: {out['n_sample']} rows (walked {out['n_walked']}, skipped {len(out['skipped_no_beam'])} "
          f"without a completed beam); predicted {out['total_predicted_hours']:.2f} h of {out['budget_hours']} h; "
          f"first excluded {out['first_excluded']}")
    print(f"overlap with the demo set: {out['overlap_with_demo_set']}; exact vs label {out['n_exact']}/{out['n_sample']}")
    print(f"one batch: {t['n_deltas']} ablations in {t['batch_seconds']:.2f} s = {t['measured_per_row_s']:.4f} s/row "
          f"vs model c({t['n_peaks']}) = {t['model_per_row_s']:.4f} s/row (ratio {t['ratio_measured_to_model']:.2f})")
    return 0


def _cmd_screen_ablate(args, inspector: Inspector) -> int:
    res = screen_ablate(inspector, args.parquet, args.v1_5_dir, args.budget_hours, force=args.force)
    print(f"screen-ablate: {len(res['done'])} rows ablated, {len(res['skipped_existing'])} already complete, "
          f"{res['seconds'] / 3600:.2f} h")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["time", "compare-preprocessing", "accuracy", "audit-cases",
                                       "audit-checks", "audit-notation", "audit-evidence-by",
                                       "audit-part-b", "demo-explain", "screen-sample", "screen-ablate",
                                       "screen-verify"])
    p.add_argument("--v1-5-dir", type=Path, default=V1_5_DIR,
                   help="demo-explain / screen-sample / screen-ablate / screen-verify: V1-5 output directory")
    p.add_argument("--budget-hours", type=float, default=SCREEN_BUDGET_HOURS,
                   help="screen-sample / screen-ablate: predicted-cost budget (registered: 9)")
    p.add_argument("--force", action="store_true",
                   help="screen-sample: recompute an existing sample; screen-ablate: run over budget")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    p.add_argument("--parquet", type=Path,
                   help="raw spectra parquet (InstaNovo columns); every command but audit-evidence-by")
    p.add_argument("--rows", type=int, nargs="+", help="rows to use (accuracy: default a random sample)")
    p.add_argument("--n", type=int, default=200, help="accuracy: sample size")
    p.add_argument("--seed", type=int, default=0, help="accuracy: sampling seed")
    p.add_argument("--decoders", nargs="+", default=["beam", "greedy"], choices=["beam", "greedy"])
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--ablation", action="store_true",
                   help="time: also time exact ablation (both scopes, per-peak and latent panels)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS_ROOT / "analysis" / "v1_4c",
                   help="audit-cases / -checks / -notation / -evidence-by / -part-b: V1-4c output directory")
    p.add_argument("--n-top", type=int, default=5, help="audit-cases: cases taken by rank")
    p.add_argument("--min-evidence", type=int, default=2,
                   help=f"audit-cases: selected cases that must be '{EVIDENCE_FOR_TRUE}'")
    p.add_argument("--isotope-rows", type=int, nargs="*", default=[],
                   help="audit-checks: rows re-annotated with max_isotope=1 under the prediction")
    p.add_argument("--decoy-rows", type=int, nargs="*", default=[],
                   help="audit-checks: rows annotated against shuffled-sequence decoys")
    args = p.parse_args(argv)
    if args.command == "audit-evidence-by":           # reads candidates.csv only; no model
        return _cmd_audit_evidence_by(args)
    if args.parquet is None:
        p.error(f"{args.command} needs --parquet")
    if args.command not in ("accuracy", "audit-cases", "audit-checks", "audit-notation",
                            "audit-part-b", "demo-explain", "screen-sample", "screen-ablate",
                            "screen-verify") and not args.rows:
        p.error(f"{args.command} needs --rows")
    inspector = Inspector(args.checkpoint, args.outputs_root, top_k=args.top_k)
    rc = {"time": _cmd_time, "compare-preprocessing": _cmd_compare, "accuracy": _cmd_accuracy,
          "audit-cases": _cmd_audit_cases, "audit-checks": _cmd_audit_checks,
          "audit-notation": _cmd_audit_notation, "audit-part-b": _cmd_audit_part_b,
          "demo-explain": _cmd_demo_explain, "screen-sample": _cmd_screen_sample,
          "screen-ablate": _cmd_screen_ablate, "screen-verify": _cmd_screen_verify}[args.command](args, inspector)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
