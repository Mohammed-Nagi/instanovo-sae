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
then jointly, and ``ablate`` takes any features in either scope.
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
                   "top_concept": "x" * longest_concept, "n_tokens_active": 999}
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["time", "compare-preprocessing", "accuracy", "audit-cases",
                                       "audit-checks", "audit-notation"])
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    p.add_argument("--parquet", type=Path, required=True, help="raw spectra parquet (InstaNovo columns)")
    p.add_argument("--rows", type=int, nargs="+", help="rows to use (accuracy: default a random sample)")
    p.add_argument("--n", type=int, default=200, help="accuracy: sample size")
    p.add_argument("--seed", type=int, default=0, help="accuracy: sampling seed")
    p.add_argument("--decoders", nargs="+", default=["beam", "greedy"], choices=["beam", "greedy"])
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--ablation", action="store_true",
                   help="time: also time exact ablation (both scopes, per-peak and latent panels)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS_ROOT / "analysis" / "v1_4c",
                   help="audit-cases: output directory")
    p.add_argument("--n-top", type=int, default=5, help="audit-cases: cases taken by rank")
    p.add_argument("--min-evidence", type=int, default=2,
                   help=f"audit-cases: selected cases that must be '{EVIDENCE_FOR_TRUE}'")
    p.add_argument("--isotope-rows", type=int, nargs="*", default=[],
                   help="audit-checks: rows re-annotated with max_isotope=1 under the prediction")
    p.add_argument("--decoy-rows", type=int, nargs="*", default=[],
                   help="audit-checks: rows annotated against shuffled-sequence decoys")
    args = p.parse_args(argv)
    if args.command not in ("accuracy", "audit-cases", "audit-checks", "audit-notation") and not args.rows:
        p.error(f"{args.command} needs --rows")
    inspector = Inspector(args.checkpoint, args.outputs_root, top_k=args.top_k)
    rc = {"time": _cmd_time, "compare-preprocessing": _cmd_compare, "accuracy": _cmd_accuracy,
          "audit-cases": _cmd_audit_cases, "audit-checks": _cmd_audit_checks,
          "audit-notation": _cmd_audit_notation}[args.command](args, inspector)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
