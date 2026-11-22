"""runtime.py -- explain(spectrum): what InstaNovo's encoder extracted from one spectrum.

Scope: the encoder. The output says which SAE features fire on each peak at
layers 2/4/6/8, what the feature dictionary knows about them, and how well each
SAE reconstructs this spectrum. It does not explain how the decoder chose each
residue; attribution (V1-4b) is not implemented yet and ``attribution=True``
raises.

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
  6. Dictionary rows for every feature that reaches a top-k list.

Dependencies on the pipeline: instanovo_io, annotate, schema, the dictionary,
and the SAE checkpoints (read by train.load_sae_from_checkpoint, the checkpoint
format's own loader). Nothing here touches the network: the checkpoint must be
a local file (no model-id download path) and modified residues are given to
spectrum_utils as delta masses, never UNIMOD tags.

CLI (from the repo root)::

    PYTHONPATH=src uv run python -m inspector.runtime time \\
        --parquet <raw spectra parquet> --rows 4506 4507 4508
    PYTHONPATH=src uv run python -m inspector.runtime compare-preprocessing \\
        --parquet <raw spectra parquet> --rows 4506 ...

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
from inspector.dictionary import FeatureDictionary
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
ATTRIBUTION_TODO = "attribution is V1-4b and not implemented yet"

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
    attribution: dict | None
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


def _validate(data: dict) -> None:
    """Structural checks beyond the top-level keys; raises ValueError."""
    n_peaks = data["input"]["n_processed_peaks"]
    if len(data["peaks"]) != n_peaks:
        raise ValueError(f"{len(data['peaks'])} peaks listed, input says {n_peaks}")
    for peak in data["peaks"]:
        if set(peak) != _PEAK_KEYS:
            raise ValueError(f"peak entry keys {sorted(peak)}")
    for layer, block in data["layers"].items():
        if not layer.isdigit() or set(block) != {"fit", "tokens"}:
            raise ValueError(f"layer block {layer!r} malformed")
        if len(block["tokens"]) != n_peaks + 1:
            raise ValueError(f"layer {layer}: {len(block['tokens'])} tokens for {n_peaks} peaks")
        for tok in block["tokens"]:
            if set(tok) != _TOKEN_KEYS:
                raise ValueError(f"layer {layer} token keys {sorted(tok)}")
        listed = {str(f["feature"]) for tok in block["tokens"] for f in tok["top"]}
        if listed != set(data["features"].get(layer, {})):
            raise ValueError(f"layer {layer}: features block does not match the top-k lists")


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


@contextlib.contextmanager
def _capture_encoder_layers(model: Any, layers: tuple[int, ...]):
    """Collect the outputs of the given encoder layers during a forward pass.

    The same capture point as extract.MultiLayerCapture (the layer module's
    output, first element if a tuple); kept local because the tool does not
    depend on extract.py. tests/test_inspector_runtime.py checks the captured
    activations against the stored chunks.
    """
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer):
        def hook(_module, _inputs, output):
            captured[layer] = (output[0] if isinstance(output, tuple) else output).detach()
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
            "dictionary_schema": {str(L): d.metadata.get("schema_version") for L, d in self.dictionaries.items()},
            "dictionary_blinded_run": {str(L): d.metadata.get("blinded_run") for L, d in self.dictionaries.items()},
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

    def explain(self, spectrum: Spectrum, attribution: bool = False) -> Explanation:
        if attribution:
            raise NotImplementedError(ATTRIBUTION_TODO)
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

        layers_out, features_out = {}, {}
        for L in self.layers:
            x = fwd.activations[L]
            with torch.inference_mode():
                out = self.saes[L].forward_inference(x.to(self.device))
            feats, x_hat = out["features"].cpu(), out["x_hat"].cpu()
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
            entries = {}
            for f in distinct:
                row = self.dictionaries[L].lookup(f)
                entries[str(f)] = {"in_dictionary": False} if row is None else {"in_dictionary": True, **row}
            features_out[str(L)] = entries
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
            "attribution": None,
            "timings_s": timings,
        }
        lap("assemble", t)
        payload["timings_s"] = dict(timings)
        return Explanation.from_dict(_finalise(payload))

    # -- size bound ----------------------------------------------------------

    @property
    def json_size_bound(self) -> int:
        """Upper bound on len(explanation.to_json()) for this configuration.

        Built from worst-case entries: n_peaks peaks with the longest label and
        every concept; max_length residues; per layer, (n_peaks + 1) tokens with
        k features each, plus one dictionary row per distinct feature, at most
        min((n_peaks + 1) * k, live features), each no longer than the layer's
        longest row. 64 KB covers provenance, input, prediction scalars,
        timings and notes.
        """
        if self._json_size_bound is not None:
            return self._json_size_bound
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
        n_tokens = self.n_peaks + 1
        bound = 64_000 + self.n_peaks * size(peak) + self.max_length * (size(residue) + 2 * 18)
        for d in self.dictionaries.values():
            row_max = max(size(_finalise({"in_dictionary": True, **d.lookup(f)}))
                          for f in d) + 8                       # + '"12345": '
            bound += n_tokens * size(token) + min(n_tokens * self.top_k, len(d)) * row_max
        self._json_size_bound = int(bound)
        return self._json_size_bound


_DEFAULT: Inspector | None = None


def explain(spectrum: Spectrum, attribution: bool = False, inspector: Inspector | None = None,
            **inspector_kwargs) -> Explanation:
    """explain() with a cached default Inspector, built from inspector_kwargs on first use."""
    global _DEFAULT
    if inspector is None:
        if _DEFAULT is None:
            _DEFAULT = Inspector(**inspector_kwargs)
        inspector = _DEFAULT
    return inspector.explain(spectrum, attribution=attribution)


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["time", "compare-preprocessing", "accuracy"])
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    p.add_argument("--parquet", type=Path, required=True, help="raw spectra parquet (InstaNovo columns)")
    p.add_argument("--rows", type=int, nargs="+", help="rows to use (accuracy: default a random sample)")
    p.add_argument("--n", type=int, default=200, help="accuracy: sample size")
    p.add_argument("--seed", type=int, default=0, help="accuracy: sampling seed")
    p.add_argument("--decoders", nargs="+", default=["beam", "greedy"], choices=["beam", "greedy"])
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = p.parse_args(argv)
    if args.command != "accuracy" and not args.rows:
        p.error(f"{args.command} needs --rows")
    inspector = Inspector(args.checkpoint, args.outputs_root, top_k=args.top_k)
    {"time": _cmd_time, "compare-preprocessing": _cmd_compare,
     "accuracy": _cmd_accuracy}[args.command](args, inspector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
