"""analysis.py -- post-hoc analysis over the released evaluate.py reports.

Everything here reads each layer's ``report.json`` (Phase 8 results) and, with
``--descriptions``, the interpret.py description runs. Those modes need neither
torch nor the activation chunks, so they run on any machine; only
``--deamidation-precursor`` and ``--label-audit`` (below) read the extract
chunks (torch, imported lazily).

Hub inventory
    Phase 8 ablates, for every concept, its top ``ablation_per_feature_top``
    detectors one at a time (``per_feature_ablation``) and as a group of
    ``ablation_top_n`` (``top_n_features``). The inventory has one row per
    distinct single-ablated feature:

    * ranked by mean delta-CE, rank 1 = the feature whose removal hurts most;
    * ``ratio_to_median``: delta-CE over the layer median |delta-CE|, a
      descriptive scale only -- there is no hub flag or threshold, because any
      cut would be an arbitrary label on a continuous effect size;
    * listing the concepts whose single-ablation list contains it
      (``single_concepts``) and the concepts whose group ablation contains it
      (``group_concepts``). Group membership is what lets one large-effect
      feature dominate a group result.

    The median is taken over the ablated features, not over all d_dict features:
    those are the top F1dom detectors of each concept, a selected population, and
    only they have a single-feature delta-CE in the report. It is the median of
    |delta-CE|, because at layer 8 the signed median is slightly negative.

    A feature's delta-CE does not depend on which concept it was listed under;
    ``per_feature_delta_ce`` checks this and fails if a report disagrees.

F1dom vs ablation selectivity
    ``f1dom_selectivity_correlation`` is the Spearman correlation, across
    concepts, between a concept's best F1dom among its ablated features and the
    selectivity of that concept's group ablation. Concepts whose selectivity is
    NaN (a tercile split that is empty) are dropped; the count is reported.

Blinded vs unblinded descriptions (``--descriptions BLIND UNBLINDED``)
    Compares two interpret.py runs that differ only in ``--blind``: same
    features, same describe and held-out examples (checked, and an error if
    not). Per layer and stratum (plus "all"): median held-out r and detection
    AUROC of each run, each run's shuffled-description null (pooled over the
    layer: 95th percentile and median), the number of features whose r beats
    that run's threshold, and the paired median difference in r (blind minus
    unblinded) with a seeded percentile-bootstrap 95% CI over features. Reads
    ``interpret/layer_{L}/<tag>/feature_descriptions.json``. A second table
    gives each run's shuffled r per layer (median, p95), overall and split by
    whether the donor description came from a feature in the same stratum.

Deamidation vs 13C at the precursor (``--deamidation-precursor``, Finding 8(d))
    The one mode that reads the local extract chunks (labels only, so it needs
    torch; imported lazily). For each label carrying a deamidation: its
    precursor ppm at isotope offsets 0 and 1, and the offset-1 ppm of the same
    peptide with one deamidation removed (the 13C-misassignment explanation).
    Centre = median offset-0 ppm of labels with no modification; D = |deamidated
    - centre| - |13C-of-amidated - centre|; each explanation "fits" within 3
    robust sigma (1.4826 x MAD). Writes
    ``analysis/label_audit/deamidation_precursor{.csv,_summary.json}``.

Label-quality audit (``--label-audit``, Finding 9)
    Also reads the extract chunks, plus the stored labels. A seeded sample of
    200 local spectra (15-16 per chunk) is re-annotated from the stored
    processed peaks: first through ``annotate.annotate_spectrum``, which must
    reproduce the stored labels exactly (otherwise the audit stops), then with
    one spectrum_utils call per peptide that returns every candidate (a ions
    on, max_isotope 1), from which ``annotate.select_annotation`` derives the
    pipeline, isotope-1 and a-ion settings. The true peptide and 20
    same-composition decoys (``annotate.peptide_decoys``) give, per class and
    per concept, the chance fraction (decoy mean / true count; immonium,
    precursor, y1, b_{L-1}, a_{L-1} excluded as composition-determined), the
    isotope-1 and a-ion transitions, co-matching classes, and b/y coverage, with
    percentile-bootstrap 95% CIs over spectra. Writes ``analysis/label_audit/``:
    reproduction.json, sample.csv, per_class.csv, determined.csv, isotope.csv,
    precedence.csv, coverage.csv, a_ions.csv, concepts.csv, summary.json, and
    concept_reliability.csv (V1-3b: per concept its dependence on the peptide,
    chance fraction with CI, positives the two labelling bugs add or hide, and
    whether it meets the display rule ``DISPLAY_RULE``).

Reliable-association list (``--reliable-associations``, V1-3b / Finding 10)
    Reads the four dictionaries (no chunks) and lists the features whose best
    association with a concept meeting the display rule has F1dom >= 0.1, a
    listing cut for later use (demo set, feature cards), not a reliability
    claim. Writes ``analysis/reliable_associations.csv`` and
    ``reliable_associations_summary.json`` (per layer: live features, features
    with any reliable association, the median of their best reliable F1dom, and
    how many reach 0.1 and 0.3).

V1-5 demo set (``--demo-set``) and activation screen (``--activation-screen``)
    ``--demo-set`` applies the registered rule D1 (STATUS V1-5): V1-4c cases
    H7/H1/H5 with labels, the top 3 correct V1-4a rows in chunks 520/572 by beam
    log p, and per layer the spectrum where the feature with the highest
    reliable F1dom fires hardest on a peak (beam completion checked with
    instanovo_io); every row is checked against its HF row and, when local, the
    chunk metadata. Stops if a derived pick differs from the registered one.
    Writes ``analysis/v1_5/demo_set.json``. ``--activation-screen`` reads the
    screen ablations (``inspector.runtime screen-ablate``) and computes the
    registered statistics (a)-(c) per layer with bootstrap CIs over spectra,
    plus V1-4c's exploratory numbers recomputed; writes
    ``analysis/v1_5/screen/activation_screen.json`` and ``activation_screen_peaks.csv``.

Usage (from the repo root)::

    PYTHONPATH=src uv run python -m analysis --demo-set --parquet <HF test parquet>
    PYTHONPATH=src uv run python -m analysis --activation-screen

    PYTHONPATH=src uv run python -m analysis
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --layers 2 4
    PYTHONPATH=src uv run python -m analysis --descriptions blind_v1 unblinded_v2
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --deamidation-precursor
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --label-audit
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --reliable-associations

Outputs under ``<root>/analysis/``: ``hubs.csv`` (a ranked list, no flag) and
``f1dom_selectivity.csv``; with ``--descriptions``, only
``description_blinding.csv`` and ``description_shuffled_null.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import spearmanr

from schema import DICTIONARY_SCHEMA_VERSION, EVAL_SCHEMA_VERSION, check_run_tag

LAYERS = (2, 4, 6, 8)
DEFAULT_ROOT = Path("sae_pipeline_outputs")
DEFAULT_SEED = 0

HUB_COLUMNS = [
    "layer", "feature_idx", "rank", "mean_delta_ce", "ratio_to_median",
    "n_single_concepts", "single_concepts", "n_group_concepts", "group_concepts",
]
CORRELATION_COLUMNS = ["layer", "spearman_rho", "p_value", "n_concepts", "n_dropped_nan"]
BLINDING_COLUMNS = [
    "layer", "stratum", "n_features",
    "blind_median_r", "unblinded_median_r",
    "blind_median_auroc", "unblinded_median_auroc",
    "blind_null_p95", "unblinded_null_p95", "blind_null_median", "unblinded_null_median",
    "blind_n_beats", "blind_n_tested", "unblinded_n_beats", "unblinded_n_tested",
    "n_paired", "median_delta_r", "delta_r_ci_low", "delta_r_ci_high",
]
NULL_COLUMNS = [
    "layer", "run", "n", "n_undefined", "median", "p95",
    "same_stratum_n", "same_stratum_median", "same_stratum_p95",
    "other_stratum_n", "other_stratum_median", "other_stratum_p95",
]
N_BOOTSTRAP = 10_000


def report_path(root: Path, layer: int, seed: int = DEFAULT_SEED) -> Path:
    return root / "sae" / f"layer_{layer}" / f"seed_{seed}" / "eval" / "report.json"


def load_report(path: Path) -> dict:
    """Load a report.json and reject schema versions this module was not written for."""
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    version = report.get("schema_version")
    if version != EVAL_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: report schema_version {version!r}, expected {EVAL_SCHEMA_VERSION}"
        )
    if "per_concept" not in report.get("phase_8", {}):
        raise ValueError(f"{path}: no phase_8.per_concept; Phase 8 has not been run")
    return report


def per_feature_delta_ce(report: dict, layer: int) -> dict[int, float]:
    """Mean delta-CE of every single-ablated feature in a report (Phase 8, SAE-patched).

    A feature listed under several concepts must carry the same value under each;
    NaN or disagreement is an error, not something to average over.
    """
    delta_ce: dict[int, float] = {}
    for concept, info in report["phase_8"]["per_concept"].items():
        for entry in info["per_feature_ablation"]:
            feat, d = int(entry["feature_idx"]), float(entry["mean_delta_ce"])
            if math.isnan(d):
                raise ValueError(f"layer {layer}: feature {feat} has NaN mean_delta_ce ({concept})")
            if feat in delta_ce and not math.isclose(delta_ce[feat], d, rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError(
                    f"layer {layer}: feature {feat} has delta-CE {delta_ce[feat]} and {d} "
                    f"under different concepts; it should not depend on the concept"
                )
            delta_ce[feat] = d
    return delta_ce


def hub_inventory(report: dict, layer: int) -> list[dict]:
    """One row per single-ablated feature, ranked by mean delta-CE (see module docstring)."""
    delta_ce = per_feature_delta_ce(report, layer)
    if not delta_ce:
        raise ValueError(f"layer {layer}: report has no per-feature ablations")

    single: dict[int, list[str]] = {}
    group: dict[int, list[str]] = {}
    for concept, info in report["phase_8"]["per_concept"].items():
        for entry in info["per_feature_ablation"]:
            single.setdefault(int(entry["feature_idx"]), []).append(concept)
        for feat in info["top_n_features"]:
            group.setdefault(int(feat), []).append(concept)

    median_abs = median(abs(d) for d in delta_ce.values())
    if median_abs <= 0.0:
        raise ValueError(f"layer {layer}: median |delta-CE| is {median_abs}; no ratio to report")

    ranked = sorted(delta_ce.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {
            "layer": layer,
            "feature_idx": feat,
            "rank": rank,
            "mean_delta_ce": d,
            "ratio_to_median": d / median_abs,
            "n_single_concepts": len(single[feat]),
            "single_concepts": "|".join(sorted(single[feat])),
            "n_group_concepts": len(group.get(feat, [])),
            "group_concepts": "|".join(sorted(group.get(feat, []))),
        }
        for rank, (feat, d) in enumerate(ranked, start=1)
    ]


def f1dom_selectivity_correlation(report: dict, layer: int) -> dict:
    """Spearman rho between each concept's best ablated-feature F1dom and its group selectivity."""
    f1dom: list[float] = []
    selectivity: list[float] = []
    n_dropped = 0
    for info in report["phase_8"]["per_concept"].values():
        entries = info["per_feature_ablation"]
        sel = float(info["causal"]["selectivity"])
        if not entries or math.isnan(sel):
            n_dropped += 1
            continue
        f1dom.append(max(float(e["f1_dom"]) for e in entries))
        selectivity.append(sel)
    if len(f1dom) < 3:
        raise ValueError(f"layer {layer}: only {len(f1dom)} concepts usable for a correlation")
    result = spearmanr(f1dom, selectivity)
    return {
        "layer": layer,
        "spearman_rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_concepts": len(f1dom),
        "n_dropped_nan": n_dropped,
    }


def description_run_path(root: Path, layer: int, tag: str) -> Path:
    return root / "interpret" / f"layer_{layer}" / check_run_tag(tag) / "feature_descriptions.json"


def load_description_run(path: Path, blind: bool) -> dict[int, dict]:
    """feature_idx -> result record of one interpret.py run, checked to be blind (or not) as claimed."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if bool(payload["config"].get("blind", False)) != blind:
        raise ValueError(f"{path}: config.blind is {payload['config'].get('blind')}, expected {blind}")
    results = {int(r["feature_idx"]): r for r in payload["results"]}
    undescribed = sorted(f for f, r in results.items() if not r.get("model_served"))
    if undescribed:
        raise ValueError(f"{path}: features without a description: {undescribed[:5]}")
    return results


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def _median_or_nan(values) -> float:
    finite = _finite(values)
    return float(np.median(finite)) if finite else float("nan")


def _bootstrap_median_ci(values: list[float], seed: int,
                         n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI of the median, resampling the values."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    medians = np.median(data[rng.integers(0, len(data), size=(n_boot, len(data)))], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def _example_keys(record: dict) -> tuple[list, list]:
    holdout = [(e["chunk_idx"], e["token_in_chunk"]) for e in record["holdout_examples"]]
    describe = [tuple(k) for k in record["describe_keys"]]
    return describe, holdout


def description_blinding(blind: dict[int, dict], plain: dict[int, dict], layer: int,
                         seed: int = DEFAULT_SEED) -> list[dict]:
    """Rows of BLINDING_COLUMNS for one layer (see the module docstring)."""
    if set(blind) != set(plain):
        raise ValueError(f"layer {layer}: runs describe different features "
                         f"({sorted(set(blind) ^ set(plain))[:5]} differ)")
    for feat, record in blind.items():
        if record["stratum"] != plain[feat]["stratum"]:
            raise ValueError(f"layer {layer}: feature {feat} is in different strata")
        if _example_keys(record) != _example_keys(plain[feat]):
            raise ValueError(f"layer {layer}: feature {feat} has different describe or held-out "
                             f"examples in the two runs; they do not differ only in blinding")

    def null(run: dict[int, dict]) -> list[float]:
        return _finite(v for r in run.values()
                       for v in (r.get("shuffled") or {}).get("r_values", []))

    null_blind, null_plain = null(blind), null(plain)
    strata = sorted({r["stratum"] for r in blind.values()})
    rows = []
    for stratum in [*strata, "all"]:
        feats = sorted(f for f in blind if stratum == "all" or blind[f]["stratum"] == stratum)
        deltas = [blind[f]["pearson_r"] - plain[f]["pearson_r"] for f in feats
                  if not math.isnan(blind[f]["pearson_r"])
                  and not math.isnan(plain[f]["pearson_r"])]
        low, high = _bootstrap_median_ci(deltas, seed)
        row = {"layer": layer, "stratum": stratum, "n_features": len(feats),
               "n_paired": len(deltas), "median_delta_r": _median_or_nan(deltas),
               "delta_r_ci_low": low, "delta_r_ci_high": high}
        for name, run, null_r in (("blind", blind, null_blind), ("unblinded", plain, null_plain)):
            tested = [b for b in (run[f].get("description_beats_baseline") for f in feats)
                      if b is not None]
            row[f"{name}_median_r"] = _median_or_nan(run[f]["pearson_r"] for f in feats)
            row[f"{name}_median_auroc"] = _median_or_nan(run[f].get("detection_auroc") for f in feats)
            row[f"{name}_null_p95"] = float(np.percentile(null_r, 95)) if null_r else float("nan")
            row[f"{name}_null_median"] = _median_or_nan(null_r)
            row[f"{name}_n_beats"] = sum(tested)
            row[f"{name}_n_tested"] = len(tested)
        rows.append(row)
    return rows


def shuffled_null_by_donor(run: dict[int, dict], layer: int, run_name: str) -> dict:
    """One run's shuffled-description r, overall and split by whether the donor shares the stratum.

    A high null driven by same-stratum donors means the baseline partly measures
    how alike the features in a stratum are, not only what a description adds.
    """
    pooled: dict[str, list[float]] = {"all": [], "same": [], "other": []}
    n_undefined = 0
    for feat, record in run.items():
        shuffled = record.get("shuffled") or {}
        donors, values = shuffled.get("donors", []), shuffled.get("r_values", [])
        if len(donors) != len(values):
            raise ValueError(f"layer {layer} {run_name}: feature {feat} has {len(donors)} donors "
                             f"but {len(values)} shuffled r values")
        for donor, r in zip(donors, values):
            if int(donor) not in run:
                raise ValueError(f"layer {layer} {run_name}: donor {donor} is not in the run")
            if r is None or math.isnan(float(r)):
                # A constant prediction (usually all zeros) leaves r undefined;
                # counted here, left out of the null as interpret.py does.
                n_undefined += 1
                continue
            side = "same" if run[int(donor)]["stratum"] == record["stratum"] else "other"
            pooled["all"].append(float(r))
            pooled[side].append(float(r))

    def p95(values: list[float]) -> float:
        return float(np.percentile(values, 95)) if values else float("nan")

    return {
        "layer": layer, "run": run_name,
        "n": len(pooled["all"]), "n_undefined": n_undefined, "median": _median_or_nan(pooled["all"]), "p95": p95(pooled["all"]),
        "same_stratum_n": len(pooled["same"]), "same_stratum_median": _median_or_nan(pooled["same"]),
        "same_stratum_p95": p95(pooled["same"]),
        "other_stratum_n": len(pooled["other"]),
        "other_stratum_median": _median_or_nan(pooled["other"]),
        "other_stratum_p95": p95(pooled["other"]),
    }


# --- Deamidation vs 13C at the precursor (Finding 8(d)) -----------------------

DEAMIDATION_MASS = 0.984016
PRECURSOR_TOL_PPM = 20.0                     # InstaNovo's filter_precursor_ppm, as in V1-4a / V1-4c
DEAMIDATION_MASS_BINS = (967.0, 1500.0, 2500.0)
DEAMIDATION_COLUMNS = [
    "chunk", "spectrum", "spectrum_id", "proforma", "charge", "neutral_mass", "n_deamidation",
    "ppm_label_iso0", "ppm_label_iso1", "ppm_amidated_iso0", "ppm_amidated_iso1",
    "label_meets_only_iso1", "amidated_meets_iso1", "d_ppm",
]


def remove_one_deamidation(proforma: str) -> str | None:
    """The ProForma string with its first deamidation tag (+0.98 on N or Q)
    removed, or None if it has none."""
    import annotate
    tokens = annotate.proforma_residue_tokens(proforma)
    for i, tok in enumerate(tokens):
        m = re.search(r"\[([^\]]+)\]", tok)
        if tok[0] in "NQ" and m and abs(float(m.group(1)) - DEAMIDATION_MASS) <= annotate.PTM_MASS_TOL:
            return "".join(tokens[:i] + [tok[0]] + tokens[i + 1:])
    return None


def precursor_explanations(proforma: str, precursor_mz: float, charge: int,
                           tol: float = PRECURSOR_TOL_PPM) -> dict:
    """ppm of a label at isotope offsets 0 and 1, and, if it carries a
    deamidation, of the same peptide with one deamidation removed (the
    "amidated" form, whose offset-1 match is the 13C misassignment explanation)."""
    import annotate
    mz = annotate.proforma_mz(proforma, charge)
    out = {"neutral_mass": annotate.proforma_neutral_mass(proforma),
           "ppm_label_iso0": annotate.precursor_ppm(mz, precursor_mz, charge, 0),
           "ppm_label_iso1": annotate.precursor_ppm(mz, precursor_mz, charge, 1)}
    out["label_meets_only_iso1"] = abs(out["ppm_label_iso0"]) > tol and abs(out["ppm_label_iso1"]) <= tol
    amidated = remove_one_deamidation(proforma)
    if amidated is not None:
        amz = annotate.proforma_mz(amidated, charge)
        out["ppm_amidated_iso0"] = annotate.precursor_ppm(amz, precursor_mz, charge, 0)
        out["ppm_amidated_iso1"] = annotate.precursor_ppm(amz, precursor_mz, charge, 1)
        out["amidated_meets_iso1"] = abs(out["ppm_amidated_iso1"]) <= tol
    return out


MAD_TO_SIGMA = 1.4826                        # MAD -> sigma for a normal distribution
FIT_SIGMAS = 3.0
D_OVER_GAP_EDGES = (-math.inf, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, math.inf)
D_OVER_GAP_LABELS = ("<-1.25", "-1.25..-0.75", "-0.75..-0.25", "-0.25..0.25", "0.25..0.75", "0.75..1.25", ">1.25")


def gap_ppm(neutral_mass: float) -> float:
    """The 13C-minus-deamidation gap (19.3 mDa) in ppm of the peptide's mass."""
    return (1.00335 - DEAMIDATION_MASS) / neutral_mass * 1e6


def explanation_fit(ppm_deamidated_iso0: float, ppm_amidated_iso1: float, centre: float, width: float) -> str:
    """Which explanation lies within `width` ppm of the instrument's centre:
    'deamidated_only', '13c_only', 'both' (unresolvable) or 'neither'."""
    d, c = abs(ppm_deamidated_iso0 - centre) <= width, abs(ppm_amidated_iso1 - centre) <= width
    return {(True, False): "deamidated_only", (False, True): "13c_only",
            (True, True): "both", (False, False): "neither"}[(d, c)]


def deamidation_d(ppm_deamidated_iso0: float, ppm_amidated_iso1: float, centre: float) -> float:
    """D = |ppm of the deamidated explanation - centre| - |ppm of the
    13C-of-amidated explanation - centre|; D > 0 means the 13C explanation sits
    closer to the instrument's typical error (centre = median ppm of unmodified labels)."""
    return abs(ppm_deamidated_iso0 - centre) - abs(ppm_amidated_iso1 - centre)


def mass_bin(mass: float, cuts: tuple[float, ...] = DEAMIDATION_MASS_BINS) -> str:
    edges = (0.0, *cuts, math.inf)
    for lo, hi in zip(edges, edges[1:]):
        if lo <= mass < hi:
            return f"<{hi:g}" if lo == 0 else (f">={lo:g}" if hi == math.inf else f"{lo:g}-{hi:g}")
    raise ValueError(mass)


def _quantiles(x) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return {"n": 0}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(x.size), "q05": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q95": q[4],
            "frac_positive": float(np.mean(x > 0))}


def deamidation_precursor(root: Path, tol: float = PRECURSOR_TOL_PPM) -> tuple[list[dict], dict]:
    """Finding 8(d): one pass over the local extract chunks (labels only, no
    annotation). Reference = labels with no modification: median and MAD of
    their offset-0 ppm, over those meeting the precursor at offset 0."""
    import torch
    import annotate
    manifest = json.loads((root / "extract" / "manifest.json").read_text())
    rows, ref_ppm, ref_only_iso1, n_ref = [], [], 0, 0
    n_spectra = 0
    for ch in manifest["chunks"]:
        meta = torch.load(root / "extract" / ch["meta"], map_location="cpu", weights_only=False)
        for s in range(meta["n_spectra"]):
            n_spectra += 1
            prof, mods = meta["proforma_strings"][s], meta["modifications"][s]
            z, pmz = int(meta["precursor_charges"][s]), float(meta["precursor_mzs"][s])
            n_deam = sum(annotate._mod_to_ptm_name(m, meta["peptides"][s]) == "deamidation" for m in mods)
            if not mods:
                e = precursor_explanations(prof, pmz, z, tol)
                n_ref += 1
                ref_only_iso1 += e["label_meets_only_iso1"]
                if abs(e["ppm_label_iso0"]) <= tol:
                    ref_ppm.append(e["ppm_label_iso0"])
            elif n_deam:
                e = precursor_explanations(prof, pmz, z, tol)
                if "ppm_amidated_iso1" not in e:
                    raise ValueError(f"deamidation in modifications but not in ProForma: {prof!r}")
                rows.append({"chunk": ch["idx"], "spectrum": s, "spectrum_id": meta["spectrum_ids"][s],
                             "proforma": prof, "charge": z, "n_deamidation": n_deam, **e})
        del meta
    centre = float(np.median(ref_ppm))
    mad = float(np.median(np.abs(np.asarray(ref_ppm) - centre)))
    for r in rows:
        r["d_ppm"] = deamidation_d(r["ppm_label_iso0"], r["ppm_amidated_iso1"], centre)

    sigma = MAD_TO_SIGMA * mad

    def block(sel):
        fit = [explanation_fit(r["ppm_label_iso0"], r["ppm_amidated_iso1"], centre, FIT_SIGMAS * sigma)
               for r in sel]
        ratio = [r["d_ppm"] / gap_ppm(r["neutral_mass"]) for r in sel]
        return {"n": len(sel),
                "fit_within_3_sigma": {k: fit.count(k) for k in ("deamidated_only", "13c_only", "both", "neither")},
                "d_over_gap_hist": dict(zip(D_OVER_GAP_LABELS, np.histogram(ratio, bins=D_OVER_GAP_EDGES)[0].tolist())),
                "label_meets_only_iso1": sum(r["label_meets_only_iso1"] for r in sel),
                "label_meets_iso0": sum(abs(r["ppm_label_iso0"]) <= tol for r in sel),
                "amidated_meets_iso1": sum(r["amidated_meets_iso1"] for r in sel),
                "both_within_tol": sum(abs(r["ppm_label_iso0"]) <= tol and r["amidated_meets_iso1"] for r in sel),
                "d_ppm": _quantiles([r["d_ppm"] for r in sel]),
                "median_mass": float(np.median([r["neutral_mass"] for r in sel])) if sel else None}

    summary = {"tol_ppm": tol, "n_spectra": n_spectra, "n_chunks": len(manifest["chunks"]),
               "reference": {"definition": "labels with no modification, meeting the precursor at offset 0",
                             "n_unmodified": n_ref, "n_used": len(ref_ppm), "median_ppm": centre, "mad_ppm": mad,
                             "n_unmodified_meeting_only_iso1": ref_only_iso1},
               "gap_mda": (annotate.CARBON_MASS_DELTA - DEAMIDATION_MASS) * 1e3,
               "fit_rule": f"|ppm - median| <= {FIT_SIGMAS} x {MAD_TO_SIGMA} x MAD = {FIT_SIGMAS * sigma:.2f} ppm",
               "all": block(rows)}
    for k in (1, 2):
        sel = [r for r in rows if (r["n_deamidation"] == 1) == (k == 1)]
        summary["one_deamidation" if k == 1 else "two_or_more"] = block(sel)
    summary["by_mass"] = {}
    for b in dict.fromkeys(mass_bin(r["neutral_mass"]) for r in sorted(rows, key=lambda r: r["neutral_mass"])):
        sel = [r for r in rows if mass_bin(r["neutral_mass"]) == b]
        blk = block(sel)
        blk["gap_ppm_at_median_mass"] = summary["gap_mda"] * 1e-3 / blk["median_mass"] * 1e6
        summary["by_mass"][b] = blk
    return rows, summary


# --- Label-quality audit (Finding 9) ------------------------------------------
# Read-only: the stored processed peaks of a seeded sample of local spectra are
# re-annotated with the pipeline's annotator. One spectrum_utils call per
# peptide (ion types abyIp + internal, max_isotope 1, all candidates); the
# pipeline setting (monoisotopic, no a ions), the isotope-1 setting and the
# a-ion setting are read off it with annotate.select_annotation.

AUDIT_N = 200
AUDIT_N_DECOYS = 20
AUDIT_N_BOOT = 2_000
AUDIT_ION_TYPES = "abyIp"                     # + internal ('m', added by run_spectrum_utils)
CHANCE_FLAG = 0.20                            # concepts above this chance fraction are flagged
FAMILY = {"b": "b", "y": "y", "a": "a", "internal": "internal", "I": "immonium",
          "precursor": "precursor", "noise": "noise"}
CHANCE_FAMILIES = ("b", "y", "a", "internal")

# Which label classes each of the 50 concepts depends on (Finding 9).
_COVER_CLASSES = "b, y, internal, precursor, immonium (residue only)"
_NOT_LABEL = "not label-derived"
_STRUCTURAL = "structural (latent token)"
CONCEPT_LABEL_CLASSES: dict[str, str] = {
    "is_b_ion": "b", "is_y_ion": "y", "is_I_ion": "immonium", "is_internal_fragment": "internal",
    "is_precursor_related": "precursor", "is_matched_peak": "any class",
    "is_noise_peak": "no class (complement of any)", "is_latent_token": _STRUCTURAL,
    "is_H2O_loss": "any class with an H2O loss", "is_NH3_loss": "any class with an NH3 loss",
    "has_neutral_loss": "any class with a loss",
    "is_fragment_charge_1": "any class at charge 1", "is_fragment_charge_2": "any class at charge 2",
    **{c: _NOT_LABEL for c in ("precursor_charge_2", "precursor_charge_3", "precursor_charge_4",
                               "mz_low", "mz_mid", "mz_high", "top_decile_intensity",
                               "peptide_length_short", "peptide_length_medium", "peptide_length_long",
                               "spectrum_contains_oxidation", "spectrum_contains_cam_cys",
                               "spectrum_contains_deamidation")},
    **{c: "b, y" for c in ("position_Nterm", "position_middle", "position_Cterm", "is_first_ion",
                           "is_last_ion", "cleaves_C_to_Lys", "cleaves_C_to_Arg", "cleaves_N_to_Pro",
                           "cleaves_C_to_Asp", "cleaves_C_to_Glu")},
    **{f"covers_{r}": _COVER_CLASSES for r in "KRWFYPMCDEN"},
    **{f"ion_contains_{p}": "b, y, internal, precursor" for p in ("oxidation", "cam_cys", "deamidation")},
}

# --- Concept reliability (V1-3b) ------------------------------------------------
# One record per concept from the label audit's concepts table: how much of the
# concept is chance, whether a labelling bug changes it, and whether it depends
# on the peptide at all. The dictionary carries these records beside every
# association, and the display rule picks which concepts may be shown as a
# feature's "reliable" associations.

PEPTIDE_DEPENDENT = "peptide-dependent"
PEPTIDE_INDEPENDENT = "peptide-independent m/z window"
COMPOSITION_DETERMINED = "composition-determined"
COMPLEMENT = "complement of is_matched_peak"
_DEPENDENCE_OVERRIDES = {
    # spectrum_utils generates immonium ions of all 23 residue letters whatever
    # the peptide, so the label is a fixed set of m/z values (Finding 9).
    "is_I_ion": PEPTIDE_INDEPENDENT,
    # b1 or y1; nearly all positives are y1, whose m/z is fixed by the C-terminal
    # residue (two values for tryptic K/R peptides) (Finding 9).
    "is_first_ion": PEPTIDE_INDEPENDENT,
    # precursor-related ions: m/z fixed by the precursor, which the encoder receives.
    "is_precursor_related": COMPOSITION_DETERMINED,
    "is_noise_peak": COMPLEMENT,
}
CONCEPT_DEPENDENCE: dict[str, str] = {
    c: (_DEPENDENCE_OVERRIDES.get(c)
        or (_NOT_LABEL if cls == _NOT_LABEL else _STRUCTURAL if cls == _STRUCTURAL else PEPTIDE_DEPENDENT))
    for c, cls in CONCEPT_LABEL_CLASSES.items()
}
# Re-documented concepts: the key stays (it is in the annotation vocabulary, the
# Phase 4 CSVs and the description prompts); the tool shows this name instead.
CONCEPT_DISPLAY_NAMES = {
    "is_I_ion": "immonium m/z window (all 23 residue letters, peptide-independent)",
    "is_first_ion": "y1/b1 m/z window (mostly y1 of a C-terminal K or R)",
}
CONCEPT_NOTES = {
    "is_I_ion": "'I' is spectrum_utils' immonium code, not isoleucine; immonium ions are generated for "
                "every residue letter whatever the peptide, so the label marks fixed m/z values (Finding 9)",
    "is_first_ion": "positives are almost all y1, whose m/z is fixed by the C-terminal residue; the chance "
                    "fraction is computed on the small b1 remainder only (Finding 9)",
    "is_precursor_related": "m/z fixed by the precursor m/z, which the encoder receives",
    "is_noise_peak": "complement of is_matched_peak: decoys match fewer peaks, so its chance fraction "
                     "exceeds 1 and is not comparable with the others",
    "ion_contains_deamidation": "part of the local deamidated labels fit a 13C misassignment better than "
                                "deamidation (Finding 8(d))",
    "spectrum_contains_deamidation": "part of the local deamidated labels fit a 13C misassignment better "
                                     "than deamidation (Finding 8(d))",
}
DISPLAY_RULE_MAX_CHANCE = 0.2
DISPLAY_RULE = (
    "display rule, not a statistical test: a concept is shown as a feature's reliable association when its "
    f"chance fraction (point estimate) is <= {DISPLAY_RULE_MAX_CHANCE}, no labelling bug adds any of its "
    "positives in the audit sample (positives a bug hides are reported, not counted), and it depends on the "
    "peptide. Why 0.2: at that level at most about one "
    "in five of the concept's positives would be expected from a random same-composition sequence; it is the "
    "cut Finding 9 used, and it falls in the gap between the b/y position and cleavage concepts (0.09-0.18) "
    "and internal fragments, neutral losses and most covers_* concepts (0.23-0.64). Concepts close to it can "
    "fall on either side by chance, so the chance fraction and its CI are always shown."
)
RELIABILITY_COLUMNS = [
    "concept", "display_name", "label_classes", "dependence", "positives", "determined_fraction",
    "chance_fraction", "ci_low", "ci_high", "internal_end_positives", "immonium_bug_positives",
    "immonium_bug_missed", "bug_share", "meets_display_rule", "reasons", "note",
]


def concept_reliability(concepts) -> list[dict]:
    """Reliability records (RELIABILITY_COLUMNS) from the label audit's
    concepts table (rows of concepts.csv, as dicts or a DataFrame). Requires
    exactly the concepts of CONCEPT_LABEL_CLASSES, in any order, and returns
    them in the table's order.

    Bug counts: internal_end_positives are positives due only to the
    internal-fragment end residue; immonium_bug_positives are positives that an
    immonium ion of an absent residue adds. Both are wrong labels, and
    bug_share is their sum over the concept's positives; the display rule
    requires it to be 0. immonium_bug_missed counts positives the immonium bug
    hides (the next candidate's concepts, e.g. a y ion with a loss outranked by
    an immonium ion without one): the concept's labelled positives are still
    right, so it is reported in the note, not counted against the rule. NaN
    chance fractions (not label-derived, structural, no positives outside
    composition-determined ions) stay NaN."""
    rows = concepts.to_dict("records") if hasattr(concepts, "to_dict") else list(concepts)
    names = [r["concept"] for r in rows]
    if len(set(names)) != len(names) or set(names) != set(CONCEPT_LABEL_CLASSES):
        raise ValueError(f"concept table does not match the 50 registry concepts: missing "
                         f"{sorted(set(CONCEPT_LABEL_CLASSES) - set(names))}, extra "
                         f"{sorted(set(names) - set(CONCEPT_LABEL_CLASSES))}, "
                         f"{len(names) - len(set(names))} duplicated")
    out = []
    for r in rows:
        c = r["concept"]
        positives = float(r["positives"])
        chance = float(r["chance_fraction"])
        int_end, imm_add, imm_miss = (float(r[k]) for k in
                                      ("internal_end_only", "immonium_fix_lost", "immonium_fix_gained"))
        bug = int_end + imm_add
        share = bug / positives if positives else math.nan
        dependence = CONCEPT_DEPENDENCE[c]
        label_derived = dependence not in (_NOT_LABEL, _STRUCTURAL)
        det = float(r["from_determined"]) / positives if positives and label_derived else math.nan
        reasons = []
        if dependence != PEPTIDE_DEPENDENT:
            reasons.append(dependence)
        elif not math.isfinite(chance):
            reasons.append("no chance fraction")
        elif chance > DISPLAY_RULE_MAX_CHANCE:
            reasons.append(f"chance fraction {chance:.2f} > {DISPLAY_RULE_MAX_CHANCE}")
        if bug:
            reasons.append(f"labelling bugs add {share:.1%} of positives")
        notes = [CONCEPT_NOTES[c]] if c in CONCEPT_NOTES else []
        if dependence == PEPTIDE_DEPENDENT and det > 0:
            notes.append(f"{det:.0%} of positives are on composition-determined ions (immonium, precursor, "
                         f"y1, b_(L-1)), whose m/z does not depend on the residue order")
        if imm_miss:
            notes.append(f"the immonium bug hides {int(imm_miss)} positive(s) in the audit sample")
        note = "; ".join(notes)
        out.append({
            "concept": c, "display_name": CONCEPT_DISPLAY_NAMES.get(c, c),
            "label_classes": CONCEPT_LABEL_CLASSES[c], "dependence": dependence,
            "positives": int(positives), "determined_fraction": det,
            "chance_fraction": chance, "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
            "internal_end_positives": int(int_end), "immonium_bug_positives": int(imm_add),
            "immonium_bug_missed": int(imm_miss), "bug_share": share,
            "meets_display_rule": not reasons, "reasons": "; ".join(reasons), "note": note,
        })
    return out


def allocate_sample(n_chunks: int, n: int, rng: np.random.Generator) -> list[int]:
    """Spectra per chunk: n // n_chunks each, plus one in n % n_chunks chunks
    drawn without replacement."""
    k = [n // n_chunks] * n_chunks
    for i in rng.choice(n_chunks, n % n_chunks, replace=False):
        k[int(i)] += 1
    return k


def _family(ann: dict) -> str:
    return FAMILY.get(ann["ion_type"], ann["ion_type"])


def _mz_bin(mz: float) -> str:
    import annotate
    low, high = annotate.MZ_LOW_HIGH_CUT
    return "mz_low" if mz < low else ("mz_mid" if mz < high else "mz_high")


def _ion_key(ann: dict) -> tuple:
    return ann["ion_type"], ann["ion_index"], ann["charge"], ann["neutral_loss"]


def _class_keys(fam: str, loss: str, charge: int) -> list[str]:
    loss = loss or "none"
    keys = [f"{fam}|{loss}|{charge}", f"{fam}|{loss}|*", f"{fam}|*|{charge}", f"{fam}|*|*"]
    if fam in ("b", "y", "internal"):
        keys.append("b+y+internal|*|*")
    return keys


def _determined_reason(ann: dict) -> str:
    t, k = ann["ion_type"], ann["ion_index"]
    if t in ("I", "precursor"):
        return FAMILY[t]
    return "y1" if (t == "y" and k == 1) else f"{t}_L-1"


def _lost_reason(best: dict, winner: dict) -> str:
    """Why a b (or y) candidate lost to the winner, by _candidate_sort_key's order."""
    if bool(best["neutral_loss"]) and not winner["neutral_loss"]:
        return "loss"
    if bool(best["neutral_loss"]) == bool(winner["neutral_loss"]) and winner["ion_type"] in ("b", "y"):
        return "charge" if best["charge"] > winner["charge"] else "ppm"
    return "other"


def peptide_label_metrics(cands: list[list[dict]], mz, intensity, peptide: str, mods: list[dict],
                          precursor_charge: int, registry) -> dict[str, float]:
    """Counts for one (spectrum, peptide) annotation, from per-peak candidate
    lists (run_spectrum_utils(return_candidates=True)) of a run with a ions and
    max_isotope 1. Keys:
      cls|family|loss|charge  pipeline-setting winners (a: a-ion setting),
                              composition-determined winners excluded
      det|reason              composition-determined winners
      iso|noise|bin, iso|became|bin, iso|became_m0|bin, iso|became_fam|family
                              pipeline-noise peaks that become isotope-1
                              matches (m0: the same ion's monoisotopic peak is
                              labelled in the spectrum)
      prec|...                classes co-matching a peak, and why b/y lost
      cov|series|charge|losses|indices|num/den   series coverage
      aion|den|family, aion|from|family[|a2/other]  peaks that become a ions
      con|cur/det/nodet|c     concept positives (pipeline setting; det = on a
                              composition-determined winner)
      con|diso|c, con|da|c    change under isotope-1 / a ions
      con|intextra|c          positives due only to the internal end residue
      con|imm_absent|c, imm|peaks, imm|absent   immonium of a residue the
                              peptide lacks (I, L and J count as one)
      con|immfix_lost|c, con|immfix_gained|c   concept positives lost / gained
                              on those peaks when such immonium candidates
                              are dropped (the next candidate wins)."""
    import annotate
    import torch
    out: dict[str, float] = {}

    def add(key, v=1):
        out[key] = out.get(key, 0) + v

    L = len(peptide)
    zmax = max(1, precursor_charge - 1)
    thr = annotate.compute_top_decile_threshold(torch.as_tensor(intensity))
    spec_c = annotate.compute_spectrum_concepts(mods, precursor_charge, peptide)

    def concepts(ann, j):
        c = annotate.compute_peak_concepts(ann, float(mz[j]), float(intensity[j]), thr, peptide, mods, L, registry)
        annotate.merge_spectrum_concepts(c, spec_c)
        return {k for k, v in c.items() if v}

    cur = [annotate.select_annotation(c, 0, ("a",)) for c in cands]
    iso = [annotate.select_annotation(c, 1, ("a",)) for c in cands]
    pla = [annotate.select_annotation(c, 0, ()) for c in cands]
    m0_keys = {_ion_key(a) for a in cur if a["ion_type"] != "noise"}
    covered = set()
    present = set(peptide) | ({"I", "L", "J"} if set(peptide) & {"I", "L"} else set())
    for j, (c, a, ai, aa) in enumerate(zip(cands, cur, iso, pla, strict=True)):
        fam = _family(a)
        det = fam != "noise" and annotate.is_composition_determined(a, L)
        # (1) classes
        if fam != "noise":
            if det:
                add(f"det|{_determined_reason(a)}")
            elif fam in CHANCE_FAMILIES:
                for k in _class_keys(fam, a["neutral_loss"], a["charge"]):
                    add("cls|" + k)
        if _family(aa) == "a" and not annotate.is_composition_determined(aa, L):
            for k in _class_keys("a", aa["neutral_loss"], aa["charge"]):
                add("cls|" + k)
        # (2) isotope peaks among pipeline noise
        if fam == "noise":
            b = _mz_bin(float(mz[j]))
            for tag in (b, "all"):
                add(f"iso|noise|{tag}")
            if ai["ion_type"] != "noise":
                for tag in (b, "all"):
                    add(f"iso|became|{tag}")
                    if _ion_key(ai) in m0_keys:
                        add(f"iso|became_m0|{tag}")
                add(f"iso|became_fam|{_family(ai)}")
        # (3) precedence among pipeline-setting candidates
        cc = [x for x in c if x["isotope"] == 0 and x["ion_type"] != "a"]
        if cc:
            fams = {_family(x) for x in cc}
            add("prec|matched")
            if len(fams) > 1:
                add("prec|multi")
            for f in fams:
                add(f"prec|co|{f}|{fam}")
            for s in ("b", "y"):
                sc = [x for x in cc if x["ion_type"] == s]
                if sc:
                    add(f"prec|{s}cand")
                    if fam != s:
                        add(f"prec|{s}cand_win|{fam}")
                        add(f"prec|{s}cand_reason|{_lost_reason(sc[0], a)}")
        # (4) series coverage, a ions
        if a["ion_type"] in ("b", "y"):
            covered.add(_ion_key(a))
        add(f"aion|den|{fam}")
        if _family(aa) == "a" and fam != "a":
            add(f"aion|from|{fam}")
            add(f"aion|from|{fam}|{'a2' if aa['ion_index'] == 2 else 'other'}")
        # (5) concepts
        base = concepts(a, j)
        for k in base:
            add(f"con|cur|{k}")
            add(f"con|{'det' if det else 'nodet'}|{k}")
        for tag, alt in (("diso", ai), ("da", aa)):
            if _ion_key(alt) != _ion_key(a):
                new = concepts(alt, j)
                for k in new - base:
                    add(f"con|{tag}|{k}")
                for k in base - new:
                    add(f"con|{tag}|{k}", -1)
        if fam == "internal":
            s0, e0 = a["ion_index"]
            fixed = concepts(dict(a, ion_index=(s0, e0 - 1)), j)   # spectrum_utils' end is exclusive
            for k in base - fixed:
                add(f"con|intextra|{k}")
        if fam == "immonium":
            add("imm|peaks")
            if a["ion_index"] not in present:
                add("imm|absent")
                for k in base:
                    if k.startswith("covers_") and k[-1] not in peptide:
                        add(f"con|imm_absent|{k}")
                # The label under an immonium filter (V1-3b): the next candidate
                # once immonium ions of absent residues are dropped.
                rest = [x for x in c if not (x["ion_type"] == "I" and x["ion_index"] not in present)]
                fixed = concepts(annotate.select_annotation(rest, 0, ("a",)), j)
                for k in base - fixed:
                    add(f"con|immfix_lost|{k}")
                for k in fixed - base:
                    add(f"con|immfix_gained|{k}")
    for s in ("b", "y"):
        idx_all = set(range(1, L))
        idx_ch = {k for k in idx_all if not annotate.is_composition_determined({"ion_type": s, "ion_index": k}, L)}
        for z in [*range(1, zmax + 1), "*"]:
            for mode in ("none", "any"):
                ks = {k for (t, k, ch, nl) in covered
                      if t == s and (z == "*" or ch == z) and (mode == "any" or not nl)}
                for name, idx in (("all", idx_all), ("chance", idx_ch)):
                    add(f"cov|{s}|{z}|{mode}|{name}|num", len(ks & idx))
                    add(f"cov|{s}|{z}|{mode}|{name}|den", len(idx))
    return out


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _safe_div(a, b):
    """a / b elementwise, NaN where b == 0 (a resample with no denominator)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b != 0)


def ratio_ci(num, den, idx: np.ndarray) -> tuple[float, float, float]:
    """sum(num) / sum(den) with a percentile bootstrap 95% CI over the resamples
    `idx` (rows of indices into the spectra)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    est = num.sum() / den.sum() if den.sum() else math.nan
    boot = _safe_div(num[idx].sum(1), den[idx].sum(1))
    if not np.isfinite(boot).any():
        return float(est), math.nan, math.nan
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def ratio_diff_ci(n1, d1, n2, d2, idx: np.ndarray) -> tuple[float, float, float]:
    """sum(n1)/sum(d1) - sum(n2)/sum(d2), paired bootstrap 95% CI over spectra."""
    n1, d1, n2, d2 = (np.asarray(x, float) for x in (n1, d1, n2, d2))
    est = float(_safe_div(n1.sum(), d1.sum()) - _safe_div(n2.sum(), d2.sum()))
    boot = _safe_div(n1[idx].sum(1), d1[idx].sum(1)) - _safe_div(n2[idx].sum(1), d2[idx].sum(1))
    if not np.isfinite(boot).any():
        return float(est), math.nan, math.nan
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def label_audit(root: Path, n: int = AUDIT_N, seed: int = DEFAULT_SEED, n_decoys: int = AUDIT_N_DECOYS,
                n_boot: int = AUDIT_N_BOOT) -> dict:
    """Run the audit and write its tables under ``root/analysis/label_audit/``.
    Stops after writing reproduction.json if re-annotating the true peptides
    does not reproduce the stored labels exactly."""
    import time
    import torch
    import annotate
    t_start = time.perf_counter()
    out_dir = root / "analysis" / "label_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "extract" / "manifest.json").read_text())
    registry = annotate.ConceptRegistry.build()
    config = annotate.AnnotationConfig(extract_dir=Path(), output_dir=Path())
    rng = np.random.default_rng(seed)
    per_chunk = allocate_sample(len(manifest["chunks"]), n, rng)
    timings = {"reproduction": 0.0, "annotate": 0.0, "metrics": 0.0}
    sample, true_m, decoy_m, mismatches = [], [], [], []

    for ch, k in zip(manifest["chunks"], per_chunk, strict=True):
        meta = torch.load(root / "extract" / ch["meta"], map_location="cpu", weights_only=False)
        labels = torch.load(root / "annotation" / "labels" / f"chunk_{ch['idx']:05d}.pt",
                            map_location="cpu", weights_only=False)
        if labels["chunk_idx"] != ch["idx"] or labels["total_tokens"] != meta["total_tokens"]:
            raise RuntimeError(f"chunk {ch['idx']}: labels do not align with the extract metadata")
        for s in sorted(rng.choice(meta["n_spectra"], k, replace=False).tolist()):
            pep, prof, mods = meta["peptides"][s], meta["proforma_strings"][s], meta["modifications"][s]
            mz, inten = meta["mz_arrays"][s], meta["intensity_arrays"][s]
            pmz, z = float(meta["precursor_mzs"][s]), int(meta["precursor_charges"][s])
            sid = int(meta["spectrum_ids"][s])
            # (0) reproduction through the pipeline's own path
            t0 = time.perf_counter()
            anns, tok = annotate.annotate_spectrum(pep, prof, mods, mz, inten, pmz, z, config, registry,
                                                   strict=True)
            rows = (meta["token_to_spectrum"] == s).nonzero().flatten()
            fields = {"token_labels": tok.labels, "ion_type_ids": tok.ion_type, "ion_indices": tok.ion_index,
                      "fragment_charges": tok.fragment_charge, "neutral_loss_ids": tok.neutral_loss_id}
            bad = [f for f, v in fields.items() if not torch.equal(v, labels[f][rows])]
            if bad:
                mismatches.append({"chunk": ch["idx"], "spectrum": s, "spectrum_id": sid, "fields": bad})
            timings["reproduction"] += time.perf_counter() - t0

            def candidates(proforma):
                return annotate.run_spectrum_utils(
                    proforma=proforma, mz_array=mz, intensity_array=inten, precursor_mz=pmz,
                    precursor_charge=z, ion_types=AUDIT_ION_TYPES, fragment_tol_mass=config.fragment_tol_mass,
                    fragment_tol_mode=config.fragment_tol_mode, enable_internal=config.enable_internal,
                    strict=True, max_isotope=1, return_candidates=True)

            t0 = time.perf_counter()
            cands = candidates(prof)
            derived = [annotate.select_annotation(c, 0, ("a",)) for c in cands]
            if any({kk: d[kk] for kk in annotate._LABEL_KEYS} != a for d, a in zip(derived, anns, strict=True)):
                mismatches.append({"chunk": ch["idx"], "spectrum": s, "spectrum_id": sid,
                                   "fields": ["derived pipeline-setting winners"]})
            tokens = annotate.proforma_residue_tokens(prof)
            if not tokens[0][0].isalpha():
                raise RuntimeError(f"spectrum {sid}: N-terminal modification (Finding 6) in {prof!r}")
            try:
                decoys = annotate.peptide_decoys(tokens, n_decoys, np.random.default_rng([seed, sid]),
                                                 exclude=[tokens])
            except RuntimeError:
                decoys = []
            decoy_inputs = [annotate.annotation_inputs_from_tokens(d, mods) for d in decoys]
            decoy_cands = [candidates(dp) for _db, dp, _dm in decoy_inputs]
            timings["annotate"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            true_m.append(peptide_label_metrics(cands, mz, inten, pep, mods, z, registry))
            dm: dict[str, float] = {}
            for (db, _dp, dmods), dc in zip(decoy_inputs, decoy_cands, strict=True):
                for kk, v in peptide_label_metrics(dc, mz, inten, db, dmods, z, registry).items():
                    dm[kk] = dm.get(kk, 0) + v / len(decoys)
            decoy_m.append(dm if decoys else None)
            timings["metrics"] += time.perf_counter() - t0
            sample.append({"chunk": ch["idx"], "spectrum": s, "spectrum_id": sid, "peptide": pep,
                           "proforma": prof, "length": len(pep), "precursor_charge": z, "n_peaks": len(mz),
                           "n_decoys": len(decoys)})
        del meta, labels

    settings = {
        "fragment_tol": f"{config.fragment_tol_mass:g} {config.fragment_tol_mode}",
        "pipeline_ion_types": config.ion_types + ("m" if config.enable_internal else ""),
        "neutral_losses": annotate.NEUTRAL_LOSSES, "max_isotope": 0,
        "fragment_charges": "1..max(1, z-1) (spectrum_utils default max_ion_charge; never passed)",
        "precedence": "monoisotopic > no loss > b/y > a/c/x/z > immonium > internal > precursor > lower charge "
                      "> smaller |ppm| (annotate._candidate_sort_key)",
        "audit_run": f"ion types {AUDIT_ION_TYPES}+m, max_isotope 1, all candidates; settings derived with "
                     "annotate.select_annotation",
        "decoys": f"{n_decoys} per spectrum, annotate.peptide_decoys of the true peptide (C-terminal residue "
                  "fixed, rest shuffled with modifications), default_rng([seed, spectrum_id])",
        "excluded_from_chance": "immonium, precursor, y1, b_{L-1}, a_{L-1} (any loss/isotope): the same for "
                                "every same-composition decoy with the C-terminal residue fixed",
        "bootstrap": f"{n_boot} resamples of spectra, percentile 95% CI, default_rng({seed}); the decoy mean "
                     "per spectrum is the unit",
        "seed": seed, "n": n, "per_chunk": dict(zip([c["idx"] for c in manifest["chunks"]], per_chunk)),
    }
    repro = {"ok": not mismatches, "n_spectra": len(sample), "mismatches": mismatches}
    (out_dir / "reproduction.json").write_text(json.dumps(repro, indent=1))
    _write_csv(out_dir / "sample.csv", list(sample[0]), sample)
    if mismatches:
        summary = {"settings": settings, "reproduction": repro, "stopped": "labels not reproduced",
                   "timings_s": timings}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary

    t0 = time.perf_counter()
    keep = [i for i, d in enumerate(decoy_m) if d is not None]
    idx_dec = bootstrap_indices(len(keep), n_boot, seed)

    def arr(key, which="true", subset=None):
        src = true_m if which == "true" else decoy_m
        ids = range(len(true_m)) if subset is None else subset
        return np.array([(src[i] or {}).get(key, 0.0) for i in ids], float)

    def T(key):                                   # true values, spectra with decoys
        return arr(key, "true", keep)

    def D(key):                                   # decoy means, spectra with decoys
        return arr(key, "decoy", keep)

    keys = set().union(*true_m)
    # (1) per class
    cls_rows = []
    for key in sorted(k[4:] for k in keys if k.startswith("cls|")):
        fam, loss, charge = key.split("|")
        t, d = T("cls|" + key), D("cls|" + key)
        cf, lo, hi = ratio_ci(d, t, idx_dec)
        cls_rows.append({"family": fam, "loss": loss, "charge": charge,
                         "setting": "a ions on" if fam == "a" else "pipeline", "n_spectra": len(keep),
                         "true_mean": t.mean(), "decoy_mean": d.mean(), "chance_fraction": cf,
                         "ci_low": lo, "ci_high": hi, "excess_mean": (t - d).mean()})
    det_rows = [{"reason": k[4:], "true_mean": arr(k).mean(), "true_total": arr(k).sum()}
                for k in sorted(k for k in keys if k.startswith("det|"))]
    # (2) isotope
    iso_rows = []
    for tag in ("all", "mz_low", "mz_mid", "mz_high"):
        nt, nd = T(f"iso|noise|{tag}"), D(f"iso|noise|{tag}")
        row = {"mz_bin": tag, "n_noise_true": nt.sum()}
        for kind in ("became", "became_m0"):
            bt, bd = T(f"iso|{kind}|{tag}"), D(f"iso|{kind}|{tag}")
            diff, lo, hi = ratio_diff_ci(bt, nt, bd, nd, idx_dec)
            row |= {f"{kind}_true": bt.sum() / nt.sum(), f"{kind}_decoy": bd.sum() / nd.sum(),
                    f"{kind}_excess": diff, f"{kind}_excess_ci_low": lo, f"{kind}_excess_ci_high": hi}
        iso_rows.append(row)
    iso_fam = {k.split("|")[2]: (T(k).sum(), D(k).sum()) for k in keys if k.startswith("iso|became_fam|")}
    # (3) precedence
    prec_rows = [{"key": k[5:], "true_total": arr(k).sum(), "true_total_decoy_spectra": T(k).sum(),
                  "decoy_mean_total": D(k).sum()}
                 for k in sorted(k for k in keys if k.startswith("prec|"))]
    # (4) coverage and a ions
    cov_rows = []
    for k in sorted(k for k in keys if k.startswith("cov|") and k.endswith("|num")):
        _c, s, zc, mode, name, _n = k.split("|")
        base = k[:-4]
        tn, td, dn = T(base + "|num"), T(base + "|den"), D(base + "|num")
        ct, clo, chi = ratio_ci(tn, td, idx_dec)
        cf, flo, fhi = ratio_ci(dn, tn, idx_dec)
        cov_rows.append({"series": s, "charge": zc, "losses": mode, "indices": name,
                         "n_spectra": int((td > 0).sum()), "coverage_true": ct, "ci_low": clo, "ci_high": chi,
                         "coverage_decoy": dn.sum() / td.sum() if td.sum() else math.nan,
                         "chance_fraction": cf, "cf_ci_low": flo, "cf_ci_high": fhi})
    a_rows = []
    for src in ("noise", "internal", "immonium", "precursor", "b", "y"):
        nt, nd = T(f"aion|den|{src}"), D(f"aion|den|{src}")
        if nt.sum() == 0:
            continue
        for sub in ("", "|a2", "|other"):
            bt, bd = T(f"aion|from|{src}{sub}"), D(f"aion|from|{src}{sub}")
            diff, lo, hi = ratio_diff_ci(bt, nt, bd, nd, idx_dec)
            a_rows.append({"from": src, "a_index": sub.strip("|") or "any", "n_from_true": nt.sum(),
                           "became_a_true": bt.sum(), "frac_true": bt.sum() / nt.sum(),
                           "frac_decoy": bd.sum() / nd.sum() if nd.sum() else math.nan,
                           "excess": diff, "ci_low": lo, "ci_high": hi})
    # (5) concepts
    con_rows = []
    for c in registry.names:
        nodet_t, nodet_d = T(f"con|nodet|{c}"), D(f"con|nodet|{c}")
        label_derived = CONCEPT_LABEL_CLASSES[c] not in (_NOT_LABEL, _STRUCTURAL)
        if label_derived and nodet_t.sum():
            cf, lo, hi = ratio_ci(nodet_d, nodet_t, idx_dec)
        else:
            cf = lo = hi = math.nan
        con_rows.append({
            "concept": c, "label_classes": CONCEPT_LABEL_CLASSES[c], "positives": arr(f"con|cur|{c}").sum(),
            "from_determined": arr(f"con|det|{c}").sum(),
            "d_isotope_true": arr(f"con|diso|{c}").sum(), "d_isotope_decoy_mean": D(f"con|diso|{c}").sum(),
            "d_a_ions_true": arr(f"con|da|{c}").sum(), "d_a_ions_decoy_mean": D(f"con|da|{c}").sum(),
            "chance_fraction": cf, "ci_low": lo, "ci_high": hi,
            "flag_chance_over_20pct": bool(label_derived and cf > CHANCE_FLAG),
            "internal_end_only": arr(f"con|intextra|{c}").sum(),
            "immonium_absent_residue": arr(f"con|imm_absent|{c}").sum(),
            "immonium_fix_lost": arr(f"con|immfix_lost|{c}").sum(),
            "immonium_fix_gained": arr(f"con|immfix_gained|{c}").sum()})
    timings["tables"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_start

    for name, rows_ in (("per_class", cls_rows), ("determined", det_rows), ("isotope", iso_rows),
                        ("precedence", prec_rows), ("coverage", cov_rows), ("a_ions", a_rows),
                        ("concepts", con_rows)):
        _write_csv(out_dir / f"{name}.csv", list(rows_[0]), rows_)
    reliability = concept_reliability(con_rows)
    _write_csv(out_dir / "concept_reliability.csv", RELIABILITY_COLUMNS, reliability)
    summary = {
        "settings": settings, "reproduction": repro, "timings_s": timings,
        "n_spectra": len(true_m), "n_with_decoys": len(keep),
        "n_peaks": int(sum(r["n_peaks"] for r in sample)),
        "pipeline_label_counts": {f: arr(f"aion|den|{f}").sum() for f in FAMILY.values() if f != "a"},
        "immonium": {"peaks": arr("imm|peaks").sum(), "absent_residue": arr("imm|absent").sum()},
        "isotope_became_by_family": {f: {"true": v[0], "decoy_mean": v[1]} for f, v in sorted(iso_fam.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


# --- Reliable-association list (V1-3b, Finding 10) --------------------------------

RELIABLE_LIST_MIN_F1DOM = 0.1
RELIABLE_LIST_CUT = (
    "listing cut F1dom >= 0.1 on the best reliable association, chosen by the user (V1-3b) to collect the "
    "few features with a non-negligible association to a concept meeting the display rule, for later use "
    "(demo set, feature cards); not a reliability claim -- the F1dom, precision, recall and chance fraction "
    "are listed")
FINDING_10_LEVELS = (0.1, 0.3)
RELIABLE_LIST_COLUMNS = [
    "layer", "feature_idx", "firing_rate", "n_reliable_associations",
    *[f"reliable_{name}_{k}" for k in (1, 2, 3)
      for name in ("concept", "f1_dom", "precision", "recall", "chance_fraction")],
    "concept_1", "f1_dom_1", "chance_fraction_1", "phase8_delta_ce", "causal_provenance",
    "description_blinded", "description_r", "description_beats_baseline", "description",
]


def _is_number(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def reliable_association_list(rows_by_layer: dict[int, list[dict]],
                              min_f1dom: float = RELIABLE_LIST_MIN_F1DOM) -> tuple[list[dict], dict]:
    """Rows (RELIABLE_LIST_COLUMNS) of the features whose reliable_f1_dom_1 is
    >= min_f1dom, by layer and then F1dom descending (ties by feature), and a
    per-layer summary over all live features. ``rows_by_layer`` holds each
    layer's dictionary rows as dicts."""
    out: list[dict] = []
    summary = {"min_reliable_f1_dom": min_f1dom, "cut": RELIABLE_LIST_CUT, "by_layer": {}}
    for layer in sorted(rows_by_layer):
        rows = rows_by_layer[layer]
        best = [float(r["reliable_f1_dom_1"]) for r in rows if _is_number(r.get("reliable_f1_dom_1"))]
        listed = sorted((r for r in rows if _is_number(r.get("reliable_f1_dom_1"))
                         and r["reliable_f1_dom_1"] >= min_f1dom),
                        key=lambda r: (-r["reliable_f1_dom_1"], r["feature_idx"]))
        summary["by_layer"][str(layer)] = {
            "n_live_features": len(rows), "n_with_reliable_association": len(best),
            "median_best_reliable_f1_dom": float(np.median(best)) if best else math.nan,
            **{f"n_best_reliable_f1_dom_ge_{level:g}": sum(b >= level for b in best)
               for level in FINDING_10_LEVELS},
            "n_listed": len(listed),
        }
        out += [{c: (layer if c == "layer" else r.get(c)) for c in RELIABLE_LIST_COLUMNS} for r in listed]
    return out, summary


def load_dictionary_rows(root: Path, layer: int) -> list[dict]:
    """One layer's dictionary rows (``dictionary/layer_{L}.parquet``), schema-checked."""
    import pyarrow.parquet as pq
    path = root / "dictionary" / f"layer_{layer}.parquet"
    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    if meta.get("schema_version") != str(DICTIONARY_SCHEMA_VERSION) or meta.get("layer") != str(layer):
        raise ValueError(f"{path}: dictionary schema {meta.get('schema_version')!r} / layer "
                         f"{meta.get('layer')!r}, expected {DICTIONARY_SCHEMA_VERSION} / {layer}; rebuild it")
    return table.to_pylist()


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# --- V1-5: demo set (--demo-set) ---------------------------------------------------------

HF_TEST_OFFSET = 527_974            # merged row of HF test row 0 (train 499,402 + validation 28,572)
CHUNK_SIZE = 1024
LOCAL_TEST_CHUNKS = (520, 572)
DEMO_GROUP_A = (
    {"id": "H7", "row": 43804,
     "label": "InstaNovo error resolvable from the spectrum: b/y peaks support the true peptide",
     "reason": "V1-4c H7, the one b/y-resolvable error: mixed-sign features on the true-only y peaks, a set "
               "effect dominated by one item, joint vs single differences"},
    {"id": "H1", "row": 19475,
     "label": "notation-equivalent to the label (N[UNIMOD:7] vs D, same molecule); not a model error",
     "reason": "V1-4c H1: a large single effect that the controls showed belongs to the peak, not to the "
               "named features"},
    {"id": "H5", "row": 60182,
     "label": "notation choice: D vs N[UNIMOD:7], same molecule; InstaNovo's metric counts it correct",
     "reason": "V1-4c H5: a clear measured effect on a notation choice, not a model error"},
)
DEMO_GROUP_B_REGISTERED = (5398, 5406, 58409)
DEMO_GROUP_C_REGISTERED = {2: 7522, 4: 11376, 6: 9282, 8: 5107}
DEMO_RULE = (
    "D1 (STATUS V1-5): group A = V1-4c H7, H1, H5 with labels; group B = V1-4a seed-0 rows (v1_4c/"
    "predictions.csv) that are exact, meet the precursor and lie in chunks 520/572, top 3 by beam log p; "
    "group C = per layer the feature in reliable_associations.csv with the highest best reliable F1dom, "
    "and the chunk 520/572 spectrum where its peak-token activation is highest; if that spectrum is "
    "already chosen or its beam does not complete, the next one.")


def hf_test_location(row: int) -> tuple[int, int]:
    """(chunk, index within the chunk) of an HF test row in the merged corpus."""
    merged = HF_TEST_OFFSET + int(row)
    return merged // CHUNK_SIZE, merged % CHUNK_SIZE


def _truthy(v) -> bool:
    return str(v).strip().lower() == "true"


def demo_group_b(predictions: list[dict], n: int = 3, chunks=LOCAL_TEST_CHUNKS) -> list[int]:
    """Exact, precursor-meeting rows in the given chunks, top n by beam log p (ties by row)."""
    ok = [p for p in predictions if _truthy(p["exact"]) and _truthy(p["meets_precursor"])
          and hf_test_location(int(p["row"]))[0] in chunks]
    ok.sort(key=lambda p: (-float(p["beam_log_prob"]), int(p["row"])))
    return [int(p["row"]) for p in ok[:n]]


def demo_group_c_feature(rows: list[dict], layer: int) -> tuple[int, float]:
    """The layer's feature with the highest best reliable F1dom (ties by feature id)."""
    cand = [(float(r["reliable_f1_dom_1"]), -int(r["feature_idx"])) for r in rows
            if int(r["layer"]) == layer and _is_number(r.get("reliable_f1_dom_1"))]
    if not cand:
        raise ValueError(f"no reliable association for layer {layer}")
    f1, neg = max(cand)
    return -neg, f1


def rank_spectra_by_peak_max(feature_col, token_to_spectrum, token_to_position, spectrum_ids) -> list[tuple[int, float]]:
    """(merged spectrum id, max activation over its peak tokens), strongest first,
    ties by spectrum id; spectra where the feature never fires on a peak are left out."""
    col = np.asarray(feature_col, dtype=np.float64)
    t2s, pos = np.asarray(token_to_spectrum), np.asarray(token_to_position)
    peak = pos >= 1
    best = np.full(len(spectrum_ids), -np.inf)
    np.maximum.at(best, t2s[peak], col[peak])
    out = [(int(spectrum_ids[s]), float(best[s])) for s in range(len(spectrum_ids)) if best[s] > 0]
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def pick_demo_spectrum(ranked: list[tuple[int, float]], taken: set[int], beam_completes) -> tuple[dict, list[dict]]:
    """The first ranked HF test row not taken whose beam completes; the rows
    passed over, with the reason."""
    passed = []
    for rank, (merged, act) in enumerate(ranked, 1):
        row = merged - HF_TEST_OFFSET
        if row in taken:
            passed.append({"row": row, "rank": rank, "activation": act, "reason": "already chosen"})
            continue
        if not beam_completes(row):
            passed.append({"row": row, "rank": rank, "activation": act, "reason": "beam did not complete"})
            continue
        return {"row": row, "rank": rank, "activation": act}, passed
    raise RuntimeError("no eligible spectrum")


def read_parquet_rows(path: Path, rows: list[int], columns: list[str]) -> dict[int, dict]:
    """The given rows of a parquet, reading each row group once."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    out = {}
    for g in range(pf.metadata.num_row_groups):
        members = [r for r in rows if starts[g] <= r < starts[g + 1]]
        if members:
            table = pf.read_row_group(g, columns=columns)
            for r in members:
                out[r] = table.slice(int(r - starts[g]), 1).to_pylist()[0]
    missing = set(rows) - set(out)
    if missing:
        raise IndexError(f"rows {sorted(missing)[:5]} outside {path}")
    return out


class _BeamModel:
    """InstaNovo through instanovo_io, decoding as explain() does (5 beams, the
    config's max_length, spectrum_utils preprocessing)."""

    def __init__(self, checkpoint: Path):
        import instanovo_io
        self.io = instanovo_io
        self.model, self.config, self.residue_set = instanovo_io.load_instanovo(checkpoint, device="cpu", by_id=False)
        self.processor = instanovo_io.make_processor(self.residue_set, annotated=False)

    def predict(self, rec: dict) -> list[str]:
        batch = self.io.preprocess_spectrum(self.processor, rec["mz_array"], rec["intensity_array"],
                                            rec["precursor_mz"], rec["precursor_charge"])
        out = self.io.beam_search(self.model, batch, "cpu", 5, int(self.config["max_length"]))
        return list(out["predictions"][0])

    def instanovo_notation(self, modified_sequence: str) -> str:
        rs = self.residue_set
        return "".join(rs.residue_remapping.get(t, t) for t in rs.tokenize(modified_sequence))


def demo_set(root: Path, parquet: Path, checkpoint: Path, log=print) -> dict:
    """Apply D1 and write analysis/v1_5/demo_set.json. Stops if a derived pick
    differs from the registered one (a deviation is logged, never absorbed)."""
    import time
    import torch
    from evaluate import ChunkStream
    from train import load_sae_from_checkpoint

    t_start = time.perf_counter()
    with open(root / "analysis" / "v1_4c" / "predictions.csv", newline="", encoding="utf-8") as fh:
        predictions = {int(r["row"]): r for r in csv.DictReader(fh)}
    with open(root / "analysis" / "reliable_associations.csv", newline="", encoding="utf-8") as fh:
        reliable = list(csv.DictReader(fh))
    group_b = demo_group_b(list(predictions.values()))
    if tuple(group_b) != DEMO_GROUP_B_REGISTERED:
        raise RuntimeError(f"group B derives {group_b}, registered {DEMO_GROUP_B_REGISTERED}: deviation")
    model = _BeamModel(checkpoint)
    hf_cols = ["mz_array", "intensity_array", "precursor_mz", "precursor_charge", "sequence", "modified_sequence"]
    beam_cache: dict[int, list[str]] = {}

    def beam(row: int) -> list[str]:
        if row not in beam_cache:
            beam_cache[row] = model.predict(read_parquet_rows(parquet, [row], hf_cols)[row])
        return beam_cache[row]

    rows = [{"group": "A", **a} for a in DEMO_GROUP_A]
    rows += [{"group": "B", "id": f"B{i + 1}", "row": r, "label": "correct prediction (exact match to the label)",
              "reason": f"rank {i + 1} by beam log p ({float(predictions[r]['beam_log_prob']):.4f}) among exact, "
                        f"precursor-meeting V1-4a rows in chunks 520/572"} for i, r in enumerate(group_b)]
    taken = {int(r["row"]) for r in rows}
    group_c_log = {}
    for layer in LAYERS:
        f, f1 = demo_group_c_feature(reliable, layer)
        if f != DEMO_GROUP_C_REGISTERED[layer]:
            raise RuntimeError(f"layer {layer}: group C derives F{f}, registered F{DEMO_GROUP_C_REGISTERED[layer]}")
        concept = next(r["reliable_concept_1"] for r in reliable if int(r["layer"]) == layer and int(r["feature_idx"]) == f)
        sae = load_sae_from_checkpoint(root / "sae" / f"layer_{layer}" / f"seed_{DEFAULT_SEED}" / "checkpoint.pt",
                                       device="cpu")
        stream = ChunkStream(extract_dir=root / "extract", annotation_dir=root / "annotation", target_layer=layer,
                             sae=sae, device="cpu", batch_size=4096, dtype=torch.float32,
                             chunk_indices=list(LOCAL_TEST_CHUNKS), feature_subset=[f])
        ranked = []
        for chunk in stream:
            ranked += rank_spectra_by_peak_max(chunk.features[:, 0], chunk.token_to_spectrum,
                                               chunk.token_to_position, chunk.spectrum_ids)
        ranked.sort(key=lambda x: (-x[1], x[0]))
        pick, passed = pick_demo_spectrum(ranked, taken, lambda r: bool(beam(r)))
        taken.add(pick["row"])
        group_c_log[str(layer)] = {"feature": f, "reliable_f1_dom": f1, "reliable_concept": concept,
                                   "n_ranked": len(ranked), "top5": ranked[:5], "passed_over": passed}
        rows.append({"group": "C", "id": f"C-L{layer}", "row": pick["row"], "layer": layer, "feature": f,
                     "label": f"top peak-token activation of L{layer} F{f} (reliable association {concept}, "
                              f"F1dom {f1:.3f})",
                     "reason": f"rank {pick['rank']} by F{f}'s peak-token activation ({pick['activation']:.4g}) in "
                               f"chunks 520/572" + (f"; passed over {len(passed)}" if passed else "")})
        log(f"layer {layer}: F{f} ({concept}, F1dom {f1:.3f}) -> row {pick['row']} (rank {pick['rank']}, "
            f"activation {pick['activation']:.4g}); passed over {passed}")
        del sae, stream

    # Checks against the HF rows: every local row against the chunk metadata, every
    # row in predictions.csv against its recorded label.
    recs = read_parquet_rows(parquet, [int(r["row"]) for r in rows], hf_cols)
    metas = {}
    for r in rows:
        row, rec = int(r["row"]), recs[int(r["row"])]
        chunk, s = hf_test_location(row)
        r.update({"chunk": chunk, "spectrum_index": s, "spectrum_id": str(HF_TEST_OFFSET + row),
                  "local_chunk": chunk in LOCAL_TEST_CHUNKS,
                  "label_peptide": model.instanovo_notation(rec["modified_sequence"])})
        if row in predictions and predictions[row]["truth"] != r["label_peptide"]:
            raise RuntimeError(f"row {row}: HF label {r['label_peptide']} != predictions.csv {predictions[row]['truth']}")
        if r["local_chunk"]:
            if chunk not in metas:
                metas[chunk] = torch.load(root / "extract" / "chunks" / f"meta_{chunk:05d}.pt", map_location="cpu",
                                          weights_only=False)
            m = metas[chunk]
            if (m["spectrum_ids"][s] != r["spectrum_id"] or m["peptides"][s] != rec["sequence"]
                    or int(m["precursor_charges"][s]) != rec["precursor_charge"]
                    or abs(float(m["precursor_mzs"][s]) - rec["precursor_mz"]) > 1e-3):
                raise RuntimeError(f"row {row}: chunk {chunk} index {s} does not hold this HF row")
        r["checked_against"] = ("chunk metadata (spectrum id, peptide, precursor m/z, charge)" if r["local_chunk"]
                                else "") + (" and " if r["local_chunk"] and row in predictions else "") + \
                               ("v1_4c/predictions.csv label" if row in predictions else "")
    out = {"rule": DEMO_RULE, "hf_test_offset": HF_TEST_OFFSET, "rows": rows, "group_c": group_c_log,
           "finding_7_note": "every local test spectrum was SAE training data (Finding 7)",
           "seconds": time.perf_counter() - t_start}
    path = root / "analysis" / "v1_5" / "demo_set.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
    return out


# --- V1-5: activation as a screen for causal effect (--activation-screen) ----------------

SCREEN_TOP_ACT = 8                  # explain()'s top_k
SCREEN_TOP_EFFECT = 3
SCREEN_MIN_L0_RHO = 3
SCREEN_LOG_EPS = 1e-6               # batching-noise level (runtime.TIE_TOLERANCE)
SCREEN_N_BOOT = 10_000
SCREEN_CAVEATS = (
    "all sample spectra were SAE training data (Finding 7); single-feature effects understate redundant "
    "features (V1-4b); n spectra from two chunks; effects are exact, error-preserving, local ablations "
    "teacher-forced on the prediction, n = 1 spectrum per row")


def _top(features: np.ndarray, values: np.ndarray, k: int) -> list[int]:
    order = sorted(range(len(features)), key=lambda i: (-values[i], int(features[i])))
    return [int(features[i]) for i in order[:k]]


def peak_rho(act, abs_e) -> float:
    """Spearman rho (average ranks for ties); NaN when either side is constant."""
    act, abs_e = np.asarray(act, float), np.asarray(abs_e, float)
    if len(act) < 2 or np.all(act == act[0]) or np.all(abs_e == abs_e[0]):
        return math.nan
    return float(spearmanr(act, abs_e).statistic)


def peak_recall(features, act, abs_e, k_act: int = SCREEN_TOP_ACT, k_eff: int = SCREEN_TOP_EFFECT) -> dict | None:
    """Screening recall |T3 & A8| / 3, mass recall and chance 8 / L0 for one peak;
    None when L0 <= k_act (trivially 1). Ties broken by feature id."""
    features, act, abs_e = np.asarray(features), np.asarray(act, float), np.asarray(abs_e, float)
    if len(features) <= k_act:
        return None
    a8 = set(_top(features, act, k_act))
    t3 = _top(features, abs_e, k_eff)
    e_of = dict(zip(features.tolist(), abs_e.tolist()))
    hit = [f for f in t3 if f in a8]
    den = sum(e_of[f] for f in t3)
    return {"recall": len(hit) / k_eff, "mass_recall": sum(e_of[f] for f in hit) / den if den > 0 else math.nan,
            "chance": k_act / len(features)}


def variance_split(peaks: list[np.ndarray]) -> dict:
    """Per-spectrum split of |e| around the spectrum mean: SS_between, SS_within,
    peaks P and pairs N (for the exchangeable-null reference)."""
    allv = np.concatenate(peaks)
    mean_s = allv.mean()
    between = float(sum(len(p) * (p.mean() - mean_s) ** 2 for p in peaks))
    within = float(sum(((p - p.mean()) ** 2).sum() for p in peaks))
    return {"between": between, "within": within, "total": between + within, "P": len(peaks), "N": int(allv.size)}


def screen_spectrum_sums(peaks: list[dict]) -> dict:
    """Per-spectrum numerators and denominators of the layer statistics from its
    peaks ({features, activation, delta_total} per peak token)."""
    s = {k: 0.0 for k in ("rho_sum", "n_rho", "n_rho_nan", "recall_sum", "mass_sum", "n_mass", "chance_sum",
                          "n_recall", "n_le8", "between", "total", "between_log", "total_log", "p_minus_1",
                          "n_minus_1")}
    rows, abs_lists, log_lists = [], [], []
    for p in peaks:
        f, a, e = np.asarray(p["features"]), np.asarray(p["activation"], float), np.abs(np.asarray(p["delta_total"], float))
        rho = peak_rho(a, e) if len(f) >= SCREEN_MIN_L0_RHO else math.nan
        if len(f) >= SCREEN_MIN_L0_RHO:
            if math.isnan(rho):
                s["n_rho_nan"] += 1
            else:
                s["rho_sum"] += rho
                s["n_rho"] += 1
        rec = peak_recall(f, a, e)
        if rec is None:
            s["n_le8"] += 1
        else:
            s["recall_sum"] += rec["recall"]
            s["chance_sum"] += rec["chance"]
            s["n_recall"] += 1
            if not math.isnan(rec["mass_recall"]):
                s["mass_sum"] += rec["mass_recall"]
                s["n_mass"] += 1
        abs_lists.append(e)
        log_lists.append(np.log10(e + SCREEN_LOG_EPS))
        rows.append({"token": p["token"], "l0": len(f), "rho": rho,
                     **({"recall": rec["recall"], "mass_recall": rec["mass_recall"], "chance": rec["chance"]}
                        if rec else {"recall": None, "mass_recall": None, "chance": None}),
                     "mean_abs_e": float(e.mean()), "max_abs_e": float(e.max())})
    if abs_lists:
        v, vl = variance_split(abs_lists), variance_split(log_lists)
        s.update(between=v["between"], total=v["total"], between_log=vl["between"], total_log=vl["total"],
                 p_minus_1=v["P"] - 1, n_minus_1=v["N"] - 1)
    return {"sums": s, "peaks": rows}


def screen_layer_summary(per_spectrum: list[dict], n_boot: int = SCREEN_N_BOOT, seed: int = DEFAULT_SEED) -> dict:
    """Layer statistics as ratios of per-spectrum sums with percentile bootstrap
    95% CIs over spectra (default_rng(seed), n_boot resamples)."""
    sums = [d["sums"] for d in per_spectrum]

    def col(k):
        return np.array([x[k] for x in sums], float)
    idx = bootstrap_indices(len(sums), n_boot, seed)

    def stat(num, den):
        est, lo, hi = ratio_ci(col(num), col(den), idx)
        return {"estimate": est, "ci_low": lo, "ci_high": hi}
    rhos = [p["rho"] for d in per_spectrum for p in d["peaks"] if p["rho"] is not None and not math.isnan(p["rho"])]
    return {
        "n_spectra": len(sums),
        "a_mean_rho": stat("rho_sum", "n_rho"), "a_median_rho": float(np.median(rhos)) if rhos else math.nan,
        "a_n_peaks": int(col("n_rho").sum()), "a_n_constant_excluded": int(col("n_rho_nan").sum()),
        "b_recall": stat("recall_sum", "n_recall"), "b_chance": stat("chance_sum", "n_recall"),
        "b_mass_recall": stat("mass_sum", "n_mass"), "b_n_peaks": int(col("n_recall").sum()),
        "b_n_excluded_l0_le_8": int(col("n_le8").sum()),
        "c_between_share": stat("between", "total"), "c_null_reference": stat("p_minus_1", "n_minus_1"),
        "c_between_share_log": stat("between_log", "total_log"),
    }


def v1_4c_exploratory(sweep_csv: Path, case: str = "H7", n_top: int = 20) -> dict:
    """V1-4c's first look, recomputed: among the case's n_top largest |effect on M|
    over every swept (feature, token), how many lie outside the top 8 by activation
    at their token, and the activation ranks of the 3 largest. Exploratory; a
    different metric from (b)."""
    with open(sweep_csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["id"] == case]
    top = sorted(rows, key=lambda r: -abs(float(r["effect_on_M"])))[:n_top]
    return {"case": case, "n_swept": len(rows), "n_tokens": len({r["token"] for r in rows}), "n_top": n_top,
            "n_outside_top8": sum(int(r["activation_rank"]) > SCREEN_TOP_ACT for r in top),
            "activation_ranks_top3": [int(r["activation_rank"]) for r in top[:3]],
            "note": ("exploratory: effect on M (log-odds of the prediction vs the truth in the differing region) "
                     "across two contexts, 13 peaks, L8, one spectrum; not comparable to (b)")}


def _read_screen_table(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    from schema import SCREEN_ABLATION_SCHEMA_VERSION
    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    if meta.get("schema_version") != str(SCREEN_ABLATION_SCHEMA_VERSION):
        raise ValueError(f"{path}: screen ablation schema {meta.get('schema_version')!r}, expected "
                         f"{SCREEN_ABLATION_SCHEMA_VERSION}")
    return table.select(["token", "kind", "feature", "activation", "delta_total"]).to_pylist()


def group_peaks(table: list[dict]) -> tuple[list[dict], dict | None]:
    """Single ablations grouped per token: (peak tokens, the latent token or None)."""
    by_tok: dict[int, dict] = {}
    for r in table:
        if r["kind"] != "single":
            continue
        d = by_tok.setdefault(int(r["token"]), {"token": int(r["token"]), "features": [], "activation": [],
                                                "delta_total": []})
        d["features"].append(r["feature"])
        d["activation"].append(r["activation"])
        d["delta_total"].append(r["delta_total"])
    return [by_tok[t] for t in sorted(by_tok) if t >= 1], by_tok.get(0)


def activation_screen(root: Path, layers=LAYERS, n_boot: int = SCREEN_N_BOOT) -> tuple[dict, list[dict]]:
    v15 = root / "analysis" / "v1_5"
    sample = json.loads((v15 / "screen" / "screen_sample.json").read_text(encoding="utf-8"))
    rows = [int(r) for r in sample["sample_rows"]]
    missing = [(r, L) for r in rows for L in layers
               if not (v15 / "screen" / "ablation" / f"row_{r}_L{L}.parquet").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} (row, layer) ablation files missing, e.g. {missing[:5]}; "
                                f"finish screen-ablate first")
    summaries, peak_rows = {}, []
    for L in layers:
        per_spectrum, latent = [], []
        for r in rows:
            peaks, lat = group_peaks(_read_screen_table(v15 / "screen" / "ablation" / f"row_{r}_L{L}.parquet"))
            d = screen_spectrum_sums(peaks)
            per_spectrum.append(d)
            for p in d["peaks"]:
                peak_rows.append({"row": r, "layer": L, "token": p["token"], "peak": p["token"] - 1, "l0": p["l0"],
                                  "rho": p["rho"], "recall": p["recall"], "mass_recall": p["mass_recall"],
                                  "chance": p["chance"], "mean_abs_e": p["mean_abs_e"], "max_abs_e": p["max_abs_e"]})
            if lat is not None:
                lp = screen_spectrum_sums([lat])["peaks"][0]
                latent.append({"row": r, **{k: lp[k] for k in ("l0", "rho", "recall", "mass_recall", "chance",
                                                                "mean_abs_e", "max_abs_e")}})
        summaries[str(L)] = {**screen_layer_summary(per_spectrum, n_boot), "latent_token_descriptive": latent}
    out = {"definitions": "STATUS V1-5 screen definitions (a)-(c); bootstrap over spectra, "
                          f"{n_boot} resamples, default_rng({DEFAULT_SEED}), percentile 95%",
           "mass_recall_aggregation": "per-peak ratio, mean over peaks (peaks whose top-3 |e| sum to 0 excluded)",
           "n_spectra": len(rows), "rows": rows, "n_exact": sample["n_exact"],
           "screen_sample_sha256_16": hashlib_sha16(v15 / "screen" / "screen_sample.json"),
           "layers": summaries,
           "v1_4c_exploratory": v1_4c_exploratory(root / "analysis" / "v1_4c" / "part_b_token_sweep.csv"),
           "caveats": SCREEN_CAVEATS}
    return out, peak_rows


def hashlib_sha16(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


SCREEN_PEAK_COLUMNS = ["row", "layer", "token", "peak", "l0", "rho", "recall", "mass_recall", "chance",
                       "mean_abs_e", "max_abs_e"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="pipeline output root")
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--descriptions", nargs=2, metavar=("BLIND", "UNBLINDED"),
                        help="compare two description run tags instead of the report analyses")
    parser.add_argument("--deamidation-precursor", action="store_true",
                        help="Finding 8(d): deamidated labels vs the precursor at isotope offsets 0/1 "
                             "(reads the extract chunks; needs torch)")
    parser.add_argument("--label-audit", action="store_true",
                        help="Finding 9: label-quality audit over a seeded sample of local spectra "
                             "(reads the extract chunks and labels; needs torch)")
    parser.add_argument("--reliable-associations", action="store_true",
                        help="V1-3b / Finding 10: list features whose best reliable concept association "
                             "has F1dom >= 0.1 (reads the dictionaries)")
    parser.add_argument("--demo-set", action="store_true",
                        help="V1-5 step 1: the registered demo set -> analysis/v1_5/demo_set.json (reads chunks "
                             "520/572 and runs beam search; needs --parquet)")
    parser.add_argument("--activation-screen", action="store_true",
                        help="V1-5 step 5: activation as a screen for causal effect, from the screen ablations")
    parser.add_argument("--parquet", type=Path, default=None, help="--demo-set: the HF nine-species test parquet")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / "instanovo_v1.1.0.ckpt",
                        help="--demo-set: InstaNovo checkpoint")
    args = parser.parse_args(argv)

    if args.demo_set:
        if args.parquet is None:
            parser.error("--demo-set needs --parquet")
        out = demo_set(args.root, args.parquet, args.checkpoint)
        for r in out["rows"]:
            print(f"{r['group']} {r['id']:>6} row {r['row']:>6} chunk {r['chunk']} | {r['label']}")
        print(f"wrote {args.root / 'analysis' / 'v1_5' / 'demo_set.json'} ({out['seconds']:.0f} s)")
        return

    if args.activation_screen:
        out, peak_rows = activation_screen(args.root, tuple(args.layers))
        out_dir = args.root / "analysis" / "v1_5" / "screen"
        (out_dir / "activation_screen.json").write_text(json.dumps(out, indent=1, allow_nan=True), encoding="utf-8")
        _write_csv(out_dir / "activation_screen_peaks.csv", SCREEN_PEAK_COLUMNS, peak_rows)
        for L, v in out["layers"].items():
            def f(x):
                return f"{x['estimate']:.3f} [{x['ci_low']:.3f}, {x['ci_high']:.3f}]"
            print(f"L{L}: rho {f(v['a_mean_rho'])} (median {v['a_median_rho']:.3f}, {v['a_n_peaks']} peaks); "
                  f"recall {f(v['b_recall'])} vs chance {f(v['b_chance'])}, mass {f(v['b_mass_recall'])} "
                  f"({v['b_n_peaks']} peaks, {v['b_n_excluded_l0_le_8']} with L0 <= 8 excluded); between-peak "
                  f"share {f(v['c_between_share'])} vs null {f(v['c_null_reference'])}, log {f(v['c_between_share_log'])}")
        print(f"V1-4c exploratory: {out['v1_4c_exploratory']}")
        return

    if args.reliable_associations:
        rows, summary = reliable_association_list(
            {layer: load_dictionary_rows(args.root, layer) for layer in args.layers})
        out_dir = args.root / "analysis"
        _write_csv(out_dir / "reliable_associations.csv", RELIABLE_LIST_COLUMNS, rows)
        (out_dir / "reliable_associations_summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        return

    if args.label_audit:
        summary = label_audit(args.root, seed=args.seed)
        print(json.dumps(summary, indent=1, default=float))
        return

    if args.deamidation_precursor:
        out_dir = args.root / "analysis" / "label_audit"
        rows, summary = deamidation_precursor(args.root)
        _write_csv(out_dir / "deamidation_precursor.csv", DEAMIDATION_COLUMNS, rows)
        (out_dir / "deamidation_precursor_summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        return

    if args.descriptions:
        blind_tag, plain_tag = args.descriptions
        blinding_rows: list[dict] = []
        null_rows: list[dict] = []
        for layer in args.layers:
            blind = load_description_run(description_run_path(args.root, layer, blind_tag), True)
            plain = load_description_run(description_run_path(args.root, layer, plain_tag), False)
            layer_rows = description_blinding(blind, plain, layer, args.seed)
            blinding_rows += layer_rows
            for name, run in ((blind_tag, blind), (plain_tag, plain)):
                n = shuffled_null_by_donor(run, layer, name)
                null_rows.append(n)
                print(f"layer {layer} {name:<13} shuffled r median {n['median']:.3f} p95 "
                      f"{n['p95']:.3f} (n={n['n']}, {n['n_undefined']} undefined) | same-stratum donors median "
                      f"{n['same_stratum_median']:.3f} p95 {n['same_stratum_p95']:.3f} "
                      f"(n={n['same_stratum_n']}) | other-stratum median "
                      f"{n['other_stratum_median']:.3f} p95 {n['other_stratum_p95']:.3f} "
                      f"(n={n['other_stratum_n']})")
            for r in layer_rows:
                print(f"layer {layer} {r['stratum']:<11} n={r['n_features']:>3}  "
                      f"r blind {r['blind_median_r']:.3f} / unblinded {r['unblinded_median_r']:.3f}"
                      f"  dr {r['median_delta_r']:+.3f} [{r['delta_r_ci_low']:+.3f}, "
                      f"{r['delta_r_ci_high']:+.3f}]  AUROC {r['blind_median_auroc']:.3f} / "
                      f"{r['unblinded_median_auroc']:.3f}  null p95 {r['blind_null_p95']:.3f} / "
                      f"{r['unblinded_null_p95']:.3f}  beats {r['blind_n_beats']}/"
                      f"{r['blind_n_tested']} / {r['unblinded_n_beats']}/{r['unblinded_n_tested']}")
        out = args.root / "analysis" / "description_blinding.csv"
        null_out = args.root / "analysis" / "description_shuffled_null.csv"
        _write_csv(out, BLINDING_COLUMNS, blinding_rows)
        _write_csv(null_out, NULL_COLUMNS, null_rows)
        print(f"wrote {out} and {null_out}")
        return

    hub_rows: list[dict] = []
    corr_rows: list[dict] = []
    for layer in args.layers:
        report = load_report(report_path(args.root, layer, args.seed))
        rows = hub_inventory(report, layer)
        corr = f1dom_selectivity_correlation(report, layer)
        hub_rows += rows
        corr_rows.append(corr)
        print(f"layer {layer}: {len(rows)} ablated features; top: "
              + ", ".join(f"F{r['feature_idx']} {r['mean_delta_ce']:+.3f}" for r in rows[:3]))
        print(f"         F1dom vs selectivity: rho {corr['spearman_rho']:+.3f} "
              f"(p {corr['p_value']:.2f}, n {corr['n_concepts']}, dropped {corr['n_dropped_nan']})")

    out_dir = args.root / "analysis"
    _write_csv(out_dir / "hubs.csv", HUB_COLUMNS, hub_rows)
    _write_csv(out_dir / "f1dom_selectivity.csv", CORRELATION_COLUMNS, corr_rows)
    print(f"wrote {out_dir / 'hubs.csv'} and {out_dir / 'f1dom_selectivity.csv'}")


if __name__ == "__main__":
    main()
