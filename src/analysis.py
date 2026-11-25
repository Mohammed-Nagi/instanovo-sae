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
    precedence.csv, coverage.csv, a_ions.csv, concepts.csv, summary.json.

Usage (from the repo root)::

    PYTHONPATH=src uv run python -m analysis
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --layers 2 4
    PYTHONPATH=src uv run python -m analysis --descriptions blind_v1 unblinded_v2
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --deamidation-precursor
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --label-audit

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

from schema import EVAL_SCHEMA_VERSION, check_run_tag

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
                              peptide lacks (I, L and J count as one)."""
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
            "immonium_absent_residue": arr(f"con|imm_absent|{c}").sum()})
    timings["tables"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_start

    for name, rows_ in (("per_class", cls_rows), ("determined", det_rows), ("isotope", iso_rows),
                        ("precedence", prec_rows), ("coverage", cov_rows), ("a_ions", a_rows),
                        ("concepts", con_rows)):
        _write_csv(out_dir / f"{name}.csv", list(rows_[0]), rows_)
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


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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
    args = parser.parse_args(argv)

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
