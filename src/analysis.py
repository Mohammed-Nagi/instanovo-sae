"""analysis.py -- post-hoc analysis over the released evaluate.py reports.

Everything here reads each layer's ``report.json`` (Phase 8 results) and, with
``--descriptions``, the interpret.py description runs. Those modes need neither
torch nor the activation chunks, so they run on any machine; only
``--deamidation-precursor`` (below) reads the extract chunks.

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

Usage (from the repo root)::

    PYTHONPATH=src uv run python -m analysis
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --layers 2 4
    PYTHONPATH=src uv run python -m analysis --descriptions blind_v1 unblinded_v2
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --deamidation-precursor

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
    args = parser.parse_args(argv)

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
