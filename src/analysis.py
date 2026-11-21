"""analysis.py -- post-hoc analysis over the released evaluate.py reports.

Everything here reads only each layer's ``report.json`` (Phase 8 results) and
needs neither torch nor the activation chunks, so it runs on any machine.

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

Usage (from the repo root)::

    PYTHONPATH=src uv run python -m analysis
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --layers 2 4
    PYTHONPATH=src uv run python -m analysis --descriptions blind_v1 unblinded_v2

Outputs under ``<root>/analysis/``: ``hubs.csv`` (a ranked list, no flag) and
``f1dom_selectivity.csv``; with ``--descriptions``, only
``description_blinding.csv`` and ``description_shuffled_null.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
    args = parser.parse_args(argv)

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
