"""analysis.py -- post-hoc analysis over the released evaluate.py reports.

Everything here reads only each layer's ``report.json`` (Phase 8 results) and
needs neither torch nor the activation chunks, so it runs on any machine.

Hub inventory
    Phase 8 ablates, for every concept, its top ``ablation_per_feature_top``
    detectors one at a time (``per_feature_ablation``) and as a group of
    ``ablation_top_n`` (``top_n_features``). The inventory has one row per
    distinct single-ablated feature:

    * ranked by mean delta-CE, rank 1 = the feature whose removal hurts most;
    * flagged as a hub when delta-CE exceeds ``multiplier`` x the layer median;
    * listing the concepts whose single-ablation list contains it
      (``single_concepts``) and the concepts whose group ablation contains it
      (``group_concepts``). Group membership is what lets one hub dominate a
      group result.

    The median is taken over the ablated features, not over all d_dict features:
    those are the top F1dom detectors of each concept, a selected population, and
    only they have a single-feature delta-CE in the report. It is the median of
    |delta-CE|, because at layer 8 the signed median is slightly negative and
    "10x a negative number" would flag nearly everything.

    A feature's delta-CE does not depend on which concept it was listed under;
    the code checks this and fails if a report disagrees.

F1dom vs ablation selectivity
    ``f1dom_selectivity_correlation`` is the Spearman correlation, across
    concepts, between a concept's best F1dom among its ablated features and the
    selectivity of that concept's group ablation. Concepts whose selectivity is
    NaN (a tercile split that is empty) are dropped; the count is reported.

Usage (from the repo root)::

    PYTHONPATH=src uv run python -m analysis
    PYTHONPATH=src uv run python -m analysis --root sae_pipeline_outputs --layers 2 4

Outputs under ``<root>/analysis/``: ``hubs.csv`` and ``f1dom_selectivity.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

from scipy.stats import spearmanr

from schema import EVAL_SCHEMA_VERSION

LAYERS = (2, 4, 6, 8)
HUB_MULTIPLIER = 10.0
DEFAULT_ROOT = Path("sae_pipeline_outputs")
DEFAULT_SEED = 0

HUB_COLUMNS = [
    "layer", "feature_idx", "rank", "mean_delta_ce", "ratio_to_median", "hub_threshold",
    "is_hub", "n_single_concepts", "single_concepts", "n_group_concepts", "group_concepts",
]
CORRELATION_COLUMNS = ["layer", "spearman_rho", "p_value", "n_concepts", "n_dropped_nan"]


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


def hub_inventory(report: dict, layer: int, multiplier: float = HUB_MULTIPLIER) -> list[dict]:
    """One row per single-ablated feature, ranked by mean delta-CE (see module docstring)."""
    per_concept = report["phase_8"]["per_concept"]

    delta_ce: dict[int, float] = {}
    single: dict[int, list[str]] = {}
    group: dict[int, list[str]] = {}
    for concept, info in per_concept.items():
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
            single.setdefault(feat, []).append(concept)
        for feat in info["top_n_features"]:
            group.setdefault(int(feat), []).append(concept)

    if not delta_ce:
        raise ValueError(f"layer {layer}: report has no per-feature ablations")
    median_abs = median(abs(d) for d in delta_ce.values())
    if median_abs <= 0.0:
        raise ValueError(f"layer {layer}: median |delta-CE| is {median_abs}; no hub threshold")
    threshold = multiplier * median_abs

    ranked = sorted(delta_ce.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {
            "layer": layer,
            "feature_idx": feat,
            "rank": rank,
            "mean_delta_ce": d,
            "ratio_to_median": d / median_abs,
            "hub_threshold": threshold,
            "is_hub": d > threshold,
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
    parser.add_argument("--multiplier", type=float, default=HUB_MULTIPLIER,
                        help="hub = delta-CE above this multiple of the layer median |delta-CE|")
    args = parser.parse_args(argv)

    hub_rows: list[dict] = []
    corr_rows: list[dict] = []
    for layer in args.layers:
        report = load_report(report_path(args.root, layer, args.seed))
        rows = hub_inventory(report, layer, args.multiplier)
        corr = f1dom_selectivity_correlation(report, layer)
        hub_rows += rows
        corr_rows.append(corr)
        hubs = [r for r in rows if r["is_hub"]]
        print(f"layer {layer}: {len(rows)} ablated features, {len(hubs)} hubs "
              f"(threshold {rows[0]['hub_threshold']:.2e}); top: "
              + ", ".join(f"F{r['feature_idx']} {r['mean_delta_ce']:+.3f}" for r in rows[:3]))
        print(f"         F1dom vs selectivity: rho {corr['spearman_rho']:+.3f} "
              f"(p {corr['p_value']:.2f}, n {corr['n_concepts']}, dropped {corr['n_dropped_nan']})")

    out_dir = args.root / "analysis"
    _write_csv(out_dir / "hubs.csv", HUB_COLUMNS, hub_rows)
    _write_csv(out_dir / "f1dom_selectivity.csv", CORRELATION_COLUMNS, corr_rows)
    print(f"wrote {out_dir / 'hubs.csv'} and {out_dir / 'f1dom_selectivity.csv'}")


if __name__ == "__main__":
    main()
