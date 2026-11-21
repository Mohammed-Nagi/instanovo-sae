"""Tests for analysis.py.

Run from the repo root: ``PYTHONPATH=src uv run pytest tests``.

The released reports in paper_artifacts/ are tracked, so the Finding 2 and
Finding 3 checks run anywhere. They pin the reproduced numbers, which makes a
change to the definitions in analysis.py show up here.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

import analysis
from schema import EVAL_SCHEMA_VERSION

REPO = Path(__file__).resolve().parent.parent
RELEASED = REPO / "paper_artifacts" / "eval_reports"


def _entry(feat: int, d: float, f1: float = 0.1, sel: float = 0.0) -> dict:
    return {"feature_idx": feat, "mean_delta_ce": d, "selectivity": sel, "f1_dom": f1, "lift": 2.0}


def _concept(entries: list[dict], group: list[int], sel: float = 0.0) -> dict:
    return {"per_feature_ablation": entries, "top_n_features": group,
            "causal": {"selectivity": sel}}


def _report(per_concept: dict) -> dict:
    return {"schema_version": EVAL_SCHEMA_VERSION, "phase_8": {"per_concept": per_concept}}


@pytest.fixture(scope="module")
def released() -> dict[int, dict]:
    return {layer: analysis.load_report(RELEASED / f"layer_{layer}.json") for layer in analysis.LAYERS}


# --- synthetic cases -------------------------------------------------------------


def test_hub_flag_ranking_and_membership():
    # |delta-CE| values 0.001, 0.001, 0.002, 0.5 -> median 0.0015, threshold 0.015
    report = _report({
        "a": _concept([_entry(1, 0.001), _entry(2, 0.5)], group=[2, 3]),
        "b": _concept([_entry(2, 0.5), _entry(3, 0.002), _entry(4, 0.001)], group=[4]),
    })
    rows = analysis.hub_inventory(report, layer=2)
    assert [r["feature_idx"] for r in rows] == [2, 3, 1, 4]  # rank by delta-CE, ties by index
    assert [r["rank"] for r in rows] == [1, 2, 3, 4]
    assert [r["is_hub"] for r in rows] == [True, False, False, False]
    hub = rows[0]
    assert hub["single_concepts"] == "a|b" and hub["n_single_concepts"] == 2
    assert hub["group_concepts"] == "a" and hub["n_group_concepts"] == 1
    assert hub["hub_threshold"] == pytest.approx(0.015)
    assert hub["ratio_to_median"] == pytest.approx(0.5 / 0.0015)
    assert rows[1]["feature_idx"] == 3 and rows[1]["group_concepts"] == "a"
    assert rows[2]["feature_idx"] == 1 and rows[2]["group_concepts"] == ""


def test_negative_signed_median_still_gives_a_sane_threshold():
    report = _report({"a": _concept(
        [_entry(1, -0.002), _entry(2, -0.001), _entry(3, 0.0005), _entry(4, 0.05)], group=[])})
    rows = {r["feature_idx"]: r for r in analysis.hub_inventory(report, layer=8)}
    assert rows[4]["is_hub"]
    assert not rows[1]["is_hub"] and not rows[2]["is_hub"] and not rows[3]["is_hub"]


def test_inconsistent_delta_ce_across_concepts_fails_loudly():
    report = _report({
        "a": _concept([_entry(1, 0.01)], group=[]),
        "b": _concept([_entry(1, 0.02)], group=[]),
    })
    with pytest.raises(ValueError, match="should not depend on the concept"):
        analysis.hub_inventory(report, layer=2)


def test_nan_delta_ce_and_empty_report_fail_loudly():
    with pytest.raises(ValueError, match="NaN"):
        analysis.hub_inventory(_report({"a": _concept([_entry(1, float("nan"))], [])}), layer=2)
    with pytest.raises(ValueError, match="no per-feature ablations"):
        analysis.hub_inventory(_report({"a": _concept([], [])}), layer=2)


def test_load_report_rejects_unexpected_schema(tmp_path):
    bad = copy.deepcopy(_report({"a": _concept([_entry(1, 0.1)], [])}))
    bad["schema_version"] = EVAL_SCHEMA_VERSION + 1
    path = tmp_path / "report.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="schema_version"):
        analysis.load_report(path)


def test_correlation_uses_best_f1dom_and_drops_nan_selectivity():
    # best F1dom per concept: 0.1, 0.2, 0.3, 0.4; selectivity increases with it -> rho = 1
    per_concept = {
        f"c{i}": _concept([_entry(i, 0.0, f1=f), _entry(100 + i, 0.0, f1=f / 2)], [], sel=f * 10)
        for i, f in enumerate([0.1, 0.2, 0.3, 0.4])
    }
    per_concept["nan_sel"] = _concept([_entry(50, 0.0, f1=0.9)], [], sel=float("nan"))
    per_concept["empty"] = _concept([], [], sel=1.0)
    result = analysis.f1dom_selectivity_correlation(_report(per_concept), layer=2)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["n_concepts"] == 4 and result["n_dropped_nan"] == 2


def test_correlation_needs_enough_concepts():
    with pytest.raises(ValueError, match="usable"):
        analysis.f1dom_selectivity_correlation(
            _report({"a": _concept([_entry(1, 0.0)], [], sel=0.1)}), layer=2)


# --- Finding 2: hub features in the released reports -----------------------------

FINDING_2_HUBS = {
    2: {7522: 0.100, 11907: 0.067, 12077: 0.033},
    4: {11376: 0.057},
    6: {9282: 0.034, 10689: 0.023},
    8: {2481: 0.018},
}


@pytest.mark.parametrize("layer", analysis.LAYERS)
def test_finding_2_hubs_are_flagged_with_the_reported_delta_ce(released, layer):
    rows = {r["feature_idx"]: r for r in analysis.hub_inventory(released[layer], layer)}
    for feat, expected in FINDING_2_HUBS[layer].items():
        assert rows[feat]["is_hub"], f"L{layer} F{feat} not flagged"
        assert rows[feat]["mean_delta_ce"] == pytest.approx(expected, abs=5e-4)


def test_finding_2_largest_hub_per_layer(released):
    top = {layer: analysis.hub_inventory(released[layer], layer)[0]["feature_idx"]
           for layer in analysis.LAYERS}
    assert top == {2: 7522, 4: 11376, 6: 11907, 8: 2481}


def test_finding_2_group_ablation_extremes_contain_a_hub(released):
    def groups(layer: int, feat: int) -> set[str]:
        rows = {r["feature_idx"]: r for r in analysis.hub_inventory(released[layer], layer)}
        return set(rows[feat]["group_concepts"].split("|"))

    assert "has_neutral_loss" in groups(2, 11907)
    assert "top_decile_intensity" in groups(4, 11907)
    assert "is_y_ion" in groups(4, 11376)


# --- Finding 3: F1dom vs selectivity ---------------------------------------------


def test_finding_3_correlations_reproduce(released):
    rho = {layer: analysis.f1dom_selectivity_correlation(released[layer], layer)["spearman_rho"]
           for layer in analysis.LAYERS}
    assert rho[2] == pytest.approx(0.230, abs=5e-3)
    assert rho[4] == pytest.approx(-0.148, abs=5e-3)
    assert rho[6] == pytest.approx(-0.052, abs=5e-3)
    assert rho[8] == pytest.approx(0.168, abs=5e-3)


def test_readme_correlations_are_not_reproduced(released):
    # README states +0.04 / +0.53 / +0.13 / +0.06. No code produced them; the
    # recomputation disagrees, most visibly at layer 4 (+0.53 claimed).
    rho4 = analysis.f1dom_selectivity_correlation(released[4], 4)["spearman_rho"]
    assert abs(rho4 - 0.53) > 0.3


# --- CLI -------------------------------------------------------------------------


def test_main_writes_both_csvs(tmp_path):
    for layer in analysis.LAYERS:
        dest = analysis.report_path(tmp_path, layer).parent
        dest.mkdir(parents=True)
        (dest / "report.json").write_bytes((RELEASED / f"layer_{layer}.json").read_bytes())
    analysis.main(["--root", str(tmp_path)])
    with open(tmp_path / "analysis" / "hubs.csv", newline="", encoding="utf-8") as fh:
        hubs = list(csv.DictReader(fh))
    assert list(hubs[0]) == analysis.HUB_COLUMNS
    assert {int(r["layer"]) for r in hubs} == set(analysis.LAYERS)
    with open(tmp_path / "analysis" / "f1dom_selectivity.csv", newline="", encoding="utf-8") as fh:
        corr = list(csv.DictReader(fh))
    assert [int(r["layer"]) for r in corr] == list(analysis.LAYERS)
