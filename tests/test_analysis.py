"""Tests for analysis.py.

Run from the repo root: ``PYTHONPATH=src uv run python -m pytest tests``.

The released reports in paper_artifacts/ are tracked, so the Finding 2 and
Finding 3 checks run anywhere. They pin the reproduced numbers, which makes a
change to the definitions in analysis.py show up here.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
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


def test_ranking_and_membership():
    # |delta-CE| values 0.001, 0.001, 0.002, 0.5 -> median 0.0015
    report = _report({
        "a": _concept([_entry(1, 0.001), _entry(2, 0.5)], group=[2, 3]),
        "b": _concept([_entry(2, 0.5), _entry(3, 0.002), _entry(4, 0.001)], group=[4]),
    })
    rows = analysis.hub_inventory(report, layer=2)
    assert [r["feature_idx"] for r in rows] == [2, 3, 1, 4]  # rank by delta-CE, ties by index
    assert [r["rank"] for r in rows] == [1, 2, 3, 4]
    top = rows[0]
    assert top["single_concepts"] == "a|b" and top["n_single_concepts"] == 2
    assert top["group_concepts"] == "a" and top["n_group_concepts"] == 1
    assert top["ratio_to_median"] == pytest.approx(0.5 / 0.0015)
    assert "is_hub" not in top and "hub_threshold" not in top
    assert rows[1]["feature_idx"] == 3 and rows[1]["group_concepts"] == "a"
    assert rows[2]["feature_idx"] == 1 and rows[2]["group_concepts"] == ""


def test_ratio_uses_median_of_absolute_values():
    # signed median is -0.00025, |median| = 0.0015; a signed median would flip the ratios' sign
    report = _report({"a": _concept(
        [_entry(1, -0.002), _entry(2, -0.001), _entry(3, 0.0005), _entry(4, 0.05)], group=[])})
    rows = {r["feature_idx"]: r for r in analysis.hub_inventory(report, layer=8)}
    assert rows[4]["ratio_to_median"] == pytest.approx(0.05 / 0.0015)
    assert rows[1]["ratio_to_median"] == pytest.approx(-0.002 / 0.0015)


def test_per_feature_delta_ce_returns_one_value_per_feature():
    report = _report({
        "a": _concept([_entry(1, 0.01), _entry(2, 0.02)], group=[]),
        "b": _concept([_entry(2, 0.02)], group=[]),
    })
    assert analysis.per_feature_delta_ce(report, layer=2) == {1: 0.01, 2: 0.02}


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
def test_finding_2_hubs_rank_at_the_top_with_the_reported_delta_ce(released, layer):
    rows = {r["feature_idx"]: r for r in analysis.hub_inventory(released[layer], layer)}
    for feat, expected in FINDING_2_HUBS[layer].items():
        assert rows[feat]["rank"] <= 3, f"L{layer} F{feat} ranks {rows[feat]['rank']}"
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


def test_readme_correlations_match_the_code(released):
    # The README quotes these values to two decimals; they must be what the code gives.
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for layer in analysis.LAYERS:
        rho = analysis.f1dom_selectivity_correlation(released[layer], layer)["spearman_rho"]
        quoted = f"{rho:+.2f}"
        assert quoted in readme, f"README does not quote layer {layer}'s rho {quoted}"
    assert "+0.53" not in readme  # the earlier, unreproducible claim


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


# --- blinded vs unblinded descriptions ----------------------------------------------


def _desc_record(feat: int, stratum: str, r: float, auroc: float, shuffled: list[float],
                 beats: bool | None, holdout_tokens: tuple[int, ...] = (1, 2)) -> dict:
    return {
        "feature_idx": feat, "stratum": stratum, "pearson_r": r, "detection_auroc": auroc,
        "model_served": "gpt-4o-2024-08-06", "description_beats_baseline": beats,
        "shuffled": {"r_values": shuffled},
        "holdout_examples": [{"chunk_idx": 0, "token_in_chunk": t} for t in holdout_tokens],
        "describe_keys": [[0, 100 + feat]],
    }


def _desc_runs():
    blind = {
        1: _desc_record(1, "concept", 0.5, 0.8, [0.1, 0.0], True),
        2: _desc_record(2, "concept", 0.3, 0.7, [0.2, -0.1], False),
        3: _desc_record(3, "unexplained", 0.6, 0.9, [0.05, float("nan")], True),
    }
    plain = {
        1: _desc_record(1, "concept", 0.7, 0.9, [0.3, 0.2], True),
        2: _desc_record(2, "concept", 0.6, 0.8, [0.1, 0.4], True),
        3: _desc_record(3, "unexplained", 0.65, 0.95, [0.0, 0.1], None),
    }
    return blind, plain


def test_description_blinding_per_stratum_numbers():
    blind, plain = _desc_runs()
    rows = {r["stratum"]: r for r in analysis.description_blinding(blind, plain, 2)}
    assert set(rows) == {"concept", "unexplained", "all"}
    concept = rows["concept"]
    assert concept["n_features"] == 2 and concept["n_paired"] == 2
    assert concept["blind_median_r"] == pytest.approx(0.4)
    assert concept["unblinded_median_r"] == pytest.approx(0.65)
    assert concept["median_delta_r"] == pytest.approx(-0.25)   # deltas -0.2, -0.3
    assert concept["delta_r_ci_low"] <= -0.25 <= concept["delta_r_ci_high"]
    assert concept["blind_median_auroc"] == pytest.approx(0.75)
    assert (concept["blind_n_beats"], concept["blind_n_tested"]) == (1, 2)
    # the null is pooled over the whole layer, NaN dropped: 0.1, 0.0, 0.2, -0.1, 0.05
    assert concept["blind_null_median"] == pytest.approx(0.05)
    assert rows["all"]["blind_null_p95"] == pytest.approx(
        float(np.percentile([0.1, 0.0, 0.2, -0.1, 0.05], 95)))
    assert (rows["unexplained"]["unblinded_n_beats"], rows["unexplained"]["unblinded_n_tested"]) \
        == (0, 0)
    assert rows["all"]["n_features"] == 3


def test_description_blinding_is_deterministic():
    blind, plain = _desc_runs()
    # json, because NaN != NaN (the one-feature stratum has no CI)
    assert json.dumps(analysis.description_blinding(blind, plain, 2)) == \
        json.dumps(analysis.description_blinding(blind, plain, 2))


def test_description_blinding_rejects_runs_with_different_examples():
    blind, plain = _desc_runs()
    plain[2] = _desc_record(2, "concept", 0.6, 0.8, [0.1], True, holdout_tokens=(1, 3))
    with pytest.raises(ValueError, match="different describe or held-out"):
        analysis.description_blinding(blind, plain, 2)


def test_description_blinding_rejects_different_feature_sets():
    blind, plain = _desc_runs()
    del plain[3]
    with pytest.raises(ValueError, match="different features"):
        analysis.description_blinding(blind, plain, 2)


def test_shuffled_null_split_by_donor_stratum():
    blind, _ = _desc_runs()
    # features 1, 2 are concept; 3 is unexplained
    blind[1]["shuffled"] = {"donors": [2, 3], "r_values": [0.8, 0.1]}
    blind[2]["shuffled"] = {"donors": [1, 3], "r_values": [0.6, float("nan")]}
    blind[3]["shuffled"] = {"donors": [1, 2], "r_values": [0.2, 0.0]}
    row = analysis.shuffled_null_by_donor(blind, 2, "blind_v1")
    assert row["n"] == 5 and row["n_undefined"] == 1 and row["median"] == pytest.approx(0.2)
    assert row["same_stratum_n"] == 2 and row["same_stratum_median"] == pytest.approx(0.7)
    assert row["other_stratum_n"] == 3 and row["other_stratum_median"] == pytest.approx(0.1)
    assert row["p95"] == pytest.approx(float(np.percentile([0.8, 0.1, 0.6, 0.2, 0.0], 95)))


def test_shuffled_null_with_mismatched_donors_raises():
    blind, _ = _desc_runs()
    blind[1]["shuffled"] = {"donors": [2], "r_values": [0.8, 0.1]}
    with pytest.raises(ValueError, match="donors"):
        analysis.shuffled_null_by_donor(blind, 2, "blind_v1")


def test_load_description_run_checks_the_blind_flag(tmp_path):
    path = tmp_path / "feature_descriptions.json"
    blind, _ = _desc_runs()
    path.write_text(json.dumps({"config": {"blind": True}, "results": list(blind.values())}))
    assert set(analysis.load_description_run(path, blind=True)) == {1, 2, 3}
    with pytest.raises(ValueError, match="expected False"):
        analysis.load_description_run(path, blind=False)


@pytest.mark.parametrize("tag", ["archive", "archive/unblinded_v1", "../layer_4/blind_v1"])
def test_description_runs_are_never_read_from_the_archive(tmp_path, tag):
    with pytest.raises(ValueError, match="archive"):
        analysis.description_run_path(tmp_path, 2, tag)
    assert analysis.description_run_path(tmp_path, 2, "blind_v1").parent.name == "blind_v1"


# --- Reliable-association list (V1-3b, Finding 10) ----------------------------------

def test_reliable_association_list_cut_order_and_summary():
    import math
    import analysis

    def row(f, best, concept="is_y_ion"):
        return {"feature_idx": f, "firing_rate": 0.01, "n_reliable_associations": 0 if best is None else 1,
                "reliable_concept_1": None if best is None else concept, "reliable_f1_dom_1": best,
                "concept_1": "mz_low", "description": None}

    rows = {8: [row(1, 0.1), row(2, 0.35), row(3, 0.0999), row(4, None), row(5, 0.35)],
            2: [row(9, float("nan")), row(10, 0.01)]}
    out, summary = analysis.reliable_association_list(rows)
    assert [(r["layer"], r["feature_idx"]) for r in out] == [(8, 2), (8, 5), (8, 1)]   # inclusive cut
    assert list(out[0]) == analysis.RELIABLE_LIST_COLUMNS and out[0]["reliable_concept_2"] is None
    s8, s2 = summary["by_layer"]["8"], summary["by_layer"]["2"]
    assert s8["n_live_features"] == 5 and s8["n_with_reliable_association"] == 4
    assert s8["median_best_reliable_f1_dom"] == pytest.approx((0.1 + 0.35) / 2)
    assert s8["n_best_reliable_f1_dom_ge_0.1"] == 3 and s8["n_best_reliable_f1_dom_ge_0.3"] == 2
    assert s2["n_listed"] == 0 and s2["n_with_reliable_association"] == 1
    assert summary["min_reliable_f1_dom"] == 0.1 and "not a reliability claim" in summary["cut"]
    assert math.isnan(analysis.reliable_association_list({4: [row(1, None)]})[1]["by_layer"]["4"][
        "median_best_reliable_f1_dom"])


# --- V1-5: demo rule and activation screen -------------------------------------------------

import math  # noqa: E402

import numpy as np  # noqa: E402

from analysis import (  # noqa: E402
    HF_TEST_OFFSET,
    demo_group_b,
    demo_group_c_feature,
    group_peaks,
    hf_test_location,
    peak_recall,
    peak_rho,
    pick_demo_spectrum,
    rank_spectra_by_peak_max,
    screen_layer_summary,
    screen_spectrum_sums,
    v1_4c_exploratory,
    variance_split,
)


def _pred(row, exact, meets, lp):
    return {"row": str(row), "exact": str(exact), "meets_precursor": str(meets), "beam_log_prob": str(lp)}


def test_hf_test_location_matches_the_recorded_local_chunks():
    assert hf_test_location(4506) == (520, 0) and hf_test_location(5529) == (520, 1023)
    assert hf_test_location(57754) == (572, 0) and hf_test_location(58777) == (572, 1023)


def test_demo_group_b_ranking():
    preds = [_pred(4552, True, True, -0.74), _pred(5398, True, True, -0.03), _pred(5406, True, True, -0.13),
             _pred(58409, True, True, -0.50), _pred(58427, True, True, -1.35),
             _pred(5000, False, True, -0.01), _pred(5001, True, False, -0.01),   # wrong / not meeting
             _pred(304, True, True, -0.001)]                                     # outside chunks 520/572
    assert demo_group_b(preds) == [5398, 5406, 58409]


def test_demo_group_c_feature_takes_the_best_reliable_f1dom():
    rows = [{"layer": "2", "feature_idx": "7", "reliable_f1_dom_1": "0.2"},
            {"layer": "2", "feature_idx": "3", "reliable_f1_dom_1": "0.4"},
            {"layer": "2", "feature_idx": "9", "reliable_f1_dom_1": "0.4"},
            {"layer": "4", "feature_idx": "1", "reliable_f1_dom_1": "0.9"}]
    assert demo_group_c_feature(rows, 2) == (3, 0.4)            # tie -> lower feature id
    with pytest.raises(ValueError):
        demo_group_c_feature(rows, 6)


def test_rank_by_peak_max_ignores_the_latent_token():
    col = [9.0, 1.0, 2.0, 0.0, 5.0, 0.0, 0.0]
    t2s = [0, 0, 0, 1, 1, 2, 2]
    pos = [0, 1, 2, 0, 1, 0, 1]
    ranked = rank_spectra_by_peak_max(col, t2s, pos, ["100", "101", "102"])
    assert ranked == [(101, 5.0), (100, 2.0)]                    # spectrum 102 never fires on a peak


def test_pick_demo_spectrum_collision_and_beam_go_to_the_next():
    ranked = [(HF_TEST_OFFSET + r, a) for r, a in ((10, 9.0), (11, 8.0), (12, 7.0))]
    pick, passed = pick_demo_spectrum(ranked, {10}, lambda r: r != 11)
    assert pick == {"row": 12, "rank": 3, "activation": 7.0}
    assert [(p["row"], p["reason"]) for p in passed] == [(10, "already chosen"), (11, "beam did not complete")]
    with pytest.raises(RuntimeError):
        pick_demo_spectrum(ranked, {10, 11, 12}, lambda r: True)


def test_peak_rho_average_ranks_and_constant():
    assert peak_rho([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert peak_rho([1, 2, 3], [1, 1, 2]) == pytest.approx(0.8660254, rel=1e-6)   # tied |e| -> average ranks
    assert math.isnan(peak_rho([1, 2, 3], [5, 5, 5]))


def test_peak_recall_exclusion_ties_mass_and_chance():
    assert peak_recall(list(range(8)), [1.0] * 8, [1.0] * 8) is None             # L0 <= 8
    feats = list(range(10))
    act = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]                                        # A8 = features 0..7
    e = [0, 0, 0, 0, 0, 0, 0, 5, 4, 3]                                           # T3 = 7, 8, 9
    r = peak_recall(feats, act, e)
    assert r["recall"] == pytest.approx(1 / 3) and r["mass_recall"] == pytest.approx(5 / 12)
    assert r["chance"] == pytest.approx(0.8)
    # ties in activation broken by feature id: features 8 and 9 tie with 7 at the cut
    r2 = peak_recall(feats, [1] * 10, [0, 0, 0, 0, 0, 0, 0, 0, 2, 1])
    assert r2["recall"] == pytest.approx(1 / 3)                                  # T3 = 8, 9, 0; A8 = 0..7


def test_variance_split_hand_computed():
    v = variance_split([np.array([1.0, 3.0]), np.array([5.0])])
    # spectrum mean 3; peak means 2 and 5: between = 2*1 + 1*4 = 6; within = 2; total = 8
    assert (v["between"], v["within"], v["total"], v["P"], v["N"]) == (6.0, 2.0, 8.0, 2, 3)


def _toy_peaks(seed):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(1, 6):
        n = int(rng.integers(1, 12))
        out.append({"token": t, "features": list(range(n)), "activation": rng.random(n).tolist(),
                    "delta_total": (rng.standard_normal(n) * 0.1).tolist()})
    return out


def test_screen_summary_is_a_ratio_of_spectrum_sums_and_deterministic():
    per = [screen_spectrum_sums(_toy_peaks(s)) for s in range(6)]
    a = screen_layer_summary(per, n_boot=500)
    b = screen_layer_summary(per, n_boot=500)
    assert a == b
    rhos = [p["rho"] for d in per for p in d["peaks"] if p["l0"] >= 3 and not math.isnan(p["rho"])]
    assert a["a_mean_rho"]["estimate"] == pytest.approx(np.mean(rhos))
    assert a["a_mean_rho"]["ci_low"] <= a["a_mean_rho"]["estimate"] <= a["a_mean_rho"]["ci_high"]
    p_sum = sum(d["sums"]["p_minus_1"] for d in per)
    n_sum = sum(d["sums"]["n_minus_1"] for d in per)
    assert a["c_null_reference"]["estimate"] == pytest.approx(p_sum / n_sum)
    n_le8 = sum(1 for d in per for p in d["peaks"] if p["l0"] <= 8)
    assert a["b_n_excluded_l0_le_8"] == n_le8


def test_group_peaks_separates_latent_and_joint():
    table = [{"token": 0, "kind": "single", "feature": 4, "activation": 1.0, "delta_total": 0.1},
             {"token": 0, "kind": "joint", "feature": -1, "activation": 1.0, "delta_total": 0.1},
             {"token": 2, "kind": "single", "feature": 1, "activation": 2.0, "delta_total": -0.3},
             {"token": 2, "kind": "single", "feature": 3, "activation": 1.0, "delta_total": 0.2}]
    peaks, latent = group_peaks(table)
    assert [p["token"] for p in peaks] == [2] and peaks[0]["features"] == [1, 3]
    assert latent["features"] == [4]


SWEEP = REPO / "sae_pipeline_outputs" / "analysis" / "v1_4c" / "part_b_token_sweep.csv"


@pytest.mark.skipif(not SWEEP.is_file(), reason="local V1-4c outputs not present")
def test_v1_4c_exploratory_numbers_reproduce():
    ex = v1_4c_exploratory(SWEEP)
    assert ex["n_outside_top8"] == 8 and ex["activation_ranks_top3"] == [4, 5, 6] and ex["n_tokens"] == 13
