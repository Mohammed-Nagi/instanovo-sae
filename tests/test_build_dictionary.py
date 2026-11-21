"""Tests for build_dictionary.py.

Run from the repo root: ``PYTHONPATH=src uv run python -m pytest tests``.

The synthetic tree has 6 features over 3 concepts, with counts chosen so that
every derived number can be checked by hand:

    n_tokens = 1000, firing_count = [10, 0, 20, 5, 0, 8]  -> live: 0, 2, 3, 5
    concept token counts: A = 100, B = 50, is_latent_token = 30
    (feature, concept, n_co): (0,A,8) (0,B,2) (2,latent,15) (2,A,10) (2,B,5)
                              (3,B,5) (5,latent,1; not BH-significant)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

import build_dictionary as bd
from schema import DICTIONARY_SCHEMA_VERSION, EVAL_SCHEMA_VERSION

REPO = Path(__file__).resolve().parent.parent

N_TOKENS = 1000
FIRING = [10, 0, 20, 5, 0, 8]
CONCEPT_COUNT = {"A": 100, "B": 50, "is_latent_token": 30}


def _pair(feat: int, concept: str, n_co: int, f1_dom: float, lift: float = 2.0,
          significant: bool = True) -> dict:
    f1 = 2 * n_co / (FIRING[feat] + CONCEPT_COUNT[concept])
    return {"feature_idx": feat, "concept": concept, "f1": f1, "f1_dom": f1_dom, "lift": lift,
            "n_co": n_co, "p_value_bh_significant": significant}


PAIRS = [
    _pair(0, "A", 8, 0.5, lift=3.0), _pair(0, "B", 2, 0.1),
    _pair(2, "is_latent_token", 15, 0.4), _pair(2, "A", 10, 0.6), _pair(2, "B", 5, 0.0),
    _pair(3, "B", 5, 0.2),
    _pair(5, "is_latent_token", 1, 0.05, significant=False),
]


def _ablation(feat: int, d: float) -> dict:
    return {"feature_idx": feat, "mean_delta_ce": d, "selectivity": 0.0, "f1_dom": 0.1, "lift": 2.0}


def _concept(entries: list[dict], group: list[int], n_spectra: int = 5000) -> dict:
    return {"per_feature_ablation": entries, "top_n_features": group,
            "causal": {"selectivity": 0.0, "n_spectra": n_spectra}}


def _report() -> dict:
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "config": {"ablation_spectra": 5000},
        "phase_1_2": {"firing_count": FIRING, "n_tokens": N_TOKENS},
        "phase_8": {"per_concept": {
            "A": _concept([_ablation(0, 0.01), _ablation(2, 0.03)], group=[3]),
            "B": _concept([_ablation(2, 0.03)], group=[]),
        }},
    }


def _descriptions(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    return pd.DataFrame({
        "feature_idx": [r[0] for r in rows], "stratum": "concept",
        "pearson_r": [r[2] for r in rows], "n_predicted": 12, "summary": [r[1] for r in rows],
    })


@pytest.fixture
def tree(tmp_path) -> Path:
    ev = tmp_path / "sae" / "layer_2" / "seed_0" / "eval"
    ev.mkdir(parents=True)
    (ev / "report.json").write_text(json.dumps(_report()))
    assoc = pd.DataFrame(PAIRS)
    assoc.to_csv(ev / "feature_label_associations.csv", index=False)
    n_sig = assoc[assoc["p_value_bh_significant"]].groupby("feature_idx").size()
    pd.DataFrame({
        "feature_idx": range(6), "firing_rate": [c / N_TOKENS for c in FIRING],
        "max_f1_dom": 0.0, "best_concept": "",
        "n_significant_concepts": [int(n_sig.get(i, 0)) for i in range(6)],
        "unexplained_mass_fraction": [0.5, "", "", 0.9, "", 0.2], "unexplained_enrichment": "",
    }).to_csv(ev / "per_feature_stats.csv", index=False)
    for tag, rows in {"unblinded_v1": [(0, "u0", 0.5), (2, "u2", 0.4)],
                      "blinded_v1": [(2, "b2", 0.3)]}.items():
        run = tmp_path / "interpret" / "layer_2" / tag
        run.mkdir(parents=True)
        frame = _descriptions(rows)
        if tag == "blinded_v1":
            frame["blind"] = True  # interpret.py writes this column; pre-V1-2 runs lack it
        frame.to_csv(run / "feature_descriptions.csv", index=False)
    return tmp_path


def _build(root: Path, **kwargs) -> pd.DataFrame:
    return bd.build_layer(root, 2, **kwargs).to_pandas().set_index("feature_idx")


def _rewrite_csv(path: Path, edit) -> None:
    frame = pd.read_csv(path)
    edit(frame)
    frame.to_csv(path, index=False)


# --- association columns -----------------------------------------------------------


def test_only_live_features_get_rows(tree):
    assert list(_build(tree).index) == [0, 2, 3, 5]


def test_precision_recall_and_ranking_match_hand_computed_counts(tree):
    d = _build(tree)
    # feature 0: A (F1dom 0.5) before B (0.1); precision = n_co / firing_count (10)
    assert (d.loc[0, "concept_1"], d.loc[0, "concept_2"]) == ("A", "B")
    assert d.loc[0, "precision_1"] == pytest.approx(8 / 10)
    assert d.loc[0, "recall_1"] == pytest.approx(8 / 100)
    assert d.loc[0, "precision_2"] == pytest.approx(2 / 10)
    assert d.loc[0, "recall_2"] == pytest.approx(2 / 50)
    assert d.loc[0, "lift_1"] == 3.0 and d.loc[0, "f1_dom_1"] == 0.5
    assert pd.isna(d.loc[0, "concept_3"]) and pd.isna(d.loc[0, "recall_3"])
    # feature 2: A (0.6), latent (0.4); B has F1dom 0 and is left out
    assert (d.loc[2, "concept_1"], d.loc[2, "concept_2"]) == ("A", "is_latent_token")
    assert d.loc[2, "recall_2"] == pytest.approx(15 / 30)
    assert pd.isna(d.loc[2, "concept_3"])


def test_features_without_a_significant_association(tree):
    d = _build(tree)
    assert list(d["has_significant_association"]) == [True, True, True, False]
    assert d.loc[5, "n_significant_concepts"] == 0
    assert pd.isna(d.loc[5, "concept_1"]) and pd.isna(d.loc[5, "precision_1"])


def test_latent_token_fraction_only_when_the_pair_is_significant(tree):
    d = _build(tree)
    assert d.loc[2, "latent_token_fraction"] == pytest.approx(15 / 20)
    assert pd.isna(d.loc[0, "latent_token_fraction"])
    assert pd.isna(d.loc[5, "latent_token_fraction"])  # pair exists but is not BH-significant


def test_unexplained_mass_fraction_is_null_when_blank(tree):
    d = _build(tree)
    assert d.loc[0, "unexplained_mass_fraction"] == pytest.approx(0.5)
    assert pd.isna(d.loc[2, "unexplained_mass_fraction"])


def test_inconsistent_concept_counts_fail_loudly(tree):
    ev = tree / "sae" / "layer_2" / "seed_0" / "eval"

    def corrupt(frame):
        frame.loc[(frame.feature_idx == 2) & (frame.concept == "A"), "f1"] *= 2

    _rewrite_csv(ev / "feature_label_associations.csv", corrupt)
    with pytest.raises(ValueError, match="concept counts"):
        _build(tree)


def test_firing_rate_disagreement_fails_loudly(tree):
    ev = tree / "sae" / "layer_2" / "seed_0" / "eval"
    _rewrite_csv(ev / "per_feature_stats.csv", lambda f: f.__setitem__("firing_rate", 0.5))
    with pytest.raises(ValueError, match="firing_rate"):
        _build(tree)


def test_significant_pair_count_disagreement_fails_loudly(tree):
    ev = tree / "sae" / "layer_2" / "seed_0" / "eval"
    _rewrite_csv(ev / "per_feature_stats.csv",
                 lambda f: f.__setitem__("n_significant_concepts", 7))
    with pytest.raises(ValueError, match="n_significant_concepts"):
        _build(tree)


# --- causal columns ----------------------------------------------------------------


def test_causal_measured_vs_untested(tree):
    d = _build(tree)
    assert list(d["causal_measured"]) == [True, True, False, False]
    assert d.loc[0, "phase8_delta_ce"] == pytest.approx(0.01)
    assert d.loc[2, "phase8_delta_ce"] == pytest.approx(0.03)
    assert d.loc[0, "causal_provenance"] == "SAE-patched, mean over 5,000 spectra, no CI"
    assert d.loc[0, "phase8_n_spectra"] == 5000
    # feature 3 is only in a group ablation: untested on its own, never zero
    assert d.loc[3, "causal_provenance"] == "untested"
    assert pd.isna(d.loc[3, "phase8_delta_ce"]) and pd.isna(d.loc[3, "phase8_n_spectra"])


def test_spectrum_count_comes_from_the_report(tree):
    ev = tree / "sae" / "layer_2" / "seed_0" / "eval"
    report = json.loads((ev / "report.json").read_text())
    report["config"]["ablation_spectra"] = 1234
    for info in report["phase_8"]["per_concept"].values():
        info["causal"]["n_spectra"] = 1234
    (ev / "report.json").write_text(json.dumps(report))
    assert _build(tree).loc[0, "causal_provenance"] == "SAE-patched, mean over 1,234 spectra, no CI"


def test_differing_spectrum_counts_fail_loudly(tree):
    ev = tree / "sae" / "layer_2" / "seed_0" / "eval"
    report = json.loads((ev / "report.json").read_text())
    report["phase_8"]["per_concept"]["B"]["causal"]["n_spectra"] = 100
    (ev / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="spectrum counts"):
        _build(tree)


# --- descriptions ------------------------------------------------------------------


def test_unblinded_only_carries_its_provenance_and_never_beats_baseline(tree):
    d = _build(tree)
    assert d.loc[0, "description"] == "u0" and not d.loc[0, "description_blinded"]
    assert d.loc[0, "description_run_tag"] == "unblinded_v1"
    assert d.loc[0, "description_provenance"] == (
        "written with access to annotation labels and the true peptide")
    assert d.loc[0, "description_r"] == pytest.approx(0.5)
    assert d.loc[0, "description_n_predicted"] == 12
    assert d["description_beats_baseline"].isna().all()
    assert d["description_shuffled_null_r"].isna().all()
    assert d["unblinded_description"].isna().all()  # nothing secondary without a blinded primary
    # features without a description
    assert pd.isna(d.loc[3, "description"]) and not d.loc[3, "description_blinded"]


def test_blinded_description_takes_precedence_and_unblinded_is_secondary(tree):
    d = _build(tree, blinded_run="blinded_v1")
    assert d.loc[2, "description"] == "b2" and d.loc[2, "description_blinded"]
    assert d.loc[2, "description_run_tag"] == "blinded_v1"
    assert d.loc[2, "description_provenance"] == bd.BLINDED_PROVENANCE
    assert d.loc[2, "unblinded_description"] == "u2"
    assert d.loc[2, "unblinded_description_r"] == pytest.approx(0.4)
    assert d.loc[2, "unblinded_description_run_tag"] == "unblinded_v1"
    assert d.loc[2, "unblinded_description_provenance"] == bd.UNBLINDED_PROVENANCE
    # feature 0 has only an unblinded description: it stays primary, flagged unblinded
    assert d.loc[0, "description"] == "u0" and not d.loc[0, "description_blinded"]
    assert pd.isna(d.loc[0, "unblinded_description"])
    assert d["description_beats_baseline"].isna().all()


def test_blinded_run_alone(tree):
    d = _build(tree, unblinded_run=None, blinded_run="blinded_v1")
    assert list(d["description_blinded"]) == [False, True, False, False]
    assert d["unblinded_description"].isna().all()


def test_blind_run_fills_the_baseline_columns(tree):
    run = tree / "interpret" / "layer_2" / "blinded_v1" / "feature_descriptions.csv"

    def add(frame):
        frame["shuffled_null_p95"] = 0.21
        frame["description_beats_baseline"] = True
    _rewrite_csv(run, add)
    d = _build(tree, blinded_run="blinded_v1")
    assert d.loc[2, "description_shuffled_null_r"] == pytest.approx(0.21)
    assert d.loc[2, "description_beats_baseline"] == True
    # feature 0's primary is unblinded: no baseline, never "beats"
    assert pd.isna(d.loc[0, "description_shuffled_null_r"])
    assert pd.isna(d.loc[0, "description_beats_baseline"])


def test_blind_run_with_undefined_baseline_stays_null(tree):
    run = tree / "interpret" / "layer_2" / "blinded_v1" / "feature_descriptions.csv"

    def add(frame):
        frame["shuffled_null_p95"] = float("nan")
        frame["description_beats_baseline"] = None
    _rewrite_csv(run, add)
    d = _build(tree, blinded_run="blinded_v1")
    assert pd.isna(d.loc[2, "description_shuffled_null_r"])
    assert pd.isna(d.loc[2, "description_beats_baseline"])


def test_unblinded_run_baseline_columns_are_ignored(tree):
    run = tree / "interpret" / "layer_2" / "unblinded_v1" / "feature_descriptions.csv"

    def add(frame):
        frame["blind"] = False
        frame["shuffled_null_p95"] = 0.1
        frame["description_beats_baseline"] = True
    _rewrite_csv(run, add)
    d = _build(tree)
    assert d["description_beats_baseline"].isna().all()
    assert d["description_shuffled_null_r"].isna().all()


def test_runs_passed_in_the_wrong_role_raise(tree):
    with pytest.raises(ValueError, match="passed as unblinded"):
        _build(tree, unblinded_run="blinded_v1")
    with pytest.raises(ValueError, match="passed as blinded"):
        _build(tree, unblinded_run=None, blinded_run="unblinded_v1")


def test_missing_named_run_raises(tree):
    with pytest.raises(FileNotFoundError, match="no_such_run"):
        _build(tree, blinded_run="no_such_run")
    with pytest.raises(FileNotFoundError):
        _build(tree, unblinded_run="also_missing")


def test_description_of_a_dead_feature_raises(tree):
    run = tree / "interpret" / "layer_2" / "unblinded_v1" / "feature_descriptions.csv"
    _descriptions([(1, "dead", 0.1)]).to_csv(run, index=False)  # feature 1 never fired
    with pytest.raises(ValueError, match="never fired"):
        _build(tree)


def test_empty_summary_raises(tree):
    run = tree / "interpret" / "layer_2" / "unblinded_v1" / "feature_descriptions.csv"
    _descriptions([(0, "", 0.1)]).to_csv(run, index=False)
    with pytest.raises(ValueError, match="empty summary"):
        _build(tree)


def test_missing_r_stays_null_but_the_description_is_kept(tree):
    run = tree / "interpret" / "layer_2" / "unblinded_v1" / "feature_descriptions.csv"
    _descriptions([(0, "u0", float("nan"))]).to_csv(run, index=False)
    d = _build(tree)
    assert d.loc[0, "description"] == "u0" and pd.isna(d.loc[0, "description_r"])


# --- output ------------------------------------------------------------------------


def test_main_writes_parquet_with_schema_metadata_and_is_idempotent(tree):
    bd.main(["--root", str(tree), "--layers", "2"])
    out = bd.dictionary_path(tree, 2)
    first = pq.read_table(out)
    meta = {k.decode(): v.decode() for k, v in first.schema.metadata.items()}
    assert meta["schema_version"] == str(DICTIONARY_SCHEMA_VERSION)
    assert meta["layer"] == "2" and meta["n_live_features"] == "4"
    assert meta["unblinded_run"] == "unblinded_v1" and meta["blinded_run"] == ""
    assert not any(p.suffix == ".tmp" for p in out.parent.iterdir())
    bd.main(["--root", str(tree), "--layers", "2"])
    assert pq.read_table(out).equals(first)


def test_main_accepts_none_for_description_runs(tree):
    bd.main(["--root", str(tree), "--layers", "2", "--unblinded-run", "none"])
    d = pq.read_table(bd.dictionary_path(tree, 2)).to_pandas()
    assert d["description"].isna().all()


def test_no_hub_flag_or_trust_columns(tree):
    columns = set(_build(tree).reset_index().columns)
    assert not {c for c in columns if "hub" in c or "trust" in c or "tier" in c}


# --- local data --------------------------------------------------------------------

LOCAL_ROOT = REPO / "sae_pipeline_outputs"
FINDING_2_HUBS = {2: [7522, 11907, 12077], 4: [11376], 6: [9282, 10689], 8: [2481]}


@pytest.mark.parametrize("layer", sorted(FINDING_2_HUBS))
def test_local_layer_has_every_live_feature_and_the_finding_2_hubs_measured(layer):
    report_file = bd.eval_dir(LOCAL_ROOT, layer, bd.DEFAULT_SEED) / "report.json"
    if not report_file.is_file() or not (report_file.parent / "feature_label_associations.csv").is_file():
        pytest.skip("local pipeline outputs not present")
    table = bd.build_layer(LOCAL_ROOT, layer, unblinded_run=None)
    report = json.loads(report_file.read_text())
    n_alive = sum(1 for c in report["phase_1_2"]["firing_count"] if c > 0)
    assert table.num_rows == n_alive
    d = table.to_pandas().set_index("feature_idx")
    for feat in FINDING_2_HUBS[layer]:
        assert d.loc[feat, "causal_measured"] and d.loc[feat, "phase8_delta_ce"] > 0.01
