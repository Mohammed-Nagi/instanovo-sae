"""Tests for build_dictionary.py.

Run from the repo root: ``PYTHONPATH=src uv run python -m pytest tests``.

The synthetic tree has 6 features over 3 concepts, with counts chosen so that
every derived number can be checked by hand:

    n_tokens = 1000, firing_count = [10, 0, 20, 5, 0, 8]  -> live: 0, 2, 3, 5
    concept token counts: A = 100, B = 50, is_latent_token = 30
    (feature, concept, n_co): (0,A,8) (0,B,2) (2,latent,15) (2,A,10) (2,B,5)
                              (3,B,5) (5,latent,1; not BH-significant)
    concept reliability: A meets the display rule (chance 0.1), B does not
    (chance 0.3), is_latent_token is structural (no chance fraction)
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


def _reliability_row(concept: str, chance: float, meets: bool, reasons: str = "") -> dict:
    return {"concept": concept, "display_name": concept, "label_classes": "b", "dependence": "peptide-dependent",
            "positives": 100, "determined_fraction": 0.0, "chance_fraction": chance, "ci_low": chance - 0.05,
            "ci_high": chance + 0.05, "internal_end_positives": 0, "immonium_bug_positives": 0,
            "immonium_bug_missed": 0, "bug_share": 0.0, "meets_display_rule": meets, "reasons": reasons,
            "note": ""}


RELIABILITY = [_reliability_row("A", 0.1, True), _reliability_row("B", 0.3, False, "chance fraction 0.30 > 0.2"),
               dict(_reliability_row("is_latent_token", float("nan"), False, "structural (latent token)"),
                    dependence="structural (latent token)", determined_fraction=float("nan"))]


def _descriptions(rows: list[tuple[int, str, float]], blind: bool = False) -> pd.DataFrame:
    # interpret.py writes the `blind` column for every run (build_dictionary requires it).
    return pd.DataFrame({
        "feature_idx": [r[0] for r in rows], "stratum": "concept",
        "pearson_r": [r[2] for r in rows], "n_predicted": 12, "summary": [r[1] for r in rows],
        "blind": blind,
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
    audit = tmp_path / "analysis" / "label_audit"
    audit.mkdir(parents=True)
    pd.DataFrame(RELIABILITY).to_csv(audit / "concept_reliability.csv", index=False)
    for tag, rows in {"unblinded_v1": [(0, "u0", 0.5), (2, "u2", 0.4)],
                      "blinded_v1": [(2, "b2", 0.3)]}.items():
        run = tmp_path / "interpret" / "layer_2" / tag
        run.mkdir(parents=True)
        _descriptions(rows, blind=tag == "blinded_v1").to_csv(run / "feature_descriptions.csv",
                                                              index=False)
    return tmp_path


def _build(root: Path, **kwargs) -> pd.DataFrame:
    # The fixture's unblinded run, unless a test names the runs itself.
    kwargs.setdefault("unblinded_run", "unblinded_v1")
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


# --- concept reliability (V1-3b) -----------------------------------------------------


def test_top_concepts_carry_their_reliability(tree):
    d = _build(tree)
    assert d.loc[0, "chance_fraction_1"] == pytest.approx(0.1) and d.loc[0, "meets_display_rule_1"] == True
    assert d.loc[0, "chance_fraction_2"] == pytest.approx(0.3) and d.loc[0, "meets_display_rule_2"] == False
    assert pd.isna(d.loc[2, "chance_fraction_2"]) and d.loc[2, "meets_display_rule_2"] == False  # latent
    assert pd.isna(d.loc[0, "meets_display_rule_3"])                   # no third concept
    assert pd.isna(d.loc[5, "meets_display_rule_1"])                   # no significant association


def test_reliable_associations_use_only_display_rule_concepts(tree):
    d = _build(tree)
    assert list(d["n_reliable_associations"]) == [1, 1, 0, 0]
    assert d.loc[0, "reliable_concept_1"] == "A" and d.loc[0, "reliable_f1_dom_1"] == 0.5
    assert d.loc[0, "reliable_precision_1"] == pytest.approx(8 / 10)
    assert d.loc[0, "reliable_recall_1"] == pytest.approx(8 / 100)
    assert d.loc[0, "reliable_lift_1"] == 3.0
    assert d.loc[0, "reliable_chance_fraction_1"] == pytest.approx(0.1)
    assert pd.isna(d.loc[0, "reliable_concept_2"])
    # feature 2: A (0.6) is reliable; the latent concept ranked second is not
    assert d.loc[2, "reliable_concept_1"] == "A" and pd.isna(d.loc[2, "reliable_concept_2"])
    # feature 3's only association (B) fails the rule: no reliable association recorded
    assert d.loc[3, "concept_1"] == "B" and pd.isna(d.loc[3, "reliable_concept_1"])
    assert pd.isna(d.loc[3, "reliable_f1_dom_1"])


def test_reliable_ranking_follows_f1dom(tree):
    path = tree / "analysis" / "label_audit" / "concept_reliability.csv"
    _rewrite_csv(path, lambda f: f.__setitem__("meets_display_rule", True))
    d = _build(tree)
    assert (d.loc[0, "reliable_concept_1"], d.loc[0, "reliable_concept_2"]) == ("A", "B")
    assert (d.loc[2, "reliable_concept_1"], d.loc[2, "reliable_concept_2"]) == ("A", "is_latent_token")
    assert d.loc[3, "reliable_concept_1"] == "B" and list(d["n_reliable_associations"]) == [2, 2, 1, 0]


def test_metadata_carries_the_reliability_records_and_rule(tree):
    import analysis
    table = bd.build_layer(tree, 2, unblinded_run=None)
    meta = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    records = {r["concept"]: r for r in json.loads(meta["concept_reliability"])}
    assert set(records) == {"A", "B", "is_latent_token"}
    assert records["A"]["meets_display_rule"] is True and records["A"]["chance_fraction"] == pytest.approx(0.1)
    assert records["is_latent_token"]["chance_fraction"] is None
    assert records["B"]["reasons"] == "chance fraction 0.30 > 0.2"
    assert meta["display_rule"] == analysis.DISPLAY_RULE
    assert meta["concept_reliability_source"].startswith("analysis/label_audit/concept_reliability.csv sha256 ")
    assert "q = " in meta["association_source"] and "1,000 tokens" in meta["association_source"]


def test_missing_reliability_table_raises(tree):
    (tree / "analysis" / "label_audit" / "concept_reliability.csv").unlink()
    with pytest.raises(FileNotFoundError, match="concept reliability"):
        _build(tree)


def test_concept_without_a_reliability_record_raises(tree):
    path = tree / "analysis" / "label_audit" / "concept_reliability.csv"
    pd.DataFrame(RELIABILITY[:1] + RELIABILITY[2:]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="without a reliability record.*'B'"):
        _build(tree)


def test_reliability_table_path_can_be_given(tree, tmp_path_factory):
    other = tmp_path_factory.mktemp("rel") / "table.csv"
    pd.DataFrame([dict(r, meets_display_rule=False) for r in RELIABILITY]).to_csv(other, index=False)
    d = bd.build_layer(tree, 2, concept_reliability=other).to_pandas()
    assert d["reliable_concept_1"].isna().all() and (d["n_reliable_associations"] == 0).all()


# --- causal columns ----------------------------------------------------------------


def test_causal_measured_vs_untested(tree):
    d = _build(tree)
    assert list(d["causal_measured"]) == [True, True, False, False]
    assert d.loc[0, "phase8_delta_ce"] == pytest.approx(0.01)
    assert d.loc[2, "phase8_delta_ce"] == pytest.approx(0.03)
    assert d.loc[0, "causal_provenance"] == (
        "SAE-patched, mean over 5,000 spectra, no CI; "
        "targets ignore modifications for ~10% of spectra (Finding 5)")
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
    assert _build(tree).loc[0, "causal_provenance"] == (
        "SAE-patched, mean over 1,234 spectra, no CI; "
        "targets ignore modifications for ~10% of spectra (Finding 5)")


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


@pytest.mark.parametrize("tag, role", [("unblinded_v1", "unblinded_run"), ("blinded_v1", "blinded_run")])
def test_run_without_blind_column_raises(tree, tag, role):
    run = tree / "interpret" / "layer_2" / tag / "feature_descriptions.csv"
    _rewrite_csv(run, lambda frame: frame.drop(columns="blind", inplace=True))
    kwargs = {"unblinded_run": None, role: tag}
    with pytest.raises(ValueError, match="no 'blind' column"):
        _build(tree, **kwargs)


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
    # Both description runs default to none: nothing is described unless named.
    assert meta["unblinded_run"] == "" and meta["blinded_run"] == ""
    assert first.column("description").null_count == first.num_rows
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


@pytest.mark.parametrize("tag", ["archive", "Archive", "archive/unblinded_v1",
                                 "archive\\unblinded_v1", ".."])
def test_no_run_tag_reads_from_the_archive(tree, tag):
    archived = tree / "interpret" / "layer_2" / "archive" / "unblinded_v1"
    archived.mkdir(parents=True)
    (tree / "interpret" / "layer_2" / "unblinded_v1" / "feature_descriptions.csv").rename(
        archived / "feature_descriptions.csv")
    with pytest.raises(ValueError, match="archive"):
        bd.build_layer(tree, 2, unblinded_run=None, blinded_run=tag)
    with pytest.raises(ValueError, match="archive"):
        bd.build_layer(tree, 2, unblinded_run=tag, blinded_run=None)
    with pytest.raises(SystemExit):
        bd.main(["--root", str(tree), "--layers", "2", "--unblinded-run", tag])


# --- V1-5: feature examples and fit reference (--examples) -----------------------------

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import torch  # noqa: E402

from evaluate import ChunkStream  # noqa: E402
from inspector.dictionary import FeatureExamples, FitReference  # noqa: E402
from inspector.runtime import fve as runtime_fve  # noqa: E402
from schema import (  # noqa: E402
    ANNOTATION_SCHEMA_VERSION,
    EXAMPLES_SCHEMA_VERSION,
    EXTRACT_SCHEMA_VERSION,
    FIT_REFERENCE_SCHEMA_VERSION,
    SAE_SCHEMA_VERSION,
)
from train import SparseAutoencoder  # noqa: E402

TOY_D_MODEL, TOY_D_DICT = 5, 7
TOY_SPECTRA = {0: [3, 6, 2, 5], 52: [4, 1, 7]}          # tokens per spectrum (latent + peaks)


def _toy_sae(seed: int = 0) -> SparseAutoencoder:
    torch.manual_seed(seed)
    sae = SparseAutoencoder(TOY_D_MODEL, TOY_D_DICT, k=2, k_aux=0)
    sae.b_enc.data = torch.randn(TOY_D_DICT) * 0.3
    sae.b_dec.data = torch.randn(TOY_D_MODEL) * 0.1
    sae.jumprelu_threshold = torch.tensor(0.2)
    sae.eval()
    return sae


def _toy_tree(root: Path) -> dict[int, torch.Tensor]:
    """A consolidated two-chunk tree (chunk ids 0 and 52 of 60) with bf16
    activations, as on disk; returns the float32 view of each chunk's activations."""
    (root / "extract" / "chunks").mkdir(parents=True)
    (root / "annotation" / "labels").mkdir(parents=True)
    gen = torch.Generator().manual_seed(1)
    acts_by_chunk = {}
    for ci, lengths in TOY_SPECTRA.items():
        n_tok = sum(lengths)
        t2s = torch.cat([torch.full((n,), s, dtype=torch.long) for s, n in enumerate(lengths)])
        pos = torch.cat([torch.arange(n) for n in lengths])
        acts = (torch.randn(n_tok, TOY_D_MODEL, generator=gen) + 0.5).to(torch.bfloat16)
        acts_by_chunk[ci] = acts.float()
        torch.save({"total_tokens": n_tok, "n_spectra": len(lengths), "token_to_spectrum": t2s,
                    "token_to_position": pos, "spectrum_ids": [str(1000 * ci + s) for s in range(len(lengths))],
                    "peptides": ["PEPTIDE"] * len(lengths), "proforma_strings": ["PEPTIDE"] * len(lengths),
                    "modifications": [[] for _ in lengths], "precursor_charges": torch.full((len(lengths),), 2),
                    "precursor_mzs": torch.full((len(lengths),), 400.0),
                    "mz_arrays": [torch.linspace(100, 900, n - 1) for n in lengths],
                    "intensity_arrays": [torch.ones(n - 1) for n in lengths]},
                   root / "extract" / "chunks" / f"meta_{ci:05d}.pt")
        torch.save({"activations": acts}, root / "extract" / "chunks" / f"acts_L2_{ci:05d}.pt")
        torch.save({"total_tokens": n_tok, "token_labels": torch.zeros(n_tok, 2, dtype=torch.bool)},
                   root / "annotation" / "labels" / f"chunk_{ci:05d}.pt")
    (root / "extract" / "manifest.json").write_text(json.dumps({
        "schema_version": EXTRACT_SCHEMA_VERSION, "n_chunks": len(TOY_SPECTRA),
        "chunks": [{"idx": ci, "meta": f"chunks/meta_{ci:05d}.pt",
                    "activations": {"2": f"chunks/acts_L2_{ci:05d}.pt"}} for ci in TOY_SPECTRA]}))
    (root / "annotation" / "annotation_manifest.json").write_text(json.dumps({
        "schema_version": ANNOTATION_SCHEMA_VERSION, "n_chunks": 60,
        "chunks": [{"idx": i, "path": f"labels/chunk_{i:05d}.pt"} for i in range(60)],
        "registry": {"names": ["a", "b"], "diagnostic": [], "family_of": {}}, "base_rates": {}}))
    return acts_by_chunk


def _toy_stream(root: Path, sae) -> ChunkStream:
    return ChunkStream(extract_dir=root / "extract", annotation_dir=root / "annotation", target_layer=2,
                       sae=sae, device="cpu", batch_size=4, dtype=torch.float32)


def test_spectrum_batches_cut_at_spectrum_boundaries():
    t2s = np.repeat([0, 1, 2, 3], [3, 6, 2, 5])
    assert bd.spectrum_batches(t2s, 8) == [(0, 3), (3, 11), (11, 16)]
    assert bd.spectrum_batches(t2s, 100) == [(0, 16)]
    assert bd.spectrum_batches(t2s, 1) == [(0, 3), (3, 9), (9, 11), (11, 16)]
    with pytest.raises(ValueError, match="contiguous"):
        bd.spectrum_batches(np.array([0, 1, 0]), 4)


@pytest.mark.parametrize("splits", [[16], [1] * 16, [5, 5, 6], [2, 9, 5], [4, 4, 4, 4]])
def test_streamed_fve_equals_runtime_fve_for_any_batching(splits):
    """Batches cut anywhere, spectra straddling them: the accumulated
    per-spectrum FVE equals runtime.fve on the spectrum's rows."""
    torch.manual_seed(3)
    lengths = [3, 6, 2, 5]
    t2s = torch.cat([torch.full((n,), s) for s, n in enumerate(lengths)])
    x = torch.randn(16, 4) + 0.3
    x_hat = x + 0.2 * torch.randn(16, 4)
    acc = bd.SpectrumFitAccumulator(len(lengths), 4)
    start = 0
    for n in splits:
        acc.add(x[start:start + n], x_hat[start:start + n], t2s[start:start + n])
        start += n
    got = acc.fve_centred()
    for s in range(len(lengths)):
        rows = t2s == s
        assert got[s] == pytest.approx(runtime_fve(x[rows], x_hat[rows])[0], abs=1e-12)


def test_fit_accumulator_rejects_a_spectrum_without_tokens():
    acc = bd.SpectrumFitAccumulator(2, 3)
    acc.add(torch.ones(2, 3), torch.zeros(2, 3), torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="no tokens"):
        acc.fve_centred()


def test_spectrum_max_and_argmax_takes_the_first_token_at_the_maximum():
    feats = torch.tensor([[0.0, 1.0], [2.0, 1.0], [2.0, 0.5], [0.0, 0.0], [3.0, 0.0]])
    vmax, arg = bd.spectrum_max_and_argmax(feats, torch.tensor([0, 0, 0, 1, 1]),
                                           torch.tensor([0, 1, 2, 0, 1]), 3)
    assert vmax[:2].tolist() == [[2.0, 1.0], [3.0, 0.0]] and arg[:2].tolist() == [[1, 0], [1, 0]]
    assert arg[2].tolist() == [-1, -1]


@pytest.mark.parametrize("batch_tokens", [1, 7, 4096])
def test_examples_equal_brute_force_top_k(tmp_path, batch_tokens):
    """Per feature: the top-K spectra by their maximum activation over every
    chunk, one per spectrum, with the argmax token; FVE per spectrum as runtime.fve."""
    acts = _toy_tree(tmp_path)
    sae = _toy_sae()
    features = [0, 2, 3, 6]
    k = 3
    res = bd.stream_layer_examples(_toy_stream(tmp_path, sae), sae, features, k=k,
                                   batch_tokens=batch_tokens, log=lambda *_: None)
    ids = {(c, s): sid for c, s, sid, _f in res["fve"]}
    rows = bd.example_rows(2, features, res["topk"], ids)

    brute, fve_expected = [], {}
    for ci, lengths in TOY_SPECTRA.items():
        out = sae.forward_inference(acts[ci])
        start = 0
        for s, n in enumerate(lengths):
            block = out["features"][start:start + n]
            fve_expected[(ci, s)] = runtime_fve(acts[ci][start:start + n], out["x_hat"][start:start + n])[0]
            for f in features:
                v = float(block[:, f].max())
                brute.append((f, v, ci, s, int(block[:, f].argmax())))
            start += n
    assert {(c, s): f for c, s, _sid, f in res["fve"]} == pytest.approx(fve_expected, abs=1e-9)
    n_with_examples = 0
    for f in features:
        expect = sorted([b for b in brute if b[0] == f and b[1] > 0], key=lambda b: -b[1])[:k]
        got = [r for r in rows if r["feature_idx"] == f]
        n_with_examples += bool(got)
        assert [(r["chunk"], r["spectrum_index"], r["token"]) for r in got] == [(b[2], b[3], b[4]) for b in expect]
        assert [r["activation"] for r in got] == pytest.approx([b[1] for b in expect], rel=1e-6)
        assert [r["rank"] for r in got] == list(range(1, len(got) + 1))
        assert len({(r["chunk"], r["spectrum_index"]) for r in got}) == len(got)          # one per spectrum
        for r in got:
            assert r["spectrum_id"] == str(1000 * r["chunk"] + r["spectrum_index"])
            assert r["peak_index"] == (r["token"] - 1 if r["token"] > 0 else None)
    assert n_with_examples >= 2                         # the toy exercises the comparison


def test_fit_quantiles():
    q = bd.fit_quantiles(np.arange(11.0), n=5)
    assert q == [0.0, 2.5, 5.0, 7.5, 10.0]
    assert len(bd.fit_quantiles(np.random.default_rng(0).random(50))) == bd.FIT_QUANTILES


def test_load_example_features_checks_the_lists(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"features": {"2": [1, 5, 9]}}))
    assert bd.load_example_features(p, [2]) == {2: [1, 5, 9]}
    with pytest.raises(ValueError, match="no feature list for layer 4"):
        bd.load_example_features(p, [2, 4])
    p.write_text(json.dumps({"features": {"2": [5, 1]}}))
    with pytest.raises(ValueError, match="sorted and unique"):
        bd.load_example_features(p, [2])


def _toy_checkpoint(root: Path, sae) -> None:
    ckpt = root / "sae" / "layer_2" / "seed_0" / "checkpoint.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"schema_version": SAE_SCHEMA_VERSION, "d_model": TOY_D_MODEL, "d_dict": TOY_D_DICT, "k": 2,
                "k_aux": 0, "W_enc": sae.W_enc.data, "b_enc": sae.b_enc.data, "W_dec": sae.W_dec.data,
                "b_dec": sae.b_dec.data, "jumprelu_threshold": sae.jumprelu_threshold,
                "firing_count": torch.zeros(TOY_D_DICT, dtype=torch.long), "tokens_seen": 0}, ckpt)


def test_build_examples_writes_schema_checked_files_and_resumes(tmp_path, monkeypatch):
    root = tmp_path / "out"
    _toy_tree(root)
    _toy_checkpoint(root, _toy_sae())
    feats_path = root / "analysis" / "v1_5" / "demo_features.json"
    feats_path.parent.mkdir(parents=True)
    feats_path.write_text(json.dumps({"features": {"2": [0, 3, 6]}}))

    def fake_annotate(_root, referenced, log=print):
        return [{"chunk": c, "spectrum_index": s, "spectrum_id": str(1000 * c + s), "precursor_mz": 400.0,
                 "precursor_charge": 2, "label_peptide": "PEPTIDE", "label_proforma": "PEPTIDE", "n_peaks": 1,
                 "mz": [100.0], "intensity": [1.0], "peak_annotations": "[]", "annotation_note": None}
                for c in sorted(referenced) for s in sorted(referenced[c])]
    monkeypatch.setattr(bd, "annotate_referenced", fake_annotate)
    logs: list[str] = []
    bd.build_examples(root, [2], feats_path, log=logs.append)

    table = pq.read_table(bd.examples_path(root, 2))
    meta = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    assert meta["schema_version"] == str(EXAMPLES_SCHEMA_VERSION) and meta["layer"] == "2"
    assert meta["n_spectra"] == "7" and json.loads(meta["chunks"]) == [0, 52]
    assert "Finding 7" in meta["finding_7_note"] and "Finding 8" in meta["label_note"]
    assert meta["activation_note"] == bd.BF16_NOTE and len(meta["sae_sha256_16"]) == 16
    assert any("referenced spectra" in line and " h at " in line for line in logs)

    ref = json.loads(bd.fit_reference_paths(root)[0].read_text())
    assert ref["schema_version"] == FIT_REFERENCE_SCHEMA_VERSION and "Finding 7" in ref["note"]
    assert ref["layers"]["2"]["n_spectra"] == 7 and len(ref["layers"]["2"]["quantiles"]) == bd.FIT_QUANTILES
    csv_lines = bd.fit_reference_paths(root)[1].read_text().splitlines()
    assert csv_lines[0] == "probability,layer_2" and len(csv_lines) == bd.FIT_QUANTILES + 1

    examples = FeatureExamples.load(root, 2)
    assert examples.metadata["layer"] == "2"
    for f in (0, 3, 6):
        assert all(e["spectrum"]["spectrum_id"] == e["spectrum_id"] for e in examples.examples(f))
    assert examples.examples(1) == [] and 1 not in examples
    fit = FitReference.load(root)
    assert 0.0 <= fit.percentile(2, float(np.median(ref["layers"]["2"]["quantiles"]))) <= 100.0

    # resume: the layer files are reused, not recomputed
    before = bd.examples_path(root, 2).stat().st_mtime_ns
    logs.clear()
    bd.build_examples(root, [2], feats_path, log=logs.append)
    assert bd.examples_path(root, 2).stat().st_mtime_ns == before
    assert any("reusing" in line for line in logs)


def test_feature_examples_reject_other_schema(tmp_path):
    base = tmp_path / "dictionary" / "examples"
    base.mkdir(parents=True)
    pq.write_table(pa.table({"x": [1]}).replace_schema_metadata({"schema_version": "999", "layer": "2"}),
                   base / "layer_2.parquet")
    with pytest.raises(ValueError, match="examples schema"):
        FeatureExamples.load(tmp_path, 2)
    with pytest.raises(FileNotFoundError):
        FeatureExamples.load(tmp_path, 4)


def test_memory_check_logs_and_guards():
    from build_dictionary import MemoryGuardError, memory_check
    assert memory_check(None).startswith("RSS ")
    assert "peak" in memory_check(1000.0)
    with pytest.raises(MemoryGuardError, match="exceeds --max-rss-gb"):
        memory_check(1e-6)
