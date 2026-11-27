"""build_dictionary.py -- the feature dictionary: one parquet per layer, one row per live feature.

The dictionary is the contract between the research pipeline and the inspector
tool. It is built only from files the pipeline already wrote -- each layer's
``report.json`` and its two per-feature CSVs under ``eval/``, and the
description runs under ``interpret/layer_{L}/<run_tag>/`` -- so it needs neither
torch nor the activation chunks.

A feature is *live* when it fired at least once (``phase_1_2.firing_count``).

Columns (``<k>`` = 1..3)
    identity     layer, feature_idx, firing_count, firing_rate
    association  n_significant_concepts, has_significant_association,
                 concept_<k>, f1_dom_<k>, precision_<k>, recall_<k>, lift_<k>
                 (the three BH-significant concepts with the highest F1dom > 0),
                 chance_fraction_<k>, meets_display_rule_<k> (that concept's
                 reliability record),
                 n_reliable_associations, reliable_concept_<k>,
                 reliable_f1_dom_<k>, reliable_precision_<k>, reliable_recall_<k>,
                 reliable_lift_<k>, reliable_chance_fraction_<k> (the same
                 ranking restricted to concepts meeting the display rule; null
                 reliable_concept_1 = no reliable association recorded),
                 latent_token_fraction (precision of the ``is_latent_token`` pair
                 when that pair is BH-significant, else null),
                 unexplained_mass_fraction (null when the CSV has none)
    causal       causal_measured, phase8_delta_ce, phase8_n_spectra, causal_provenance
    description  description, description_run_tag, description_stratum,
                 description_blinded, description_provenance, description_r,
                 description_n_predicted, description_shuffled_null_r,
                 description_beats_baseline
    secondary    unblinded_description, unblinded_description_r,
                 unblinded_description_run_tag, unblinded_description_provenance

What the columns do and do not claim
    * Associations are Phase 4's: every BH-significant (feature, concept) pair
      over the whole cluster corpus, computed from the annotator's labels, which
      Finding 9 found partly wrong (two labelling bugs) and partly chance. Each
      concept's reliability record comes from the label audit
      (``analysis/label_audit/concept_reliability.csv``, 200 local spectra):
      chance fraction with CI, positives a bug adds or hides, dependence on the
      peptide, a display name for the re-documented m/z-window concepts
      (is_I_ion, is_first_ion). The parquet metadata carries all records
      (``concept_reliability``, JSON), the display rule with its reason
      (``display_rule``) and both sources. The display rule (chance fraction
      <= 0.2, no bug-added positives, peptide-dependent) only decides which
      concepts may be shown as reliable associations; it is not a test, and the
      chance fraction is always shown beside a concept. A reliable association
      is usually weak (F1dom, precision and recall sit beside it; Finding 10).
    * precision = n_co / firing_count. recall = n_co / concept_count. The concept
      counts are not stored anywhere, so they are recovered exactly from the
      stored F1: F1 = 2 n_co / (firing_count + concept_count). Every pair of a
      concept must give the same count (up to float32 rounding); the build fails
      otherwise.
    * Causal fields are Phase 8's single-feature mean delta-CE, SAE-patched (the
      reconstruction error is dropped), over the spectra the report states, with
      no confidence interval, and teacher-forced on the bare sequence, so the
      targets of modified peptides (~10% of spectra) omit the modification;
      the provenance string says so. A feature Phase 8 did not ablate on its own is
      "untested" with a null delta-CE -- not zero and not "unimportant". A
      feature that appears only in a concept's group ablation is untested here.
    * No hub flag, tier, threshold, combined score or ordinal trust level.
      ``has_significant_association``, ``causal_measured`` and
      ``description_blinded`` record which evidence exists; the effect sizes and
      r values sit beside them.
    * A blinded description takes precedence over an unblinded one for the same
      feature, and the unblinded one is then kept in the ``unblinded_*`` fields.
      Every unblinded description carries the provenance "written with access to
      annotation labels and the true peptide".
    * ``description_shuffled_null_r`` is the threshold the blind run compared r
      against: the 95th percentile of every shuffled-description r in that layer
      and run (interpret.py ``shuffled_null_p95``; each feature's own shuffled r
      values stay in the run's CSV). ``description_beats_baseline`` is r above it.
      Both are filled only for a blinded primary description from a run that has
      them, and are null otherwise; ``description_beats_baseline`` is never true
      for an unblinded description.

Usage (from the repo root)::

    PYTHONPATH=src uv run python -m build_dictionary --layers 2 4 6 --blinded-run blind_v1
    PYTHONPATH=src uv run python -m build_dictionary --layers 8     # no descriptions yet

Both runs default to none, so a layer is described only by a run named on the
command line. The current dictionary uses blinded descriptions only. Superseded
runs sit in ``interpret/layer_{L}/archive/`` as provenance; ``schema.check_run_tag``
refuses any tag that would read from there.

Output: ``<root>/dictionary/layer_{L}.parquet``. The build is deterministic and
the write is atomic, so rerunning it is idempotent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from analysis import (
    DEFAULT_ROOT,
    DISPLAY_RULE,
    DEFAULT_SEED,
    LAYERS,
    load_report,
    per_feature_delta_ce,
    report_path,
)
from schema import DICTIONARY_SCHEMA_VERSION, check_run_tag

TOP_CONCEPTS = 3
LATENT_CONCEPT = "is_latent_token"
# float32 rounding in the stored F1 spreads the recovered concept count by ~4e-5.
CONCEPT_COUNT_REL_TOL = 1e-3

UNBLINDED_PROVENANCE = "written with access to annotation labels and the true peptide"
BLINDED_PROVENANCE = "blinded: annotation labels and the true peptide withheld from the prompt"
UNTESTED = "untested"
# Phase 8 (like Phase 7) teacher-forced the dataset's bare `sequence`, so for a
# modified peptide the delta-CE targets are the unmodified residues (STATUS
# Finding 5). Measured on the local part of Phase 8's 5,000 spectra: 208 of
# the 2,048 in chunks 0-1 carry a modification (132/1,024 and 76/1,024).
PHASE8_TARGET_CAVEAT = "targets ignore modifications for ~10% of spectra (Finding 5)"

_RELIABILITY_COLUMNS = ["concept", "display_name", "dependence", "positives", "determined_fraction",
                        "chance_fraction", "ci_low", "ci_high", "internal_end_positives",
                        "immonium_bug_positives", "immonium_bug_missed", "bug_share",
                        "meets_display_rule", "reasons", "note"]
_ASSOC_COLUMNS = ["feature_idx", "concept", "f1", "f1_dom", "lift", "n_co", "p_value_bh_significant"]
_DESCRIPTION_COLUMNS = {"feature_idx", "stratum", "pearson_r", "n_predicted", "summary"}


def eval_dir(root: Path, layer: int, seed: int) -> Path:
    return report_path(root, layer, seed).parent


def description_path(root: Path, layer: int, run_tag: str) -> Path:
    return root / "interpret" / f"layer_{layer}" / check_run_tag(run_tag) / "feature_descriptions.csv"


def dictionary_path(root: Path, layer: int) -> Path:
    return root / "dictionary" / f"layer_{layer}.parquet"


def reliability_path(root: Path) -> Path:
    """The label audit's concept reliability table (analysis.py --label-audit)."""
    return root / "analysis" / "label_audit" / "concept_reliability.csv"


def _causal_provenance(n_spectra: int) -> str:
    return f"SAE-patched, mean over {n_spectra:,} spectra, no CI; {PHASE8_TARGET_CAVEAT}"


# --- inputs ------------------------------------------------------------------------


def live_feature_table(report: dict, stats: pd.DataFrame, layer: int) -> pd.DataFrame:
    """Per-feature counts for the live features, cross-checked against per_feature_stats.csv."""
    firing_count = np.asarray(report["phase_1_2"]["firing_count"], dtype=np.int64)
    n_tokens = int(report["phase_1_2"]["n_tokens"])
    if len(stats) != len(firing_count) or not np.array_equal(
            stats["feature_idx"].to_numpy(), np.arange(len(firing_count))):
        raise ValueError(f"layer {layer}: per_feature_stats.csv does not cover features "
                         f"0..{len(firing_count) - 1} in order")
    rate = firing_count / n_tokens
    if not np.allclose(rate, stats["firing_rate"].to_numpy(), rtol=1e-9, atol=0.0):
        raise ValueError(f"layer {layer}: per_feature_stats.csv firing_rate disagrees with "
                         f"report.json firing_count / n_tokens")
    table = pd.DataFrame({
        "layer": layer,
        "feature_idx": np.arange(len(firing_count)),
        "firing_count": firing_count,
        "firing_rate": rate,
        "n_significant_concepts": stats["n_significant_concepts"].to_numpy(dtype=np.int64),
        "unexplained_mass_fraction": pd.to_numeric(stats["unexplained_mass_fraction"], errors="coerce"),
    })
    return table[table["firing_count"] > 0].reset_index(drop=True)


def concept_counts(assoc: pd.DataFrame, firing_count: np.ndarray, layer: int) -> dict[str, int]:
    """Tokens carrying each concept, recovered from F1 = 2 n_co / (firing_count + concept_count)."""
    pairs = assoc[(assoc["n_co"] > 0) & (assoc["f1"] > 0)]
    implied = 2.0 * pairs["n_co"].to_numpy() / pairs["f1"].to_numpy() - firing_count[
        pairs["feature_idx"].to_numpy()]
    frame = pd.DataFrame({"concept": pairs["concept"].to_numpy(), "implied": implied})
    grouped = frame.groupby("concept")["implied"]
    median, spread = grouped.median(), grouped.max() - grouped.min()
    bad = (spread / median > CONCEPT_COUNT_REL_TOL) | (median <= 0)
    if bad.any():
        raise ValueError(
            f"layer {layer}: concept counts recovered from F1 disagree across features for "
            f"{sorted(bad[bad].index)}; precision/recall would be wrong")
    return {concept: round(value) for concept, value in median.items()}


def top_concepts(assoc: pd.DataFrame, features: np.ndarray, firing_count: np.ndarray,
                 counts: dict[str, int], concepts: set[str] | None = None,
                 prefix: str = "") -> dict[str, pd.Series]:
    """<prefix>concept_k / f1_dom_k / precision_k / recall_k / lift_k columns,
    aligned to ``features``; ``concepts`` restricts the ranking to those concepts."""
    sig = assoc[assoc["significant"] & (assoc["f1_dom"] > 0)]
    if concepts is not None:
        sig = sig[sig["concept"].isin(concepts)]
    sig = sig.copy()
    sig = sig.sort_values(["feature_idx", "f1_dom", "concept"], ascending=[True, False, True],
                          kind="stable")
    sig["position"] = sig.groupby("feature_idx").cumcount()
    sig["precision"] = sig["n_co"] / firing_count[sig["feature_idx"].to_numpy()]
    sig["recall"] = sig["n_co"] / sig["concept"].map(counts).astype(float)
    out: dict[str, pd.Series] = {}
    for k in range(TOP_CONCEPTS):
        rank = sig[sig["position"] == k].set_index("feature_idx").reindex(features)
        out[f"{prefix}concept_{k + 1}"] = rank["concept"]
        for name in ("f1_dom", "precision", "recall", "lift"):
            out[f"{prefix}{name}_{k + 1}"] = rank[name]
    return out


def load_reliability(path: Path, assoc_concepts: set[str], layer: int) -> pd.DataFrame:
    """The concept reliability table, indexed by concept. Every concept in the
    associations must have a record; loud on anything that does not fit."""
    if not path.is_file():
        raise FileNotFoundError(f"concept reliability table not found: {path}; write it with "
                                f"`python -m analysis --label-audit`")
    frame = pd.read_csv(path)
    missing = set(_RELIABILITY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if not frame["concept"].is_unique:
        raise ValueError(f"{path}: duplicate concept rows")
    absent = sorted(assoc_concepts - set(frame["concept"]))
    if absent:
        raise ValueError(f"layer {layer}: concepts without a reliability record in {path}: {absent}")
    flags = frame["meets_display_rule"].astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError(f"{path}: meets_display_rule must be True/False")
    frame["meets_display_rule"] = flags.eq("true")
    for col in ("reasons", "note"):
        frame[col] = frame[col].fillna("").astype(str)
    return frame[_RELIABILITY_COLUMNS].set_index("concept")


def reliability_columns(top: dict[str, pd.Series], assoc: pd.DataFrame, features: np.ndarray,
                        firing_count: np.ndarray, counts: dict[str, int],
                        reliability: pd.DataFrame) -> dict[str, pd.Series]:
    """chance_fraction_k / meets_display_rule_k beside the top 3, and the
    reliable_* ranking over the concepts that meet the display rule."""
    chance = reliability["chance_fraction"]
    meets = reliability["meets_display_rule"]
    reliable = set(meets.index[meets])
    out: dict[str, pd.Series] = {}
    for k in range(1, TOP_CONCEPTS + 1):
        concept = top[f"concept_{k}"]
        out[f"chance_fraction_{k}"] = concept.map(chance)
        out[f"meets_display_rule_{k}"] = concept.map(meets).astype(object).where(concept.notna(), None)
    sig = assoc[assoc["significant"] & (assoc["f1_dom"] > 0) & assoc["concept"].isin(reliable)]
    out["n_reliable_associations"] = sig.groupby("feature_idx").size().reindex(features, fill_value=0)
    rel = top_concepts(assoc, features, firing_count, counts, concepts=reliable, prefix="reliable_")
    out.update(rel)
    for k in range(1, TOP_CONCEPTS + 1):
        out[f"reliable_chance_fraction_{k}"] = rel[f"reliable_concept_{k}"].map(chance)
    return out


def reliability_records(reliability: pd.DataFrame) -> list[dict]:
    """The reliability table as JSON-ready records (NaN -> None), for the metadata."""
    frame = reliability.reset_index()
    records = []
    for rec in frame.to_dict("records"):
        clean = {}
        for key, value in rec.items():
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                value = value.item()
            if isinstance(value, float) and np.isnan(value):
                value = None
            clean[key] = value
        records.append(clean)
    return records


def latent_token_fraction(assoc: pd.DataFrame, features: np.ndarray,
                          firing_count: np.ndarray) -> pd.Series:
    """Precision of the is_latent_token pair where it is BH-significant, else NaN."""
    pairs = assoc[(assoc["concept"] == LATENT_CONCEPT) & assoc["significant"]]
    precision = pairs.set_index("feature_idx")["n_co"] / firing_count[pairs["feature_idx"].to_numpy()]
    return precision.reindex(features)


def causal_columns(report: dict, layer: int, features: np.ndarray) -> dict[str, list]:
    """Phase 8 single-feature delta-CE with provenance, or 'untested'."""
    delta = per_feature_delta_ce(report, layer)
    n_values = {int(info["causal"]["n_spectra"])
                for info in report["phase_8"]["per_concept"].values() if info.get("causal")}
    if len(n_values) != 1:
        raise ValueError(f"layer {layer}: Phase 8 spectrum counts {sorted(n_values)}; need exactly one")
    n_spectra = n_values.pop()
    configured = report.get("config", {}).get("ablation_spectra")
    if configured is not None and int(configured) != n_spectra:
        raise ValueError(f"layer {layer}: config.ablation_spectra {configured} != causal n_spectra "
                         f"{n_spectra}")
    measured = [int(f) in delta for f in features]
    return {
        "causal_measured": measured,
        "phase8_delta_ce": [delta[int(f)] if m else None for f, m in zip(features, measured)],
        "phase8_n_spectra": [n_spectra if m else None for m in measured],
        "causal_provenance": [_causal_provenance(n_spectra) if m else UNTESTED for m in measured],
    }


def load_descriptions(path: Path, live: set[int], layer: int) -> pd.DataFrame:
    """A description run's CSV, indexed by feature. Loud on anything that does not fit."""
    if not path.is_file():
        raise FileNotFoundError(f"layer {layer}: description run not found: {path}")
    frame = pd.read_csv(path)
    missing = _DESCRIPTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if not frame["feature_idx"].is_unique:
        raise ValueError(f"{path}: duplicate feature_idx rows")
    dead = sorted(set(frame["feature_idx"]) - live)
    if dead:
        raise ValueError(f"{path}: describes features that never fired: {dead[:5]}")
    if frame["summary"].isna().any() or (frame["summary"].astype(str).str.strip() == "").any():
        raise ValueError(f"{path}: rows with an empty summary")
    return frame.set_index("feature_idx")


def description_columns(features: np.ndarray, layer: int, unblinded: pd.DataFrame | None,
                        unblinded_tag: str | None, blinded: pd.DataFrame | None,
                        blinded_tag: str | None) -> dict[str, list]:
    """Primary description (blinded wins) plus the unblinded one as a secondary field."""
    cols: dict[str, list] = {name: [] for name in (
        "description", "description_run_tag", "description_stratum", "description_blinded",
        "description_provenance", "description_r", "description_n_predicted",
        "description_shuffled_null_r", "description_beats_baseline",
        "unblinded_description", "unblinded_description_r", "unblinded_description_run_tag",
        "unblinded_description_provenance")}

    def number(row, key):
        value = row[key]
        return None if pd.isna(value) else float(value)

    def flag(row, key):
        if key not in row.index or pd.isna(row[key]):
            return None
        text = str(row[key]).strip().lower()
        if text not in ("true", "false"):
            raise ValueError(f"layer {layer}: {key} must be True/False/blank, got {row[key]!r}")
        return text == "true"

    for feat in features:
        feat = int(feat)
        blind_row = blinded.loc[feat] if blinded is not None and feat in blinded.index else None
        plain_row = unblinded.loc[feat] if unblinded is not None and feat in unblinded.index else None
        primary, tag, is_blind = ((blind_row, blinded_tag, True) if blind_row is not None
                                  else (plain_row, unblinded_tag, False))
        if primary is None:
            for name in ("description", "description_run_tag", "description_stratum",
                         "description_provenance", "description_r", "description_n_predicted"):
                cols[name].append(None)
            cols["description_blinded"].append(False)
            cols["description_shuffled_null_r"].append(None)
            cols["description_beats_baseline"].append(None)
        else:
            cols["description"].append(str(primary["summary"]))
            cols["description_run_tag"].append(tag)
            cols["description_stratum"].append(str(primary["stratum"]))
            cols["description_blinded"].append(is_blind)
            cols["description_provenance"].append(BLINDED_PROVENANCE if is_blind else UNBLINDED_PROVENANCE)
            cols["description_r"].append(number(primary, "pearson_r"))
            n_pred = primary["n_predicted"]
            cols["description_n_predicted"].append(None if pd.isna(n_pred) else int(n_pred))
            has_baseline = is_blind and "shuffled_null_p95" in primary.index
            cols["description_shuffled_null_r"].append(
                number(primary, "shuffled_null_p95") if has_baseline else None)
            cols["description_beats_baseline"].append(
                flag(primary, "description_beats_baseline") if has_baseline else None)
        secondary = plain_row if is_blind else None
        cols["unblinded_description"].append(None if secondary is None else str(secondary["summary"]))
        cols["unblinded_description_r"].append(None if secondary is None
                                               else number(secondary, "pearson_r"))
        cols["unblinded_description_run_tag"].append(None if secondary is None else unblinded_tag)
        cols["unblinded_description_provenance"].append(None if secondary is None
                                                        else UNBLINDED_PROVENANCE)
    return cols


# --- assembly ----------------------------------------------------------------------

_SCHEMA = pa.schema([
    ("layer", pa.int32()), ("feature_idx", pa.int32()), ("firing_count", pa.int64()),
    ("firing_rate", pa.float64()), ("n_significant_concepts", pa.int32()),
    ("has_significant_association", pa.bool_()),
    *[field for k in range(1, TOP_CONCEPTS + 1) for field in (
        (f"concept_{k}", pa.string()), (f"f1_dom_{k}", pa.float64()), (f"precision_{k}", pa.float64()),
        (f"recall_{k}", pa.float64()), (f"lift_{k}", pa.float64()),
        (f"chance_fraction_{k}", pa.float64()), (f"meets_display_rule_{k}", pa.bool_()))],
    ("n_reliable_associations", pa.int32()),
    *[field for k in range(1, TOP_CONCEPTS + 1) for field in (
        (f"reliable_concept_{k}", pa.string()), (f"reliable_f1_dom_{k}", pa.float64()),
        (f"reliable_precision_{k}", pa.float64()), (f"reliable_recall_{k}", pa.float64()),
        (f"reliable_lift_{k}", pa.float64()), (f"reliable_chance_fraction_{k}", pa.float64()))],
    ("latent_token_fraction", pa.float64()), ("unexplained_mass_fraction", pa.float64()),
    ("causal_measured", pa.bool_()), ("phase8_delta_ce", pa.float64()),
    ("phase8_n_spectra", pa.int64()), ("causal_provenance", pa.string()),
    ("description", pa.string()), ("description_run_tag", pa.string()),
    ("description_stratum", pa.string()), ("description_blinded", pa.bool_()),
    ("description_provenance", pa.string()), ("description_r", pa.float64()),
    ("description_n_predicted", pa.int64()), ("description_shuffled_null_r", pa.float64()),
    ("description_beats_baseline", pa.bool_()),
    ("unblinded_description", pa.string()), ("unblinded_description_r", pa.float64()),
    ("unblinded_description_run_tag", pa.string()), ("unblinded_description_provenance", pa.string()),
])


def _array(values, dtype: pa.DataType) -> pa.Array:
    if isinstance(values, pd.Series):
        values = values.astype(object).where(values.notna(), None).tolist()
    return pa.array(values, type=dtype, from_pandas=True)


def build_layer(root: Path, layer: int, seed: int = DEFAULT_SEED,
                unblinded_run: str | None = None,
                blinded_run: str | None = None,
                concept_reliability: Path | None = None) -> pa.Table:
    """Build one layer's dictionary table (see the module docstring).
    ``concept_reliability`` defaults to ``reliability_path(root)``."""
    ev = eval_dir(root, layer, seed)
    report = load_report(ev / "report.json")
    stats = pd.read_csv(ev / "per_feature_stats.csv")
    live = live_feature_table(report, stats, layer)
    features = live["feature_idx"].to_numpy()
    firing_count = np.asarray(report["phase_1_2"]["firing_count"], dtype=np.int64)

    assoc = pd.read_csv(ev / "feature_label_associations.csv", usecols=_ASSOC_COLUMNS)
    assoc["significant"] = assoc["p_value_bh_significant"].astype(str).str.lower().eq("true")
    n_sig_csv = assoc[assoc["significant"]].groupby("feature_idx").size().reindex(
        np.arange(len(firing_count)), fill_value=0).to_numpy()
    if not np.array_equal(n_sig_csv, stats["n_significant_concepts"].to_numpy()):
        raise ValueError(f"layer {layer}: significant pairs in feature_label_associations.csv do "
                         f"not match n_significant_concepts in per_feature_stats.csv")

    counts = concept_counts(assoc, firing_count, layer)
    top = top_concepts(assoc, features, firing_count, counts)
    rel_path = Path(concept_reliability) if concept_reliability is not None else reliability_path(root)
    reliability = load_reliability(rel_path, set(assoc["concept"]), layer)
    rel_cols = reliability_columns(top, assoc, features, firing_count, counts, reliability)
    latent = latent_token_fraction(assoc, features, firing_count)
    causal = causal_columns(report, layer, features)

    live_set = {int(f) for f in features}
    unblinded = load_descriptions(description_path(root, layer, unblinded_run), live_set, layer) \
        if unblinded_run else None
    blinded = load_descriptions(description_path(root, layer, blinded_run), live_set, layer) \
        if blinded_run else None
    # Every run interpret.py writes since V1-2 records `blind`. The only runs
    # without it (pre-V1-2) are archived and unreadable (schema.check_run_tag),
    # so a missing column means an unknown file, not an unblinded run.
    for frame, tag, expected in ((unblinded, unblinded_run, False), (blinded, blinded_run, True)):
        if frame is None:
            continue
        if "blind" not in frame.columns:
            raise ValueError(f"layer {layer}: run {tag!r} has no 'blind' column, so whether it "
                             f"was written blind is unknown; rerun it with interpret.py")
        flags = frame["blind"].astype(str).str.lower() == "true"
        if (flags != expected).any():
            raise ValueError(f"layer {layer}: run {tag!r} is passed as "
                             f"{'blinded' if expected else 'unblinded'} but its CSV says otherwise")
    desc = description_columns(features, layer, unblinded, unblinded_run, blinded, blinded_run)

    n = len(features)
    values: dict[str, object] = {
        "layer": live["layer"], "feature_idx": live["feature_idx"],
        "firing_count": live["firing_count"], "firing_rate": live["firing_rate"],
        "n_significant_concepts": live["n_significant_concepts"],
        "has_significant_association": live["n_significant_concepts"] > 0,
        **top, **rel_cols,
        "latent_token_fraction": latent,
        "unexplained_mass_fraction": live["unexplained_mass_fraction"],
        **causal, **desc,
    }
    table = pa.table({field.name: _array(values[field.name], field.type) for field in _SCHEMA},
                     schema=_SCHEMA)
    metadata = {
        "schema_version": str(DICTIONARY_SCHEMA_VERSION), "layer": str(layer), "seed": str(seed),
        "n_tokens": str(report["phase_1_2"]["n_tokens"]), "n_live_features": str(n),
        "report": report_path(Path(), layer, seed).as_posix(),
        "unblinded_run": unblinded_run or "", "blinded_run": blinded_run or "",
        "association_source": (
            f"Phase 4 feature_label_associations.csv: every BH-significant (feature, concept) pair at "
            f"q = {report.get('config', {}).get('fdr_q', 'unrecorded')} over "
            f"{int(report['phase_1_2']['n_tokens']):,} tokens (every chunk of the cluster run, "
            f"including the test split, Finding 7), from the annotator's labels (Finding 9)"),
        "concept_reliability_source": (
            f"{_display_path(rel_path, root)} sha256 {_sha256_16(rel_path)}: label audit of a seeded "
            f"sample of local spectra (Finding 9; STATUS V1-3b)"),
        "display_rule": DISPLAY_RULE,
        "concept_reliability": json.dumps(reliability_records(reliability), allow_nan=False),
    }
    return table.replace_schema_metadata(metadata)


# --- feature examples and fit reference (V1-5, --examples) ----------------------------
#
# One streaming pass per layer over the local chunks: the SAE runs on the stored
# activations in batches of whole spectra; per spectrum it accumulates the
# centred-FVE sums and, for the demo features only, the maximum activation and
# its token. A feature's examples are its top EXAMPLES_K spectra by that
# maximum (one per spectrum by construction). Torch is imported lazily so the
# normal build stays torch-free.

EXAMPLES_K = 20
EXAMPLE_BATCH_TOKENS = 4096
FIT_QUANTILES = 1001
BEAM_SECONDS = (2.2, 4.2)          # V1-4a/V1-4c beam cost per spectrum on this machine
BF16_NOTE = "activations from stored bf16 extraction"
FINDING7_NOTE = ("every local spectrum, test split included, was SAE training data (Finding 7); "
                 "these are not held-out examples")
LABEL_NOTE = ("database-search label, not InstaNovo's prediction; benchmark labels can be "
              "wrong (Finding 8)")
ANNOTATION_NOTE = ("peaks annotated against the label with annotate.annotate_spectrum (strict), "
                   "the pipeline's annotator; Finding 9: internal-fragment matches are largely "
                   "chance and immonium labels carry a known bug, so b/y labels are the reliable part")
FIT_REFERENCE_NOTE = ("reference spectra were SAE training data (Finding 7); an unseen spectrum "
                      "may sit lower")
FIT_DEFINITION = ("per-spectrum centred FVE = 1 - SSE / (sum x^2 - sum_d (sum_t x_d)^2 / n) over "
                  "the spectrum's tokens (latent and peaks), runtime.fve's definition")
N_TERMINAL_NOTE = "not annotated: N-terminal modification, rejected by spectrum_utils (Finding 6)"


def examples_dir(root: Path) -> Path:
    return root / "dictionary" / "examples"


def examples_path(root: Path, layer: int) -> Path:
    return examples_dir(root) / f"layer_{layer}.parquet"


def fit_reference_paths(root: Path) -> tuple[Path, Path]:
    return root / "dictionary" / "fit_reference.json", root / "dictionary" / "fit_reference.csv"


def load_example_features(path: Path, layers: list[int]) -> dict[int, list[int]]:
    """{layer: sorted feature ids} from demo_features.json (runtime demo-explain)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for layer in layers:
        feats = data.get("features", {}).get(str(layer))
        if feats is None:
            raise ValueError(f"{path}: no feature list for layer {layer}")
        if not feats or sorted(set(int(f) for f in feats)) != [int(f) for f in feats]:
            raise ValueError(f"{path}: layer {layer} feature list must be non-empty, sorted and unique")
        out[layer] = [int(f) for f in feats]
    return out


def spectrum_batches(token_to_spectrum, max_tokens: int) -> list[tuple[int, int]]:
    """Token ranges [start, end) of consecutive whole spectra holding at most
    max_tokens tokens (a longer spectrum gets a batch of its own). Tokens of a
    spectrum must be contiguous."""
    t2s = np.asarray(token_to_spectrum)
    if t2s.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, t2s[1:] != t2s[:-1]])
    if len(np.unique(t2s[starts])) != len(starts):
        raise ValueError("a spectrum's tokens are not contiguous")
    bounds = list(starts) + [t2s.size]
    out, b0 = [], 0
    for i in range(1, len(bounds)):
        if bounds[i] - bounds[b0] > max_tokens and i - 1 > b0:
            out.append((int(bounds[b0]), int(bounds[i - 1])))
            b0 = i - 1
    out.append((int(bounds[b0]), int(bounds[-1])))
    return out


class SpectrumFitAccumulator:
    """Per-spectrum centred-FVE sums (float64), for any split of the tokens into
    batches, spectra straddling batches included."""

    def __init__(self, n_spectra: int, d_model: int):
        import torch
        self.sse = torch.zeros(n_spectra, dtype=torch.float64)
        self.sq = torch.zeros(n_spectra, dtype=torch.float64)
        self.colsum = torch.zeros(n_spectra, d_model, dtype=torch.float64)
        self.n = torch.zeros(n_spectra, dtype=torch.float64)

    def add(self, x, x_hat, spectrum_idx) -> None:
        import torch
        x64, xh64 = x.double(), x_hat.double()
        idx = torch.as_tensor(spectrum_idx, dtype=torch.long)
        self.sse.index_add_(0, idx, ((x64 - xh64) ** 2).sum(dim=1))
        self.sq.index_add_(0, idx, (x64 ** 2).sum(dim=1))
        self.colsum.index_add_(0, idx, x64)
        self.n.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))

    def fve_centred(self) -> np.ndarray:
        """1 - SSE / max(sum x^2 - sum_d (sum_t x_d)^2 / n, 1e-12), per spectrum."""
        if bool((self.n == 0).any()):
            raise ValueError("a spectrum received no tokens")
        mean_energy = (self.colsum ** 2).sum(dim=1) / self.n
        return (1.0 - self.sse / (self.sq - mean_energy).clamp_min(1e-12)).numpy()


def spectrum_max_and_argmax(feats, token_to_spectrum, token_to_position, n_spectra: int):
    """Per spectrum and column: the maximum over the spectrum's tokens and the
    token position where it is reached (the first such token), as
    ([n_spectra, F] float32, [n_spectra, F] int64). Spectra without a token stay
    at -inf / -1. feats [T, F]; tokens of a spectrum contiguous."""
    import torch
    n_f = feats.size(1)
    vmax = torch.full((n_spectra, n_f), -float("inf"))
    arg = torch.full((n_spectra, n_f), -1, dtype=torch.long)
    t2s = torch.as_tensor(token_to_spectrum)
    pos = torch.as_tensor(token_to_position)
    for start, end in spectrum_batches(t2s.numpy(), 1):            # one spectrum per range
        s = int(t2s[start])
        block = feats[start:end]
        vals = block.max(dim=0).values
        first = (block == vals).to(torch.int8).argmax(dim=0)        # first token at the maximum
        vmax[s] = vals
        arg[s] = pos[start:end][first]
    return vmax, arg


def _stream(root: Path, layer: int, sae):
    import torch
    from evaluate import ChunkStream
    return ChunkStream(extract_dir=root / "extract", annotation_dir=root / "annotation",
                       target_layer=layer, sae=sae, device="cpu", batch_size=EXAMPLE_BATCH_TOKENS,
                       dtype=torch.float32)


def stream_layer_examples(stream, sae, features: list[int], k: int = EXAMPLES_K,
                          batch_tokens: int = EXAMPLE_BATCH_TOKENS, log=print) -> dict:
    """One pass over a ChunkStream's chunks: per-spectrum centred FVE for every
    spectrum and the top-k spectra per feature (TopKAccumulator with aux = argmax
    token). Never holds more than one batch of full-width SAE features."""
    import time
    import torch
    from evaluate import TopKAccumulator

    cols = torch.as_tensor(features, dtype=torch.long)
    acc = TopKAccumulator(len(features), k, aux=torch.long)
    fve_rows, spectra = [], {}
    chunks = list(stream._chunk_indices)
    t_start = time.perf_counter()
    for i, (ci, acts, (meta, _ann)) in enumerate(zip(chunks, stream.iter_activations(),
                                                     stream.iter_metadata_annotations(), strict=True)):
        t0 = time.perf_counter()
        if acts.size(0) != meta["total_tokens"]:
            raise ValueError(f"chunk {ci}: {acts.size(0)} activation rows, meta says {meta['total_tokens']}")
        t2s, pos, n_spec = meta["token_to_spectrum"], meta["token_to_position"], int(meta["n_spectra"])
        fit = SpectrumFitAccumulator(n_spec, acts.size(1))
        vmax = torch.full((n_spec, len(features)), -float("inf"))
        arg = torch.full((n_spec, len(features)), -1, dtype=torch.long)
        for start, end in spectrum_batches(t2s.numpy(), batch_tokens):
            x = acts[start:end].float()
            with torch.inference_mode():
                out = sae.forward_inference(x)
            fit.add(x, out["x_hat"], t2s[start:end])
            sub = out["features"][:, cols]
            del out
            v, a = spectrum_max_and_argmax(sub, t2s[start:end], pos[start:end], n_spec)
            hit = a >= 0
            vmax[hit], arg[hit] = v[hit], a[hit]
        if bool((arg < 0).any()):
            raise ValueError(f"chunk {ci}: a spectrum has no tokens")
        acc.add_chunk(ci, vmax, aux=arg)
        fve = fit.fve_centred()
        for s in range(n_spec):
            fve_rows.append((ci, s, str(meta["spectrum_ids"][s]), float(fve[s])))
        spectra[ci] = n_spec
        dt = time.perf_counter() - t0
        if i == 0:
            log(f"  chunk {ci}: {dt:.0f} s; {len(chunks)} chunks -> ~{dt * len(chunks) / 60:.0f} min for this layer")
        else:
            log(f"  chunk {ci}: {dt:.0f} s (layer elapsed {(time.perf_counter() - t_start) / 60:.1f} min)")
    return {"topk": acc, "fve": fve_rows, "chunks": chunks, "n_spectra_by_chunk": spectra}


def example_rows(layer: int, features: list[int], acc, spectrum_ids: dict[tuple[int, int], str]) -> list[dict]:
    """Rows of examples/layer_L.parquet: per feature its kept spectra with a
    positive maximum, rank 1 = strongest."""
    rows = []
    for j, f in enumerate(features):
        rank = 0
        for v, c, s, t in zip(acc.values[j].tolist(), acc.chunk_ids[j].tolist(),
                              acc.token_ids[j].tolist(), acc.aux_ids[j].tolist(), strict=True):
            if not v > 0:
                continue
            rank += 1
            rows.append({"layer": layer, "feature_idx": int(f), "rank": rank, "chunk": int(c),
                         "spectrum_index": int(s), "spectrum_id": spectrum_ids[(int(c), int(s))],
                         "token": int(t), "peak_index": int(t) - 1 if t > 0 else None,
                         "activation": float(v)})
    return rows


def fit_quantiles(fve: np.ndarray, n: int = FIT_QUANTILES) -> list[float]:
    return np.quantile(np.asarray(fve, dtype=np.float64), np.linspace(0.0, 1.0, n)).tolist()


_EXAMPLE_SCHEMA = pa.schema([
    ("layer", pa.int64()), ("feature_idx", pa.int64()), ("rank", pa.int64()), ("chunk", pa.int64()),
    ("spectrum_index", pa.int64()), ("spectrum_id", pa.string()), ("token", pa.int64()),
    ("peak_index", pa.int64()), ("activation", pa.float64()),
])
_SPECTRA_SCHEMA = pa.schema([
    ("chunk", pa.int64()), ("spectrum_index", pa.int64()), ("spectrum_id", pa.string()),
    ("precursor_mz", pa.float64()), ("precursor_charge", pa.int64()),
    ("label_peptide", pa.string()), ("label_proforma", pa.string()), ("n_peaks", pa.int64()),
    ("mz", pa.list_(pa.float64())), ("intensity", pa.list_(pa.float64())),
    ("peak_annotations", pa.string()), ("annotation_note", pa.string()),
])


def annotate_referenced(root: Path, referenced: dict[int, set[int]], log=print) -> list[dict]:
    """The spectrum context of every referenced (chunk, spectrum): processed
    peaks from the chunk metadata, precursor, label, and the peaks re-annotated
    against the label (strict). A spectrum with an N-terminal modification is
    kept without annotation and says why (Finding 6)."""
    import torch
    import annotate
    registry = annotate.ConceptRegistry.build()
    config = annotate.AnnotationConfig(extract_dir=Path(), output_dir=Path())
    manifest = json.loads((root / "extract" / "manifest.json").read_text())
    meta_paths = {int(c["idx"]): root / "extract" / c["meta"] for c in manifest["chunks"]}
    rows = []
    for ci in sorted(referenced):
        meta = torch.load(meta_paths[ci], map_location="cpu", weights_only=False)
        for s in sorted(referenced[ci]):
            pep, prof, mods = meta["peptides"][s], meta["proforma_strings"][s], meta["modifications"][s]
            mz, inten = meta["mz_arrays"][s], meta["intensity_arrays"][s]
            pmz, z = float(meta["precursor_mzs"][s]), int(meta["precursor_charges"][s])
            anns, note = None, None
            if any(int(m["position"]) == 0 for m in mods):
                note = N_TERMINAL_NOTE
            else:
                found, _tok = annotate.annotate_spectrum(pep, prof, mods, mz, inten, pmz, z, config,
                                                         registry, strict=True)
                anns = [{"ion_type": a["ion_type"],
                         "ion_index": list(a["ion_index"]) if isinstance(a["ion_index"], tuple) else a["ion_index"],
                         "charge": int(a["charge"]), "neutral_loss": a["neutral_loss"]} for a in found]
            rows.append({"chunk": ci, "spectrum_index": s, "spectrum_id": str(meta["spectrum_ids"][s]),
                         "precursor_mz": pmz, "precursor_charge": z, "label_peptide": pep,
                         "label_proforma": prof, "n_peaks": int(len(mz)),
                         "mz": [float(v) for v in mz], "intensity": [float(v) for v in inten],
                         "peak_annotations": None if anns is None else json.dumps(anns),
                         "annotation_note": note})
        log(f"  annotated chunk {ci}: {len(referenced[ci])} spectra")
    return rows


def _examples_metadata(layer: int | None, extra: dict) -> dict[str, str]:
    from schema import EXAMPLES_SCHEMA_VERSION
    meta = {"schema_version": str(EXAMPLES_SCHEMA_VERSION), "k": str(EXAMPLES_K),
            "activation_note": BF16_NOTE, "finding_7_note": FINDING7_NOTE, "label_note": LABEL_NOTE,
            "annotation_note": ANNOTATION_NOTE}
    if layer is not None:
        meta["layer"] = str(layer)
    meta.update({k: v if isinstance(v, str) else json.dumps(v) for k, v in extra.items()})
    return meta


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_examples(root: Path, layers: list[int], features_path: Path, seed: int = DEFAULT_SEED,
                   batch_tokens: int = EXAMPLE_BATCH_TOKENS, log=print) -> dict:
    """The --examples pass: examples/layer_L.parquet per layer, examples/spectra.parquet,
    fit_reference.{json,csv}. Layers already written are not recomputed unless
    their file is removed (the pass is resumable per layer); the fit values of a
    finished layer are kept in examples/fit_L.parquet for that."""
    import time
    import torch
    from schema import FIT_REFERENCE_SCHEMA_VERSION
    from train import load_sae_from_checkpoint

    feats_by_layer = load_example_features(features_path, layers)
    src = {"example_features": _display_path(features_path, root),
           "example_features_sha256_16": _sha256_16(features_path)}
    referenced: dict[int, set[int]] = {}
    fit_by_layer: dict[int, dict] = {}
    t_all = time.perf_counter()
    for layer in layers:
        ckpt = root / "sae" / f"layer_{layer}" / f"seed_{seed}" / "checkpoint.pt"
        out = examples_path(root, layer)
        fit_out = examples_dir(root) / f"fit_{layer}.parquet"
        if out.is_file() and fit_out.is_file():
            log(f"layer {layer}: examples exist, reusing {out.name}")
        else:
            sae = load_sae_from_checkpoint(ckpt, device="cpu")
            sae.eval()
            stream = _stream(root, layer, sae)
            log(f"layer {layer}: {len(feats_by_layer[layer])} features, chunks {list(stream._chunk_indices)}")
            t0 = time.perf_counter()
            res = stream_layer_examples(stream, sae, feats_by_layer[layer], batch_tokens=batch_tokens, log=log)
            ids = {(c, s): sid for c, s, sid, _f in res["fve"]}
            rows = example_rows(layer, feats_by_layer[layer], res["topk"], ids)
            meta = _examples_metadata(layer, {
                "sae_checkpoint": _display_path(ckpt, root), "sae_sha256_16": _sha256_16(ckpt),
                "chunks": res["chunks"], "n_spectra": len(res["fve"]), "n_features": len(feats_by_layer[layer]),
                "rule": (f"top {EXAMPLES_K} spectra per feature by the spectrum's maximum activation "
                         f"(one per spectrum), shown at the argmax token; features active at any token "
                         f"of the V1-5 demo spectra"), **src, "seconds": round(time.perf_counter() - t0, 1)})
            table = pa.Table.from_pylist(rows, schema=_EXAMPLE_SCHEMA).replace_schema_metadata(meta)
            fit_table = pa.table({"chunk": [r[0] for r in res["fve"]], "spectrum_index": [r[1] for r in res["fve"]],
                                  "spectrum_id": [r[2] for r in res["fve"]],
                                  "fve_centred": [r[3] for r in res["fve"]]}).replace_schema_metadata(
                {"layer": str(layer), "chunks": json.dumps(res["chunks"]), "definition": FIT_DEFINITION})
            write_atomic(fit_table, fit_out)
            write_atomic(table, out)
            del sae, stream, res
            log(f"layer {layer}: {len(rows)} example rows -> {out} ({(time.perf_counter() - t0) / 60:.1f} min)")
        table = pq.read_table(out)
        for c, s in zip(table["chunk"].to_pylist(), table["spectrum_index"].to_pylist(), strict=True):
            referenced.setdefault(c, set()).add(s)
        fit_table = pq.read_table(fit_out)
        fit_by_layer[layer] = {"fve": np.asarray(fit_table["fve_centred"].to_pylist()),
                               "chunks": json.loads(fit_table.schema.metadata[b"chunks"])}

    n_ref = sum(len(v) for v in referenced.values())
    log(f"{n_ref} referenced spectra; beam predictions for them (--with-predictions) would cost "
        f"{n_ref * BEAM_SECONDS[0] / 3600:.1f}-{n_ref * BEAM_SECONDS[1] / 3600:.1f} h "
        f"at {BEAM_SECONDS[0]}-{BEAM_SECONDS[1]} s each")
    t0 = time.perf_counter()
    spectra_rows = annotate_referenced(root, referenced, log=log)
    spectra_table = pa.Table.from_pylist(spectra_rows, schema=_SPECTRA_SCHEMA).replace_schema_metadata(
        _examples_metadata(None, {**src, "n_spectra": len(spectra_rows)}))
    write_atomic(spectra_table, examples_dir(root) / "spectra.parquet")
    log(f"spectra.parquet: {len(spectra_rows)} spectra, re-annotation {time.perf_counter() - t0:.0f} s")

    probs = np.linspace(0.0, 1.0, FIT_QUANTILES).tolist()
    ref = {"schema_version": FIT_REFERENCE_SCHEMA_VERSION, "definition": FIT_DEFINITION,
           "activation_note": BF16_NOTE, "note": FIT_REFERENCE_NOTE, "seed": seed,
           "probabilities": probs,
           "layers": {str(L): {"n_spectra": int(v["fve"].size), "chunks": v["chunks"],
                               "quantiles": fit_quantiles(v["fve"])} for L, v in fit_by_layer.items()}}
    json_path, csv_path = fit_reference_paths(root)
    _write_text_atomic(json.dumps(ref, allow_nan=False, indent=1), json_path)
    lines = ["probability," + ",".join(f"layer_{L}" for L in fit_by_layer)]
    for i, p in enumerate(probs):
        lines.append(f"{p:.3f}," + ",".join(repr(ref["layers"][str(L)]["quantiles"][i]) for L in fit_by_layer))
    _write_text_atomic("\n".join(lines) + "\n", csv_path)
    log(f"fit reference -> {json_path.name}, {csv_path.name}; "
        + "; ".join(f"L{L}: n {v['fve'].size}, median {np.median(v['fve']):.4f}" for L, v in fit_by_layer.items())
        + f"; total {(time.perf_counter() - t_all) / 60:.1f} min")
    return {"referenced": n_ref, "fit": {L: int(v["fve"].size) for L, v in fit_by_layer.items()}}


# Merged corpus order (STATUS V1-2 log, 2026-11-21): train, validation, test; spectrum_id = row.
HF_SPLITS = (("train", 499_402), ("validation", 28_572), ("test", 111_312))


def hf_row_locator(hf_data_dir: Path) -> list[tuple[Path, int, int]]:
    """(parquet, first merged row, rows) per HF shard in merged order, checked
    against the recorded split sizes."""
    out, start = [], 0
    for split, n_expected in HF_SPLITS:
        files = sorted(Path(hf_data_dir).glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {split}-*.parquet in {hf_data_dir}")
        n_split = 0
        for f in files:
            n = pq.ParquetFile(f).metadata.num_rows
            out.append((f, start, n))
            start += n
            n_split += n
        if n_split != n_expected:
            raise ValueError(f"{split}: {n_split} rows in {hf_data_dir}, recorded {n_expected}")
    return out


def add_predictions(root: Path, hf_data_dir: Path, checkpoint: Path, save_every: int = 25, log=print) -> int:
    """--with-predictions: beam predictions (InstaNovo, 5 beams, the config's
    max_length, as explain()) for every spectrum in examples/spectra.parquet,
    written to examples/predictions.parquet. Resumable: spectra already there
    are skipped; the file is rewritten atomically every save_every spectra."""
    import time
    import instanovo_io
    shards = hf_row_locator(hf_data_dir)
    spectra = pq.read_table(examples_dir(root) / "spectra.parquet",
                            columns=["spectrum_id", "precursor_mz", "precursor_charge"]).to_pylist()
    out = examples_dir(root) / "predictions.parquet"
    done = pq.read_table(out).to_pylist() if out.is_file() else []
    have = {r["spectrum_id"] for r in done}
    todo = [s for s in spectra if s["spectrum_id"] not in have]
    log(f"{len(todo)} of {len(spectra)} spectra need a prediction")
    if not todo:
        return 0
    model, config, residue_set = instanovo_io.load_instanovo(checkpoint, device="cpu", by_id=False)
    processor = instanovo_io.make_processor(residue_set, annotated=False)
    schema = pa.schema([("spectrum_id", pa.string()), ("prediction", pa.string()),
                        ("beam_log_prob", pa.float64())])
    meta = {"note": "InstaNovo beam search (5 beams), the prediction for the spectrum, not the label",
            "checkpoint_bytes": str(Path(checkpoint).stat().st_size)}
    t0 = time.perf_counter()
    for i, s in enumerate(todo, 1):
        row = int(s["spectrum_id"])
        shard = next((f, r0) for f, r0, n in shards if r0 <= row < r0 + n)
        rec = _read_parquet_row(shard[0], row - shard[1])
        if abs(rec["precursor_mz"] - s["precursor_mz"]) > 1e-3 or rec["precursor_charge"] != s["precursor_charge"]:
            raise ValueError(f"spectrum {row}: HF row does not match the chunk's precursor")
        batch = instanovo_io.preprocess_spectrum(processor, rec["mz_array"], rec["intensity_array"],
                                                 rec["precursor_mz"], rec["precursor_charge"])
        res = instanovo_io.beam_search(model, batch, "cpu", 5, int(config["max_length"]))
        tokens = list(res["predictions"][0])
        done.append({"spectrum_id": s["spectrum_id"], "prediction": "".join(tokens) if tokens else None,
                     "beam_log_prob": float(res["prediction_log_probability"][0]) if tokens else None})
        if i % save_every == 0 or i == len(todo):
            write_atomic(pa.Table.from_pylist(done, schema=schema).replace_schema_metadata(meta), out)
            log(f"  {i}/{len(todo)} ({(time.perf_counter() - t0) / i:.1f} s each)")
    return 0


def _read_parquet_row(path: Path, row: int) -> dict:
    pf = pq.ParquetFile(path)
    start = 0
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if row < start + n:
            table = pf.read_row_group(g, columns=["mz_array", "intensity_array", "precursor_mz", "precursor_charge"])
            return table.slice(row - start, 1).to_pylist()[0]
        start += n
    raise IndexError(f"row {row} outside {path}")


def _sha256_16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _display_path(path: Path, root: Path) -> str:
    """``path`` relative to the output root when it lies under it (no personal paths)."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).name


def write_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def _run_tag(value: str) -> str | None:
    if value.lower() == "none":
        return None
    try:
        return check_run_tag(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="pipeline output root")
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--unblinded-run", type=_run_tag, default=None,
                        help="description run tag written with labels (interpret/layer_L/<tag>); "
                             "default none")
    parser.add_argument("--blinded-run", type=_run_tag, default=None,
                        help="description run tag written blind; takes precedence; default none")
    parser.add_argument("--concept-reliability", type=Path, default=None,
                        help="concept reliability table (default <root>/analysis/label_audit/"
                             "concept_reliability.csv, written by analysis.py --label-audit)")
    parser.add_argument("--examples", action="store_true",
                        help="V1-5: feature examples and fit reference instead of the dictionary build "
                             "(streams the local chunks; needs --example-features)")
    parser.add_argument("--example-features", type=Path, default=None,
                        help="demo_features.json (inspector.runtime demo-explain)")
    parser.add_argument("--batch-tokens", type=int, default=EXAMPLE_BATCH_TOKENS,
                        help="--examples: tokens per SAE batch (whole spectra)")
    parser.add_argument("--with-predictions", action="store_true",
                        help="--examples: also beam predictions for the referenced spectra (resumable)")
    parser.add_argument("--hf-data-dir", type=Path, default=None,
                        help="--with-predictions: directory holding the HF nine-species parquets")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / "instanovo_v1.1.0.ckpt",
                        help="--with-predictions: InstaNovo checkpoint")
    args = parser.parse_args(argv)

    if args.examples:
        if args.example_features is None:
            parser.error("--examples needs --example-features")
        build_examples(args.root, args.layers, args.example_features, args.seed, args.batch_tokens)
        if args.with_predictions:
            if args.hf_data_dir is None:
                parser.error("--with-predictions needs --hf-data-dir")
            add_predictions(args.root, args.hf_data_dir, args.checkpoint)
        return

    for layer in args.layers:
        table = build_layer(args.root, layer, args.seed, args.unblinded_run, args.blinded_run,
                            args.concept_reliability)
        out = dictionary_path(args.root, layer)
        write_atomic(table, out)
        measured = int(pc.sum(table["causal_measured"]).as_py())
        described = table.num_rows - table["description"].null_count
        reliable = table.num_rows - table["reliable_concept_1"].null_count
        print(f"layer {layer}: {table.num_rows} live features, "
              f"{measured} with measured Phase 8 delta-CE, {described} described, "
              f"{reliable} with a reliable association -> {out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"error: {exc}")
