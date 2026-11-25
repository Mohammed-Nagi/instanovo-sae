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
    args = parser.parse_args(argv)

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
