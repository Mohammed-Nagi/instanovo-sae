"""interpret.py -- LLM-assisted interpretation of SAE features.

Adapts InterPLM's automated feature-description pipeline (Simon and Zou, 2025,
Methods 5.3) to mass spectrometry, and adds a causal cross-reference the
protein-language-model setting does not support.

Why. The registry labels a peak only when it matches a theoretical fragment ion,
so features firing on the unmatched majority are invisible to the F1 evaluation
however interpretable they are. F1-dom separately demotes detectors of concepts
that co-occur with others, since it penalises firing on concept-negative tokens.
An LLM reading raw examples carries neither bias.

Strata (--strata). Each feature lands in exactly one, in this precedence:

    causal       implicated by the Phase 8 ablations. Do causally necessary
                 features have describable structure?
    unexplained  activation mass concentrated on peaks the theory cannot label.
                 The discovery set.
    concept      strongest BH-significant chemical association. Positive
                 control: recovering known chemistry unprompted validates the
                 method.
    unlabelled   no BH-significant concept of any kind.

Expect unlabelled to come back empty, and read that as a result rather than a
misconfiguration. With 12,288 features tested against 50 concepts, everything
that fires appreciably picks up some significant association, leaving the pool to
the near-dead tail: on layer 2 that is 151 features, the busiest firing on 18 of
67.5M tokens. Nor does it stand in for the discovery set the way it might seem
to -- layer 2 has 576 features above 0.90 unexplained mass and none of them are
unlabelled, because scoring an is_noise_peak association is itself disqualifying.
Both pools and their overlap are logged per run.

Concept labels are withheld from the prompt unless --include-concept-labels, so
any chemistry the model names is inferred from the spectra rather than retrieved.
For the discovery set this also avoids feeding back the labels whose bias the
analysis exists to circumvent.

Validation follows InterPLM: a held-out set is shown without activations, the
model predicts each one, and the Pearson correlation against measured values is
reported (InterPLM's median is 0.72 over 1,200 protein features). A description
that cannot predict held-out activations is not evidence of anything. Each row
carries holdout_coverage, since r covers only the examples actually answered.

Inputs, all from an existing evaluate.py run:
    <eval-dir>/per_feature_stats.csv        ranking and concept association
    <eval-dir>/top_activating_tokens.csv    global top-K provenance (Phase 3)
    <eval-dir>/report.json                  Phase 8 causal results, if present
plus the extract/annotation chunks and the SAE checkpoint. The chunks are needed
because Phase 3 stores only the top-K, and it is the mid-range and zero examples
that make the prediction task non-trivial.

Blinding (--blind). By default the tables show the annotation result
(matched_ion), each neighbour's annotation and the true peptide, so a
description can be read off the labels rather than inferred from the spectrum.
--blind drops all three from both the description and prediction prompts;
neighbours become mz@rel_intensity, and a token_kind column keeps only the
latent-summary-token marker, which is model input structure, not annotation.

Split. Examples are split into describe and held-out halves by spectrum, so no
spectrum (and no peak shown as a neighbour) appears on both sides. Example ids
are a random permutation and each table is in random row order, so neither id
nor row position carries activation strength or the order examples were merged.
Everything per feature (sampling, merge, ids, split, row order) draws from an
RNG seeded by (seed, layer, feature), so examples are identical across runs that
differ only in --blind, and across resumes.

Baselines. Each feature also gets detection AUROC (do predicted activations
separate firing from non-firing held-out tokens?) from the same predictions as
r, and --n-shuffles predictions of its held-out table from other features'
descriptions in the same run. description_beats_baseline is r above the 95th
percentile of all shuffled r in the layer. A constant prediction against
varying measured activations scores r = 0 (no information), in the null and
for the feature's own description alike; r is NaN only with fewer than 3
predictions or constant measured activations.

Outputs under --output-dir/--run-tag:
    run_config.json                     settings a resume must match
    feature_descriptions.csv            one row per feature: stratum, summary, r
    feature_descriptions.json           full descriptions, examples, predictions
    feature_descriptions.partial.jsonl  resume log, appended as each feature lands
    shuffled_baseline.partial.jsonl     resume log for the shuffled baseline

Rerunning skips features already in the resume logs, so an interrupted run does
not pay for them twice; logged records are rescored under the current r rule
on load. --rescore does only that: it rewrites the outputs from the logs and
calls nothing. --dry-run builds the prompts, prints examples and a cost
estimate, and writes nothing.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import ChunkStream
from schema import check_run_tag
from train import load_sae_from_checkpoint

LOG = logging.getLogger("interpret")

# InterPLM quantises max activation into bins of 0.1 and samples examples from
# each, so the description has to explain the whole dynamic range rather than
# just the extremes. The same scheme is used here over normalised activations.
N_ACTIVATION_BINS = 10
EXAMPLES_PER_BIN = 2
EXAMPLES_TOP_BIN = 6
ZERO_EXAMPLES = 8

# Below this, a feature cannot be characterised and its r would be dominated by
# the zero examples. InterPLM's equivalent cutoff is 20.
MIN_ACTIVE_EXAMPLES = 12

# A partial reply scores the description on a subset the model may have picked
# for being easy, so a shortfall below this share is logged rather than folded
# silently into r.
MIN_PREDICTION_COVERAGE = 0.8

# Firing floor for the unexplained and unlabelled strata. A feature firing on one
# token also scores an unexplained fraction of exactly 1.0. Override per run with
# --min-firing-rate.
MIN_UNEXPLAINED_FIRING_RATE = 1e-4

# Concepts describing a token's role rather than its chemistry: is_noise_peak
# says the theory had nothing to say, is_latent_token says the token is not a
# peak. Neither is a positive control for inferring chemistry, so both are barred
# from the concept stratum.
STRUCTURAL_CONCEPTS = frozenset({"is_noise_peak", "is_latent_token"})

# Reporting only, for how far the older "no significant concept" proxy reaches
# into the discovery population. The stratum itself ranks rather than thresholds.
UNEXPLAINED_OVERLAP_FRACTION = 0.90

# Peaks shown as spectral context around the activating peak.
CONTEXT_PEAKS = 6

DEFAULT_MODEL = "gpt-4o"

# Shuffled-description predictions per feature, for the baseline null.
DEFAULT_N_SHUFFLES = 3
# description_beats_baseline: r above this percentile of the layer's shuffled r.
BASELINE_PERCENTILE = 95.0

# r is NaN below this many predictions; a correlation over two points is +-1
# whatever the description says.
MIN_SCORED_PREDICTIONS = 3
CONSTANT_STD = 1e-12

# --- Cost estimate (dry run only) ---------------------------------------------
# USD per million tokens (input, output). Check against current OpenAI pricing
# before relying on the estimate; a model missing here gets a token count only.
PRICE_PER_MTOK = {
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o": (2.50, 10.00),
}
# Calibrated once against unblinded_v1 (gpt-4o-2024-08-06, 4 layers x 80
# features): its rebuilt prompts run 2.64 characters per recorded prompt token,
# its replies 296 completion tokens per feature (describe + predict; the split
# assumes ~5.5 tokens per predicted row), and its descriptions 860 characters on
# average. These are assumptions for a budget, not measurements of the run
# being estimated.
CHARS_PER_TOKEN = 2.64
DESCRIBE_COMPLETION_TOKENS = 210
PREDICT_COMPLETION_TOKENS = 86
PLACEHOLDER_DESCRIPTION_CHARS = 860


@dataclasses.dataclass
class InterpretConfig:
    """Configuration for one interpretation run."""

    eval_dir: Path
    extract_dir: Path
    annotation_dir: Path
    sae_checkpoint: Path
    output_dir: Path            # the run directory: --output-dir / --run-tag
    target_layer: int
    run_tag: str = ""
    blind: bool = False
    chunk_indices: tuple[int, ...] | None = None   # explicit request, else spread
    n_shuffles: int = DEFAULT_N_SHUFFLES

    strata: tuple[str, ...] = ("concept", "unexplained", "unlabelled", "causal")
    n_per_stratum: int = 40

    # Chunks encoded to recover the activation distribution. Each contributes
    # ~108k tokens, so a handful is ample for sampling examples per feature.
    n_sample_chunks: int = 12
    include_concept_labels: bool = False
    min_firing_rate: float = MIN_UNEXPLAINED_FIRING_RATE

    model: str = DEFAULT_MODEL
    max_tokens: int = 2048
    seed: int = 0
    device: str = "cpu"   # resolved from --device before construction
    batch_size: int = 4096
    dry_run: bool = False       # build prompts, skip the API
    rescore: bool = False       # recompute scores from the resume logs, skip the API

    def as_jsonable(self) -> dict:
        out = dataclasses.asdict(self)
        for key in ("eval_dir", "extract_dir", "annotation_dir",
                    "sae_checkpoint", "output_dir"):
            out[key] = str(out[key])
        out["strata"] = list(self.strata)
        out["chunk_indices"] = None if self.chunk_indices is None else list(self.chunk_indices)
        return out

    def resume_key(self, chunks_used: list[int]) -> dict:
        """Settings that decide what a run's records mean. A resume must match all of them."""
        return {
            "run_tag": self.run_tag, "blind": self.blind, "model": self.model,
            "seed": self.seed, "target_layer": self.target_layer,
            "strata": list(self.strata), "n_per_stratum": self.n_per_stratum,
            "chunk_indices_used": list(chunks_used), "n_shuffles": self.n_shuffles,
            "include_concept_labels": self.include_concept_labels,
            "min_firing_rate": self.min_firing_rate, "split": "spectrum",
        }


@dataclasses.dataclass
class TokenExample:
    """One activating (or non-activating) token, with its spectral context."""

    example_id: int
    activation: float          # normalised to the feature's observed maximum
    chunk_idx: int
    token_in_chunk: int
    spectrum_in_chunk: int     # the split groups on (chunk_idx, spectrum_in_chunk)
    is_latent: bool
    peak_mz: float
    peak_intensity: float      # relative to the spectrum's most intense peak
    ion_type: str              # "noise" when the peak matched no fragment ion
    peak_rank: int             # 1 = most intense peak in its spectrum
    n_peaks: int
    peptide: str               # ProForma, so modifications are visible
    precursor_mz: float
    precursor_charge: int
    # (m/z, relative intensity, ion type) per neighbour. Intensity and ion type
    # are what let a description say "unmatched peaks just above an intense
    # y-ion" rather than only "near a peak at 1044.5".
    neighbours: list[tuple[float, float, str]]
    concepts: list[str]        # withheld from the prompt unless requested


# --- Reading the evaluation artefacts ----------------------------------------

def _optional_float(row: dict, key: str) -> float:
    """Read a column evaluate.py may leave blank, as NaN.

    The unexplained-mass columns are empty when Phase 4 was restored from cache,
    which carries no activation magnitudes. NaN keeps those features out of any
    ordering without dropping them from the table.
    """
    raw = (row.get(key) or "").strip()
    if not raw:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _read_per_feature_stats(eval_dir: Path) -> dict[int, dict]:
    """feature_idx -> firing rate, concept association, and unexplained-mass stats."""
    path = eval_dir / "per_feature_stats.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run evaluate.py Phase 4 for this layer/seed first."
        )
    stats: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            stats[int(row["feature_idx"])] = {
                "firing_rate": float(row["firing_rate"]),
                "max_f1_dom": float(row["max_f1_dom"]),
                "best_concept": row["best_concept"],
                "n_significant": int(row["n_significant_concepts"]),
                "unexplained_fraction": _optional_float(row, "unexplained_mass_fraction"),
                "unexplained_enrichment": _optional_float(row, "unexplained_enrichment"),
            }
    return stats


def _selectivity_rank(value: float) -> float:
    """|selectivity| for ordering, with NaN ranked last.

    Every comparison against NaN is False, so comparing raw values would let a
    NaN entry win by arriving first and never be replaced by a real measurement.
    """
    return abs(value) if not math.isnan(value) else -float("inf")


def _read_causal_features(eval_dir: Path) -> dict[int, dict]:
    """feature_idx -> its strongest Phase 8 result, if Phase 8 was run.

    Both the group ablations (top_n_features per concept) and the per-feature
    ablations are read. A feature appearing in several concepts keeps the entry
    with the largest |selectivity|, which is the concept it is most specific to.
    """
    path = eval_dir / "report.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    phase_8 = report.get("phase_8") or {}
    per_concept = phase_8.get("per_concept") or {}

    best: dict[int, dict] = {}

    def offer(fi: int, entry: dict) -> None:
        prev = best.get(fi)
        if prev is None or _selectivity_rank(entry["selectivity"]) > _selectivity_rank(
            prev["selectivity"]
        ):
            best[fi] = entry

    for concept, info in per_concept.items():
        causal = info.get("causal")
        for fi in info.get("top_n_features", []):
            if causal:
                offer(int(fi), {
                    "concept": concept,
                    "selectivity": _finite(causal.get("selectivity")),
                    "selectivity_z": _finite(causal.get("selectivity_z")),
                    "mean_delta_ce": _finite(causal.get("mean_delta_ce")),
                    "source": "group",
                })
        for entry in info.get("per_feature_ablation", []):
            offer(int(entry["feature_idx"]), {
                "concept": concept,
                "selectivity": _finite(entry.get("selectivity")),
                "selectivity_z": float("nan"),
                "mean_delta_ce": _finite(entry.get("mean_delta_ce")),
                "source": "per_feature",
            })

    return best


def _finite(v) -> float:
    """Coerce None and JSON's stringified NaN into a float NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _read_global_top_tokens(eval_dir: Path) -> dict[int, list[tuple[float, int, int]]]:
    """feature_idx -> [(activation, chunk_idx, token_in_chunk), ...] from Phase 3.

    These are the globally strongest activations, which a sample of chunks is
    unlikely to contain. They are merged into the top activation bin so the
    description is anchored on the feature's actual maximum.
    """
    path = eval_dir / "top_activating_tokens.csv"
    if not path.exists():
        LOG.warning("%s not found; using sampled chunks only for top examples", path)
        return {}
    out: dict[int, list[tuple[float, int, int]]] = {}
    with open(path, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            out.setdefault(int(row["feature_idx"]), []).append((
                float(row["activation"]),
                int(row["chunk_idx"]),
                int(row["token_in_chunk"]),
            ))
    return out


def _read_ion_type_vocab(annotation_dir: Path) -> dict[int, str]:
    """Invert the annotation manifest's ion-type vocabulary.

    Read from the manifest rather than imported from annotate.py, so this script
    inherits no spectrum_utils dependency.
    """
    manifest = json.loads((annotation_dir / "annotation_manifest.json").read_text(encoding="utf-8"))
    vocab = manifest.get("vocab", {}).get("ion_type", {})
    return {int(v): k for k, v in vocab.items()}


def _read_concept_names(annotation_dir: Path) -> list[str]:
    manifest = json.loads((annotation_dir / "annotation_manifest.json").read_text(encoding="utf-8"))
    return list(manifest["registry"]["names"])


# --- Feature selection --------------------------------------------------------

def select_strata(
    stats: dict[int, dict],
    causal: dict[int, dict],
    n_per_stratum: int,
    strata: tuple[str, ...],
    rng: random.Random,
    min_firing_rate: float = MIN_UNEXPLAINED_FIRING_RATE,
) -> dict[int, str]:
    """Choose features for each requested stratum. Returns feature_idx -> stratum.

    Each feature lands in exactly one stratum, in the order causal >
    unexplained > concept > unlabelled. Causal is first as the smallest and most
    informative set; unexplained precedes concept so an unexplained-peak
    specialist is not absorbed into the positive control.
    """
    assigned: dict[int, str] = {}

    if "causal" in strata and causal:
        # NaN selectivity sorts last: a concept whose prevalence terciles were
        # degenerate produced no usable contrast, so it is the weakest evidence.
        ranked = sorted(causal.items(), key=lambda kv: -_selectivity_rank(kv[1]["selectivity"]))
        for fi, _info in ranked[:n_per_stratum]:
            assigned[fi] = "causal"
        LOG.info("Stratum causal: %d features", sum(1 for s in assigned.values() if s == "causal"))
    elif "causal" in strata:
        LOG.warning(
            "Stratum 'causal' requested but report.json has no phase_8 results; "
            "run evaluate.py with RUN_PHASE_8=1 to populate it. Skipping."
        )

    if "unexplained" in strata:
        # The discovery set: features concentrating their activation mass on
        # peaks the theory cannot label.
        missing = [fi for fi, s in stats.items()
                   if np.isnan(s["unexplained_fraction"])]
        if len(missing) == len(stats):
            raise ValueError(
                "per_feature_stats.csv has no unexplained_mass_fraction values, so "
                "the 'unexplained' stratum cannot be selected. This column is "
                "written by evaluate.py Phase 4 and is blank when Phase 4 was "
                "restored from cache. Re-run evaluate.py for this layer/seed "
                "without --phase4-cache-dir, or drop 'unexplained' from --strata."
            )

        # Latent-token detectors are dropped. UnexplainedMass is a ratio over
        # peak-token mass, correctly, but nothing requires that mass to be large:
        # a feature answering almost entirely to the latent summary token clears
        # the firing floor on latent tokens alone, then scores ~1.0 on the sliver
        # of peak mass left over. It is not a peak specialist. is_noise_peak
        # stays -- an unmatched peak is exactly the discovery case.
        candidates = [
            fi for fi, s in stats.items()
            if s["firing_rate"] >= min_firing_rate
            and not np.isnan(s["unexplained_fraction"])
            and s["best_concept"] != "is_latent_token"
            and fi not in assigned
        ]
        # Ties at exactly 1.0 are common, so the tie-break decides much of the
        # set: firing rate gives more examples to describe from, where the
        # default would just favour low feature ids.
        candidates.sort(key=lambda fi: (-stats[fi]["unexplained_fraction"],
                                        -stats[fi]["firing_rate"]))
        for fi in candidates[:n_per_stratum]:
            assigned[fi] = "unexplained"
        LOG.info("Stratum unexplained: %d features (firing rate >= %g; %d candidates)",
                 sum(1 for s in assigned.values() if s == "unexplained"),
                 min_firing_rate, len(candidates))

    if "concept" in strata:
        # Positive control: highest F1-dom among features the registry already
        # describes chemically. Features whose best concept is structural rather
        # than chemical are excluded -- see STRUCTURAL_CONCEPTS.
        candidates = sorted(
            (fi for fi, s in stats.items()
             if s["n_significant"] > 0
             and s["best_concept"] not in STRUCTURAL_CONCEPTS
             and fi not in assigned),
            key=lambda fi: -stats[fi]["max_f1_dom"],
        )
        for fi in candidates[:n_per_stratum]:
            assigned[fi] = "concept"
        LOG.info("Stratum concept: %d features", sum(1 for s in assigned.values() if s == "concept"))

    if "unlabelled" in strata:
        # Features the evaluation is silent about, sampled rather than ranked so
        # the set stays unbiased with respect to firing rate within the pool.
        # The firing floor matters here: without it this stratum selects the
        # near-dead tail, which would be chosen, encoded, then dropped by
        # MIN_ACTIVE_EXAMPLES having produced nothing. See the module docstring
        # for why an empty result is expected and is itself informative.
        pool = [fi for fi, s in stats.items()
                if s["n_significant"] == 0 and s["firing_rate"] > 0 and fi not in assigned]
        candidates = [fi for fi in pool if stats[fi]["firing_rate"] >= min_firing_rate]
        rng.shuffle(candidates)
        for fi in candidates[:n_per_stratum]:
            assigned[fi] = "unlabelled"
        n_taken = sum(1 for s in assigned.values() if s == "unlabelled")
        LOG.info("Stratum unlabelled: %d features (uniform sample of %d candidates "
                 "clearing firing rate %g; %d in the pool before the floor)",
                 n_taken, len(candidates), min_firing_rate, len(pool))
        if pool and not candidates:
            hottest = max(stats[fi]["firing_rate"] for fi in pool)
            LOG.warning(
                "Stratum 'unlabelled' is empty: all %d features without a significant "
                "concept fire below the floor (busiest %.2e). Every feature that fires "
                "appreciably in this layer has some significant association, so this "
                "stratum has no describable members -- lower --min-firing-rate only if "
                "you want the near-dead tail.",
                len(pool), hottest,
            )

    if "unexplained" in strata and "unlabelled" in strata:
        # How far "no significant concept" reaches into the discovery
        # population. Over the full pools, so it is independent of
        # n_per_stratum.
        pool_unexp = {
            fi for fi, s in stats.items()
            if s["firing_rate"] >= min_firing_rate
            and not np.isnan(s["unexplained_fraction"])
            and s["unexplained_fraction"] >= UNEXPLAINED_OVERLAP_FRACTION
        }
        pool_unlab = {fi for fi, s in stats.items()
                      if s["n_significant"] == 0 and s["firing_rate"] > 0}
        both = pool_unexp & pool_unlab
        LOG.info(
            "Pools at unexplained fraction >= %.2f: unexplained=%d, unlabelled=%d, "
            "overlap=%d (%d unexplained specialists the 'no significant concept' "
            "proxy misses)",
            UNEXPLAINED_OVERLAP_FRACTION, len(pool_unexp), len(pool_unlab),
            len(both), len(pool_unexp - pool_unlab),
        )

    return assigned


# --- Building example tables --------------------------------------------------

def _relative_intensity(intensity: float, spectrum_max: float) -> float:
    return float(intensity / spectrum_max) if spectrum_max > 0 else 0.0


def available_chunk_indices(extract_dir: Path, target_layer: int) -> tuple[list[int], int]:
    """Chunk ids whose activation file for this layer is still on disk.

    A run with KEEP_CHUNKS=0 deletes activations after evaluating and keeps only
    KEEP_CHUNK_SAMPLE per layer as a cross-layer token sample, while leaving the
    manifest listing all of them. Selecting by manifest alone would then pick
    chunks that no longer exist and die on the first torch.load. Returns
    (available ids, total in manifest) so the caller can say how much survived.
    """
    manifest = json.loads((extract_dir / "manifest.json").read_text(encoding="utf-8"))
    layer_key = str(target_layer)
    available = [
        c["idx"] for c in manifest["chunks"]
        if layer_key in c.get("activations", {})
        and (extract_dir / c["activations"][layer_key]).exists()
        and (extract_dir / c["meta"]).exists()
    ]
    return available, manifest["n_chunks"]


def spread_chunk_indices(n_chunks: int, n_sample: int) -> list[int]:
    """Evenly spaced chunk ids, rather than the leading n_sample.

    Chunks follow the merged parquet, which is not shuffled. Mean peptide length
    drifts from about 14.7 residues at the start of the file to 21.8 at the end,
    so the first dozen chunks of 625 are a different population from the corpus
    and would describe every feature from one end of the dataset.
    """
    if n_sample >= n_chunks:
        return list(range(n_chunks))
    step = n_chunks / n_sample
    return sorted({min(int(i * step), n_chunks - 1) for i in range(n_sample)})


def resolve_chunk_indices(
    available: list[int],
    requested: tuple[int, ...] | None,
    n_sample: int,
) -> list[int]:
    """The chunk ids to encode: the explicit request, else an even spread of what is on disk.

    An explicit request is honoured exactly or not at all. Silently dropping a
    missing chunk would change every feature's examples while the run still
    claimed the requested set.
    """
    if requested is None:
        return [available[i] for i in spread_chunk_indices(len(available), n_sample)]
    wanted = sorted({int(c) for c in requested})
    missing = [c for c in wanted if c not in set(available)]
    if missing:
        raise ValueError(
            f"--chunk-indices names {len(missing)} chunk(s) without activations and meta "
            f"on disk for this layer: {missing}. On disk: {available}"
        )
    return wanted


def _collect_activations(
    stream: ChunkStream,
    feature_ids: list[int],
    n_chunks: int,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Encode a sample of chunks and keep each target feature's activation column.

    Returns (feature_idx -> activations over the sampled tokens, chunk records).
    The chunk records hold the per-token and per-spectrum metadata needed to turn
    a token index back into a readable example.

    MEMORY. The stream is built with feature_subset=feature_ids, so it encodes
    straight into the selected columns and the full [n_tokens, d_dict] matrix
    never exists outside one batch. Peak is
    n_chunks * n_tokens * len(feature_ids) * 4 bytes, about 400 MB for 12 chunks
    and 80 features. Without the subset it would be ~5 GB per chunk at
    d_dict = 12,288, enough to be OOM-killed on a 16 GB machine -- which is what
    happened before that argument existed.

    chunk.features therefore arrives already narrowed, with columns in
    feature_ids order rather than indexed by feature id.
    """
    feature_index = {fi: i for i, fi in enumerate(feature_ids)}
    columns: list[np.ndarray] = []
    records: list[dict] = []
    offset = 0

    for chunk in stream:
        feats = chunk.features.to(torch.float32).numpy()
        if feats.shape[1] != len(feature_ids):
            raise RuntimeError(
                f"stream yielded {feats.shape[1]} feature columns, expected "
                f"{len(feature_ids)}; build ChunkStream with feature_subset=feature_ids"
            )
        columns.append(feats)
        records.append({
            "chunk_idx": chunk.chunk_idx,
            "offset": offset,
            "n_tokens": feats.shape[0],
            "labels": chunk.labels,
            "token_to_spectrum": chunk.token_to_spectrum.numpy(),
            "token_to_position": chunk.token_to_position.numpy(),
            "ion_type_ids": chunk.ion_type_ids.numpy(),
            "peak_mzs": chunk.peak_mzs.numpy(),
            "peak_intensities": chunk.peak_intensities.numpy(),
            "peptides": chunk.proforma_strings,
            "precursor_mzs": chunk.precursor_mzs.numpy(),
            "precursor_charges": chunk.precursor_charges.numpy(),
        })
        offset += feats.shape[0]
        LOG.info("Encoded chunk %d (%d tokens)", chunk.chunk_idx, feats.shape[0])
        if len(records) >= n_chunks:
            break

    if not columns:
        raise RuntimeError("No chunks were read; check --extract-dir and --annotation-dir.")

    stacked = np.concatenate(columns, axis=0)
    return {fi: stacked[:, feature_index[fi]] for fi in feature_ids}, records


def _locate(records: list[dict], global_token: int) -> tuple[dict, int]:
    """Map a token index in the concatenated sample back to (chunk, local index)."""
    for rec in records:
        if rec["offset"] <= global_token < rec["offset"] + rec["n_tokens"]:
            return rec, global_token - rec["offset"]
    raise IndexError(f"token {global_token} is outside the sampled chunks")


def _make_example(
    rec: dict,
    local_token: int,
    activation: float,
    example_id: int,
    ion_vocab: dict[int, str],
    concept_names: list[str],
) -> TokenExample:
    """Turn one token into a readable example with its spectral context."""
    spectrum = int(rec["token_to_spectrum"][local_token])
    position = int(rec["token_to_position"][local_token])
    is_latent = position == 0

    # Peaks of this spectrum, in stored order, for intensity rank and neighbours.
    same_spectrum = np.flatnonzero(rec["token_to_spectrum"] == spectrum)
    peak_tokens = same_spectrum[rec["token_to_position"][same_spectrum] > 0]
    intensities = rec["peak_intensities"][peak_tokens]
    spectrum_max = float(intensities.max()) if intensities.size else 0.0

    mz = float(rec["peak_mzs"][local_token])
    intensity = float(rec["peak_intensities"][local_token])

    if is_latent or peak_tokens.size == 0:
        peak_rank = 0
        neighbours: list[tuple[float, float, str]] = []
    else:
        order = np.argsort(-intensities)
        rank_of = {int(peak_tokens[o]): r + 1 for r, o in enumerate(order)}
        peak_rank = rank_of.get(local_token, 0)
        mzs = rec["peak_mzs"][peak_tokens]
        near = np.argsort(np.abs(mzs - mz))[1:CONTEXT_PEAKS + 1]
        neighbours = [
            (
                round(float(mzs[i]), 4),
                _relative_intensity(float(intensities[i]), spectrum_max),
                ion_vocab.get(int(rec["ion_type_ids"][int(peak_tokens[i])]), "unknown"),
            )
            for i in sorted(near)
        ]

    labels = rec["labels"][local_token]
    concepts = [concept_names[i] for i in np.flatnonzero(labels.numpy())] \
        if len(concept_names) else []

    return TokenExample(
        example_id=example_id,
        activation=activation,
        chunk_idx=rec["chunk_idx"],
        token_in_chunk=local_token,
        spectrum_in_chunk=spectrum,
        is_latent=is_latent,
        peak_mz=mz,
        peak_intensity=_relative_intensity(intensity, spectrum_max),
        ion_type=ion_vocab.get(int(rec["ion_type_ids"][local_token]), "unknown"),
        peak_rank=peak_rank,
        n_peaks=int(peak_tokens.size),
        peptide=str(rec["peptides"][spectrum]),
        precursor_mz=float(rec["precursor_mzs"][spectrum]),
        precursor_charge=int(rec["precursor_charges"][spectrum]),
        neighbours=neighbours,
        concepts=concepts,
    )


def feature_peak(activations: np.ndarray, global_top: list[tuple[float, int, int]]) -> float:
    """The activation that normalises to 1.0 for this feature.

    Sampled chunks and Phase 3's global top-K are two views of the same feature
    and must share one scale, or the same physical activation would carry two
    different labels in one table -- corrupting both the description and the
    prediction target it is scored against. Phase 3 scanned every chunk, so its
    maximum is usually the larger of the two.
    """
    sampled = float(activations.max()) if activations.size else 0.0
    global_max = max((a for a, _c, _t in global_top), default=0.0)
    return max(sampled, global_max)


def build_examples(
    activations: np.ndarray,
    peak: float,
    records: list[dict],
    ion_vocab: dict[int, str],
    concept_names: list[str],
    rng: random.Random,
) -> list[TokenExample]:
    """Sample examples across the feature's activation range, plus zeros.

    Stratifying by activation decile is what makes the held-out prediction task
    discriminative: a set drawn only from the top would let a description score
    well by predicting "high" everywhere.
    """
    if peak <= 0:
        return []

    normalised = activations / peak
    active = np.flatnonzero(normalised > 0)
    if active.size < MIN_ACTIVE_EXAMPLES:
        return []

    chosen: list[int] = []
    for b in range(N_ACTIVATION_BINS):
        lo, hi = b / N_ACTIVATION_BINS, (b + 1) / N_ACTIVATION_BINS
        in_bin = active[(normalised[active] > lo) & (normalised[active] <= hi)]
        if in_bin.size == 0:
            continue
        want = EXAMPLES_TOP_BIN if b == N_ACTIVATION_BINS - 1 else EXAMPLES_PER_BIN
        take = min(want, in_bin.size)
        chosen.extend(rng.sample(list(in_bin), take))

    # Sampled by position: zeros holds most of the layer, and materialising it as
    # a Python list once per feature costs more than the sampling does.
    zeros = np.flatnonzero(normalised == 0)
    if zeros.size:
        take = min(ZERO_EXAMPLES, zeros.size)
        chosen.extend(int(zeros[i]) for i in rng.sample(range(zeros.size), take))

    rng.shuffle(chosen)
    examples = []
    for i, token in enumerate(chosen):
        rec, local = _locate(records, int(token))
        examples.append(_make_example(
            rec, local, float(normalised[token]), i, ion_vocab, concept_names,
        ))
    return examples


def merge_global_top(
    examples: list[TokenExample],
    global_top: list[tuple[float, int, int]],
    peak: float,
    records: list[dict],
    ion_vocab: dict[int, str],
    concept_names: list[str],
    limit: int = 4,
) -> list[TokenExample]:
    """Add Phase 3's globally strongest activations when they fall in the sample.

    The strongest examples carry the most information about what a feature
    detects, and `peak` is the shared scale from feature_peak, so these rows are
    directly comparable to the sampled ones.

    Expect few. Phase 3 ranked over every chunk while this run reads a sample of
    them, so on the nine-species layers only 1 to 2 per cent of top-K rows land in
    a 12-of-625 sample: under one row per feature. The consequence is that most
    features are described from the lower part of their range, and raising
    --n-sample-chunks is the only lever on it.
    """
    if peak <= 0:
        return examples

    by_chunk = {rec["chunk_idx"]: rec for rec in records}
    seen = {(e.chunk_idx, e.token_in_chunk) for e in examples}

    added = 0
    next_id = len(examples)
    for activation, chunk_idx, token in sorted(global_top, reverse=True):
        if added >= limit:
            break
        rec = by_chunk.get(chunk_idx)
        if rec is None or (chunk_idx, token) in seen:
            continue
        examples.append(_make_example(
            rec, token, float(activation / peak), next_id, ion_vocab, concept_names,
        ))
        seen.add((chunk_idx, token))
        next_id += 1
        added += 1
    return examples


def feature_rng(seed: int, layer: int, feature_idx: int, purpose: str = "examples") -> random.Random:
    """An RNG that depends only on (purpose, seed, layer, feature).

    Hashed with sha256 rather than Python's hash(), which is salted per process.
    A shared stream would make a feature's examples depend on which features
    came before it, so a resumed run, a partial failure or a second run with a
    different --blind would see different examples for the same feature.
    """
    digest = hashlib.sha256(f"{purpose}:{seed}:{layer}:{feature_idx}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def reassign_example_ids(examples: list[TokenExample], rng: random.Random) -> list[TokenExample]:
    """Replace example ids with a random permutation.

    build_examples numbers its sample in shuffled order, but merge_global_top
    appends the globally strongest activations after it, so before this step the
    highest ids were the strongest examples: a side channel the prediction pass
    could read without understanding the description.
    """
    ids = list(range(len(examples)))
    rng.shuffle(ids)
    return [dataclasses.replace(e, example_id=i) for e, i in zip(examples, ids)]


def split_by_spectrum(
    examples: list[TokenExample], rng: random.Random,
) -> tuple[list[TokenExample], list[TokenExample]]:
    """Split into (describe, held-out) with no spectrum on both sides, each in random row order.

    Spectra are taken in random order and each goes whole to the currently
    smaller side. Splitting by example instead lets a held-out peak appear as a
    neighbour in the describe table, and lets the same spectrum's context be
    seen on both sides.
    """
    groups: dict[tuple[int, int], list[TokenExample]] = {}
    for e in examples:
        groups.setdefault((e.chunk_idx, e.spectrum_in_chunk), []).append(e)
    keys = sorted(groups)
    rng.shuffle(keys)
    describe: list[TokenExample] = []
    holdout: list[TokenExample] = []
    for key in keys:
        (describe if len(describe) < len(holdout) else holdout).extend(groups[key])
    rng.shuffle(describe)
    rng.shuffle(holdout)
    return describe, holdout


def prepare_feature(
    feature_idx: int,
    activations: np.ndarray,
    global_top: list[tuple[float, int, int]],
    records: list[dict],
    ion_vocab: dict[int, str],
    concept_names: list[str],
    seed: int,
    layer: int,
) -> tuple[list[TokenExample], list[TokenExample]] | None:
    """One feature's (describe, held-out) tables, or None when it cannot be described.

    Sampling, merge, id permutation, split and row order all draw from this
    feature's own RNG, in that order.
    """
    rng = feature_rng(seed, layer, feature_idx)
    peak = feature_peak(activations, global_top)
    examples = build_examples(activations, peak, records, ion_vocab, concept_names, rng)
    if not examples:
        return None
    examples = merge_global_top(examples, global_top, peak, records, ion_vocab, concept_names)
    examples = reassign_example_ids(examples, rng)
    describe, holdout = split_by_spectrum(examples, rng)
    if not describe or not holdout:
        return None
    return describe, holdout


# --- Prompt construction ------------------------------------------------------

def _csv_safe(value: str) -> str:
    """Strip separators from a free-text cell.

    The table is joined on commas by hand, since the model reads it as plain
    text. A comma or newline inside a field would shift every column after it and
    the row would still parse, describing the wrong peak. Nothing in the
    nine-species benchmark contains either; this guards a different dataset.
    """
    return str(value).replace(",", ";").replace("\n", " ").replace("\r", " ")


LATENT_MARKER = "LATENT_SUMMARY_TOKEN"

# The only columns a blind table may carry. Everything here is model input: the
# peak's own m/z and intensity, its spectrum's size and precursor, and whether
# the token is the latent summary token. No annotation, no peptide.
BLIND_COLUMNS = (
    "example_id", "token_kind", "peak_mz", "rel_intensity", "intensity_rank",
    "n_peaks", "precursor_mz", "precursor_z", "nearby_peaks",
)


def format_example_table(
    examples: list[TokenExample],
    include_activation: bool,
    include_concepts: bool,
    blind: bool = False,
) -> str:
    """Render examples as a compact table for the model.

    blind drops matched_ion, the neighbours' annotations and peptide_proforma,
    and writes token_kind (latent marker or "peak") in matched_ion's place.
    """
    if blind and include_concepts:
        raise ValueError("concept labels are annotation; they cannot be shown in a blind table")
    if blind:
        header = list(BLIND_COLUMNS)
    else:
        header = [
            "example_id", "peak_mz", "rel_intensity", "intensity_rank",
            "n_peaks", "matched_ion", "peptide_proforma",
            "precursor_mz", "precursor_z", "nearby_peaks",
        ]
    if include_concepts:
        header.append("concepts")
    if include_activation:
        header.insert(1, "activation")

    rows = [",".join(header)]
    for e in examples:
        if blind:
            cells = [
                str(e.example_id),
                LATENT_MARKER if e.is_latent else "peak",
                f"{e.peak_mz:.4f}",
                f"{e.peak_intensity:.3f}",
                str(e.peak_rank),
                str(e.n_peaks),
                f"{e.precursor_mz:.4f}",
                str(e.precursor_charge),
                " ".join(f"{m:.2f}@{i:.2f}" for m, i, _t in e.neighbours),
            ]
        else:
            token_kind = LATENT_MARKER if e.is_latent else e.ion_type
            cells = [
                str(e.example_id),
                f"{e.peak_mz:.4f}",
                f"{e.peak_intensity:.3f}",
                str(e.peak_rank),
                str(e.n_peaks),
                _csv_safe(token_kind),
                _csv_safe(e.peptide),
                f"{e.precursor_mz:.4f}",
                str(e.precursor_charge),
                " ".join(f"{m:.2f}@{i:.2f}/{t}" for m, i, t in e.neighbours),
            ]
        if include_concepts:
            cells.append(_csv_safe(" ".join(e.concepts)))
        if include_activation:
            # `or 0.0` collapses negative zero, which the SAE's masked multiply
            # produces for about a third of inactive tokens. "-0.000" in the
            # column the model reasons over invites a sign that is not there.
            cells.insert(1, f"{e.activation or 0.0:.3f}")
        rows.append(",".join(cells))
    return "\n".join(rows)


DOMAIN_PREAMBLE = """\
You are analysing a sparse autoencoder feature trained on the encoder activations
of InstaNovo, a transformer that sequences peptides de novo from tandem mass
spectra.

Each row below is one encoder token. Almost all tokens are a single observed peak
in one MS/MS spectrum; the exception is the latent summary token, which is
prepended to every spectrum and represents the spectrum as a whole. Columns:

  peak_mz          observed mass-to-charge ratio of the peak
  rel_intensity    intensity relative to the most intense peak in that spectrum
  intensity_rank   1 = most intense peak in its spectrum
  n_peaks          number of peaks in that spectrum
  matched_ion      fragment-ion series this peak matched (b, y, I = immonium,
                   internal, precursor), or "noise" if it matched no theoretical
                   fragment of the labelled peptide. Roughly 60% of peaks are
                   unmatched, so "noise" does not mean uninformative.
  peptide_proforma the peptide that produced the spectrum, in ProForma notation;
                   bracketed values are modification delta masses
  precursor_mz/z   mass-to-charge and charge of the intact peptide
  nearby_peaks     the nearest other peaks in the same spectrum, each written
                   mz@rel_intensity/matched_ion -- so "1044.52@0.91/y" is an
                   intense y-ion neighbour. Use these for mass differences and
                   for patterns defined by what a peak sits next to.
"""

BLIND_PREAMBLE = """\
You are analysing a sparse autoencoder feature trained on the encoder activations
of InstaNovo, a transformer that sequences peptides de novo from tandem mass
spectra.

Each row below is one encoder token. Almost all tokens are a single observed peak
in one MS/MS spectrum; the exception is the latent summary token, which is
prepended to every spectrum and represents the spectrum as a whole. The table
shows only what the model itself receives: no fragment-ion assignments and no
peptide sequence are given. Columns:

  token_kind       "peak" for an observed peak, LATENT_SUMMARY_TOKEN for the
                   spectrum-level summary token
  peak_mz          observed mass-to-charge ratio of the peak
  rel_intensity    intensity relative to the most intense peak in that spectrum
  intensity_rank   1 = most intense peak in its spectrum
  n_peaks          number of peaks in that spectrum
  precursor_mz/z   mass-to-charge and charge of the intact peptide
  nearby_peaks     the nearest other peaks in the same spectrum, each written
                   mz@rel_intensity -- so "1044.52@0.91" is an intense
                   neighbour. Use these for mass differences and for patterns
                   defined by what a peak sits next to.
"""

_CONSIDER_UNBLINDED = """\
  - which physical properties (m/z region, relative intensity, charge, position
    in the spectrum) separate high from medium from zero activation
  - whether activation tracks a fragment-ion series, a neutral loss, a mass
    difference between peaks, or a specific residue or modification in the
    peptide
  - whether the feature responds to the peak itself or to its spectral context
  - if the pattern is a mass relationship, state the approximate mass difference"""

_CONSIDER_BLIND = """\
  - which physical properties (m/z region, relative intensity, intensity rank,
    precursor m/z and charge, position in the spectrum) separate high from
    medium from zero activation
  - whether activation tracks a mass relationship: a mass difference to nearby
    peaks (for example an amino-acid residue mass or a neutral loss), or a
    relation to the precursor mass
  - whether the feature responds to the peak itself, to its spectral context,
    or to the latent summary token
  - if the pattern is a mass relationship, state the approximate mass difference"""


def build_description_prompt(examples: list[TokenExample], include_concepts: bool,
                             blind: bool = False) -> str:
    table = format_example_table(examples, include_activation=True,
                                 include_concepts=include_concepts, blind=blind)
    preamble = BLIND_PREAMBLE if blind else DOMAIN_PREAMBLE
    consider = _CONSIDER_BLIND if blind else _CONSIDER_UNBLINDED
    return f"""{preamble}
Analyse this table to determine what predicts the 'activation' column. Activation
is normalised so 1.0 is the strongest response observed for this feature and 0.0
means the feature did not fire.

Your description will be used to predict activations for held-out tokens given
only the other columns, so include only what is useful for that. Consider:

{consider}

State honestly if the activating tokens have no coherent pattern you can identify.

Reply in exactly this format:

DESCRIPTION: The activation patterns are characterized by: <detailed description>
SUMMARY: The feature activates on <one sentence>

Table:
{table}
"""


def build_prediction_prompt(
    description: str,
    examples: list[TokenExample],
    include_concepts: bool,
    blind: bool = False,
) -> str:
    table = format_example_table(examples, include_activation=False,
                                 include_concepts=include_concepts, blind=blind)
    preamble = BLIND_PREAMBLE if blind else DOMAIN_PREAMBLE
    return f"""{preamble}
Below is a description of what a sparse autoencoder feature detects, and a table
of held-out tokens with the activation column removed. Predict the feature's
normalised activation (0.0 to 1.0) for each token, judging how well it matches
the described pattern. Many tokens will be 0.0.

Output nothing but a CSV table, starting with the header line exactly as shown:

example_id,predicted_activation

Feature description:
{description}

Tokens to predict:
{table}
"""


# --- Model calls --------------------------------------------------------------

def load_dotenv(path: Path) -> int:
    """Load KEY=value pairs from a .env file into os.environ. Returns the count.

    Handles one KEY=value per line, # comments, blank lines, an optional `export`
    prefix and optional quotes -- small enough not to warrant python-dotenv.

    The environment wins over the file, so an exported key overrides it and a
    stray local file never shadows a deployment secret. Read as utf-8-sig
    because PowerShell and Notepad write a BOM by default, which would otherwise
    make the first key parse as "\\ufeffOPENAI_API_KEY" and silently not match.
    """
    if not path.exists():
        return 0

    loaded = 0
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def _make_client():
    """OpenAI client, or None when the SDK or key is unavailable."""
    try:
        from openai import OpenAI
    except ImportError:
        LOG.error("The 'openai' package is required. Install with: uv add openai")
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        LOG.error(
            "OPENAI_API_KEY is not set. Put it in a .env file beside this script "
            "(see .env.example), export it, or pass --env-file."
        )
        return None
    return OpenAI()


# HTTP statuses worth a second attempt: rate limits, and the server's own faults.
# Anything else (bad key, unknown model, malformed request) will fail the same
# way forever, and 160 features each burning three retries turns a typo into an
# hour of sleeping.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class PermanentAPIError(RuntimeError):
    """An API failure that retrying cannot fix."""


def _create_completion(client, model: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
    """One chat completion, returned as (text, provenance).

    The output-length parameter was renamed: older chat models take max_tokens,
    while the reasoning models reject it and require max_completion_tokens. The
    first call decides which this model wants, so the caller need not know.

    Provenance travels with the reply rather than accumulating in a module-level
    set. A set is only populated by calls this process actually made, so a run
    resumed from the log would report no model and no tokens at all, despite
    reporting the descriptions those calls produced.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
        )
    except TypeError:
        response = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=max_tokens,
        )
    except Exception as exc:  # inspect the message, then re-raise
        if "max_completion_tokens" not in str(exc):
            raise
        response = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=max_tokens,
        )
    usage = getattr(response, "usage", None)
    meta = {
        # The alias asked for floats between versions; this is what answered.
        "model": getattr(response, "model", None) or model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }
    return response.choices[0].message.content or "", meta


def call_model(client, model: str, prompt: str, max_tokens: int,
               retries: int = 3) -> tuple[str, dict]:
    """One completion, retrying transient failures with linear backoff.

    A permanent failure is raised immediately: the run is unattended and every
    feature would otherwise repeat the same doomed attempt.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return _create_completion(client, model, prompt, max_tokens)
        except Exception as exc:  # classify, then retry or give up
            status = getattr(exc, "status_code", None)
            if status is not None and status not in RETRYABLE_STATUS:
                raise PermanentAPIError(f"{type(exc).__name__}: {exc}") from exc
            last = exc
            wait = 5 * (attempt + 1)
            LOG.warning("Model call failed (attempt %d/%d): %s; retrying in %ds",
                        attempt + 1, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Model call failed after {retries} attempts: {last}")


_DESCRIPTION_RE = re.compile(r"DESCRIPTION:\s*(.*?)\s*SUMMARY:", re.DOTALL)
_SUMMARY_RE = re.compile(r"SUMMARY:\s*(.*)", re.DOTALL)


def parse_description(text: str) -> tuple[str, str]:
    """Split the reply into (description, one-sentence summary)."""
    description = _DESCRIPTION_RE.search(text)
    summary = _SUMMARY_RE.search(text)
    return (
        description.group(1).strip() if description else text.strip(),
        summary.group(1).strip().split("\n")[0] if summary else "",
    )


def parse_predictions(text: str, valid_ids: set[int]) -> dict[int, float]:
    """Parse the predicted-activation CSV, ignoring prose around it."""
    out: dict[int, float] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            example_id, value = int(parts[0]), float(parts[1])
        except ValueError:
            continue  # header or commentary
        if example_id in valid_ids:
            out[example_id] = min(max(value, 0.0), 1.0)
    return out


def pearson_r(pairs: list[tuple[float, float]]) -> float:
    """Correlation between predicted and measured activations.

    NaN only when the pairs cannot test a description: fewer than
    MIN_SCORED_PREDICTIONS, or measured activations that are constant. A
    constant prediction against varying measurements scores 0: the description
    led the model to predict nothing about this feature, which is a result
    (no information), not a missing value. Dropping it as NaN would remove
    exactly the uninformative descriptions from the shuffled null and inflate
    its percentiles.
    """
    if len(pairs) < MIN_SCORED_PREDICTIONS:
        return float("nan")
    predicted = np.array([p for p, _ in pairs], dtype=np.float64)
    measured = np.array([m for _, m in pairs], dtype=np.float64)
    if measured.std() < CONSTANT_STD:
        return float("nan")
    if predicted.std() < CONSTANT_STD:
        return 0.0
    return float(np.corrcoef(predicted, measured)[0, 1])


def detection_auroc(pairs: list[tuple[float, float]]) -> float:
    """AUROC of predicted activation for separating firing (measured > 0) from non-firing tokens.

    Mann-Whitney with ties counted as half. NaN unless both classes are present,
    since a held-out set with no zeros (or no firing tokens) cannot test
    detection at all.
    """
    pos = np.array([p for p, m in pairs if m > 0], dtype=np.float64)
    neg = np.array([p for p, m in pairs if not m > 0], dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def score_predictions(predictions: dict[int, float], holdout: list[TokenExample]) -> dict:
    """r, AUROC and coverage of one set of predictions against the measured activations."""
    measured = {e.example_id: e.activation for e in holdout}
    pairs = [(predictions[i], measured[i]) for i in predictions]
    return {
        "pearson_r": pearson_r(pairs),
        "detection_auroc": detection_auroc(pairs),
        "n_predicted": len(pairs),
        "holdout_coverage": len(pairs) / len(holdout) if holdout else 0.0,
    }


def baseline_threshold(null_values: list[float]) -> float:
    """BASELINE_PERCENTILE of the finite shuffled-description r values; NaN if there are none."""
    finite = [v for v in null_values if not math.isnan(v)]
    if not finite:
        return float("nan")
    return float(np.percentile(finite, BASELINE_PERCENTILE))


def beats_baseline(r: float, threshold: float) -> bool | None:
    """r strictly above the layer's shuffled-description threshold; None when either is NaN."""
    if math.isnan(r) or math.isnan(threshold):
        return None
    return bool(r > threshold)


# --- Per-feature driver -------------------------------------------------------

PLACEHOLDER_DESCRIPTION = "<the description written in the first call goes here>"


def interpret_feature(
    client,
    config: InterpretConfig,
    feature_idx: int,
    stratum: str,
    describe_set: list[TokenExample],
    holdout_set: list[TokenExample],
) -> dict:
    """Describe one feature from describe_set and score the description on holdout_set.

    The split comes from split_by_spectrum. Tables are rendered in the order
    given, which is already random. A dry run returns both prompts (the
    prediction prompt with a placeholder description) and calls nothing.
    """
    result: dict = {
        "feature_idx": feature_idx,
        "stratum": stratum,
        "blind": config.blind,
        "n_describe": len(describe_set),
        "n_holdout": len(holdout_set),
        "description": "",
        "summary": "",
        "pearson_r": float("nan"),
        "detection_auroc": float("nan"),
        "n_predicted": 0,
    }

    description_prompt = build_description_prompt(
        describe_set, config.include_concept_labels, blind=config.blind,
    )
    if config.dry_run:
        result["description_prompt"] = description_prompt
        result["prediction_prompt"] = build_prediction_prompt(
            PLACEHOLDER_DESCRIPTION, holdout_set, config.include_concept_labels,
            blind=config.blind,
        )
        return result

    reply, describe_meta = call_model(
        client, config.model, description_prompt, config.max_tokens,
    )
    description, summary = parse_description(reply)
    result["description"] = description
    result["summary"] = summary

    prediction_prompt = build_prediction_prompt(
        description, holdout_set, config.include_concept_labels, blind=config.blind,
    )
    reply, predict_meta = call_model(
        client, config.model, prediction_prompt, config.max_tokens,
    )
    predictions = parse_predictions(reply, {e.example_id for e in holdout_set})

    # Stored per feature so it survives a resume: both calls used the same model,
    # and the token counts are what the compute statement is built from.
    result["model_served"] = describe_meta["model"]
    result["prompt_tokens"] = describe_meta["prompt_tokens"] + predict_meta["prompt_tokens"]
    result["completion_tokens"] = (
        describe_meta["completion_tokens"] + predict_meta["completion_tokens"]
    )
    measured = {e.example_id: e.activation for e in holdout_set}
    scores = score_predictions(predictions, holdout_set)

    # A reply that skips the examples it found hard would score on an easier
    # subset than the one it was given, so record the coverage alongside r.
    if scores["holdout_coverage"] < MIN_PREDICTION_COVERAGE:
        LOG.warning(
            "Feature %d: model predicted only %d/%d held-out examples (%.0f%%); "
            "its r is over that subset, not the full holdout.",
            feature_idx, scores["n_predicted"], len(holdout_set),
            100 * scores["holdout_coverage"],
        )

    result.update(scores)
    result["predictions"] = [
        {"example_id": i, "predicted": predictions[i], "measured": measured[i]}
        for i in sorted(predictions)
    ]
    # In row order: the shuffled baseline re-renders this exact table.
    result["holdout_examples"] = [dataclasses.asdict(e) for e in holdout_set]
    result["describe_keys"] = [[e.chunk_idx, e.token_in_chunk] for e in describe_set]
    return result


# --- Shuffled-description baseline --------------------------------------------

def _example_from_dict(d: dict) -> TokenExample:
    d = dict(d)
    d["neighbours"] = [tuple(n) for n in d["neighbours"]]
    return TokenExample(**d)


def draw_donors(feature_idx: int, pool: list[int], n: int, seed: int, layer: int) -> list[int]:
    """n other features whose descriptions stand in for this one's, without replacement.

    Drawn from every described feature in the layer (both strata), never the
    feature itself, from an RNG keyed on the feature so the draw does not depend
    on the order features are processed or on the main RNG.
    """
    others = sorted(f for f in pool if f != feature_idx)
    if len(others) < n:
        raise ValueError(
            f"feature {feature_idx}: only {len(others)} other described features, "
            f"need {n} for the shuffled baseline"
        )
    return feature_rng(seed, layer, feature_idx, purpose="shuffle").sample(others, n)


def _shuffled_resume_path(output_dir: Path) -> Path:
    return output_dir / "shuffled_baseline.partial.jsonl"


def run_shuffled_baseline(client, config: InterpretConfig, results: list[dict]) -> None:
    """Predict each feature's held-out table from other features' descriptions.

    Adds a "shuffled" record to each result in place. Records are appended to a
    resume log as they finish, and logged ones are rescored under the current r
    rule (rescore_shuffled). With client=None a missing record is an error. A logged record whose donors differ from a fresh
    draw means the set of described features changed, and is an error rather
    than a silent mix of two nulls.
    """
    described = {r["feature_idx"]: r for r in results if r.get("model_served")}
    pool = sorted(described)
    path = _shuffled_resume_path(config.output_dir)
    done: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOG.warning("Ignoring a truncated line in %s", path.name)
                continue
            done[int(record["feature_idx"])] = record

    for n, fi in enumerate(pool, start=1):
        donors = draw_donors(fi, pool, config.n_shuffles, config.seed, config.target_layer)
        if fi in done:
            if done[fi]["donors"] != donors:
                raise RuntimeError(
                    f"feature {fi}: logged shuffled donors {done[fi]['donors']} differ from "
                    f"a fresh draw {donors}; the set of described features changed since "
                    f"{path.name} was written"
                )
            holdout = [_example_from_dict(d) for d in described[fi]["holdout_examples"]]
            rescore_shuffled(done[fi], holdout)
            described[fi]["shuffled"] = done[fi]
            continue
        if client is None:
            raise RuntimeError(
                f"feature {fi} has no record in {path.name}; --rescore makes no API calls"
            )
        holdout = [_example_from_dict(d) for d in described[fi]["holdout_examples"]]
        record = {"feature_idx": fi, "donors": donors, "r_values": [], "auroc_values": [],
                  "n_predicted": [], "prompt_tokens": 0, "completion_tokens": 0}
        for donor in donors:
            prompt = build_prediction_prompt(
                described[donor]["description"], holdout, config.include_concept_labels,
                blind=config.blind,
            )
            reply, meta = call_model(client, config.model, prompt, config.max_tokens)
            predictions = parse_predictions(reply, {e.example_id for e in holdout})
            scores = score_predictions(predictions, holdout)
            record["r_values"].append(scores["pearson_r"])
            record["auroc_values"].append(scores["detection_auroc"])
            record["n_predicted"].append(scores["n_predicted"])
            record["prompt_tokens"] += meta["prompt_tokens"]
            record["completion_tokens"] += meta["completion_tokens"]
        LOG.info("[shuffled %d/%d] feature %d: r %s with donors %s", n, len(pool), fi,
                 ", ".join(f"{v:.3f}" for v in record["r_values"]), donors)
        described[fi]["shuffled"] = record
        append_jsonl(path, record)


def apply_baseline(results: list[dict]) -> float:
    """Fill shuffled_null_p95 and description_beats_baseline from the pooled shuffled r. Returns the threshold.

    The threshold is NaN (and every beats_baseline None) unless every described
    feature has its shuffled record: a null pooled over part of the layer would
    shift as the rest arrived.
    """
    described = [r for r in results if r.get("model_served")]
    if not described or any(not r.get("shuffled") for r in described):
        threshold = float("nan")
    else:
        threshold = baseline_threshold([v for r in described for v in r["shuffled"]["r_values"]])
    for r in results:
        r["shuffled_null_p95"] = threshold
        r["description_beats_baseline"] = beats_baseline(r["pearson_r"], threshold)
    return threshold


def rescore_result(result: dict) -> bool:
    """Recompute a stored feature's r and AUROC from its logged predictions. Returns True if r changed.

    Logs written before the constant-prediction rule (pearson_r) hold NaN where
    r is now 0; rescoring on load brings them under the current rule without
    an API call.
    """
    preds = result.get("predictions")
    if not preds:
        return False
    pairs = [(p["predicted"], p["measured"]) for p in preds]
    old = result["pearson_r"]
    result["pearson_r"] = pearson_r(pairs)
    result["detection_auroc"] = detection_auroc(pairs)
    return not _same_float(old, result["pearson_r"])


def rescore_shuffled(record: dict, holdout: list[TokenExample]) -> int:
    """Reapply the constant-prediction rule to a logged shuffled record. Returns NaNs reclassified.

    The shuffled log keeps r and n_predicted per donor, not the predictions.
    With at least MIN_SCORED_PREDICTIONS predictions and measured activations
    that vary over the predicted subset, pearson_r's only NaN case is a
    constant prediction, so that NaN becomes 0. The subset is known to vary
    when the full held-out set varies and n_predicted exceeds the size of its
    largest group of equal activations. Anything else stays NaN, with a
    warning when the log cannot decide the case.
    """
    measured = [e.activation for e in holdout]
    varies = len(measured) >= 2 and float(np.std(measured)) >= CONSTANT_STD
    largest_tie = max(Counter(measured).values()) if measured else 0
    changed = 0
    for i, (r, n) in enumerate(zip(record["r_values"], record["n_predicted"])):
        if not math.isnan(r) or n < MIN_SCORED_PREDICTIONS or not varies:
            continue
        if n > largest_tie:
            record["r_values"][i] = 0.0
            changed += 1
        else:
            LOG.warning("Feature %d, donor %d: shuffled r is NaN over %d of %d held-out "
                        "examples and the log cannot tell a constant prediction from a "
                        "constant measured subset; left NaN.",
                        record["feature_idx"], record["donors"][i], n, len(measured))
    return changed


def _same_float(a: float, b: float) -> bool:
    """Equal, or both NaN. Logged predictions are re-summed in id order, so allow rounding."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


# --- Output -------------------------------------------------------------------

def _resume_path(output_dir: Path) -> Path:
    return output_dir / "feature_descriptions.partial.jsonl"


def load_completed(output_dir: Path) -> dict[int, dict]:
    """Features already described in an earlier attempt, keyed by feature_idx.

    Every stage in this pipeline resumes rather than repeating work, and here the
    work costs money: an interrupted run has already paid for the features it
    finished. A malformed trailing line is dropped, since a run killed mid-write
    leaves one.
    """
    path = _resume_path(output_dir)
    if not path.exists():
        return {}
    done: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            LOG.warning("Ignoring a truncated line in %s", path.name)
            continue
        done[int(record["feature_idx"])] = record
    return done


def append_completed(output_dir: Path, result: dict) -> None:
    """Persist one feature as soon as it is done, before the run can die."""
    append_jsonl(_resume_path(output_dir), result)


def append_jsonl(path: Path, result: dict) -> None:
    """Append one record to a resume log.

    A run killed mid-write leaves a line with no newline on the end. Appending
    straight onto it would fuse the two records into one unparseable line and
    lose this feature as well as that one, so the separator is restored first.
    """
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as fp:
            fp.seek(-1, os.SEEK_END)
            needs_newline = fp.read(1) != b"\n"
        if needs_newline:
            with open(path, "a", encoding="utf-8") as fp:
                fp.write("\n")
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(result, default=str) + "\n")


def _blank_if_nan(value) -> str:
    """Write an absent unexplained-mass statistic as an empty cell, not 'nan'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def check_run_config(output_dir: Path, key: dict) -> None:
    """Write run_config.json on the first call; on later calls require it to match.

    Resume logs are only meaningful under the settings that wrote them: a resumed
    blind run must not pick up unblinded descriptions, nor a run on other chunks
    reuse examples drawn from these.
    """
    path = output_dir / "run_config.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        diff = {k: (stored.get(k), key.get(k)) for k in set(stored) | set(key)
                if stored.get(k) != key.get(k)}
        if diff:
            raise ValueError(
                f"{path} was written with different settings; resuming would mix two runs. "
                f"(stored, now): {diff}. Use a new --run-tag."
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(key, indent=2), encoding="utf-8")


def write_outputs(
    results: list[dict],
    stats: dict[int, dict],
    causal: dict[int, dict],
    config: InterpretConfig,
    chunks_used: list[int] | None = None,
) -> None:
    """Write the summary CSV and the full JSON record."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Feature order, so a resumed run is byte-comparable with an uninterrupted one.
    results = sorted(results, key=lambda r: r["feature_idx"])

    csv_path = config.output_dir / "feature_descriptions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "feature_idx", "stratum", "blind", "pearson_r", "detection_auroc",
            "n_predicted", "holdout_coverage",
            "n_shuffles", "shuffled_r_values", "shuffled_null_p95",
            "description_beats_baseline",
            "firing_rate", "max_f1_dom", "best_concept", "n_significant_concepts",
            "unexplained_mass_fraction", "unexplained_enrichment",
            "causal_concept", "causal_selectivity", "causal_selectivity_z",
            "causal_mean_delta_ce", "summary",
        ])
        for r in results:
            fi = r["feature_idx"]
            s = stats.get(fi, {})
            c = causal.get(fi, {})
            shuffled = r.get("shuffled") or {}
            beats = r.get("description_beats_baseline")
            writer.writerow([
                fi, r["stratum"], r.get("blind", config.blind), r["pearson_r"],
                r.get("detection_auroc", ""), r["n_predicted"],
                r.get("holdout_coverage", ""),
                len(shuffled.get("r_values", [])),
                ";".join(repr(float(v)) for v in shuffled.get("r_values", [])),
                _blank_if_nan(r.get("shuffled_null_p95")),
                "" if beats is None else beats,
                s.get("firing_rate", ""), s.get("max_f1_dom", ""),
                s.get("best_concept", ""), s.get("n_significant", ""),
                _blank_if_nan(s.get("unexplained_fraction")),
                _blank_if_nan(s.get("unexplained_enrichment")),
                c.get("concept", ""), c.get("selectivity", ""),
                c.get("selectivity_z", ""), c.get("mean_delta_ce", ""),
                r["summary"],
            ])

    # Aggregated from the records themselves, so a run rebuilt entirely from the
    # resume log still reports the model that produced it and what it cost.
    shuffled_records = [r["shuffled"] for r in results if r.get("shuffled")]
    usage = {
        "models_served": sorted({r["model_served"] for r in results if r.get("model_served")}),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in results)
                         + sum(s["prompt_tokens"] for s in shuffled_records),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in results)
                             + sum(s["completion_tokens"] for s in shuffled_records),
        "n_api_calls": 2 * sum(1 for r in results if r.get("model_served"))
                       + sum(len(s["donors"]) for s in shuffled_records),
    }

    json_path = config.output_dir / "feature_descriptions.json"
    json_path.write_text(json.dumps(
        {"config": config.as_jsonable(), "chunk_indices_used": chunks_used,
         "usage": usage, "results": results},
        indent=2, default=str,
    ), encoding="utf-8")
    LOG.info("Wrote %s and %s", csv_path, json_path)
    if usage["prompt_tokens"] or usage["completion_tokens"]:
        LOG.info("API usage: %d calls to %s, %s prompt + %s completion tokens",
                 usage["n_api_calls"], ", ".join(usage["models_served"]) or "?",
                 f"{usage['prompt_tokens']:,}", f"{usage['completion_tokens']:,}")

    by_stratum: dict[str, list[dict]] = {}
    for r in results:
        by_stratum.setdefault(r["stratum"], []).append(r)
    LOG.info("Median description-prediction r by stratum "
             "(InterPLM reports 0.72 for protein features):")
    for stratum, rows in sorted(by_stratum.items()):
        rs = [r["pearson_r"] for r in rows if not math.isnan(r["pearson_r"])]
        aurocs = [r.get("detection_auroc", float("nan")) for r in rows]
        aurocs = [a for a in aurocs if not math.isnan(a)]
        beats = [r.get("description_beats_baseline") for r in rows]
        beats = [b for b in beats if b is not None]
        LOG.info("  %-11s n=%3d  median r = %s  median AUROC = %s  beats baseline %s",
                 stratum, len(rows),
                 f"{np.median(rs):.3f}" if rs else "n/a",
                 f"{np.median(aurocs):.3f}" if aurocs else "n/a",
                 f"{sum(beats)}/{len(beats)}" if beats else "n/a")


# --- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="layer_{L}/seed_{S}/eval directory from evaluate.py")
    p.add_argument("--extract-dir", type=Path, required=True)
    p.add_argument("--annotation-dir", type=Path, required=True)
    p.add_argument("--sae-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="interpret/layer_{L}; the run writes into <output-dir>/<run-tag>/")
    p.add_argument("--target-layer", type=int, required=True)
    p.add_argument("--run-tag", required=True,
                   help="Name of this description run, e.g. blind_v1. Separate runs never "
                        "share a directory, and a resume must match run_config.json.")
    p.add_argument("--blind", action="store_true",
                   help="Withhold matched_ion, neighbour labels and the peptide from both "
                        "prompts; keep only model inputs and the latent-token marker.")
    p.add_argument("--chunk-indices", type=int, nargs="+", default=None,
                   help="Encode exactly these chunk ids (all must be on disk for this "
                        "layer) instead of an even spread of --n-sample-chunks.")
    p.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES,
                   help="Shuffled-description predictions per feature for the baseline "
                        f"(default {DEFAULT_N_SHUFFLES}; 0 skips the baseline).")

    p.add_argument("--strata", nargs="+",
                   default=["concept", "unexplained", "unlabelled", "causal"],
                   choices=["concept", "unexplained", "unlabelled", "causal"])
    p.add_argument("--n-per-stratum", type=int, default=40)
    p.add_argument("--n-sample-chunks", type=int, default=12)
    p.add_argument("--include-concept-labels", action="store_true",
                   help="Show the 50 concept labels to the model. Off by default so "
                        "any chemistry it names is inferred from the spectra alone.")
    p.add_argument("--min-firing-rate", type=float, default=MIN_UNEXPLAINED_FIRING_RATE,
                   help="Firing-rate floor for the unexplained and unlabelled strata "
                        f"(default {MIN_UNEXPLAINED_FIRING_RATE:g}). Drops features too "
                        "rare to describe, including those whose unexplained-mass "
                        "fraction of 1.0 comes from firing on a handful of tokens.")

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--env-file", type=Path, default=None,
                   help="File of KEY=value pairs holding OPENAI_API_KEY. Defaults to "
                        "'.env' beside this script. Existing environment variables "
                        "take precedence over the file.")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto",
                   help="'auto' (default) uses cuda when a GPU is visible and cpu "
                        "otherwise. An explicit value is honoured as given.")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--dry-run", action="store_true",
                   help="Build the prompts, print an example of each and a cost "
                        "estimate, call nothing and write nothing.")
    p.add_argument("--rescore", action="store_true",
                   help="Recompute r, AUROC and the shuffled baseline from this run's resume "
                        "logs under the current scoring rule and rewrite its outputs. Calls "
                        "nothing; fails if any feature or shuffled record is missing.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    """Follow the hardware for 'auto'; honour anything explicit.

    Matches run_pipeline.sh: the default adapts so a CPU box works untouched,
    while an explicit --device cuda still fails loudly on a machine without one
    rather than silently running orders of magnitude slower.
    """
    if requested != "auto":
        return requested
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("Device auto-detected: %s", device)
    return device


def estimate_cost(
    prepared: dict[int, tuple[list[TokenExample], list[TokenExample]]],
    config: InterpretConfig,
) -> dict:
    """Tokens and dollars for the calls a real run would make on these prepared features.

    Prompt tokens come from the exact describe prompts and from the prediction
    prompts with a placeholder description of PLACEHOLDER_DESCRIPTION_CHARS,
    converted at CHARS_PER_TOKEN. Completion tokens are per-call averages. All
    three constants were calibrated against unblinded_v1, so this is a budget,
    not a measurement.
    """
    placeholder = "x" * PLACEHOLDER_DESCRIPTION_CHARS
    describe_chars = predict_chars = 0
    for describe, holdout in prepared.values():
        describe_chars += len(build_description_prompt(
            describe, config.include_concept_labels, blind=config.blind))
        predict_chars += len(build_prediction_prompt(
            placeholder, holdout, config.include_concept_labels, blind=config.blind))
    n = len(prepared)
    n_predict = 1 + config.n_shuffles
    prompt_tokens = (describe_chars + n_predict * predict_chars) / CHARS_PER_TOKEN
    completion_tokens = n * (DESCRIBE_COMPLETION_TOKENS + n_predict * PREDICT_COMPLETION_TOKENS)
    price = PRICE_PER_MTOK.get(config.model)
    return {
        "n_features": n,
        "n_calls": n * (1 + n_predict),
        "describe_prompt_chars": describe_chars,
        "predict_prompt_chars": predict_chars,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usd": None if price is None
        else (prompt_tokens * price[0] + completion_tokens * price[1]) / 1e6,
    }


def report_dry_run(
    prepared: dict[int, tuple[list[TokenExample], list[TokenExample]]],
    assigned: dict[int, str],
    config: InterpretConfig,
) -> int:
    """Print one example of each prompt and the cost estimate. Writes nothing."""
    if not prepared:
        LOG.error("Dry run: no feature could be prepared, so there is nothing to estimate.")
        return 1
    first = min(prepared)
    describe, holdout = prepared[first]
    mode = "blind" if config.blind else "unblinded"
    print(f"\n===== Description prompt: feature {first} ({assigned[first]}), {mode} =====")
    print(build_description_prompt(describe, config.include_concept_labels, blind=config.blind))
    print(f"\n===== Prediction prompt: feature {first}, {mode}, placeholder description =====")
    print(build_prediction_prompt(PLACEHOLDER_DESCRIPTION, holdout,
                                  config.include_concept_labels, blind=config.blind))

    est = estimate_cost(prepared, config)
    sizes = [(len(d), len(h)) for d, h in prepared.values()]
    print(f"\n===== Cost estimate: layer {config.target_layer}, run {config.run_tag} ({mode}) =====")
    print(f"features prepared       {est['n_features']} "
          f"(mean {np.mean([s[0] for s in sizes]):.1f} describe / "
          f"{np.mean([s[1] for s in sizes]):.1f} held-out examples)")
    print(f"API calls               {est['n_calls']} (1 describe + 1 predict + "
          f"{config.n_shuffles} shuffled per feature)")
    print(f"prompt tokens (est.)    {est['prompt_tokens']:,.0f} "
          f"(at {CHARS_PER_TOKEN} chars/token)")
    print(f"completion tokens (est.) {est['completion_tokens']:,.0f}")
    if est["usd"] is None:
        print(f"cost                    unknown: no price for model {config.model!r} in "
              f"PRICE_PER_MTOK")
    else:
        price = PRICE_PER_MTOK[config.model]
        print(f"cost (est.)             ${est['usd']:.2f} at ${price[0]:.2f} / "
              f"${price[1]:.2f} per M input / output tokens ({config.model})")
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    if args.blind and args.include_concept_labels:
        LOG.error("--blind and --include-concept-labels contradict each other: concept "
                  "labels are annotation.")
        return 2
    if args.n_shuffles < 0:
        LOG.error("--n-shuffles must be >= 0")
        return 2
    try:
        check_run_tag(args.run_tag)
    except ValueError as exc:
        LOG.error("--run-tag: %s", exc)
        return 2
    if args.rescore and args.dry_run:
        LOG.error("--rescore and --dry-run are separate modes; pass one.")
        return 2

    config = InterpretConfig(
        eval_dir=args.eval_dir,
        extract_dir=args.extract_dir,
        annotation_dir=args.annotation_dir,
        sae_checkpoint=args.sae_checkpoint,
        output_dir=args.output_dir / args.run_tag,
        target_layer=args.target_layer,
        run_tag=args.run_tag,
        blind=args.blind,
        chunk_indices=tuple(args.chunk_indices) if args.chunk_indices else None,
        n_shuffles=args.n_shuffles,
        strata=tuple(args.strata),
        n_per_stratum=args.n_per_stratum,
        n_sample_chunks=args.n_sample_chunks,
        include_concept_labels=args.include_concept_labels,
        min_firing_rate=args.min_firing_rate,
        model=args.model,
        max_tokens=args.max_tokens,
        seed=args.seed,
        device=resolve_device(args.device),
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        rescore=args.rescore,
    )
    LOG.info("Run %s (%s), layer %d, output %s%s", config.run_tag,
             "blind" if config.blind else "unblinded", config.target_layer,
             config.output_dir, " [dry run: nothing is written]" if config.dry_run else "")

    # Strata selection only. Everything per feature draws from feature_rng.
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    stats = _read_per_feature_stats(config.eval_dir)
    causal = _read_causal_features(config.eval_dir)
    global_top = _read_global_top_tokens(config.eval_dir)
    ion_vocab = _read_ion_type_vocab(config.annotation_dir)
    concept_names = _read_concept_names(config.annotation_dir)

    assigned = select_strata(stats, causal, config.n_per_stratum, config.strata, rng,
                             min_firing_rate=config.min_firing_rate)
    if not assigned:
        LOG.error("No features selected; check --strata and the eval directory.")
        return 1
    feature_ids = sorted(assigned)
    LOG.info("Interpreting %d features across %d strata",
             len(feature_ids), len(set(assigned.values())))

    available, total_chunks = available_chunk_indices(config.extract_dir, config.target_layer)
    if not available:
        LOG.error(
            "No activation files for layer %d survive under %s, though the manifest "
            "lists %d chunks. A KEEP_CHUNKS=0 run deletes them after evaluating. "
            "Re-extract this layer to interpret it.",
            config.target_layer, config.extract_dir, total_chunks,
        )
        return 1
    if len(available) < total_chunks and config.chunk_indices is None:
        # Say where the survivors sit, not just how many. A KEEP_CHUNKS=0 prune
        # leaves the first few, clustered at the head of an unshuffled corpus; a
        # deliberately published sample is spread across it. Those two cases carry
        # very different weight, and only the span distinguishes them.
        span = (available[-1] - available[0] + 1) / total_chunks if len(available) > 1 else 0.0
        LOG.warning(
            "Only %d of %d chunks hold layer-%d activations, so sampling is confined "
            "to them: %.1f%% of the corpus, spanning chunks %d to %d (%.0f%% of its "
            "length). Chunks are in dataset order and the dataset is not shuffled, so "
            "a narrow span means a narrow slice of the data, not a random one.",
            len(available), total_chunks, config.target_layer,
            100 * len(available) / total_chunks,
            available[0], available[-1], 100 * span,
        )
    try:
        chosen = resolve_chunk_indices(available, config.chunk_indices, config.n_sample_chunks)
    except ValueError as exc:
        LOG.error("%s", exc)
        return 1
    LOG.info("Chunk indices used (%s, %d of %d on disk): %s",
             "explicit --chunk-indices" if config.chunk_indices else "even spread",
             len(chosen), len(available), " ".join(str(c) for c in chosen))

    # Both of these come before the encode, which is the expensive part: a
    # missing key or an already-finished feature set should not cost a pass over
    # the chunks first.
    client = None
    completed: dict[int, dict] = {}
    if not config.dry_run:
        if not config.rescore:
            env_file = args.env_file or Path(__file__).parent / ".env"
            n_loaded = load_dotenv(env_file)
            if n_loaded:
                LOG.info("Loaded %d variable(s) from %s", n_loaded, env_file)
            client = _make_client()
            if client is None:
                return 1
        try:
            check_run_config(config.output_dir, config.resume_key(chosen))
        except ValueError as exc:
            LOG.error("%s", exc)
            return 1
        completed = load_completed(config.output_dir)
        if completed:
            LOG.info("Resuming: %d feature(s) already described in %s",
                     len(completed), _resume_path(config.output_dir).name)
        n_rescored = sum(rescore_result(r) for r in completed.values())
        if n_rescored:
            LOG.info("Rescored %d logged feature(s) under the current r rule", n_rescored)
        # A feature can change stratum between runs without its description
        # becoming wrong: adding Phase 8 promotes features into `causal` that a
        # Phase-8-less run filed under `concept` or `unexplained`, since causal
        # takes precedence. The description was written from the same examples
        # either way, so relabel it rather than paying to regenerate it.
        for fi, record in completed.items():
            if fi in assigned and record.get("stratum") != assigned[fi]:
                LOG.info("Feature %d moved from stratum %s to %s; keeping its description",
                         fi, record.get("stratum"), assigned[fi])
                record["stratum"] = assigned[fi]

    todo = [fi for fi in feature_ids if fi not in completed]
    results: list[dict] = [completed[fi] for fi in feature_ids if fi in completed]
    if config.rescore and todo:
        LOG.error("--rescore: %d selected feature(s) have no record in %s (first: %s); "
                  "rescoring would need API calls.", len(todo),
                  _resume_path(config.output_dir).name, todo[:5])
        return 1
    prepared: dict[int, tuple[list[TokenExample], list[TokenExample]]] = {}
    skipped = failed = 0
    stopped = False

    if todo:
        sae = load_sae_from_checkpoint(config.sae_checkpoint, device=config.device)
        stream = ChunkStream(
            extract_dir=config.extract_dir,
            annotation_dir=config.annotation_dir,
            target_layer=config.target_layer,
            sae=sae,
            device=config.device,
            batch_size=config.batch_size,
            dtype=torch.float32,
            chunk_indices=chosen,
            # Encode straight into the columns we need. The full dictionary is 5 GB
            # per chunk and we keep 80 columns of it.
            feature_subset=todo,
        )
        activations, records = _collect_activations(stream, todo, len(chosen))

        for n, fi in enumerate(todo, start=1):
            split = prepare_feature(
                fi, activations[fi], global_top.get(fi, []), records, ion_vocab,
                concept_names, config.seed, config.target_layer,
            )
            if split is None:
                LOG.warning("Feature %d: fewer than %d active examples in the sample (or one "
                            "spectrum holds them all); skipping", fi, MIN_ACTIVE_EXAMPLES)
                skipped += 1
                continue
            describe_set, holdout_set = split
            if config.dry_run:
                prepared[fi] = split
                continue

            LOG.info("[%d/%d] feature %d (%s), %d describe / %d held-out examples",
                     n, len(todo), fi, assigned[fi], len(describe_set), len(holdout_set))
            try:
                result = interpret_feature(client, config, fi, assigned[fi],
                                           describe_set, holdout_set)
            except PermanentAPIError as exc:
                # Nothing about the next feature would go differently, and the
                # partial file already holds everything paid for so far.
                LOG.error("Feature %d hit a permanent API error, stopping: %s", fi, exc)
                stopped = True
                break
            except RuntimeError as exc:
                LOG.error("Feature %d failed: %s", fi, exc)
                failed += 1
                continue
            results.append(result)
            append_completed(config.output_dir, result)

    if config.dry_run:
        return report_dry_run(prepared, assigned, config)

    if skipped:
        LOG.info("Skipped %d/%d features with too few active examples; raise "
                 "--n-sample-chunks to reach rarer features.", skipped, len(todo))

    # The donors are drawn from the set of described features, so the baseline
    # waits until that set is final: a failed feature would otherwise change it
    # between this attempt and the resume that fills it in.
    complete = not stopped and failed == 0
    if config.n_shuffles > 0:
        if complete:
            try:
                run_shuffled_baseline(client, config, results)
            except (PermanentAPIError, RuntimeError, ValueError) as exc:
                LOG.error("Shuffled baseline stopped: %s", exc)
                if config.rescore:
                    # Rescoring rewrites finished outputs; never replace them
                    # with a version missing its baseline.
                    return 1
                complete = False
        else:
            LOG.warning("Shuffled baseline not run: %d feature(s) failed or the run stopped. "
                        "Rerun with the same arguments to finish both.", failed + int(stopped))
    threshold = apply_baseline(results)
    if not math.isnan(threshold):
        LOG.info("Shuffled-description null: %.1fth percentile r = %.3f",
                 BASELINE_PERCENTILE, threshold)
    write_outputs(results, stats, causal, config, chosen)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
