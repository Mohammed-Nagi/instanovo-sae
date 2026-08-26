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

Outputs under --output-dir:
    feature_descriptions.csv            one row per feature: stratum, summary, r
    feature_descriptions.json           full descriptions, examples, predictions
    feature_descriptions.partial.jsonl  resume log, appended as each feature lands

Rerunning skips features already in the resume log, so an interrupted run does
not pay for them twice.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import ChunkStream
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


@dataclasses.dataclass
class InterpretConfig:
    """Configuration for one interpretation run."""

    eval_dir: Path
    extract_dir: Path
    annotation_dir: Path
    sae_checkpoint: Path
    output_dir: Path
    target_layer: int

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

    def as_jsonable(self) -> dict:
        out = dataclasses.asdict(self)
        for key in ("eval_dir", "extract_dir", "annotation_dir",
                    "sae_checkpoint", "output_dir"):
            out[key] = str(out[key])
        out["strata"] = list(self.strata)
        return out


@dataclasses.dataclass
class TokenExample:
    """One activating (or non-activating) token, with its spectral context."""

    example_id: int
    activation: float          # normalised to the feature's observed maximum
    chunk_idx: int
    token_in_chunk: int
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
    return abs(value) if value == value else -float("inf")


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


def _collect_activations(
    stream: ChunkStream,
    feature_ids: list[int],
    n_chunks: int,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Encode a sample of chunks and keep each target feature's activation column.

    Returns (feature_idx -> activations over the sampled tokens, chunk records).
    The chunk records hold the per-token and per-spectrum metadata needed to turn
    a token index back into a readable example.

    MEMORY. ChunkStream materialises one chunk's dense [n_tokens, d_dict] feature
    matrix at a time, about 5 GB at d_dict = 12,288, freed as the loop advances.
    Only the selected columns persist: n_chunks * n_tokens * len(feature_ids) * 4
    bytes, roughly 600 MB for 12 chunks and 120 features. That transient 5 GB is
    the binding constraint on a small machine, and --n-sample-chunks does not
    shrink it; it is set by the chunk size chosen at extraction.
    """
    feature_index = {fi: i for i, fi in enumerate(feature_ids)}
    columns: list[np.ndarray] = []
    records: list[dict] = []
    offset = 0

    for chunk in stream:
        feats = chunk.features[:, feature_ids].to(torch.float32).numpy()
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


# --- Prompt construction ------------------------------------------------------

def _csv_safe(value: str) -> str:
    """Strip separators from a free-text cell.

    The table is joined on commas by hand, since the model reads it as plain
    text. A comma or newline inside a field would shift every column after it and
    the row would still parse, describing the wrong peak. Nothing in the
    nine-species benchmark contains either; this guards a different dataset.
    """
    return str(value).replace(",", ";").replace("\n", " ").replace("\r", " ")


def format_example_table(
    examples: list[TokenExample],
    include_activation: bool,
    include_concepts: bool,
) -> str:
    """Render examples as a compact table for the model."""
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
        token_kind = "LATENT_SUMMARY_TOKEN" if e.is_latent else e.ion_type
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


def build_description_prompt(examples: list[TokenExample], include_concepts: bool) -> str:
    table = format_example_table(examples, include_activation=True,
                                 include_concepts=include_concepts)
    return f"""{DOMAIN_PREAMBLE}
Analyse this table to determine what predicts the 'activation' column. Activation
is normalised so 1.0 is the strongest response observed for this feature and 0.0
means the feature did not fire.

Your description will be used to predict activations for held-out tokens given
only the other columns, so include only what is useful for that. Consider:

  - which physical properties (m/z region, relative intensity, charge, position
    in the spectrum) separate high from medium from zero activation
  - whether activation tracks a fragment-ion series, a neutral loss, a mass
    difference between peaks, or a specific residue or modification in the
    peptide
  - whether the feature responds to the peak itself or to its spectral context
  - if the pattern is a mass relationship, state the approximate mass difference

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
) -> str:
    table = format_example_table(examples, include_activation=False,
                                 include_concepts=include_concepts)
    return f"""{DOMAIN_PREAMBLE}
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
    """Correlation between predicted and measured activations."""
    if len(pairs) < 3:
        return float("nan")
    predicted = np.array([p for p, _ in pairs], dtype=np.float64)
    measured = np.array([m for _, m in pairs], dtype=np.float64)
    if predicted.std() < 1e-12 or measured.std() < 1e-12:
        # A constant prediction has no correlation defined; report NaN rather
        # than a spurious 0, which would read as "uncorrelated" instead of
        # "uninformative".
        return float("nan")
    return float(np.corrcoef(predicted, measured)[0, 1])


# --- Per-feature driver -------------------------------------------------------

def interpret_feature(
    client,
    config: InterpretConfig,
    feature_idx: int,
    stratum: str,
    examples: list[TokenExample],
    rng: random.Random,
) -> dict:
    """Describe one feature and score the description on held-out examples.

    The split is random rather than by activation level, so both halves span the
    dynamic range and the prediction task is the same task the description was
    written for.
    """
    shuffled = list(examples)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    describe_set, holdout_set = shuffled[:half], shuffled[half:]

    result: dict = {
        "feature_idx": feature_idx,
        "stratum": stratum,
        "n_describe": len(describe_set),
        "n_holdout": len(holdout_set),
        "description": "",
        "summary": "",
        "pearson_r": float("nan"),
        "n_predicted": 0,
    }

    description_prompt = build_description_prompt(describe_set, config.include_concept_labels)
    if config.dry_run:
        result["description_prompt"] = description_prompt
        return result

    reply, describe_meta = call_model(
        client, config.model, description_prompt, config.max_tokens,
    )
    description, summary = parse_description(reply)
    result["description"] = description
    result["summary"] = summary

    prediction_prompt = build_prediction_prompt(
        description, holdout_set, config.include_concept_labels,
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
    pairs = [(predictions[i], measured[i]) for i in predictions]

    # A reply that skips the examples it found hard would score on an easier
    # subset than the one it was given, so record the coverage alongside r.
    coverage = len(pairs) / len(holdout_set) if holdout_set else 0.0
    result["holdout_coverage"] = coverage
    if coverage < MIN_PREDICTION_COVERAGE:
        LOG.warning(
            "Feature %d: model predicted only %d/%d held-out examples (%.0f%%); "
            "its r is over that subset, not the full holdout.",
            feature_idx, len(pairs), len(holdout_set), 100 * coverage,
        )

    result["pearson_r"] = pearson_r(pairs)
    result["n_predicted"] = len(pairs)
    result["predictions"] = [
        {"example_id": i, "predicted": predictions[i], "measured": measured[i]}
        for i in sorted(predictions)
    ]
    result["holdout_examples"] = [dataclasses.asdict(e) for e in holdout_set]
    return result


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
    """Persist one feature as soon as it is done, before the run can die.

    A run killed mid-write leaves a line with no newline on the end. Appending
    straight onto it would fuse the two records into one unparseable line and
    lose this feature as well as that one, so the separator is restored first.
    """
    path = _resume_path(output_dir)
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


def write_outputs(
    results: list[dict],
    stats: dict[int, dict],
    causal: dict[int, dict],
    config: InterpretConfig,
) -> None:
    """Write the summary CSV and the full JSON record."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Feature order, so a resumed run is byte-comparable with an uninterrupted one.
    results = sorted(results, key=lambda r: r["feature_idx"])

    csv_path = config.output_dir / "feature_descriptions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "feature_idx", "stratum", "pearson_r", "n_predicted", "holdout_coverage",
            "firing_rate", "max_f1_dom", "best_concept", "n_significant_concepts",
            "unexplained_mass_fraction", "unexplained_enrichment",
            "causal_concept", "causal_selectivity", "causal_selectivity_z",
            "causal_mean_delta_ce", "summary",
        ])
        for r in results:
            fi = r["feature_idx"]
            s = stats.get(fi, {})
            c = causal.get(fi, {})
            writer.writerow([
                fi, r["stratum"], r["pearson_r"], r["n_predicted"],
                r.get("holdout_coverage", ""),
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
    usage = {
        "models_served": sorted({r["model_served"] for r in results if r.get("model_served")}),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in results),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in results),
        "n_api_calls": 2 * sum(1 for r in results if r.get("model_served")),
    }

    json_path = config.output_dir / "feature_descriptions.json"
    json_path.write_text(json.dumps(
        {"config": config.as_jsonable(), "usage": usage, "results": results},
        indent=2, default=str,
    ), encoding="utf-8")
    LOG.info("Wrote %s and %s", csv_path, json_path)
    if usage["prompt_tokens"] or usage["completion_tokens"]:
        LOG.info("API usage: %d calls to %s, %s prompt + %s completion tokens",
                 usage["n_api_calls"], ", ".join(usage["models_served"]) or "?",
                 f"{usage['prompt_tokens']:,}", f"{usage['completion_tokens']:,}")

    by_stratum: dict[str, list[float]] = {}
    for r in results:
        if r["pearson_r"] == r["pearson_r"]:
            by_stratum.setdefault(r["stratum"], []).append(r["pearson_r"])
    LOG.info("Median description-prediction r by stratum "
             "(InterPLM reports 0.72 for protein features):")
    for stratum, values in sorted(by_stratum.items()):
        LOG.info("  %-11s n=%3d  median r = %.3f", stratum, len(values), float(np.median(values)))


# --- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="layer_{L}/seed_{S}/eval directory from evaluate.py")
    p.add_argument("--extract-dir", type=Path, required=True)
    p.add_argument("--annotation-dir", type=Path, required=True)
    p.add_argument("--sae-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-layer", type=int, required=True)

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
                   help="Build prompts and write them out without calling the API.")
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


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    config = InterpretConfig(
        eval_dir=args.eval_dir,
        extract_dir=args.extract_dir,
        annotation_dir=args.annotation_dir,
        sae_checkpoint=args.sae_checkpoint,
        output_dir=args.output_dir,
        target_layer=args.target_layer,
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
    )

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

    # Both of these come before the encode, which is the expensive part: a
    # missing key or an already-finished feature set should not cost a pass over
    # the chunks first.
    client = None
    completed: dict[int, dict] = {}
    if not config.dry_run:
        env_file = args.env_file or Path(__file__).parent / ".env"
        n_loaded = load_dotenv(env_file)
        if n_loaded:
            LOG.info("Loaded %d variable(s) from %s", n_loaded, env_file)
        client = _make_client()
        if client is None:
            return 1
        config.output_dir.mkdir(parents=True, exist_ok=True)
        completed = load_completed(config.output_dir)
        if completed:
            LOG.info("Resuming: %d feature(s) already described in %s",
                     len(completed), _resume_path(config.output_dir).name)
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
    if not todo:
        LOG.info("Every selected feature is already described; rewriting outputs only.")
        write_outputs([completed[fi] for fi in feature_ids if fi in completed],
                      stats, causal, config)
        return 0

    available, total_chunks = available_chunk_indices(config.extract_dir, config.target_layer)
    if not available:
        LOG.error(
            "No activation files for layer %d survive under %s, though the manifest "
            "lists %d chunks. A KEEP_CHUNKS=0 run deletes them after evaluating. "
            "Re-extract this layer to interpret it.",
            config.target_layer, config.extract_dir, total_chunks,
        )
        return 1
    if len(available) < total_chunks:
        LOG.warning(
            "Only %d of %d chunks still hold layer-%d activations (KEEP_CHUNKS=0 "
            "prunes to a cross-layer sample). Sampling is confined to those, so the "
            "examples come from %.1f%% of the corpus and from its beginning rather "
            "than spread across it.",
            len(available), total_chunks, config.target_layer,
            100 * len(available) / total_chunks,
        )
    picks = spread_chunk_indices(len(available), config.n_sample_chunks)
    chosen = [available[i] for i in picks]
    LOG.info("Sampling %d chunk(s) of the %d available: %s",
             len(chosen), len(available),
             ", ".join(str(c) for c in chosen[:8]) + (", ..." if len(chosen) > 8 else ""))

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
    )
    activations, records = _collect_activations(stream, todo, len(chosen))

    results: list[dict] = [completed[fi] for fi in feature_ids if fi in completed]
    skipped = 0
    for n, fi in enumerate(todo, start=1):
        top_k = global_top.get(fi, [])
        peak = feature_peak(activations[fi], top_k)
        examples = build_examples(
            activations[fi], peak, records, ion_vocab, concept_names, rng,
        )
        if not examples:
            LOG.warning("Feature %d: fewer than %d active examples in the sample; skipping",
                        fi, MIN_ACTIVE_EXAMPLES)
            skipped += 1
            continue
        examples = merge_global_top(
            examples, top_k, peak, records, ion_vocab, concept_names,
        )

        LOG.info("[%d/%d] feature %d (%s), %d examples",
                 n, len(todo), fi, assigned[fi], len(examples))
        try:
            result = interpret_feature(client, config, fi, assigned[fi], examples, rng)
        except PermanentAPIError as exc:
            # Nothing about the next feature would go differently, and the
            # partial file already holds everything paid for so far.
            LOG.error("Feature %d hit a permanent API error, stopping: %s", fi, exc)
            break
        except RuntimeError as exc:
            LOG.error("Feature %d failed: %s", fi, exc)
            continue
        results.append(result)
        if not config.dry_run:
            append_completed(config.output_dir, result)

    if skipped:
        LOG.info("Skipped %d/%d features with too few active examples; raise "
                 "--n-sample-chunks to reach rarer features.", skipped, len(todo))
    write_outputs(results, stats, causal, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
