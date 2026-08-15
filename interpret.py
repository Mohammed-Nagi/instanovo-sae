"""interpret.py -- LLM-assisted interpretation of SAE features.

Adapts the automated feature-description pipeline of InterPLM (Simon and Zou,
2025, Methods 5.3) to the mass-spectrometry setting, and adds a causal
cross-reference that the protein-language-model setting does not support.

Motivation. The concept registry in annotate.py labels a peak only when it
matches a theoretical fragment ion; the remaining peaks carry structural labels
only. Features that fire preferentially on those peaks are invisible to the
F1-based evaluation even when they are perfectly interpretable. Separately,
F1-dom penalises a feature for firing on concept-negative tokens, which
systematically demotes detectors of concepts that co-occur with others -- so the
correlational ranking is biased in a way an LLM reading raw examples is not.
This script probes both gaps.

Three feature strata are interpreted (see --strata):

    A  concept    features with the strongest BH-significant concept association.
                  POSITIVE CONTROL: the chemistry is already known from Phase 4,
                  so recovering it without being told validates the pipeline.
    B  unlabelled features with no BH-significant concept at all. DISCOVERY SET:
                  these are the features the registry cannot describe, and the
                  reason the reported interpretable fraction is a lower bound.
    C  causal     features implicated by the Phase 8 ablations. CROSS-REFERENCE:
                  asks whether causally necessary features have describable
                  structure, including ones the F1 ranking demoted.

Concept labels are WITHHELD from the prompt by default (--include-concept-labels
to override). InterPLM supplies Swiss-Prot metadata, so its descriptions are
partly retrieved rather than inferred; here the model sees only physical
observables -- m/z, intensity, peptide, precursor -- so any chemistry it names is
inferred from the spectra. For stratum B this also avoids feeding back the very
labels whose bias the analysis is trying to circumvent.

Validation. Descriptions are scored the way InterPLM scores them: a held-out set
of examples is shown WITHOUT activations, the model predicts each activation, and
the Pearson correlation against the measured values is reported. A description
that cannot predict held-out activations is not evidence of anything. InterPLM
reports a median r of 0.72 across 1,200 protein features.

Inputs (all produced by an existing evaluate.py run):
    <eval-dir>/per_feature_stats.csv        feature ranking and concept association
    <eval-dir>/top_activating_tokens.csv    global top-K provenance (Phase 3)
    <eval-dir>/report.json                  Phase 8 causal results, if present
plus the extract/annotation chunks and the SAE checkpoint, to recover the full
activation distribution (Phase 3 stores only the top-K, and the mid-range and
zero examples are what make the prediction task non-trivial).

Output under --output-dir:
    feature_descriptions.csv    one row per feature: stratum, summary, r, n
    feature_descriptions.json   full descriptions, examples, and predictions
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

# A feature with too few active examples cannot be characterised, and its
# prediction r would be dominated by the zero examples. InterPLM excludes
# features with fewer than 20 examples across the top activation ranges.
MIN_ACTIVE_EXAMPLES = 12

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

    strata: tuple[str, ...] = ("concept", "unlabelled", "causal")
    n_per_stratum: int = 40

    # Chunks encoded to recover the activation distribution. Each contributes
    # ~108k tokens, so a handful is ample for sampling examples per feature.
    n_sample_chunks: int = 12
    include_concept_labels: bool = False

    model: str = DEFAULT_MODEL
    max_tokens: int = 2048
    seed: int = 0
    device: str = "cuda"
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
    neighbour_mzs: list[float]
    concepts: list[str]        # withheld from the prompt unless requested


# --- Reading the evaluation artefacts ----------------------------------------

def _read_per_feature_stats(eval_dir: Path) -> dict[int, dict]:
    """feature_idx -> {firing_rate, max_f1_dom, best_concept, n_significant}."""
    path = eval_dir / "per_feature_stats.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run evaluate.py Phase 4 for this layer/seed first."
        )
    stats: dict[int, dict] = {}
    with open(path, newline="") as fp:
        for row in csv.DictReader(fp):
            stats[int(row["feature_idx"])] = {
                "firing_rate": float(row["firing_rate"]),
                "max_f1_dom": float(row["max_f1_dom"]),
                "best_concept": row["best_concept"],
                "n_significant": int(row["n_significant_concepts"]),
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
    report = json.loads(path.read_text())
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
    with open(path, newline="") as fp:
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
    manifest = json.loads((annotation_dir / "annotation_manifest.json").read_text())
    vocab = manifest.get("vocab", {}).get("ion_type", {})
    return {int(v): k for k, v in vocab.items()}


def _read_concept_names(annotation_dir: Path) -> list[str]:
    manifest = json.loads((annotation_dir / "annotation_manifest.json").read_text())
    return list(manifest["registry"]["names"])


# --- Feature selection --------------------------------------------------------

def select_strata(
    stats: dict[int, dict],
    causal: dict[int, dict],
    n_per_stratum: int,
    strata: tuple[str, ...],
    rng: random.Random,
) -> dict[int, str]:
    """Choose features for each requested stratum. Returns feature_idx -> stratum.

    A feature is assigned to exactly one stratum, with causal taking precedence:
    a causally implicated feature is more informative there than as one more
    concept-associated example, and the causal set is the smallest.
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

    if "concept" in strata:
        # Highest F1-dom among features the registry can already describe.
        candidates = sorted(
            (fi for fi, s in stats.items()
             if s["n_significant"] > 0 and fi not in assigned),
            key=lambda fi: -stats[fi]["max_f1_dom"],
        )
        for fi in candidates[:n_per_stratum]:
            assigned[fi] = "concept"
        LOG.info("Stratum concept: %d features", sum(1 for s in assigned.values() if s == "concept"))

    if "unlabelled" in strata:
        # Features with no significant concept, sampled rather than ranked:
        # there is no meaningful ordering, and sampling keeps the set unbiased
        # with respect to firing rate.
        candidates = [
            fi for fi, s in stats.items()
            if s["n_significant"] == 0 and s["firing_rate"] > 0 and fi not in assigned
        ]
        rng.shuffle(candidates)
        for fi in candidates[:n_per_stratum]:
            assigned[fi] = "unlabelled"
        LOG.info("Stratum unlabelled: %d features",
                 sum(1 for s in assigned.values() if s == "unlabelled"))

    return assigned


# --- Building example tables --------------------------------------------------

def _relative_intensity(intensity: float, spectrum_max: float) -> float:
    return float(intensity / spectrum_max) if spectrum_max > 0 else 0.0


def _collect_activations(
    stream: ChunkStream,
    feature_ids: list[int],
    n_chunks: int,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Encode a sample of chunks and keep each target feature's activation column.

    Returns (feature_idx -> activations over the sampled tokens, chunk records).
    The chunk records hold the per-token and per-spectrum metadata needed to turn
    a token index back into a readable example.

    MEMORY. ChunkStream materialises the full dense [n_tokens, d_dict] feature
    matrix for one chunk at a time -- about 5 GB at the default chunk size and
    d_dict = 12,288 -- which is freed as the loop advances. Only the selected
    columns are retained, so the lasting cost is
    n_chunks * n_tokens * len(feature_ids) * 4 bytes (roughly 600 MB for 12
    chunks and 120 features). Lower --n-sample-chunks on a small machine.
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
        neighbours: list[float] = []
    else:
        order = np.argsort(-intensities)
        rank_of = {int(peak_tokens[o]): r + 1 for r, o in enumerate(order)}
        peak_rank = rank_of.get(local_token, 0)
        mzs = rec["peak_mzs"][peak_tokens]
        near = np.argsort(np.abs(mzs - mz))[1:CONTEXT_PEAKS + 1]
        neighbours = [round(float(mzs[i]), 4) for i in sorted(near)]

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
        neighbour_mzs=neighbours,
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

    zeros = np.flatnonzero(normalised == 0)
    if zeros.size:
        chosen.extend(rng.sample(list(zeros), min(ZERO_EXAMPLES, zeros.size)))

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

    The sampled chunks rarely contain a feature's global maximum, and the
    strongest examples carry the most information about what it detects. `peak`
    is the shared scale from feature_peak, so these rows are directly comparable
    to the sampled ones.
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

    The table is joined on commas by hand rather than via csv.writer, because the
    model reads it as plain text. A comma or newline inside a field would shift
    every column after it, so the row would still parse but describe the wrong
    peak. No ProForma string in the nine-species benchmark contains either; this
    is here so a different dataset cannot corrupt the table silently.
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
        "precursor_mz", "precursor_z", "nearby_mz",
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
            " ".join(f"{m:.2f}" for m in e.neighbour_mzs),
        ]
        if include_concepts:
            cells.append(_csv_safe(" ".join(e.concepts)))
        if include_activation:
            cells.insert(1, f"{e.activation:.3f}")
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
  nearby_mz        m/z of the nearest other peaks in the same spectrum
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

def _make_client():
    """OpenAI client, or None when the SDK or key is unavailable."""
    try:
        from openai import OpenAI
    except ImportError:
        LOG.error("The 'openai' package is required. Install with: uv add openai")
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        LOG.error("OPENAI_API_KEY is not set.")
        return None
    return OpenAI()


def _create_completion(client, model: str, prompt: str, max_tokens: int) -> str:
    """One chat completion.

    The output-length parameter was renamed: older chat models take max_tokens,
    while the reasoning models reject it and require max_completion_tokens. The
    first call decides which this model wants, so the caller need not know.
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
    except Exception as exc:  # noqa: BLE001 -- inspect the message, then re-raise
        if "max_completion_tokens" not in str(exc):
            raise
        response = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=max_tokens,
        )
    return response.choices[0].message.content or ""


def call_model(client, model: str, prompt: str, max_tokens: int, retries: int = 3) -> str:
    """One completion, retrying on transient failures with linear backoff."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return _create_completion(client, model, prompt, max_tokens)
        except Exception as exc:  # noqa: BLE001 -- surface any API failure to the caller
            last = exc
            wait = 5 * (attempt + 1)
            LOG.warning("Model call failed (attempt %d/%d): %s; retrying in %ds",
                        attempt + 1, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Model call failed after {retries} attempts: {last}")


_DESCRIPTION_RE = re.compile(r"DESCRIPTION:\s*(.*?)\s*SUMMARY:", re.S)
_SUMMARY_RE = re.compile(r"SUMMARY:\s*(.*)", re.S)


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

    description, summary = parse_description(
        call_model(client, config.model, description_prompt, config.max_tokens)
    )
    result["description"] = description
    result["summary"] = summary

    prediction_prompt = build_prediction_prompt(
        description, holdout_set, config.include_concept_labels,
    )
    predictions = parse_predictions(
        call_model(client, config.model, prediction_prompt, config.max_tokens),
        {e.example_id for e in holdout_set},
    )
    measured = {e.example_id: e.activation for e in holdout_set}
    pairs = [(predictions[i], measured[i]) for i in predictions]

    result["pearson_r"] = pearson_r(pairs)
    result["n_predicted"] = len(pairs)
    result["predictions"] = [
        {"example_id": i, "predicted": predictions[i], "measured": measured[i]}
        for i in sorted(predictions)
    ]
    result["holdout_examples"] = [dataclasses.asdict(e) for e in holdout_set]
    return result


# --- Output -------------------------------------------------------------------

def write_outputs(
    results: list[dict],
    stats: dict[int, dict],
    causal: dict[int, dict],
    config: InterpretConfig,
) -> None:
    """Write the summary CSV and the full JSON record."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = config.output_dir / "feature_descriptions.csv"
    with open(csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "feature_idx", "stratum", "pearson_r", "n_predicted",
            "firing_rate", "max_f1_dom", "best_concept", "n_significant_concepts",
            "causal_concept", "causal_selectivity", "causal_selectivity_z",
            "causal_mean_delta_ce", "summary",
        ])
        for r in results:
            fi = r["feature_idx"]
            s = stats.get(fi, {})
            c = causal.get(fi, {})
            writer.writerow([
                fi, r["stratum"], r["pearson_r"], r["n_predicted"],
                s.get("firing_rate", ""), s.get("max_f1_dom", ""),
                s.get("best_concept", ""), s.get("n_significant", ""),
                c.get("concept", ""), c.get("selectivity", ""),
                c.get("selectivity_z", ""), c.get("mean_delta_ce", ""),
                r["summary"],
            ])

    json_path = config.output_dir / "feature_descriptions.json"
    json_path.write_text(json.dumps(
        {"config": config.as_jsonable(), "results": results},
        indent=2, default=str,
    ))
    LOG.info("Wrote %s and %s", csv_path, json_path)

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

    p.add_argument("--strata", nargs="+", default=["concept", "unlabelled", "causal"],
                   choices=["concept", "unlabelled", "causal"])
    p.add_argument("--n-per-stratum", type=int, default=40)
    p.add_argument("--n-sample-chunks", type=int, default=12)
    p.add_argument("--include-concept-labels", action="store_true",
                   help="Show the 50 concept labels to the model. Off by default so "
                        "any chemistry it names is inferred from the spectra alone.")

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--dry-run", action="store_true",
                   help="Build prompts and write them out without calling the API.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
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
        model=args.model,
        max_tokens=args.max_tokens,
        seed=args.seed,
        device=args.device,
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

    assigned = select_strata(stats, causal, config.n_per_stratum, config.strata, rng)
    if not assigned:
        LOG.error("No features selected; check --strata and the eval directory.")
        return 1
    feature_ids = sorted(assigned)
    LOG.info("Interpreting %d features across %d strata",
             len(feature_ids), len(set(assigned.values())))

    sae = load_sae_from_checkpoint(config.sae_checkpoint, device=config.device)
    stream = ChunkStream(
        extract_dir=config.extract_dir,
        annotation_dir=config.annotation_dir,
        target_layer=config.target_layer,
        sae=sae,
        device=config.device,
        batch_size=config.batch_size,
        dtype=torch.float32,
    )
    activations, records = _collect_activations(stream, feature_ids, config.n_sample_chunks)

    client = None if config.dry_run else _make_client()
    if client is None and not config.dry_run:
        return 1

    results: list[dict] = []
    skipped = 0
    for n, fi in enumerate(feature_ids, start=1):
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
                 n, len(feature_ids), fi, assigned[fi], len(examples))
        try:
            results.append(interpret_feature(
                client, config, fi, assigned[fi], examples, rng,
            ))
        except RuntimeError as exc:
            LOG.error("Feature %d failed: %s", fi, exc)

    if skipped:
        LOG.info("Skipped %d/%d features with too few active examples; raise "
                 "--n-sample-chunks to reach rarer features.", skipped, len(feature_ids))
    write_outputs(results, stats, causal, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
