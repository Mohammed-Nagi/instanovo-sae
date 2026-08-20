"""evaluate.py: SAE evaluation pipeline for InstaNovo interpretability.

The evaluation is organised around four questions:
  Q1: Is the SAE working?
    P1 Reconstruction: centred fraction of variance explained (FVE),
        uncentred FVE, elementwise MSE, and legacy per-token SSE.
    P2 Sparsity: mean L0, dead-feature percentages, and firing-rate Gini.
    P5 Geometry: encoder/decoder alignment, near-duplicate decoder directions,
        and decoder effective rank.
    P6 Threshold sweep: FVE/L0/dead trade-off across JumpReLU threshold
        multipliers.
  Q2: Does the model survive SAE substitution?
    P7 Loss recovered: clean, SAE-patched, zero-ablated, and mean-ablated CE.
  Q3: What might each feature mean?
    P3 Top activating tokens per feature, with chunk/token provenance.
    P4 Feature-concept associations: F1, F1-dom, lift, chi-square, BH-FDR.
  Q4: Do features causally matter?
    P8 Causal ablation: SAE-reconstruction baseline, selectivity vs concept
        prevalence, firing-rate-matched random controls, correlated-concept
        controls, and a permutation-test sanity check.

Phase definitions below are ordered to follow the thesis results chapter:

    Definition order          Phase                     Thesis
    ----------------------------------------------------------------------
    1                         P1+2 reconstruction       4.1.1 + 4.2.1
    2                         P6 threshold sweep        4.1.2 (Figure 4.1)
    3                         P5 geometry               4.2.2 (Table 4.3)
    4                         P3 top-activating         4.3
    5                         P4 associations           4.3   (Tables 4.4-4.5)
    6                         P7 loss recovered         4.4   (Table 4.6)
    7                         cross-layer matching      4.5   (Table 4.7)
    8                         P8 causal ablation        4.6

Phase NUMBERS are unchanged: they are a cross-file contract, appearing as
report.json keys (parsed by run_pipeline.sh), as the --skip CLI choices, and in
the Phase 4 resume cache. Evaluator.run executes in a different order from the
definitions above because Phase 8 and the cross-layer/cross-seed checks all
consume Phase 4's output, which must therefore be computed (or loaded from
cache) before them.

Streaming model:
  - ChunkStream.__iter__ loads activations, annotations, metadata, and computes
    post-JumpReLU features. Use it only for phases that need joined features and
    labels, currently P3 and P4.
  - iter_activations(), iter_metadata(), and iter_metadata_annotations() are
    lightweight alternatives for phases that do not need the full joined chunk.
    P1+2 and P6 use activation-only passes to avoid unnecessary annotation loads
    and wasted default-threshold feature encodes.
  - P6 is a single-pass threshold sweep: each activation batch is encoded to SAE
    preactivations once, then reused for every threshold multiplier.

Causal validity (Phase 8). Phases 3-4 establish correlation: feature F fires when
concept C is labelled. Phase 8 adds interventions that can falsify that reading:
  - Patch the target encoder layer with the SAE reconstruction and zero feature F
    before decoding, a rank-1 edit along F's decoder direction.
  - Compare ablations against the SAE reconstruction without ablation, not the
    clean model, so SAE reconstruction error cancels out of delta-CE.
  - Retain a concept-feature claim only when the ablation is important,
    concept-selective, stronger than firing-rate-matched controls, and still
    selective when spectra carrying correlated concepts are removed.

Phases 7-8 require the InstaNovo model and original spectra. Pass --instanovo-path
and optionally --spectra-path. If --spectra-path is omitted, the dataset_path in
the extract manifest is used. The model is rerun with a shuffle=False DataLoader
using the same processor and n_peaks as extraction, so per-spectrum CE aligns with
per-spectrum concept prevalence by global order. Both sides are truncated to the
shorter length as a safeguard.

Output layout under --output-dir / layer_{L} / seed_{S} / eval/:
    report.json                       # top-level summary, all phase outputs
    per_feature_stats.csv             # per-feature firing rate and best concept
    feature_label_associations.csv    # significant (feature, concept) pairs
    top_activating_tokens.csv         # per-feature top-K token provenance
    causal_ablation.csv               # per-concept causal-ablation summary
    cross_layer_matches.csv           # per-anchor cross-layer matches
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import chi2 as scipy_chi2

# Import sibling modules from the pipeline.
sys.path.insert(0, str(Path(__file__).parent))
from schema import (
    ANNOTATION_SCHEMA_VERSION,  # annotate.py label chunks
    EVAL_SCHEMA_VERSION,        # this module's report.json + CSVs
    EXTRACT_SCHEMA_VERSION,     # extract.py manifest/layout
)
from train import SparseAutoencoder, load_sae_from_checkpoint

LOG = logging.getLogger("evaluate")

# Default JumpReLU threshold multipliers for the Phase 6 sweep.
THRESHOLD_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

# Cosine similarity above which two decoder directions count as near-duplicates.
NEAR_DUPLICATE_COSINE = 0.95
# Cap on the pairwise decoder-similarity sample; the full matrix is d_dict^2.
NEAR_DUPLICATE_SAMPLE = 2000

# |phi| above which a concept is treated as correlated with the target concept
# and excluded from the orthogonalised selectivity contrast (thesis 3.5.4).
CORRELATED_CONCEPT_PHI = 0.3
# Firing-rate matching window for random ablation controls, in log10 decades.
FIRING_RATE_MATCH_DEX = 0.3
# Minimum spectra needed for a tercile selectivity contrast to be meaningful.
MIN_SPECTRA_FOR_SELECTIVITY = 6
# Rows kept in the per-concept and per-family Phase 4 rankings.
TOP_FEATURES_PER_CONCEPT = 20
# Anchor features carried into cross-layer matching (thesis 4.5).
CROSS_LAYER_ANCHORS = 100
# Top-N features per concept compared across seeds for the Jaccard overlap.
CROSS_SEED_TOP_N = 10
# Tokens sampled when estimating the layer mean for the Phase 7 mean-ablation.
LAYER_MEAN_MAX_TOKENS = 200_000
# Progress is logged every this many chunks during the long streaming phases.
CHUNK_LOG_INTERVAL = 50


def _preserve_hook_output(original_output, patched_first: torch.Tensor):
    """Return patched_first with the same outer contract as a forward hook output."""
    if isinstance(original_output, tuple):
        return (patched_first, *original_output[1:])
    return patched_first


# --- Configuration ------------------------------------------------------------

@dataclasses.dataclass
class EvaluationConfig:
    """All knobs for one evaluation run."""

    # I/O
    extract_dir: Path
    annotation_dir: Path
    sae_checkpoint: Path
    output_dir: Path
    target_layer: int
    seed: int
    phase4_cache_dir: Path | None = None

    # Optional: InstaNovo model for Phases 7-8, and the spectra it runs on.
    instanovo_path: Path | None = None
    # Original spectra source (parquet / glob / mzML) for Phase 7-8 CE. If None,
    # the extract manifest's recorded dataset_path is used. MUST be the same
    # spectra, read shuffle=False, so per-spectrum CE from the model aligns with
    # the per-spectrum concept prevalence from the chunks by global order.
    spectra_path: Path | None = None
    # Peaks per spectrum for the Phase 7/8 loader. None (the default) means read
    # it from the extract manifest, which is what keeps the re-run loader
    # producing the same spectra extraction saw. Set it only to override a
    # manifest that predates the field.
    n_peaks: int | None = None

    # Optional: paths to other-layer SAE checkpoints for cross-layer matching.
    other_layer_checkpoints: dict[int, Path] = dataclasses.field(default_factory=dict)

    # Optional: paths to other-seed checkpoints for cross-seed verification.
    other_seed_checkpoints: list[Path] = dataclasses.field(default_factory=list)

    # Phase enable flags.
    run_phase_1_2: bool = True
    run_phase_3: bool = True
    run_phase_4: bool = True
    run_phase_5: bool = True
    run_phase_6: bool = True
    run_phase_7: bool = True
    run_phase_8: bool = True
    run_cross_layer: bool = True
    run_cross_seed: bool = True

    # Statistical thresholds.
    fdr_q: float = 0.05                       # Benjamini-Hochberg target FDR
    top_k_tokens: int = 20                    # top-K activating tokens per feature

    # Causal ablation parameters.
    ablation_spectra: int = 5_000             # spectra used per ablation pass
    ablation_top_n: int = 10                  # group ablation: features per concept
    ablation_per_feature_top: int = 100       # per-feature ablation: features per concept
    n_random_controls: int = 5                # matched random ablations per concept
    n_firing_rate_deciles: int = 5            # stratification bins for top-N selection

    # Fixed-marginal null sanity-check parameters.
    permutation_n_features: int = 120         # top features to check (about 1%)
    permutation_n_shuffles: int = 100         # hypergeometric samples per feature

    # Cross-layer matching parameters.
    cross_layer_token_sample: int = 100_000   # tokens used for correlation
    cross_layer_top_k: int = 5                # best matches per anchor feature

    # Compute parameters.
    device: str = "cuda"
    batch_size: int = 4096
    dtype: torch.dtype = torch.float32

    def output_subdir(self) -> Path:
        """Per-(layer, seed) output directory: output_dir/layer_{L}/seed_{S}/eval."""
        return self.output_dir / f"layer_{self.target_layer}" / f"seed_{self.seed}" / "eval"

    def as_jsonable(self) -> dict:
        """Config as a JSON-serialisable dict (Paths -> str, dtype -> name)."""
        out = dataclasses.asdict(self)
        for key in ("extract_dir", "annotation_dir", "sae_checkpoint", "output_dir"):
            out[key] = str(out[key])
        if out["phase4_cache_dir"]:
            out["phase4_cache_dir"] = str(out["phase4_cache_dir"])
        if out["instanovo_path"]:
            out["instanovo_path"] = str(out["instanovo_path"])
        if out["spectra_path"]:
            out["spectra_path"] = str(out["spectra_path"])
        out["other_layer_checkpoints"] = {
            str(k): str(v) for k, v in self.other_layer_checkpoints.items()
        }
        out["other_seed_checkpoints"] = [str(p) for p in self.other_seed_checkpoints]
        out["dtype"] = str(self.dtype).replace("torch.", "")
        return out


# --- Statistical primitives ---------------------------------------------------

def benjamini_hochberg(p_values: torch.Tensor, q: float = 0.05) -> torch.Tensor:
    """BH-FDR control. Returns a boolean tensor (same shape) of rejected nulls.

    Controls false discovery rate at level q across all tests in p_values.
    Less conservative than Bonferroni -- surfaces real signals that the
    family-wise error rate would have over-corrected away.
    """
    flat = p_values.flatten().contiguous()
    m = flat.numel()
    if m == 0:
        return torch.zeros_like(p_values, dtype=torch.bool)

    sorted_idx = torch.argsort(flat)
    sorted_p = flat[sorted_idx]
    ranks = torch.arange(1, m + 1, dtype=sorted_p.dtype, device=sorted_p.device)
    thresholds = (ranks / m) * q

    # Find largest k such that p_(k) <= (k/m) * q.
    below = sorted_p <= thresholds
    if not below.any():
        return torch.zeros_like(p_values, dtype=torch.bool)

    k_max = int(below.nonzero(as_tuple=False).max().item())
    rejected_sorted = torch.zeros(m, dtype=torch.bool)
    rejected_sorted[: k_max + 1] = True

    rejected_flat = torch.zeros(m, dtype=torch.bool)
    rejected_flat[sorted_idx] = rejected_sorted
    return rejected_flat.reshape(p_values.shape)


def compute_contingency_stats(
    n11: torch.Tensor,
    marginal_f: torch.Tensor,
    marginal_c: torch.Tensor,
    n_total: int,
) -> dict[str, torch.Tensor]:
    """Derive F1, lift, chi^2, and F1-dom from accumulated counts.

    All inputs are [n_features, n_concepts] except marginals which are
    1-D and broadcast. Outputs are dense [n_features, n_concepts] tensors.
    """
    mf = marginal_f.unsqueeze(1).to(torch.float64)
    mc = marginal_c.unsqueeze(0).to(torch.float64)
    n11_f = n11.to(torch.float64)
    N = float(n_total)

    n10 = mf - n11_f
    n01 = mc - n11_f
    n00 = N - mf - mc + n11_f

    # Clamp denominators to avoid division by zero.
    safe = lambda x: x.clamp_min(1e-12)

    precision = n11_f / safe(n11_f + n10)
    recall = n11_f / safe(n11_f + n01)
    f1 = 2 * precision * recall / safe(precision + recall)

    # Lift: P(c | f) / P(c) = (n11/N) / ((mf/N)(mc/N)) = n11 * N / (mf * mc)
    lift = n11_f * N / safe(mf * mc)

    # chi^2 test statistic with 1 degree of freedom.
    numerator = N * (n11_f * n00 - n10 * n01) ** 2
    denominator = safe((n11_f + n10) * (n00 + n01) * (n11_f + n01) * (n00 + n10))
    chi2_stat = numerator / denominator

    # F1-dom: penalise broad features.
    # Conditional firing rates inside positive vs negative class.
    cond_pos = n11_f / safe(n11_f + n01)
    cond_neg = n10 / safe(n10 + n00)
    # Multiplicative dampening: F1-dom = F1 * max(0, 1 - cond_neg / cond_pos).
    dominance = torch.clamp(1.0 - cond_neg / safe(cond_pos), min=0.0)
    f1_dom = f1 * dominance

    return {
        # Keep raw counts in float64 so large co-occurrence counts are not rounded
        # when compacting Phase 4 to CSV. Derived metrics are stored as float32.
        "n11": n11_f,
        "precision": precision.to(torch.float32),
        "recall": recall.to(torch.float32),
        "f1": f1.to(torch.float32),
        "f1_dom": f1_dom.to(torch.float32),
        "lift": lift.to(torch.float32),
        "chi2_stat": chi2_stat.to(torch.float32),
    }


def chi2_pvalues_from_stat(chi2_stat: torch.Tensor) -> torch.Tensor:
    """One-tailed chi^2 p-value with df=1 from the test statistic."""
    stat_np = chi2_stat.cpu().numpy().astype(np.float64)
    p = 1.0 - scipy_chi2.cdf(stat_np, df=1)
    return torch.from_numpy(p.astype(np.float32))


def _centred_ss_total(sq_total: float, colsum: torch.Tensor | None, n_tokens: int) -> float:
    """Centred total sum of squares: sum x^2 - sum_d (sum_t x_d)^2 / n.

    Shared by Phase 1+2 and Phase 6 so both report FVE on the same (centred)
    convention -- improvement over predicting the per-dimension mean, rather
    than over predicting zero.
    """
    mean_energy = (
        float((colsum ** 2).sum().item()) / max(n_tokens, 1)
        if colsum is not None else 0.0
    )
    return max(sq_total - mean_energy, 1e-12)


def _gini(rates: torch.Tensor) -> float:
    """Gini coefficient of a non-negative distribution.

    0 means every feature fires equally often; values approaching 1 mean a few
    features account for nearly all activations. Standard sorted-ascending
    estimator: G = (2 sum i*x_i - (n+1) sum x_i) / (n sum x_i).
    """
    rates_sorted, _ = torch.sort(rates)
    n_f = rates_sorted.numel()
    total = rates_sorted.sum()
    if float(total) <= 0:
        return 0.0
    idx = torch.arange(1, n_f + 1, dtype=torch.float64)
    return float((2.0 * (idx * rates_sorted).sum() - (n_f + 1) * total) / (n_f * total))


# --- Streaming data over chunks -----------------------------------------------

@dataclasses.dataclass
class JoinedChunk:
    """One chunk's worth of (activations, labels, metadata) joined by row."""

    chunk_idx: int
    activations: torch.Tensor      # [n_tokens, d_model]  raw layer activations
    features: torch.Tensor         # [n_tokens, d_dict]   post-JumpReLU SAE features
    labels: torch.Tensor           # [n_tokens, n_concepts] bool
    token_to_spectrum: torch.Tensor   # [n_tokens]  row of the spectrum each token belongs to
    token_to_position: torch.Tensor   # [n_tokens]  0 = latent summary token, 1.. = peaks
    # Carried from the annotation chunk for traceback / the visualisation tool;
    # the phases here use `labels`, not these per-token peak fields.
    ion_type_ids: torch.Tensor
    peak_mzs: torch.Tensor
    peak_intensities: torch.Tensor

    # Optional baseline predictions if available in the extract chunk.
    baseline_top1: torch.Tensor | None
    baseline_ce: torch.Tensor | None
    baseline_decoder_mask: torch.Tensor | None

    # Per-spectrum metadata (one row per spectrum in this chunk).
    n_spectra: int
    spectrum_ids: list[str]
    peptides: list[str]
    proforma_strings: list[str]
    modifications: list[list[dict]]
    precursor_charges: torch.Tensor
    precursor_mzs: torch.Tensor


class ChunkStream:
    """Iterates extract chunks (per-layer activation file + ChunkMeta) paired
    with the matching annotation label chunk, joined row-for-row.

    The full iterator loads metadata, activations, and annotation labels, then
    encodes activations through the SAE at the checkpoint's current JumpReLU
    threshold. It is intended for phases that need joined features and labels.
    Lightweight iter_* helpers avoid that encode for phases that only need
    activations, metadata, or labels.
    """

    def __init__(
        self,
        extract_dir: Path,
        annotation_dir: Path,
        target_layer: int,
        sae: SparseAutoencoder,
        device: str,
        batch_size: int,
        dtype: torch.dtype,
    ):
        self.extract_dir = extract_dir
        self.annotation_dir = annotation_dir
        self.target_layer = target_layer
        self.sae = sae
        self.device = device
        self.batch_size = batch_size
        self.dtype = dtype

        extract_manifest = json.loads((extract_dir / "manifest.json").read_text())
        annotation_manifest = json.loads((annotation_dir / "annotation_manifest.json").read_text())
        self._check_manifests(extract_manifest, annotation_manifest)

        self.meta_paths, self.acts_paths = self._resolve_layer_paths(
            extract_dir, extract_manifest, target_layer,
        )
        self.annotation_paths = [annotation_dir / c["path"] for c in annotation_manifest["chunks"]]
        self.n_chunks = extract_manifest["n_chunks"]
        self.concept_names: list[str] = annotation_manifest["registry"]["names"]
        self.diagnostic_concepts: set[str] = set(annotation_manifest["registry"]["diagnostic"])
        self.family_of: dict[str, str] = annotation_manifest["registry"]["family_of"]
        self.base_rates: dict[str, float] = annotation_manifest["base_rates"]

    @staticmethod
    def _check_manifests(extract_manifest: dict, annotation_manifest: dict) -> None:
        """Reject manifests that cannot be row-joined.

        Every phase depends on activations, labels, and metadata sharing a token
        ordering, so a schema or chunk-count disagreement is fatal rather than a
        warning: the join would otherwise pair each token with another's labels.
        """
        if extract_manifest["schema_version"] != EXTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"Extract schema {extract_manifest['schema_version']} != {EXTRACT_SCHEMA_VERSION}"
            )
        if annotation_manifest["schema_version"] != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(
                f"Annotation schema {annotation_manifest['schema_version']} != {ANNOTATION_SCHEMA_VERSION}"
            )
        if extract_manifest["n_chunks"] != annotation_manifest["n_chunks"]:
            raise ValueError(
                f"Chunk count mismatch: extract={extract_manifest['n_chunks']}, "
                f"annotation={annotation_manifest['n_chunks']}"
            )

    @staticmethod
    def _resolve_layer_paths(
        extract_dir: Path, extract_manifest: dict, target_layer: int
    ) -> tuple[list[Path], list[Path]]:
        """Meta and activation file paths for the target layer, one per chunk.

        Each chunk stores metadata (ChunkMeta) and one activation file per
        extracted layer separately, so only the target layer's bytes are read.
        """
        layer_key = str(target_layer)
        meta_paths: list[Path] = []
        acts_paths: list[Path] = []
        for c in extract_manifest["chunks"]:
            if layer_key not in c["activations"]:
                raise KeyError(
                    f"Layer {target_layer} not extracted for chunk {c['idx']}; "
                    f"available: {sorted(int(k) for k in c['activations'])}"
                )
            meta_paths.append(extract_dir / c["meta"])
            acts_paths.append(extract_dir / c["activations"][layer_key])
        return meta_paths, acts_paths

    def _encode_chunk(self, activations: torch.Tensor) -> torch.Tensor:
        """Stream the SAE forward pass over activations, batched to fit memory.

        Writes each mini-batch directly into a pre-allocated output tensor
        rather than collecting a list and torch.cat-ing at the end. A
        list-then-cat holds both the list of parts and the freshly allocated
        concatenated tensor at once -- roughly double the final tensor's
        memory -- which matters here since a full chunk's dense feature
        matrix (n_tokens x d_dict) can already be several GB on its own.
        """
        n_tokens = activations.size(0)
        out = torch.empty((n_tokens, self.sae.d_dict), dtype=self.dtype)
        for start in range(0, n_tokens, self.batch_size):
            end = min(start + self.batch_size, n_tokens)
            x = activations[start:end].to(self.device, dtype=self.dtype, non_blocking=True)
            with torch.no_grad():
                batch_out = self.sae.forward_inference(x)
            out[start:end] = batch_out["features"].detach().to(self.dtype).cpu()
            del x, batch_out
        return out

    def __iter__(self) -> Iterator[JoinedChunk]:
        for ci in range(self.n_chunks):
            meta = torch.load(self.meta_paths[ci], map_location="cpu", weights_only=False)
            acts_obj = torch.load(self.acts_paths[ci], map_location="cpu", weights_only=False)
            annotation = torch.load(self.annotation_paths[ci], map_location="cpu", weights_only=False)

            activations = acts_obj["activations"]

            # All three files must agree on token count or the row-join is invalid.
            if not (activations.size(0) == meta["total_tokens"] == annotation["total_tokens"]):
                raise ValueError(
                    f"Chunk {ci} token-count mismatch: activations={activations.size(0)}, "
                    f"meta={meta['total_tokens']}, annotation={annotation['total_tokens']}"
                )

            features = self._encode_chunk(activations)

            yield JoinedChunk(
                chunk_idx=ci,
                activations=activations,
                features=features,
                labels=annotation["token_labels"],
                token_to_spectrum=meta["token_to_spectrum"],
                token_to_position=meta["token_to_position"],
                ion_type_ids=annotation["ion_type_ids"],
                peak_mzs=annotation["peak_mzs"],
                peak_intensities=annotation["peak_intensities"],
                baseline_top1=meta.get("baseline_top1"),
                baseline_ce=meta.get("baseline_ce"),
                baseline_decoder_mask=meta.get("baseline_decoder_mask"),
                n_spectra=meta["n_spectra"],
                spectrum_ids=meta["spectrum_ids"],
                peptides=meta["peptides"],
                proforma_strings=meta["proforma_strings"],
                modifications=meta["modifications"],
                precursor_charges=meta["precursor_charges"],
                precursor_mzs=meta["precursor_mzs"],
            )

    # Lightweight stream helpers.
    def iter_activations(self) -> Iterator[torch.Tensor]:
        """Iterate only raw activation chunks.

        Phases that deliberately re-encode under modified SAE settings should
        not pay ChunkStream.__iter__'s default-threshold feature encode.
        """
        for acts_path in self.acts_paths:
            acts_obj = torch.load(acts_path, map_location="cpu", weights_only=False)
            yield acts_obj["activations"]

    def iter_metadata(self) -> Iterator[dict]:
        """Iterate extract metadata without loading activations or SAE features."""
        for meta_path in self.meta_paths:
            yield torch.load(meta_path, map_location="cpu", weights_only=False)

    def iter_metadata_annotations(self) -> Iterator[tuple[dict, dict]]:
        """Iterate metadata and annotation labels without loading activations."""
        for meta_path, annotation_path in zip(self.meta_paths, self.annotation_paths):
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            annotation = torch.load(annotation_path, map_location="cpu", weights_only=False)
            yield meta, annotation


class TopKAccumulator:
    """Tracks the top-K activation values per feature across all chunks,
    along with provenance (chunk idx, token-in-chunk idx, value).
    """

    def __init__(self, n_features: int, k: int):
        self.n_features = n_features
        self.k = k
        self.values = torch.full((n_features, k), -float("inf"))
        self.chunk_ids = torch.full((n_features, k), -1, dtype=torch.long)
        self.token_ids = torch.full((n_features, k), -1, dtype=torch.long)

    def add_chunk(self, chunk_idx: int, features: torch.Tensor) -> None:
        """Merge one chunk's per-feature top-K into the running top-K, keeping
        provenance. features is [n_tokens, n_features] for this chunk.
        """
        n_tokens = features.size(0)
        if n_tokens == 0:
            return

        # Per-feature top-K within this chunk.
        k_use = min(self.k, n_tokens)
        chunk_top_vals, chunk_top_idx = torch.topk(features, k_use, dim=0)  # [k_use, n_features]
        chunk_top_vals = chunk_top_vals.t()  # [n_features, k_use]
        chunk_top_idx = chunk_top_idx.t()    # [n_features, k_use]

        # Merge with running top-K: stack along k dim and take top-K of the union.
        merged_vals = torch.cat([self.values, chunk_top_vals], dim=1)
        merged_chunk_ids = torch.cat(
            [self.chunk_ids, torch.full_like(chunk_top_idx, chunk_idx)], dim=1,
        )
        merged_token_ids = torch.cat([self.token_ids, chunk_top_idx], dim=1)

        new_vals, new_idx = torch.topk(merged_vals, self.k, dim=1)
        self.values = new_vals
        # Gather provenance using the same indices.
        self.chunk_ids = torch.gather(merged_chunk_ids, 1, new_idx)
        self.token_ids = torch.gather(merged_token_ids, 1, new_idx)


def _log_chunk_progress(label: str, chunk_i: int, n_chunks: int) -> None:
    """Log streaming progress at a fixed interval and on the final chunk."""
    if chunk_i == n_chunks or chunk_i % CHUNK_LOG_INTERVAL == 0:
        LOG.info("%s: processed %d/%d chunks", label, chunk_i, n_chunks)


# =============================================================================
# Thesis 4.1.1 + 4.2.1 -- Phase 1+2: reconstruction and sparsity
# Combined into one activation-only pass: both metrics need the same forward
# encode over every token, and the dataset is ~67.5M tokens per layer.
# =============================================================================

class _ReconSparsityAccumulator:
    """Streaming accumulators for reconstruction quality and sparsity."""

    def __init__(self, d_dict: int):
        self.sq_resid = 0.0
        self.sq_total = 0.0   # uncentred second moment: sum_{t,d} x^2
        self.sum_l0 = 0.0
        self.n_tokens = 0
        self.colsum = None    # sum_t x (per d_model dim), float64 -- for centred FVE
        self.firing_count = torch.zeros(d_dict, dtype=torch.long)

    def add_batch(self, x: torch.Tensor, x_hat: torch.Tensor, features: torch.Tensor) -> None:
        resid = x - x_hat
        self.sq_resid += float((resid ** 2).sum().item())
        self.sq_total += float((x ** 2).sum().item())
        xs = x.sum(dim=0).to(torch.float64).cpu()
        self.colsum = xs if self.colsum is None else self.colsum + xs

        fired = features > 0
        self.sum_l0 += float(fired.float().sum().item())
        self.firing_count += fired.sum(dim=0).long().cpu()
        self.n_tokens += x.size(0)

    def metrics(self, sae: SparseAutoencoder) -> dict:
        """Finalise into the phase_1_2 report block.

        `fve_overall` is the centred fraction of variance explained: improvement
        over predicting the per-dimension mean. `fve_uncentered` is retained for
        continuity with older runs. `mse_total` is conventional elementwise MSE
        over tokens and hidden dimensions; `sse_per_token` preserves the older
        per-token summed-squared-error scale.
        """
        ss_tot_centered = _centred_ss_total(self.sq_total, self.colsum, self.n_tokens)
        near_dead_cut = max(1, self.n_tokens // 100_000)
        rates = self.firing_count.to(torch.float64) / max(self.n_tokens, 1)

        return {
            "fve_overall": 1.0 - self.sq_resid / ss_tot_centered,
            "fve_uncentered": 1.0 - self.sq_resid / max(self.sq_total, 1e-12),
            "mse_total": self.sq_resid / max(self.n_tokens * sae.d_model, 1),
            "sse_per_token": self.sq_resid / max(self.n_tokens, 1),
            "l0_mean": self.sum_l0 / max(self.n_tokens, 1),
            "strict_dead_pct": 100.0 * int((self.firing_count == 0).sum().item()) / sae.d_dict,
            "near_dead_pct": 100.0 * int((self.firing_count < near_dead_cut).sum().item()) / sae.d_dict,
            "firing_rate_gini": _gini(rates),
            "n_tokens": self.n_tokens,
            "firing_count": self.firing_count.tolist(),
        }


def phase_1_2_reconstruction_and_sparsity(
    stream: ChunkStream,
    sae: SparseAutoencoder,
    device: str,
) -> dict:
    """Compute reconstruction quality and sparsity in one activation-only pass."""
    acc = _ReconSparsityAccumulator(sae.d_dict)

    for chunk_i, activations in enumerate(stream.iter_activations(), start=1):
        # Process raw activations directly. This avoids loading annotations and
        # avoids materialising a full dense chunk.features tensor on CPU.
        for start in range(0, activations.size(0), stream.batch_size):
            end = min(start + stream.batch_size, activations.size(0))
            x = activations[start:end].to(device, dtype=stream.dtype, non_blocking=True)
            with torch.no_grad():
                out = sae.forward_inference(x)
                features = out["features"]
                x_hat = out["x_hat"]

            acc.add_batch(x, x_hat, features)
            del x, out, features, x_hat

        _log_chunk_progress("Phase 1+2", chunk_i, stream.n_chunks)

    return acc.metrics(sae)


# =============================================================================
# Thesis 4.1.2 (Figure 4.1) -- Phase 6: JumpReLU threshold sweep
# =============================================================================

class _ThresholdSweepAccumulator:
    """Per-multiplier residual and firing accumulators for the threshold sweep.

    The activations x are identical across multipliers, so the total sum of
    squares is shared and only the residual changes per threshold.
    """

    def __init__(self, n_mult: int, d_dict: int):
        self.sq_resid = torch.zeros(n_mult, dtype=torch.float64)
        self.sum_l0 = torch.zeros(n_mult, dtype=torch.float64)
        self.firing = torch.zeros((n_mult, d_dict), dtype=torch.long)
        self.sq_total = 0.0
        self.n_tokens = 0
        self.colsum = None

    def add_shared(self, x: torch.Tensor) -> None:
        """Accumulate the multiplier-independent totals for one batch."""
        self.sq_total += float((x ** 2).sum().item())
        xs = x.sum(dim=0).to(torch.float64).cpu()
        self.colsum = xs if self.colsum is None else self.colsum + xs
        self.n_tokens += x.size(0)

    def add_threshold(self, i: int, x: torch.Tensor, x_hat: torch.Tensor,
                      features: torch.Tensor) -> None:
        self.sq_resid[i] += float(((x - x_hat) ** 2).sum().item())
        fired = features > 0
        self.sum_l0[i] += float(fired.float().sum().item())
        self.firing[i] += fired.sum(dim=0).long().cpu()

    def results(self, multipliers: tuple[float, ...], d_dict: int) -> list[dict]:
        """Centred FVE per multiplier, consistent with Phase 1+2."""
        ss_tot_centered = _centred_ss_total(self.sq_total, self.colsum, self.n_tokens)
        return [
            {
                "multiplier": mult,
                "fve": 1.0 - float(self.sq_resid[i].item()) / ss_tot_centered,
                "l0_mean": float(self.sum_l0[i].item()) / max(self.n_tokens, 1),
                "dead_pct": 100.0 * int((self.firing[i] == 0).sum().item()) / d_dict,
            }
            for i, mult in enumerate(multipliers)
        ]


def phase_6_threshold_sweep(
    stream: ChunkStream,
    sae: SparseAutoencoder,
    multipliers: tuple[float, ...] = THRESHOLD_MULTIPLIERS,
) -> dict:
    """Sweep JumpReLU threshold multipliers and report FVE/L0/dead.

    Each activation batch is encoded to preactivations once, then all threshold
    multipliers are applied to that shared tensor. This preserves the same metrics
    as separate passes while avoiding repeated disk scans.
    """
    original_threshold = sae.jumprelu_threshold.detach().clone()
    thresholds = [original_threshold * mult for mult in multipliers]
    acc = _ThresholdSweepAccumulator(len(multipliers), sae.d_dict)

    for chunk_i, activations in enumerate(stream.iter_activations(), start=1):
        for start in range(0, activations.size(0), stream.batch_size):
            end = min(start + stream.batch_size, activations.size(0))
            x = activations[start:end].to(stream.device, dtype=stream.dtype, non_blocking=True)

            # Compute preactivations once and reuse them for every threshold.
            with torch.no_grad():
                h = sae.preactivations(x)
                acc.add_shared(x)

                for i, threshold in enumerate(thresholds):
                    features = h * (h > threshold).to(h.dtype)
                    x_hat = sae.decode(features)
                    acc.add_threshold(i, x, x_hat, features)
                    del features, x_hat

            del x, h

        _log_chunk_progress("Phase 6 threshold sweep", chunk_i, stream.n_chunks)

    results = acc.results(multipliers, sae.d_dict)

    sae.jumprelu_threshold.copy_(original_threshold)  # restore defensively
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sweep": results}


# =============================================================================
# Thesis 4.2.2 (Table 4.3) -- Phase 5: dictionary geometry
# =============================================================================

def _encoder_decoder_alignment(W_enc: torch.Tensor, W_dec: torch.Tensor) -> float:
    """Mean |cos| between each feature's encoder and decoder direction.

    Near 1 would mean the encoder is essentially the decoder transpose, leaving
    no room for a learned sparse projection; near 0 would mean detection and
    reconstruction directions are geometrically unrelated.
    """
    enc_t = W_enc.t()  # [d_dict, d_model]
    enc_norm = enc_t.norm(dim=1).clamp_min(1e-12)
    dec_norm = W_dec.norm(dim=1).clamp_min(1e-12)
    cos = (enc_t * W_dec).sum(dim=1) / (enc_norm * dec_norm)
    return float(cos.abs().mean().item())


def _near_duplicate_pairs(W_dec: torch.Tensor, d_dict: int) -> tuple[int, int]:
    """Count near-duplicate decoder directions within a random subsample.

    The full pairwise matrix is d_dict^2, so this is estimated on a subsample and
    returned as a count WITHIN the sample -- a lower bound on the global count,
    not an estimate of it.
    """
    n_sample = min(NEAR_DUPLICATE_SAMPLE, d_dict)
    sample_idx = torch.randperm(d_dict)[:n_sample]
    sample = W_dec[sample_idx]
    sample = sample / sample.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = sample @ sample.t()
    sim.fill_diagonal_(0.0)
    return int((sim > NEAR_DUPLICATE_COSINE).sum().item() // 2), n_sample  # undirected pairs


def _effective_rank(W_dec: torch.Tensor) -> float:
    """Entropy-perplexity of the decoder singular-value spectrum.

    Equals the full rank when all singular values are equal and collapses toward
    one when a single direction dominates, so it measures how many independent
    directions the dictionary actually uses.
    """
    try:
        sv = torch.linalg.svdvals(W_dec.to(torch.float32))
        normalized = sv / sv.sum().clamp_min(1e-12)
        entropy = -(normalized * (normalized + 1e-12).log()).sum()
        return float(entropy.exp().item())
    except Exception:
        return float("nan")


def phase_5_geometric(sae: SparseAutoencoder) -> dict:
    """Encoder/decoder alignment and near-duplicate feature analysis."""
    with torch.no_grad():
        W_enc = sae.W_enc.detach().cpu()  # [d_model, d_dict]
        W_dec = sae.W_dec.detach().cpu()  # [d_dict, d_model]

        alignment = _encoder_decoder_alignment(W_enc, W_dec)
        near_dup_pairs, n_sample = _near_duplicate_pairs(W_dec, sae.d_dict)
        effective_rank = _effective_rank(W_dec)

    return {
        "encoder_decoder_alignment": alignment,
        "near_duplicate_pairs_in_sample": near_dup_pairs,
        "near_duplicate_sample_size": n_sample,
        "near_duplicate_cosine_threshold": NEAR_DUPLICATE_COSINE,
        "effective_rank": effective_rank,
    }


# =============================================================================
# Thesis 4.3 -- Phase 3: top-K activating tokens per feature
# =============================================================================

def phase_3_top_activating(
    stream: ChunkStream,
    sae: SparseAutoencoder,
    k: int,
) -> dict:
    """Collect the top-K activating tokens per feature with provenance."""
    accumulator = TopKAccumulator(n_features=sae.d_dict, k=k)
    chunk_metadata: list[dict] = []  # one row per chunk for traceback

    for chunk in stream:
        accumulator.add_chunk(chunk.chunk_idx, chunk.features)
        chunk_metadata.append({
            "chunk_idx": chunk.chunk_idx,
            "n_spectra": chunk.n_spectra,
            "spectrum_ids": chunk.spectrum_ids,
            "token_to_spectrum": chunk.token_to_spectrum.tolist(),
            "token_to_position": chunk.token_to_position.tolist(),
        })

    return {
        "values": accumulator.values,
        "chunk_ids": accumulator.chunk_ids,
        "token_ids": accumulator.token_ids,
        "chunk_metadata": chunk_metadata,
        "k": k,
    }


# =============================================================================
# Thesis 4.3 (Tables 4.4-4.5) -- Phase 4: feature-concept associations
# =============================================================================

def _accumulate_contingency(
    stream: ChunkStream, n_features: int, n_concepts: int,
    row_batch: int = 8192,  # caps peak memory of the dense feat_bool cast below
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Stream the feature x concept co-occurrence table over every chunk.

    Accumulators are float64: summing indicator products over ~67.5M tokens
    would lose precision in float32. The float64 cast and matmul are done in
    row_batch-sized slices rather than over a whole chunk at once, since a
    dense [n_tokens_in_chunk, n_features] float64 cast of a large chunk can
    exceed available RAM (a ~130k-token chunk needs ~13 GB just for that cast).
    """
    n11 = torch.zeros((n_features, n_concepts), dtype=torch.float64)
    marginal_f = torch.zeros(n_features, dtype=torch.float64)
    marginal_c = torch.zeros(n_concepts, dtype=torch.float64)
    n_total = 0

    for chunk in stream:
        n_tokens = chunk.features.size(0)
        for start in range(0, n_tokens, row_batch):
            end = min(start + row_batch, n_tokens)
            feat_bool = (chunk.features[start:end] > 0).to(torch.float64)
            labels_f = chunk.labels[start:end].to(torch.float64)

            n11 += feat_bool.t() @ labels_f
            marginal_f += feat_bool.sum(dim=0)
            marginal_c += labels_f.sum(dim=0)
            n_total += feat_bool.size(0)

    return n11, marginal_f, marginal_c, n_total


def _per_concept_top_features(
    stats: dict[str, torch.Tensor],
    rejected: torch.Tensor,
    p_values: torch.Tensor,
    score: torch.Tensor,
    concept_names: list[str],
) -> dict[str, list[dict]]:
    """Best BH-significant features for each concept, ranked by the composite
    score (F1-dom weighted by log-lift), as in thesis Table 4.5."""
    per_concept_top: dict[str, list[dict]] = {}
    for ci, cname in enumerate(concept_names):
        rejected_mask = rejected[:, ci]
        if not rejected_mask.any():
            per_concept_top[cname] = []
            continue
        scores_c = score[:, ci].clone()
        scores_c[~rejected_mask] = -float("inf")
        top_vals, top_idx = torch.topk(
            scores_c, k=min(TOP_FEATURES_PER_CONCEPT, int(rejected_mask.sum())),
        )
        per_concept_top[cname] = [
            {
                "feature_idx": int(top_idx[i]),
                "f1_dom": float(stats["f1_dom"][top_idx[i], ci]),
                "lift": float(stats["lift"][top_idx[i], ci]),
                "n_co": int(stats["n11"][top_idx[i], ci]),
                "score": float(top_vals[i]),
                "p_value": float(p_values[top_idx[i], ci]),
            }
            for i in range(top_vals.numel()) if top_vals[i] > -float("inf")
        ]
    return per_concept_top


def _per_family_top_features(
    stats: dict[str, torch.Tensor],
    rejected: torch.Tensor,
    concept_names: list[str],
    family_of: dict[str, str],
) -> dict[str, list[dict]]:
    """For each concept family, the features scoring highest against ANY concept
    in that family, with the concept that produced the score."""
    family_to_concept_indices: dict[str, list[int]] = defaultdict(list)
    for ci, cname in enumerate(concept_names):
        family_to_concept_indices[family_of[cname]].append(ci)

    per_family_top: dict[str, list[dict]] = {}
    for family, concept_indices in family_to_concept_indices.items():
        family_score = stats["f1_dom"][:, concept_indices].max(dim=1)
        family_best_concept = torch.tensor(concept_indices)[family_score.indices]
        family_mask = rejected[:, concept_indices].any(dim=1)
        if not family_mask.any():
            per_family_top[family] = []
            continue
        family_scores_masked = family_score.values.clone()
        family_scores_masked[~family_mask] = -float("inf")
        top_vals, top_idx = torch.topk(
            family_scores_masked, k=min(TOP_FEATURES_PER_CONCEPT, int(family_mask.sum())),
        )
        per_family_top[family] = [
            {
                "feature_idx": int(top_idx[i]),
                "best_concept_idx": int(family_best_concept[top_idx[i]]),
                "best_concept_name": concept_names[int(family_best_concept[top_idx[i]])],
                "f1_dom": float(top_vals[i]),
            }
            for i in range(top_vals.numel()) if top_vals[i] > -float("inf")
        ]
    return per_family_top


def phase_4_feature_concept_associations(
    stream: ChunkStream,
    sae: SparseAutoencoder,
    fdr_q: float,
) -> dict:
    """Streaming accumulation of feature-concept contingency tables.

    Computes F1, F1-dom, lift, chi^2, and applies Benjamini-Hochberg at level
    fdr_q across all (feature, concept) pairs. Returns the stats matrices,
    the significance mask, and per-concept and per-family best-feature
    rankings.
    """
    n_features = sae.d_dict
    n_concepts = len(stream.concept_names)

    n11, marginal_f, marginal_c, n_total = _accumulate_contingency(
        stream, n_features, n_concepts,
    )

    stats = compute_contingency_stats(n11, marginal_f, marginal_c, n_total)
    p_values = chi2_pvalues_from_stat(stats["chi2_stat"])
    rejected = benjamini_hochberg(p_values, q=fdr_q)

    # Composite ranking score: F1-dom up-weighted by enrichment over base rate.
    score = stats["f1_dom"] * torch.log1p(stats["lift"])

    return {
        "stats": stats,
        "p_values": p_values,
        "rejected": rejected,
        "n_significant_pairs": int(rejected.sum().item()),
        "n_features_with_concept": int(rejected.any(dim=1).sum().item()),
        "per_concept_top": _per_concept_top_features(
            stats, rejected, p_values, score, stream.concept_names,
        ),
        "per_family_top": _per_family_top_features(
            stats, rejected, stream.concept_names, stream.family_of,
        ),
        "n_total_tokens": n_total,
        "concept_names": stream.concept_names,
        "marginal_f": marginal_f,
        "marginal_c": marginal_c,
        "n11": n11,
    }


def _read_cached_token_total(report_path: Path) -> tuple[dict, int]:
    """Load a cached report and its Phase 1+2 token count.

    The token total is what converts the cached per-feature firing RATES back
    into the marginal COUNTS the contingency maths needs, so a report without it
    cannot support a resume.

    The report's schema version is checked first: the resume path reads this
    report together with the CSVs written beside it, so a report from an
    incompatible layout would be half-parsed into a silently wrong Phase 4 rather
    than failing.
    """
    cached_report = json.loads(report_path.read_text())

    cached_version = cached_report.get("schema_version")
    if cached_version != EVAL_SCHEMA_VERSION:
        raise ValueError(
            f"{report_path} has evaluation schema {cached_version!r}, but this "
            f"loader expects {EVAL_SCHEMA_VERSION}. Re-run evaluate.py for this "
            "layer/seed instead of resuming from the cache."
        )

    phase_1_2 = cached_report.get("phase_1_2", {})
    n_total = int(phase_1_2.get("n_tokens", 0) or 0)
    if n_total <= 0:
        raise ValueError(
            f"{report_path} does not contain phase_1_2.n_tokens; cannot resume "
            "Phase 4-dependent phases from cache."
        )
    return cached_report, n_total


def _read_association_csv(
    assoc_path: Path, n_features: int, concept_to_idx: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Rebuild the Phase 4 stat matrices from the significant-pair CSV.

    Only BH-significant pairs were written, so non-significant entries stay at
    neutral defaults (p=1, not rejected). Downstream consumers select exclusively
    from the rejected mask, so those defaults are never read as real metrics.
    """
    n_concepts = len(concept_to_idx)
    out = {
        "n11": torch.zeros((n_features, n_concepts), dtype=torch.float64),
        "f1": torch.zeros((n_features, n_concepts), dtype=torch.float32),
        "f1_dom": torch.zeros((n_features, n_concepts), dtype=torch.float32),
        "lift": torch.zeros((n_features, n_concepts), dtype=torch.float32),
        "chi2_stat": torch.zeros((n_features, n_concepts), dtype=torch.float32),
        "p_values": torch.ones((n_features, n_concepts), dtype=torch.float32),
        "rejected": torch.zeros((n_features, n_concepts), dtype=torch.bool),
    }

    with open(assoc_path, newline="") as fp:
        for row in csv.DictReader(fp):
            fi = int(row["feature_idx"])
            concept = row["concept"]
            if not (0 <= fi < n_features) or concept not in concept_to_idx:
                raise ValueError(
                    f"Phase 4 cache row references unknown feature/concept: "
                    f"feature={fi}, concept={concept!r}"
                )
            ci = concept_to_idx[concept]
            out["f1"][fi, ci] = float(row["f1"])
            out["f1_dom"][fi, ci] = float(row["f1_dom"])
            out["lift"][fi, ci] = float(row["lift"])
            out["n11"][fi, ci] = float(row["n_co"])
            out["chi2_stat"][fi, ci] = float(row["chi2"])
            out["p_values"][fi, ci] = float(row["p_value"])
            out["rejected"][fi, ci] = row.get("p_value_bh_significant", "True").lower() == "true"

    return out


def _read_feature_marginals(
    per_feature_path: Path, n_features: int, n_total: int,
) -> torch.Tensor:
    """Recover per-feature marginal counts from the cached firing rates."""
    marginal_f = torch.zeros(n_features, dtype=torch.float64)
    with open(per_feature_path, newline="") as fp:
        for row in csv.DictReader(fp):
            fi = int(row["feature_idx"])
            if not (0 <= fi < n_features):
                raise ValueError(f"Phase 4 cache row references unknown feature: {fi}")
            marginal_f[fi] = float(row["firing_rate"]) * n_total
    return marginal_f


def _resolve_concept_marginals(
    stream: ChunkStream, concept_names: list[str], n_total: int,
) -> torch.Tensor:
    """Concept marginal counts, preferring the annotation phi blob's exact counts
    over the manifest's rounded base rates."""
    phi_path = stream.annotation_dir / "concept_phi.pt"
    if phi_path.exists():
        phi_blob = torch.load(phi_path, map_location="cpu", weights_only=False)
        phi_names = list(phi_blob.get("concept_names", []))
        phi_marginal = phi_blob.get("marginal")
        if phi_names == concept_names and phi_marginal is not None:
            return phi_marginal.to(torch.float64)
    return torch.tensor(
        [float(stream.base_rates.get(name, 0.0)) * n_total for name in concept_names],
        dtype=torch.float64,
    )


def load_phase_4_cache(
    cache_dir: Path,
    stream: ChunkStream,
    sae: SparseAutoencoder,
) -> tuple[dict, dict]:
    """Reconstruct the Phase 4 tensors needed by downstream phases from CSV cache.

    This is intentionally a resume helper, not a replacement for Phase 4. It loads
    the significant-pair CSV, per-feature firing rates, and existing report.json so
    Phase 8 / cross checks can run without rescanning every chunk for Phase 4.
    Non-significant pair metrics are left at neutral defaults because downstream
    consumers only select from the BH-significant mask.
    """
    assoc_path = cache_dir / "feature_label_associations.csv"
    per_feature_path = cache_dir / "per_feature_stats.csv"
    report_path = cache_dir / "report.json"
    missing = [
        str(path) for path in (assoc_path, per_feature_path, report_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Phase 4 cache is incomplete; missing: " + ", ".join(missing)
        )

    cached_report, n_total = _read_cached_token_total(report_path)

    concept_names = stream.concept_names
    concept_to_idx = {name: idx for idx, name in enumerate(concept_names)}
    n_features = sae.d_dict

    loaded = _read_association_csv(assoc_path, n_features, concept_to_idx)
    marginal_f = _read_feature_marginals(per_feature_path, n_features, n_total)
    marginal_c = _resolve_concept_marginals(stream, concept_names, n_total)

    rejected = loaded["rejected"]
    n11 = loaded["n11"]
    phase_4_summary = cached_report.get("phase_4", {})
    return {
        "stats": {
            "n11": n11,
            "f1": loaded["f1"],
            "f1_dom": loaded["f1_dom"],
            "lift": loaded["lift"],
            "chi2_stat": loaded["chi2_stat"],
        },
        "p_values": loaded["p_values"],
        "rejected": rejected,
        "n_significant_pairs": int(rejected.sum().item()),
        "n_features_with_concept": int(rejected.any(dim=1).sum().item()),
        "per_concept_top": phase_4_summary.get("per_concept_top", {}),
        "per_family_top": phase_4_summary.get("per_family_top", {}),
        "n_total_tokens": n_total,
        "concept_names": concept_names,
        "marginal_f": marginal_f,
        "marginal_c": marginal_c,
        "n11": n11,
    }, cached_report


# =============================================================================
# Thesis 4.4 (Table 4.6) -- Phase 7: loss recovered under SAE substitution
# =============================================================================

def make_sae_substitution_hook(
    sae: SparseAutoencoder,
    ablate_features: torch.Tensor | None = None,
):
    """Returns a forward hook that replaces a layer's output with the SAE
    reconstruction. If ablate_features is provided, those feature indices
    are zeroed before decoding.
    """
    ablate_set = None
    if ablate_features is not None and ablate_features.numel() > 0:
        ablate_set = ablate_features.to(sae.W_dec.device).long()

    def hook(_module, _inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        original_shape = x.shape
        sae_dtype = next(sae.parameters()).dtype
        flat = x.reshape(-1, x.size(-1)).to(dtype=sae_dtype)
        h = sae.preactivations(flat)
        gate = (h > sae.jumprelu_threshold.unsqueeze(0))
        features = h * gate.to(h.dtype)
        if ablate_set is not None:
            features[:, ablate_set] = 0
        x_hat = sae.decode(features).to(dtype=x.dtype)
        return _preserve_hook_output(output, x_hat.reshape(original_shape))

    return hook


def _make_zero_hook():
    """Forward hook that replaces a layer's output with zeros (ablate the layer)."""
    def hook(_module, _inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        return _preserve_hook_output(output, torch.zeros_like(x))
    return hook


def _make_mean_hook(layer_mean: torch.Tensor):
    """Forward hook that replaces a layer's output with the dataset mean activation."""
    def hook(_module, _inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        mean = layer_mean.to(device=x.device, dtype=x.dtype)
        return _preserve_hook_output(output, mean.expand_as(x).clone())
    return hook


def _compute_layer_mean(
    stream: ChunkStream, target_layer: int, device: str, max_tokens: int,
) -> torch.Tensor:
    """Compute the mean activation vector for the target layer over a sample."""
    sums = None
    n = 0
    for x in stream.iter_activations():
        if n + x.size(0) > max_tokens:
            x = x[: max_tokens - n]
        if sums is None:
            sums = x.sum(dim=0)
        else:
            sums += x.sum(dim=0)
        n += x.size(0)
        if n >= max_tokens:
            break
    return (sums / max(n, 1)).to(device)


def _ce_batch_stats(
    model, batch, hook_factory, encoder_layer, device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-spectrum (mean CE, correct-token count, valid-token count) for one batch.

    A fresh hook is installed for this batch and removed in a finally block, so a
    failure mid-forward cannot leave the model permanently patched.
    """
    import instanovo_io

    factory_hook = hook_factory()
    handle = encoder_layer.register_forward_hook(factory_hook) if factory_hook is not None else None
    try:
        with torch.no_grad():
            logits = instanovo_io.model_forward_logits(model, batch, device)
            ce, top1, targets, valid = instanovo_io.per_token_ce_and_top1(
                logits, batch["peptides"], pad_index=instanovo_io.PAD_INDEX,
            )
    finally:
        if handle is not None:
            handle.remove()

    v = valid.sum(dim=1)                                  # [B] valid tokens / spectrum
    return (
        (ce.sum(dim=1) / v.clamp_min(1)).cpu(),
        ((top1 == targets) & valid).sum(dim=1).cpu(),
        v.cpu(),
    )


def _ce_per_spectrum(
    model, loader, target_layer: int, hook_factory, device: str, max_spectra: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run InstaNovo over the DataLoader (capped at max_spectra spectra) under the
    encoder hook produced by hook_factory(), and return flat per-spectrum tensors
    (mean_ce[S], correct[S], valid[S]) in global (shuffle=False) spectrum order.

    hook_factory() returns a forward hook to install on the target encoder layer,
    or None for the clean (un-hooked) pass.

    ALIGNMENT CONTRACT. The loader reads the same spectra that extraction used,
    with shuffle=False and the same processor, so the i-th spectrum here is the
    i-th spectrum in the chunks. That is what lets per-spectrum CE be compared, by
    position, against the per-spectrum concept prevalence computed from the chunks.
    Both sides are truncated to the shorter length, so a prefix mismatch (e.g. an
    extraction max_spectra cap) degrades to a shorter aligned prefix rather than a
    silent misalignment.

    Per-spectrum (not aggregate) CE is what lets Phase 8 measure SELECTIVITY --
    whether ablating a feature hurts concept-bearing spectra more than others.
    """
    import instanovo_io

    encoder_layer = instanovo_io.get_encoder_layer(model, target_layer)
    ce_means, corrects, valids = [], [], []
    seen = 0
    loader_iter = iter(loader)
    if loader_iter is loader:
        raise TypeError(
            "Phase 7/8 requires a re-iterable DataLoader, not a one-shot iterator. "
            "Pass a DataLoader object so clean, SAE, zero, mean, and ablation modes "
            "all run over the same spectrum prefix."
        )
    for batch in loader_iter:
        if seen >= max_spectra:
            break
        ce_mean, correct, valid = _ce_batch_stats(
            model, batch, hook_factory, encoder_layer, device,
        )
        ce_means.append(ce_mean)
        corrects.append(correct)
        valids.append(valid)
        seen += int(ce_mean.shape[0])

    if not ce_means:
        z = torch.zeros(0)
        return z, z, z
    return (
        torch.cat(ce_means)[:max_spectra],
        torch.cat(corrects).to(torch.float32)[:max_spectra],
        torch.cat(valids).to(torch.float32)[:max_spectra],
    )


def _cached_clean_ce_per_spectrum(stream: ChunkStream, max_spectra: int) -> torch.Tensor | None:
    """Per-spectrum clean CE from extract's cached baseline (baseline_ce masked by
    baseline_decoder_mask), flat and in global order, capped at max_spectra. Returns
    None if any chunk lacks a cached baseline.

    Extract computed this with the same instanovo_io.per_token_ce_and_top1 path the
    Phase 7 loader uses, so on the same spectra the two should agree to float
    precision. Comparing them (in phase_7_loss_recovered) is a free check that the
    re-run loader is aligned, in order, with the chunks.
    """
    parts: list[torch.Tensor] = []
    seen = 0
    for meta in stream.iter_metadata():
        if seen >= max_spectra:
            break
        baseline_ce = meta.get("baseline_ce")
        baseline_decoder_mask = meta.get("baseline_decoder_mask")
        if baseline_ce is None or baseline_decoder_mask is None:
            return None
        mask = baseline_decoder_mask.to(torch.float32)
        per_spectrum = (baseline_ce * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        parts.append(per_spectrum)
        seen += meta["n_spectra"]
    if not parts:
        return None
    return torch.cat(parts)[:max_spectra]


def _token_weighted_mean(ce_means: torch.Tensor, valids: torch.Tensor) -> float:
    """Token-weighted CE: sum_s(mean_ce_s * valid_s) / sum_s(valid_s).

    Weighting by valid tokens rather than averaging the per-spectrum means keeps
    long peptides from being under-counted relative to short ones.
    """
    n = min(ce_means.numel(), valids.numel())
    if n == 0:
        return float("nan")
    return float((ce_means[:n] * valids[:n]).sum() / valids[:n].sum().clamp_min(1))


def _clean_ce_alignment(
    ce_clean_m: torch.Tensor, stream: ChunkStream, n_spectra_cap: int,
) -> dict | None:
    """Cross-check the re-run loader's clean CE against extract's cached baseline.

    Both were computed by the same code path on the same spectra, so on an aligned
    run they agree to float precision. A discrepancy on the CE scale means the
    loader is reordered or differently filtered relative to the chunks, which
    would silently invalidate every Phase 8 selectivity contrast.
    """
    cached = _cached_clean_ce_per_spectrum(stream, n_spectra_cap)
    if cached is None or not cached.numel() or not ce_clean_m.numel():
        return None

    n = min(cached.numel(), ce_clean_m.numel())
    diff = (ce_clean_m[:n] - cached[:n]).abs()
    cached_mean = float(cached[:n].mean())
    mean_abs = float(diff.mean())
    return {
        "n_spectra_compared": int(n),
        "loader_clean_ce": float(ce_clean_m[:n].mean()),
        "cached_clean_ce": cached_mean,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": mean_abs,
        # Heuristic: aligned runs agree to well under 1% of the CE scale; a
        # reordered / differently-filtered loader differs on the CE scale.
        "aligned": bool(mean_abs <= max(1e-2, 0.01 * abs(cached_mean))),
    }


def phase_7_loss_recovered(
    model, loader, sae: SparseAutoencoder, target_layer: int, device: str,
    layer_mean: torch.Tensor, stream: ChunkStream | None = None, n_spectra_cap: int = 1024,
) -> dict:
    """Compute token-weighted CE in four modes -- clean, SAE-patched, zero-ablated,
    mean-ablated -- and the loss recovered by the SAE substitution.

    loss_recovered = (CE_zero - CE_sae) / (CE_zero - CE_clean): the fraction of the
    layer's contribution (clean vs destroyed) that survives replacing the layer
    with its SAE reconstruction. ~1 means the SAE preserves the model's behaviour;
    ~0 means it is no better than deleting the layer. The clean baseline is the
    right reference here (this is a reconstruction-fidelity check), unlike Phase 8
    where per-feature effects use the SAE-full baseline.

    If `stream` is given, the loader-computed clean CE is cross-checked per spectrum
    against extract's cached baseline (clean_ce_alignment in the result).
    """
    ce_clean_m, corr_clean, valid_clean = _ce_per_spectrum(
        model, loader, target_layer, lambda: None, device, n_spectra_cap)
    ce_sae_m, corr_sae, valid_sae = _ce_per_spectrum(
        model, loader, target_layer, lambda: make_sae_substitution_hook(sae), device, n_spectra_cap)
    ce_zero_m, _, valid_zero = _ce_per_spectrum(
        model, loader, target_layer, _make_zero_hook, device, n_spectra_cap)
    ce_mean_m, _, valid_mean = _ce_per_spectrum(
        model, loader, target_layer, lambda: _make_mean_hook(layer_mean), device, n_spectra_cap)

    ce_clean = _token_weighted_mean(ce_clean_m, valid_clean)
    ce_sae = _token_weighted_mean(ce_sae_m, valid_sae)
    ce_zero = _token_weighted_mean(ce_zero_m, valid_zero)
    ce_mean = _token_weighted_mean(ce_mean_m, valid_mean)

    denom = ce_zero - ce_clean
    loss_recovered = (ce_zero - ce_sae) / denom if abs(denom) > 1e-12 else float("nan")
    n_tokens = float(valid_clean.sum())
    top1_clean = float(corr_clean.sum()) / max(n_tokens, 1.0)
    top1_sae = float(corr_sae.sum()) / max(float(valid_sae.sum()), 1.0)

    alignment = (
        _clean_ce_alignment(ce_clean_m, stream, n_spectra_cap)
        if stream is not None else None
    )

    return {
        "ce_clean": ce_clean,
        "ce_sae": ce_sae,
        "ce_zero": ce_zero,
        "ce_mean": ce_mean,
        "loss_recovered_vs_zero": loss_recovered,
        "top1_acc_clean": top1_clean,
        "top1_acc_sae": top1_sae,
        "top1_drop_pp": 100.0 * (top1_clean - top1_sae),
        "clean_ce_alignment": alignment,
        "n_tokens": int(n_tokens),
        "n_spectra": int(ce_clean_m.numel()),
    }


# =============================================================================
# Thesis 4.5 (Table 4.7) -- cross-layer feature stability
# =============================================================================

def _encode_layer_sample(
    extract_dir: Path,
    manifest: dict,
    layer: int,
    sae: SparseAutoencoder,
    n_tokens: int,
    device: str,
    batch_size: int,
) -> torch.Tensor | None:
    """Encode up to n_tokens of a layer's activations into SAE features.

    Chunks are read in manifest order and every layer stops at the same token
    count, so the returned matrices are row-aligned across layers -- which is what
    makes the per-token correlation between layers meaningful.
    """
    n_collected = 0
    layer_key = str(layer)
    sae_dtype = next(sae.parameters()).dtype
    out = torch.empty((n_tokens, sae.d_dict), dtype=sae_dtype)

    for chunk_info in manifest["chunks"]:
        if layer_key not in chunk_info["activations"]:
            continue
        acts_obj = torch.load(
            extract_dir / chunk_info["activations"][layer_key],
            map_location="cpu", weights_only=False,
        )
        x_cpu = acts_obj["activations"]
        for start in range(0, x_cpu.size(0), batch_size):
            remaining = n_tokens - n_collected
            if remaining <= 0:
                break
            end = min(start + batch_size, x_cpu.size(0), start + remaining)
            x = x_cpu[start:end].to(device=device, dtype=sae_dtype)
            with torch.no_grad():
                batch_out = sae.forward_inference(x)
            take = end - start
            out[n_collected:n_collected + take] = batch_out["features"].detach().cpu()
            n_collected += take
            del x, batch_out
            if n_collected >= n_tokens:
                break
        if n_collected >= n_tokens:
            break

    if n_collected == 0:
        return None
    return out[:n_collected]


def _match_anchor_across_layers(
    feat_idx: int,
    target_acts_centred: torch.Tensor,
    target_norms: torch.Tensor,
    other_centred_norms: dict[int, tuple[torch.Tensor, torch.Tensor]],
    target_layer: int,
    top_k: int,
) -> dict:
    """Best Pearson correlates of one anchor feature at every other layer.

    Correlation is the mean-centred dot product normalised by the product of
    norms, computed over the shared token sample. Centring/norms for the
    other layers are precomputed once by the caller (see cross_layer_matching)
    rather than recomputed here: recomputing a full [n_tokens, d_dict] centred
    copy per anchor feature -- potentially hundreds of times, once per anchor
    -- both wastes compute and repeatedly allocates/frees multi-GB tensors,
    which fragments memory on memory-constrained machines. The dot product
    itself is computed as a matrix-vector product (anchor_col @ other_centred)
    rather than an explicit elementwise multiply followed by a sum: the
    elementwise form materialises a full [n_tokens, d_dict] intermediate
    (identical in size to other_centred) on every call, whereas the matmul
    form never allocates more than its [d_dict]-sized output.
    """
    anchor_col = target_acts_centred[:, feat_idx]
    anchor_norm = target_norms[feat_idx]

    match_row = {"anchor_layer": target_layer, "anchor_feature": int(feat_idx)}
    for L, (other_centred, other_norms) in other_centred_norms.items():
        corrs = (anchor_col @ other_centred) / (anchor_norm * other_norms)
        k_use = min(top_k, corrs.numel())
        top = torch.topk(corrs, k_use)
        match_row[f"L{L}_best_feature"] = int(top.indices[0])
        match_row[f"L{L}_best_corr"] = float(top.values[0])
        match_row[f"L{L}_top{top_k}"] = ";".join(
            f"F{int(top.indices[i])}:{float(top.values[i]):.3f}" for i in range(k_use)
        )
    return match_row


def cross_layer_matching(
    extract_dir: Path,
    target_sae: SparseAutoencoder,
    target_layer: int,
    other_saes: dict[int, SparseAutoencoder],
    anchor_features: torch.Tensor,
    n_tokens: int,
    top_k: int,
    device: str,
    batch_size: int,
) -> dict:
    """For each anchor feature at target_layer, find its best Pearson correlate
    at every other available layer using activation co-firing on a token sample.
    """
    if not other_saes:
        return {"matches": []}
    if anchor_features.numel() == 0:
        return {"matches": [], "n_anchors": 0, "other_layers": list(other_saes.keys())}

    # Encode a token sample at every layer.
    manifest = json.loads((extract_dir / "manifest.json").read_text())
    feature_acts: dict[int, torch.Tensor] = {}
    skipped_layers: list[int] = []

    for L, sae in {target_layer: target_sae, **other_saes}.items():
        acts = _encode_layer_sample(
            extract_dir, manifest, L, sae, n_tokens, device, batch_size,
        )
        if acts is None:
            LOG.warning("Cross-layer: no activation chunks collected at L%d; skipping matching", L)
            skipped_layers.append(L)
            continue
        feature_acts[L] = acts
        LOG.info("Cross-layer: encoded %d tokens at L%d", acts.size(0), L)

    if target_layer not in feature_acts:
        LOG.warning("Cross-layer: no activation chunks collected for anchor layer L%d", target_layer)
        return {"matches": [], "n_anchors": 0, "other_layers": list(other_saes.keys())}

    target_acts = feature_acts.pop(target_layer)
    target_acts_centred = target_acts - target_acts.mean(dim=0, keepdim=True)
    target_norms = target_acts_centred.norm(dim=0).clamp_min(1e-12)
    del target_acts  # raw copy no longer needed once centred

    # Centre/normalise every other layer once up front instead of once per
    # anchor feature, and drop each raw tensor as soon as it's consumed --
    # both cut peak memory substantially when several layers' token samples
    # (each several GB) would otherwise all be resident at once.
    other_centred_norms: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for L in list(feature_acts.keys()):
        other_acts = feature_acts.pop(L)
        other_centred = other_acts - other_acts.mean(dim=0, keepdim=True)
        other_norms = other_centred.norm(dim=0).clamp_min(1e-12)
        del other_acts
        other_centred_norms[L] = (other_centred, other_norms)

    matches: list[dict] = []
    for feat_idx in anchor_features.tolist():
        match_row = _match_anchor_across_layers(
            feat_idx, target_acts_centred, target_norms, other_centred_norms,
            target_layer, top_k,
        )
        # Two keys means no other layer produced a match for this anchor.
        if len(match_row) > 2:
            matches.append(match_row)

    matched_layers = [L for L in other_saes if L in other_centred_norms]
    return {
        "matches": matches,
        "n_anchors": len(matches),
        "other_layers": matched_layers,
        "skipped_layers": skipped_layers,
    }


def cross_seed_verification(
    target_per_concept: dict,
    other_seed_evaluations: list[dict],
) -> dict:
    """Compute Jaccard overlap of top-N features per concept across seed runs."""
    if not other_seed_evaluations:
        return {"results": [], "note": "No other seed evaluations available."}

    results = []
    for concept_name, info in target_per_concept.items():
        target_top = {f["feature_idx"] for f in info[:CROSS_SEED_TOP_N]}
        per_seed = []
        for other in other_seed_evaluations:
            other_top = {
                f["feature_idx"] for f in other.get(concept_name, [])[:CROSS_SEED_TOP_N]
            }
            if not other_top:
                continue
            intersection = len(target_top & other_top)
            union = len(target_top | other_top)
            per_seed.append({
                "intersection": intersection,
                "union": union,
                "jaccard": intersection / max(union, 1),
            })
        if per_seed:
            results.append({
                "concept": concept_name,
                "mean_jaccard": float(np.mean([s["jaccard"] for s in per_seed])),
                "per_seed": per_seed,
            })

    return {"results": results, "n_other_seeds": len(other_seed_evaluations)}


# =============================================================================
# Thesis 4.6 -- Phase 8: causal ablation
# =============================================================================

def stratified_feature_selection(
    f1_dom: torch.Tensor,
    firing_rate: torch.Tensor,
    rejected_mask: torch.Tensor,
    n_total: int,
    n_deciles: int,
) -> torch.Tensor:
    """Sample features stratified by firing-rate decile, weighted by F1-dom.

    For each of n_deciles firing-rate bins, sample (n_total / n_deciles)
    features weighted by F1-dom within that bin. Surfaces narrow detectors
    that a pure F1-dom ranking would have demoted out of the top-N.
    """
    rejected_idx = rejected_mask.nonzero(as_tuple=False).squeeze(-1)
    if rejected_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    rates_rej = firing_rate[rejected_idx]
    scores_rej = f1_dom[rejected_idx]

    # Define deciles on log-firing-rate (rates vary by orders of magnitude).
    log_rates = torch.log10(rates_rej.clamp_min(1e-12))
    quantiles = torch.linspace(0, 1, n_deciles + 1)
    bin_edges = torch.quantile(log_rates, quantiles)

    per_bin = max(1, n_total // n_deciles)
    chosen: list[int] = []
    for b in range(n_deciles):
        in_bin = (log_rates >= bin_edges[b]) & (log_rates < bin_edges[b + 1] + 1e-9)
        bin_idx = rejected_idx[in_bin]
        bin_scores = scores_rej[in_bin]
        if bin_idx.numel() == 0:
            continue
        # Sample without replacement weighted by score; if everyone is zero,
        # fall back to uniform.
        weights = bin_scores.clamp_min(1e-6)
        sample_size = min(per_bin, bin_idx.numel())
        sampled = torch.multinomial(weights, sample_size, replacement=False)
        chosen.extend(bin_idx[sampled].tolist())

    return torch.tensor(chosen[:n_total], dtype=torch.long)


def _match_random_features(
    target_features: torch.Tensor,
    firing_rate: torch.Tensor,
    n_replicates: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Sample control feature sets matched to the target features by firing rate.

    Matching on firing rate (+/-0.3 dex per feature) is what makes the random control
    fair: a more active feature would tend to perturb more on ablation regardless of
    meaning, so an unmatched control would understate the target's specific effect.
    """
    target_rates = firing_rate[target_features]
    log_target = torch.log10(target_rates.clamp_min(1e-12))
    log_all = torch.log10(firing_rate.clamp_min(1e-12))

    matched_sets: list[torch.Tensor] = []
    n_features = firing_rate.numel()
    for _ in range(n_replicates):
        chosen: list[int] = []
        for lt in log_target:
            # Sample uniformly from features within +/-0.3 dex of the target rate.
            close = (
                ((log_all - lt).abs() < FIRING_RATE_MATCH_DEX)
                & (~torch.isin(torch.arange(n_features), target_features))
            )
            close_idx = close.nonzero(as_tuple=False).squeeze(-1)
            if close_idx.numel() == 0:
                continue
            pick = close_idx[torch.randint(close_idx.numel(), (1,), generator=generator)]
            chosen.append(int(pick))
        if chosen:
            matched_sets.append(torch.tensor(chosen, dtype=torch.long))
    return matched_sets


def _select_per_feature_targets(
    rejected: torch.Tensor, f1_dom: torch.Tensor, n: int,
) -> torch.Tensor:
    """Top-n features by F1-dom among the BH-rejected set."""
    f1_masked = f1_dom.clone()
    f1_masked[~rejected] = -1.0
    top = torch.topk(f1_masked, k=min(n, int(rejected.sum().item())))
    return top.indices


def _find_correlated_concepts(
    ci: int, phi_matrix: torch.Tensor | None, concept_names: list[str], threshold: float,
) -> list[int]:
    """Indices of concepts whose |phi| with concept ci exceeds the threshold."""
    n_concepts = len(concept_names)
    if (
        phi_matrix is None
        or phi_matrix.numel() == 0
        or phi_matrix.ndim != 2
        or ci >= phi_matrix.shape[0]
        or phi_matrix.shape[1] != n_concepts
    ):
        return []
    row = phi_matrix[ci]
    correlated = (row.abs() > threshold) & (torch.arange(row.numel()) != ci)
    return correlated.nonzero(as_tuple=False).squeeze(-1).tolist()


def _per_spectrum_prevalence(chunk: JoinedChunk, n_concepts: int) -> torch.Tensor:
    """Per-spectrum concept prevalence: the fraction of each spectrum's encoder
    tokens carrying each concept. Shape [n_spectra, n_concepts].

    Concepts are defined per encoder token (peaks of a spectrum), but CE is defined
    per spectrum (its predicted peptide). Aggregating concept labels to per-spectrum
    prevalence is what makes the two commensurable for the selectivity contrast.
    """
    return _per_spectrum_prevalence_from_tensors(
        chunk.labels, chunk.token_to_spectrum, chunk.n_spectra,
    )


def _per_spectrum_prevalence_from_tensors(
    labels: torch.Tensor,
    token_to_spectrum: torch.Tensor,
    n_spectra: int,
) -> torch.Tensor:
    """Per-spectrum concept prevalence from metadata and annotation tensors."""
    labels = labels.to(torch.float32)
    t2s = token_to_spectrum.long()
    sums = torch.zeros(n_spectra, labels.size(1))
    counts = torch.zeros(n_spectra, 1)
    sums.index_add_(0, t2s, labels)
    counts.index_add_(0, t2s, torch.ones(labels.size(0), 1))
    return sums / counts.clamp_min(1.0)


def _prevalence_per_spectrum_flat(
    stream: ChunkStream, n_concepts: int, max_spectra: int,
) -> torch.Tensor:
    """Per-spectrum concept prevalence over the stream, flat and in global order,
    capped at max_spectra. Aligns by position with _ce_per_spectrum (same source,
    same shuffle=False order). Shape [<=max_spectra, n_concepts]."""
    parts = []
    seen = 0
    for meta, annotation in stream.iter_metadata_annotations():
        if seen >= max_spectra:
            break
        parts.append(_per_spectrum_prevalence_from_tensors(
            annotation["token_labels"], meta["token_to_spectrum"], meta["n_spectra"],
        ))
        seen += meta["n_spectra"]
    if not parts:
        return torch.zeros(0, n_concepts)
    return torch.cat(parts)[:max_spectra]


def _compute_sae_full_baseline(
    model, loader, stream: ChunkStream, sae: SparseAutoencoder, target_layer: int,
    n_concepts: int, config: EvaluationConfig,
):
    """Per-spectrum CE under the FULL SAE reconstruction (no ablation) -- the
    reference every ablation delta is measured against -- plus per-spectrum concept
    prevalence. Computed once and reused across all ablations.

    Measuring against the SAE reconstruction (not the clean model) means the SAE's
    own reconstruction error is common to baseline and ablation and cancels in the
    delta, isolating the marginal causal effect of the ablated feature(s). CE comes
    from the model (loader) and prevalence from the chunks; both are in global
    shuffle=False order and truncated to the shorter length so they stay aligned.
    """
    ce_full, _corr, _valid = _ce_per_spectrum(
        model, loader, target_layer,
        lambda: make_sae_substitution_hook(sae), config.device, config.ablation_spectra,
    )
    prevalence = _prevalence_per_spectrum_flat(stream, n_concepts, config.ablation_spectra)
    n = min(ce_full.shape[0], prevalence.shape[0])
    if ce_full.shape[0] != prevalence.shape[0]:
        LOG.warning(
            "Phase 8 length mismatch (model CE=%d, chunk prevalence=%d); using aligned prefix of %d",
            ce_full.shape[0], prevalence.shape[0], n,
        )
    return ce_full[:n], prevalence[:n]


def _ablation_deltas(
    model, loader, sae: SparseAutoencoder, target_layer: int,
    features_to_ablate: torch.Tensor, config: EvaluationConfig, ce_full: torch.Tensor,
) -> torch.Tensor:
    """Per-spectrum delta-CE = CE(SAE, features ablated) - CE(SAE, full).

    ce_full is the cached per-spectrum SAE-full CE in global order; the ablation
    pass uses the same loader and cap, so subtraction is position-aligned.
    """
    if features_to_ablate.numel() == 0 or ce_full.numel() == 0:
        return torch.zeros_like(ce_full)
    ce_ablated, _c, _v = _ce_per_spectrum(
        model, loader, target_layer,
        lambda: make_sae_substitution_hook(sae, features_to_ablate),
        config.device, config.ablation_spectra,
    )
    n = min(ce_ablated.shape[0], ce_full.shape[0])
    return ce_ablated[:n] - ce_full[:n]


def _selectivity(delta_ce: torch.Tensor, prevalence_col: torch.Tensor) -> float:
    """Concept selectivity of an ablation: mean delta-CE on high-concept-prevalence
    spectra minus low-prevalence spectra (tercile split on prevalence). Positive
    means ablating the feature(s) hurts concept-bearing spectra specifically -- the
    falsifiable signature of a concept-specific feature. NaN if a tercile is empty
    (e.g. a concept present in every spectrum, so prevalence cannot be contrasted).
    """
    n = min(delta_ce.numel(), prevalence_col.numel())
    delta_ce = delta_ce[:n]
    prevalence_col = prevalence_col[:n]
    if delta_ce.numel() < MIN_SPECTRA_FOR_SELECTIVITY:
        return float("nan")
    lo_cut = torch.quantile(prevalence_col, 1.0 / 3.0)
    hi_cut = torch.quantile(prevalence_col, 2.0 / 3.0)
    hi = prevalence_col >= hi_cut
    lo = prevalence_col <= lo_cut
    if hi.sum() == 0 or lo.sum() == 0 or float(hi_cut) == float(lo_cut):
        return float("nan")
    return float(delta_ce[hi].mean() - delta_ce[lo].mean())


def _zscore(value: float, distribution: list[float]) -> float:
    """Standardise a value against a control distribution (NaNs dropped)."""
    vals = [v for v in distribution if v == v]
    if value != value or len(vals) < 2:
        return float("nan")
    mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    return (value - mu) / sd if sd > 1e-12 else float("nan")


def _orthogonalised_selectivity(
    delta_target: torch.Tensor,
    prevalence: torch.Tensor,
    prev_col: torch.Tensor,
    correlated_cis: list[int],
) -> tuple[float, int]:
    """Selectivity restricted to spectra free of every correlated concept.

    This removes the possibility that the selectivity signal is borrowed from a
    co-occurring concept rather than the target. When too few such spectra remain
    the result is NaN -- reported honestly, since it means the concepts are not
    separable in this data rather than that the test passed.
    """
    keep = torch.ones(delta_target.numel(), dtype=torch.bool)
    for cci in correlated_cis:
        keep &= prevalence[:, cci] <= 0.0
    n_kept = int(keep.sum().item())
    if n_kept < MIN_SPECTRA_FOR_SELECTIVITY:
        return float("nan"), n_kept
    return _selectivity(delta_target[keep], prev_col[keep]), n_kept


def _causal_report(
    delta_target: torch.Tensor, prevalence: torch.Tensor, target_ci: int,
    correlated_cis: list[int], control_deltas: list[torch.Tensor],
) -> dict:
    """Reduce a feature set's per-spectrum delta-CE to falsifiable causal metrics.

    The headline evidence is selectivity_z -- the target's concept-selectivity
    standardised against the matched random-feature selectivity distribution. The
    claim 'these features encode the concept' is retained only if selectivity > 0,
    selectivity_z is large and positive, and orthogonalised_selectivity (selectivity
    restricted to spectra free of the correlated concepts) stays positive. magnitude
    metrics are reported too, but on their own show importance, not specificity.
    """
    if prevalence.ndim != 2 or target_ci >= prevalence.shape[1]:
        prevalence = torch.zeros(delta_target.numel(), max(target_ci + 1, 1))
    n = min(delta_target.numel(), prevalence.shape[0])
    delta_target = delta_target[:n]
    prevalence = prevalence[:n]
    prev_col = prevalence[:, target_ci] if prevalence.numel() else torch.zeros(0)
    target_mean = float(delta_target.mean().item()) if delta_target.numel() else float("nan")
    sel = _selectivity(delta_target, prev_col)

    orth_sel, n_orthogonal = _orthogonalised_selectivity(
        delta_target, prevalence, prev_col, correlated_cis,
    )

    ctrl_means = [float(d.mean().item()) for d in control_deltas if d.numel()]
    ctrl_sels = [_selectivity(d[:prev_col.numel()], prev_col[:d.numel()]) for d in control_deltas]
    ctrl_sels_valid = [s for s in ctrl_sels if s == s]
    return {
        "mean_delta_ce": target_mean,
        "selectivity": sel,
        "orthogonalised_selectivity": orth_sel,
        "control_mean_delta_ce": float(np.mean(ctrl_means)) if ctrl_means else float("nan"),
        "control_mean_selectivity": float(np.mean(ctrl_sels_valid)) if ctrl_sels_valid else float("nan"),
        "magnitude_z": _zscore(target_mean, ctrl_means),
        "selectivity_z": _zscore(sel, ctrl_sels),
        "n_spectra": int(delta_target.numel()),
        "n_orthogonal_spectra": n_orthogonal,
    }


def _hypergeometric_null_chi2(
    nf: float, nc: float, n_total: int, n_shuffles: int, rng,
) -> list[float]:
    """Chi-square values under the fixed-margin null for one feature-concept pair.

    Holding both the feature's firing count and the concept's base rate fixed, the
    co-occurrence count follows Hypergeometric(N, marginal_c, marginal_f). Sampling
    it and recomputing chi-square gives a null that the marginal frequencies alone
    cannot explain away.
    """
    null_n11 = rng.hypergeometric(
        ngood=int(nc), nbad=int(n_total - nc), nsample=int(nf), size=n_shuffles,
    )
    null_chi2_values = []
    for nn11 in null_n11:
        nn10 = nf - nn11
        nn01 = nc - nn11
        nn00 = n_total - nn11 - nn10 - nn01
        num = n_total * (nn11 * nn00 - nn10 * nn01) ** 2
        denom = max((nn11 + nn10) * (nn00 + nn01) * (nn11 + nn01) * (nn00 + nn10), 1e-12)
        null_chi2_values.append(num / denom)
    return null_chi2_values


def _permutation_test_top_features(
    phase_4_results: dict, n_features: int, n_shuffles: int,
) -> dict:
    """Compare fixed-marginal null p-values to asymptotic chi-square p-values.

    A large discrepancy is a red flag that the asymptotic assumptions are breaking
    down for rare concepts or sparse features.

    Despite the historical "permutation" name, this samples the fixed-marginal
    hypergeometric null for each selected feature-concept pair.
    """
    stats = phase_4_results["stats"]
    rejected = phase_4_results["rejected"]
    marginal_f = phase_4_results["marginal_f"]
    marginal_c = phase_4_results["marginal_c"]
    n_total = phase_4_results["n_total_tokens"]

    f1_dom = stats["f1_dom"]
    max_f1 = f1_dom.max(dim=1).values
    rejected_any = rejected.any(dim=1)
    eligible = max_f1.clone()
    eligible[~rejected_any] = -1.0

    top_features = torch.topk(eligible, k=min(n_features, int(rejected_any.sum().item())))
    feature_indices = top_features.indices

    results = []
    rng = np.random.default_rng(42)
    for f in feature_indices:
        f = int(f)
        best_concept = int(f1_dom[f].argmax().item())
        observed_chi2 = float(stats["chi2_stat"][f, best_concept].item())

        nf = float(marginal_f[f])
        nc = float(marginal_c[best_concept])
        if nf == 0 or nc == 0 or n_total == 0:
            results.append({"feature": f, "concept": best_concept, "empirical_p": float("nan")})
            continue

        null_chi2_values = _hypergeometric_null_chi2(nf, nc, n_total, n_shuffles, rng)
        empirical_p = float(np.mean(np.array(null_chi2_values) >= observed_chi2))
        asymptotic_p = float(1.0 - scipy_chi2.cdf(observed_chi2, df=1))

        results.append({
            "feature_idx": f,
            "concept_idx": best_concept,
            "observed_chi2": observed_chi2,
            "empirical_p": empirical_p,
            "asymptotic_p": asymptotic_p,
            "discrepancy_ratio": empirical_p / max(asymptotic_p, 1e-12),
        })

    return {"n_tested": len(results), "results": results}


def _ablate_concept_group(
    model, loader, sae: SparseAutoencoder, target_layer: int,
    ci: int, f1_c: torch.Tensor, rejected_c: torch.Tensor,
    firing_rate_t: torch.Tensor, phi_matrix: torch.Tensor | None,
    concept_names: list[str], config: EvaluationConfig,
    ce_full: torch.Tensor, prevalence: torch.Tensor,
    random_control_generator: torch.Generator,
) -> tuple[dict, torch.Tensor, list[torch.Tensor], list[int]]:
    """Group-ablate a concept's top-N features against firing-rate-matched controls.

    Returns the causal report plus the selected features, the control sets, and the
    correlated-concept indices, so the caller can record what was tested.
    """
    top_n_features = stratified_feature_selection(
        f1_c, firing_rate_t, rejected_c, config.ablation_top_n, config.n_firing_rate_deciles,
    )
    matched_random = _match_random_features(
        top_n_features, firing_rate_t, n_replicates=config.n_random_controls,
        generator=random_control_generator,
    )
    correlated = _find_correlated_concepts(
        ci, phi_matrix, concept_names, threshold=CORRELATED_CONCEPT_PHI,
    )

    delta_target = _ablation_deltas(
        model, loader, sae, target_layer, top_n_features, config, ce_full,
    )
    control_deltas = [
        _ablation_deltas(model, loader, sae, target_layer, ctrl, config, ce_full)
        for ctrl in matched_random
    ]
    causal = _causal_report(delta_target, prevalence, ci, correlated, control_deltas)
    return causal, top_n_features, matched_random, correlated


def _ablate_individual_features(
    model, loader, sae: SparseAutoencoder, target_layer: int,
    ci: int, f1_c: torch.Tensor, rejected_c: torch.Tensor, stats: dict,
    config: EvaluationConfig, ce_full: torch.Tensor, prevalence: torch.Tensor,
) -> list[dict]:
    """Single-feature ablations: the necessity and selectivity of each feature alone.

    A null here is ambiguous rather than negative -- a concept encoded redundantly
    across many features survives losing any one of them -- which is why the group
    ablation runs alongside.
    """
    results = []
    for feat in _select_per_feature_targets(
        rejected_c, f1_c, n=config.ablation_per_feature_top,
    ):
        feat = int(feat)
        d = _ablation_deltas(
            model, loader, sae, target_layer, torch.tensor([feat]), config, ce_full,
        )
        results.append({
            "feature_idx": feat,
            "mean_delta_ce": float(d.mean().item()) if d.numel() else float("nan"),
            "selectivity": _selectivity(d, prevalence[:, ci]),
            "f1_dom": float(f1_c[feat]),
            "lift": float(stats["lift"][feat, ci]),
        })
    return results


def phase_8_causal_ablation(
    stream: ChunkStream,
    model,
    loader,
    sae: SparseAutoencoder,
    target_layer: int,
    phase_4_results: dict,
    phi_matrix: torch.Tensor | None,
    config: EvaluationConfig,
) -> dict:
    """Test whether BH-significant feature->concept associations are CAUSAL and
    CONCEPT-SPECIFIC, not merely correlational. See the module header for the
    methodology and its limits; in brief:

      - Intervention: patch encoder.layers[L] with the SAE reconstruction and zero
        the target feature(s) before decoding (a rank-k edit along their decoder
        directions).
      - Reference: the SAE reconstruction WITHOUT ablation, so the SAE's own
        reconstruction error cancels and delta-CE isolates the feature's marginal
        effect -- NOT the clean model, which would confound the two.
      - A claim is retained only if the effect is (1) selective -- larger on
        concept-bearing spectra than others; (2) above firing-rate-matched random
        controls (selectivity_z); and (3) survives controlling for correlated
        concepts (orthogonalised_selectivity). Any failure refutes or weakens it.
      - Necessity, not sufficiency: a null is ambiguous (feature redundancy), which
        the top-N group ablation partially addresses; a positive selective effect
        is the strong result.
    """
    stats = phase_4_results["stats"]
    rejected = phase_4_results["rejected"]
    firing_rate = phase_4_results["marginal_f"] / max(phase_4_results["n_total_tokens"], 1)
    firing_rate_t = torch.as_tensor(firing_rate, dtype=torch.float32)
    n_concepts = len(stream.concept_names)

    has_compatible_phi = (
        phi_matrix is not None
        and phi_matrix.ndim == 2
        and phi_matrix.shape[0] >= n_concepts
        and phi_matrix.shape[1] == n_concepts
    )
    if not has_compatible_phi:
        LOG.warning(
            "Phase 8: concept phi matrix missing or incompatible; "
            "correlated-concept controls will be skipped"
        )

    # The SAE-full per-spectrum CE is the shared reference for every ablation, and
    # yields the per-spectrum concept prevalence used for the selectivity contrast.
    # Computed once so the per-feature loop does not recompute it.
    ce_full, prevalence = _compute_sae_full_baseline(
        model, loader, stream, sae, target_layer, n_concepts, config,
    )
    LOG.info(
        "Phase 8: baseline ready for %d spectra, evaluating %d concepts "
        "(top_n=%d, controls=%d, per_feature_top=%d)",
        ce_full.numel(), n_concepts, config.ablation_top_n,
        config.n_random_controls, config.ablation_per_feature_top,
    )

    per_concept_results: dict[str, dict] = {}
    random_control_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    for ci, concept_name in enumerate(stream.concept_names):
        f1_c = stats["f1_dom"][:, ci]
        rejected_c = rejected[:, ci]
        if not rejected_c.any():
            per_concept_results[concept_name] = {
                "concept": concept_name, "n_eligible_features": 0,
                "causal": None, "per_feature_ablation": [],
            }
            continue

        LOG.info(
            "Phase 8: concept %d/%d %s (%d eligible features)",
            ci + 1, n_concepts, concept_name, int(rejected_c.sum().item()),
        )

        causal, top_n_features, matched_random, correlated = _ablate_concept_group(
            model, loader, sae, target_layer, ci, f1_c, rejected_c, firing_rate_t,
            phi_matrix, stream.concept_names, config, ce_full, prevalence,
            random_control_generator,
        )
        per_feature_results = _ablate_individual_features(
            model, loader, sae, target_layer, ci, f1_c, rejected_c, stats,
            config, ce_full, prevalence,
        )

        per_concept_results[concept_name] = {
            "concept": concept_name,
            "family": stream.family_of[concept_name],
            "diagnostic": concept_name in stream.diagnostic_concepts,
            "n_eligible_features": int(rejected_c.sum().item()),
            "top_n_features": top_n_features.tolist(),
            "matched_random_count": len(matched_random),
            "n_correlated_concepts_controlled": len(correlated),
            "causal": causal,
            "per_feature_ablation": per_feature_results,
        }

    permutation_test = _permutation_test_top_features(
        phase_4_results, config.permutation_n_features, config.permutation_n_shuffles,
    )
    return {"per_concept": per_concept_results, "permutation_test": permutation_test}


# =============================================================================
# Evaluator orchestrator
# =============================================================================

class Evaluator:
    """Runs the full eight-phase evaluation and writes the report."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.output_dir = config.output_subdir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        LOG.info("Loading SAE from %s", config.sae_checkpoint)
        self.sae = load_sae_from_checkpoint(config.sae_checkpoint, device=config.device)

        self.phi_matrix: torch.Tensor | None = None
        phi_path = config.annotation_dir / "concept_phi.pt"
        if phi_path.exists():
            phi_blob = torch.load(phi_path, map_location="cpu", weights_only=False)
            self.phi_matrix = phi_blob["phi"]

        self.instanovo = None
        self.loader = None
        if config.instanovo_path and (config.run_phase_7 or config.run_phase_8):
            self.instanovo = self._load_instanovo()
            if self.instanovo is not None:
                self.loader = self._build_loader()
                if self.loader is None:
                    LOG.warning("No spectra source for Phases 7-8; they will be skipped")

    def _load_instanovo(self):
        """Load InstaNovo for Phase 7/8 ablation passes via the integration layer.

        Uses instanovo_io.load_instanovo, which unpacks the (model, config)
        tuple that InstaNovo.load returns and pulls residue_set off the model.

        Returns None (skipping Phases 7-8) if the model uses flash attention:
        flash attention bypasses the standard encoder stack, so the layer hook
        would never fire and the substitution/ablation would silently be a no-op
        (loss_recovered ~1, all delta-CE ~0). This mirrors extract.py,
        which refuses to extract from a flash-attention checkpoint for the same
        reason. Load a non-flash checkpoint to enable the causal phases.
        """
        try:
            import instanovo_io
            model, _config, _residue_set = instanovo_io.load_instanovo(
                self.config.instanovo_path, device=self.config.device,
            )
        except ImportError as e:
            LOG.warning("InstaNovo import failed: %s; Phases 7 and 8 will be skipped", e)
            return None
        if instanovo_io.uses_flash_attention(model):
            LOG.warning(
                "InstaNovo checkpoint uses flash attention, which bypasses the standard "
                "encoder stack; the layer-%d hook would not fire and Phases 7-8 would "
                "silently measure no intervention. Skipping them. Load a non-flash "
                "checkpoint to enable the causal phases.", self.config.target_layer,
            )
            return None
        return model

    def _resolve_n_peaks(self, ecfg: dict) -> int:
        """Peaks per spectrum for the Phase 7/8 loader.

        Taken from the extract manifest so the re-run loader reproduces exactly
        the spectra extraction saw: n_peaks sets how many peak tokens each
        spectrum contributes, and Phase 8 compares per-spectrum CE from this
        loader against per-spectrum prevalence from those chunks by position. An
        explicit config value overrides it, for manifests written before the
        field existed.
        """
        if self.config.n_peaks is not None:
            manifest_n_peaks = ecfg.get("n_peaks")
            if manifest_n_peaks is not None and int(manifest_n_peaks) != self.config.n_peaks:
                LOG.warning(
                    "n_peaks override (%d) disagrees with the extract manifest (%d); "
                    "Phase 7/8 per-spectrum alignment against the chunks may break.",
                    self.config.n_peaks, int(manifest_n_peaks),
                )
            return self.config.n_peaks

        manifest_n_peaks = ecfg.get("n_peaks")
        if manifest_n_peaks is None:
            raise ValueError(
                "The extract manifest does not record n_peaks, so the Phase 7/8 "
                "loader cannot be guaranteed to rebuild the same spectra. Re-run "
                "extract.py, or pass --n-peaks with the value that extraction used."
            )
        return int(manifest_n_peaks)

    def _build_loader(self):
        """Build the shuffle=False DataLoader over the original spectra for Phase
        7/8 model forward passes, using the SAME processor and parameters as
        extraction so per-spectrum order matches the chunks.

        The spectra source is config.spectra_path, falling back to the dataset_path
        recorded in the extract manifest. Loader params (batch_size, num_workers,
        n_peaks) are read from the manifest's config so the run is reproducible.
        """
        import instanovo_io
        manifest = json.loads((self.config.extract_dir / "manifest.json").read_text())
        ecfg = manifest.get("config", {})
        source = self.config.spectra_path or ecfg.get("dataset_path")
        if source is None:
            return None
        n_peaks = self._resolve_n_peaks(ecfg)
        try:
            sdf = instanovo_io.load_spectrum_dataframe(source, annotated=True, shuffle=False)
            return instanovo_io.make_dataloader(
                sdf,
                self.instanovo.residue_set,
                batch_size=int(ecfg.get("batch_size", 32)),
                num_workers=int(ecfg.get("num_workers", 4)),
                n_peaks=n_peaks,
                annotated=True,
            )
        except Exception as e:  # noqa: BLE001 -- surface any data/loader failure clearly
            LOG.warning("Failed to build Phase 7/8 loader from %s: %s", source, e)
            return None

    def _new_stream(self) -> ChunkStream:
        """A fresh ChunkStream. Phases re-iterate from scratch, so each gets its own."""
        return ChunkStream(
            extract_dir=self.config.extract_dir,
            annotation_dir=self.config.annotation_dir,
            target_layer=self.config.target_layer,
            sae=self.sae,
            device=self.config.device,
            batch_size=self.config.batch_size,
            dtype=self.config.dtype,
        )

    def _model_phases_available(self) -> bool:
        """Phases 7 and 8 need both the InstaNovo model and a spectra loader."""
        return self.instanovo is not None and self.loader is not None

    def _run_sae_quality_phases(self, report: dict) -> None:
        """Phases 1+2, 5, and 6: everything measurable from the chunks and the SAE
        checkpoint alone (thesis 4.1-4.2)."""
        if self.config.run_phase_1_2:
            LOG.info("Running Phase 1+2 (reconstruction + sparsity)")
            report["phase_1_2"] = phase_1_2_reconstruction_and_sparsity(
                self._new_stream(), self.sae, self.config.device,
            )

        if self.config.run_phase_5:
            LOG.info("Running Phase 5 (geometric)")
            report["phase_5"] = phase_5_geometric(self.sae)

        if self.config.run_phase_6:
            LOG.info("Running Phase 6 (threshold sweep)")
            report["phase_6"] = phase_6_threshold_sweep(self._new_stream(), self.sae)

    def _run_association_phases(self, report: dict) -> dict | None:
        """Phases 3 and 4 (thesis 4.3). Returns the full Phase 4 result, which
        Phase 8 and the cross-layer/seed checks consume."""
        phase_4_full: dict | None = None

        if self.config.run_phase_3:
            LOG.info("Running Phase 3 (top-K activating tokens)")
            phase_3_results = phase_3_top_activating(
                self._new_stream(), self.sae, self.config.top_k_tokens,
            )
            report["phase_3"] = self._compact_phase_3(phase_3_results)

        if self.config.run_phase_4:
            LOG.info("Running Phase 4 (feature <-> concept associations)")
            phase_4_full = phase_4_feature_concept_associations(
                self._new_stream(), self.sae, self.config.fdr_q,
            )
            report["phase_4"] = self._compact_phase_4(phase_4_full)

        return phase_4_full

    def _resume_phase_4_from_cache(self, report: dict) -> tuple[dict | None, bool]:
        """Load a previous run's Phase 4 output so the phases that depend on it can
        run without rescanning every chunk. Returns (phase_4_full, loaded_flag)."""
        cache_dir = self.config.phase4_cache_dir or self.output_dir
        try:
            LOG.info("Loading Phase 4 cache from %s", cache_dir)
            phase_4_full, cached_report = load_phase_4_cache(
                cache_dir, self._new_stream(), self.sae,
            )
            # Carry forward the cached run's other phase blocks, but never its
            # Phase 4-dependent results: those must be recomputed for this run.
            for key, value in cached_report.items():
                if key not in {
                    "schema_version", "config", "elapsed_s",
                    "phase_8", "cross_layer", "cross_seed",
                } and key not in report:
                    report[key] = value
            LOG.info(
                "Loaded Phase 4 cache: %d significant pairs across %d features",
                phase_4_full["n_significant_pairs"],
                phase_4_full["n_features_with_concept"],
            )
            return phase_4_full, True
        except FileNotFoundError as e:
            if self.config.phase4_cache_dir is not None:
                raise
            LOG.warning("%s; Phase 4-dependent phases will be skipped", e)
            return None, False

    def _run_task_preservation(self, report: dict) -> None:
        """Phase 7 (thesis 4.4): how much sequencing behaviour survives patching
        the SAE reconstruction into the forward pass."""
        LOG.info("Running Phase 7 (loss recovered)")
        layer_mean = _compute_layer_mean(
            self._new_stream(), self.config.target_layer, self.config.device,
            max_tokens=LAYER_MEAN_MAX_TOKENS,
        )
        report["phase_7"] = phase_7_loss_recovered(
            self.instanovo, self.loader, self.sae,
            self.config.target_layer, self.config.device, layer_mean,
            stream=self._new_stream(), n_spectra_cap=self.config.ablation_spectra,
        )

    def _run_cross_layer(self, report: dict, phase_4_full: dict) -> None:
        """Cross-layer feature stability (thesis 4.5)."""
        LOG.info("Running cross-layer feature matching")
        other_saes = {
            L: load_sae_from_checkpoint(p, self.config.device)
            for L, p in self.config.other_layer_checkpoints.items()
        }
        # Anchor features: top-100 by F1-dom across any concept.
        f1_max = phase_4_full["stats"]["f1_dom"].max(dim=1).values
        f1_max[~phase_4_full["rejected"].any(dim=1)] = -1.0
        anchor_features = torch.topk(
            f1_max, k=min(CROSS_LAYER_ANCHORS, int((f1_max > 0).sum().item())),
        ).indices
        report["cross_layer"] = cross_layer_matching(
            extract_dir=self.config.extract_dir,
            target_sae=self.sae,
            target_layer=self.config.target_layer,
            other_saes=other_saes,
            anchor_features=anchor_features,
            n_tokens=self.config.cross_layer_token_sample,
            top_k=self.config.cross_layer_top_k,
            device=self.config.device,
            batch_size=self.config.batch_size,
        )

    def _run_cross_seed(self, report: dict, phase_4_full: dict) -> None:
        """Top-feature agreement across independently seeded SAE runs."""
        LOG.info("Running cross-seed verification")
        other_seed_evals = []
        for other_path in self.config.other_seed_checkpoints:
            other_report_path = other_path.parent / "eval" / "report.json"
            if other_report_path.exists():
                other_report = json.loads(other_report_path.read_text())
                other_seed_evals.append(
                    other_report.get("phase_4", {}).get("per_concept_top", {})
                )
        report["cross_seed"] = cross_seed_verification(
            phase_4_full["per_concept_top"], other_seed_evals,
        )

    def run(self) -> dict:
        """Run every enabled phase, write report.json plus the CSVs, and return the
        in-memory report. Phases 7-8 are silently skipped when the model or spectra
        are unavailable; all other phases need only the chunks and SAE checkpoint.

        Execution order differs from the thesis-ordered definitions above: Phase 8
        and the cross-layer/cross-seed checks all consume Phase 4's output, so
        Phase 4 must be computed (or loaded from cache) before them.
        """
        report: dict = {
            "schema_version": EVAL_SCHEMA_VERSION,
            "config": self.config.as_jsonable(),
        }
        t0 = time.time()

        self._run_sae_quality_phases(report)
        phase_4_full = self._run_association_phases(report)

        phase_4_loaded_from_cache = False
        phase_4_needed = (
            self.config.run_phase_8
            or self.config.run_cross_layer
            or self.config.run_cross_seed
        )
        if phase_4_full is None and phase_4_needed and not self.config.run_phase_4:
            phase_4_full, phase_4_loaded_from_cache = self._resume_phase_4_from_cache(report)

        if self.config.run_phase_7 and self._model_phases_available():
            self._run_task_preservation(report)

        if (self.config.run_phase_8 and self._model_phases_available()
                and phase_4_full is not None):
            LOG.info("Running Phase 8 (causal ablation)")
            report["phase_8"] = phase_8_causal_ablation(
                self._new_stream(), self.instanovo, self.loader, self.sae,
                self.config.target_layer, phase_4_full,
                self.phi_matrix,
                self.config,
            )

        if (self.config.run_cross_layer and self.config.other_layer_checkpoints
                and phase_4_full is not None):
            self._run_cross_layer(report, phase_4_full)

        if (self.config.run_cross_seed and self.config.other_seed_checkpoints
                and phase_4_full is not None):
            self._run_cross_seed(report, phase_4_full)

        report["elapsed_s"] = time.time() - t0
        self._write_report(report, None if phase_4_loaded_from_cache else phase_4_full)
        return report

    def _compact_phase_3(self, results: dict) -> dict:
        """Reduce Phase 3 output to a JSON-safe summary; full data goes to CSV."""
        csv_path = self.output_dir / "top_activating_tokens.csv"
        with open(csv_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["feature_idx", "rank", "activation", "chunk_idx", "token_in_chunk"])
            for f in range(results["values"].size(0)):
                for r in range(results["k"]):
                    val = float(results["values"][f, r])
                    chunk_id = int(results["chunk_ids"][f, r])
                    token_id = int(results["token_ids"][f, r])
                    if not math.isfinite(val) or chunk_id < 0 or token_id < 0:
                        continue
                    writer.writerow([f, r, val, chunk_id, token_id])
        return {"k": results["k"], "csv": csv_path.name}

    def _compact_phase_4(self, results: dict) -> dict:
        """Reduce Phase 4 output to a JSON-safe summary; full data goes to CSV."""
        rejected = results["rejected"]
        stats = results["stats"]
        p_values = results["p_values"]
        csv_path = self.output_dir / "feature_label_associations.csv"
        with open(csv_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "feature_idx", "concept", "f1", "f1_dom", "lift", "n_co",
                "chi2", "p_value", "p_value_bh_significant",
            ])
            rejected_pairs = rejected.nonzero(as_tuple=False)
            for fi, ci in rejected_pairs.tolist():
                writer.writerow([
                    fi, results["concept_names"][ci],
                    float(stats["f1"][fi, ci]),
                    float(stats["f1_dom"][fi, ci]),
                    float(stats["lift"][fi, ci]),
                    int(stats["n11"][fi, ci]),
                    float(stats["chi2_stat"][fi, ci]),
                    float(p_values[fi, ci]),
                    True,
                ])
        return {
            "n_significant_pairs": results["n_significant_pairs"],
            "n_features_with_concept": results["n_features_with_concept"],
            "per_concept_top": results["per_concept_top"],
            "per_family_top": results["per_family_top"],
            "csv": csv_path.name,
        }

    def _write_per_feature_csv(self, phase_4_full: dict) -> None:
        """Per-feature firing rate and best-scoring concept."""
        f1_max = phase_4_full["stats"]["f1_dom"].max(dim=1)
        firing_rate = phase_4_full["marginal_f"] / max(phase_4_full["n_total_tokens"], 1)
        csv_path = self.output_dir / "per_feature_stats.csv"
        with open(csv_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "feature_idx", "firing_rate", "max_f1_dom", "best_concept",
                "n_significant_concepts",
            ])
            for f in range(phase_4_full["stats"]["f1_dom"].size(0)):
                best_idx = int(f1_max.indices[f])
                writer.writerow([
                    f,
                    float(firing_rate[f]),
                    float(f1_max.values[f]),
                    phase_4_full["concept_names"][best_idx],
                    int(phase_4_full["rejected"][f].sum().item()),
                ])

    def _write_causal_csv(self, report: dict) -> None:
        """Per-concept causal-ablation summary.

        The headline columns are the selectivity metrics (concept-specificity),
        not raw delta-CE, which shows importance alone.
        """
        csv_path = self.output_dir / "causal_ablation.csv"
        with open(csv_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "concept", "family", "diagnostic", "n_eligible",
                "mean_delta_ce", "selectivity", "selectivity_z",
                "orthogonalised_selectivity", "control_mean_selectivity",
                "magnitude_z", "n_correlated_controlled",
            ])
            for concept, info in report["phase_8"]["per_concept"].items():
                causal = info.get("causal")
                if causal is None:
                    continue
                writer.writerow([
                    concept,
                    info.get("family", ""),
                    info.get("diagnostic", False),
                    info["n_eligible_features"],
                    causal["mean_delta_ce"],
                    causal["selectivity"],
                    causal["selectivity_z"],
                    causal["orthogonalised_selectivity"],
                    causal["control_mean_selectivity"],
                    causal["magnitude_z"],
                    info.get("n_correlated_concepts_controlled", 0),
                ])

    def _write_cross_layer_csv(self, report: dict) -> None:
        """One row per anchor feature, with its best match at every other layer."""
        csv_path = self.output_dir / "cross_layer_matches.csv"
        with open(csv_path, "w", newline="") as fp:
            rows = report["cross_layer"]["matches"]
            if rows:
                writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    def _write_report(self, report: dict, phase_4_full: dict | None) -> None:
        """Write report.json (tensor-free summary) and the per-feature / association /
        causal / cross-layer CSVs. Heavy tensor data lives only in the CSVs.
        """
        report_path = self.output_dir / "report.json"
        # Compact for JSON: strip the heavy tensor-bearing keys (already in CSV).
        compact = {k: v for k, v in report.items() if not isinstance(v, torch.Tensor)}
        report_path.write_text(json.dumps(compact, indent=2, default=str))
        LOG.info("Wrote report to %s", report_path)

        if phase_4_full is not None:
            self._write_per_feature_csv(phase_4_full)
        if "phase_8" in report and report["phase_8"]:
            self._write_causal_csv(report)
        if "cross_layer" in report:
            self._write_cross_layer_csv(report)


# --- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single-(layer, seed) evaluation run."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--extract-dir", type=Path, required=True)
    p.add_argument("--annotation-dir", type=Path, required=True)
    p.add_argument("--sae-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-layer", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase4-cache-dir", type=Path, default=None,
                   help="Existing eval directory containing report.json, "
                        "feature_label_associations.csv, and per_feature_stats.csv. "
                        "Used only when Phase 4 is skipped but Phase 8/cross checks need it; "
                        "defaults to this run's layer/seed eval directory.")
    p.add_argument("--instanovo-path", type=Path, default=None)
    p.add_argument("--spectra-path", type=Path, default=None,
                   help="Spectra source for Phase 7/8 CE; defaults to the extract "
                        "manifest's dataset_path. Must be the same spectra used for extraction.")
    p.add_argument("--n-peaks", type=int, default=None,
                   help="Override peaks per spectrum for the Phase 7/8 loader. "
                        "Defaults to the value recorded in the extract manifest, "
                        "which is what keeps the re-run loader aligned with the chunks.")

    p.add_argument("--other-layer-checkpoint", type=str, nargs="*", default=[],
                   help="Format: layer_idx=path/to/checkpoint.pt")
    p.add_argument("--other-seed-checkpoint", type=Path, nargs="*", default=[])

    p.add_argument("--skip", type=str, nargs="*", default=[],
                   choices=["1", "2", "3", "4", "5", "6", "7", "8",
                            "cross_layer", "cross_seed"])

    p.add_argument("--fdr-q", type=float, default=0.05)
    p.add_argument("--ablation-spectra", type=int, default=5000)
    p.add_argument("--ablation-top-n", type=int, default=10)
    p.add_argument("--ablation-per-feature-top", type=int, default=100)
    p.add_argument("--cross-layer-tokens", type=int, default=100_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    """CLI entry point: build the config from args, run the evaluation, write outputs."""
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    other_layer_checkpoints: dict[int, Path] = {}
    for spec in args.other_layer_checkpoint:
        layer_str, path_str = spec.split("=", 1)
        other_layer_checkpoints[int(layer_str)] = Path(path_str)

    config = EvaluationConfig(
        extract_dir=args.extract_dir,
        annotation_dir=args.annotation_dir,
        sae_checkpoint=args.sae_checkpoint,
        output_dir=args.output_dir,
        target_layer=args.target_layer,
        seed=args.seed,
        phase4_cache_dir=args.phase4_cache_dir,
        instanovo_path=args.instanovo_path,
        spectra_path=args.spectra_path,
        n_peaks=args.n_peaks,
        other_layer_checkpoints=other_layer_checkpoints,
        other_seed_checkpoints=list(args.other_seed_checkpoint),
        run_phase_1_2="1" not in args.skip and "2" not in args.skip,
        run_phase_3="3" not in args.skip,
        run_phase_4="4" not in args.skip,
        run_phase_5="5" not in args.skip,
        run_phase_6="6" not in args.skip,
        run_phase_7="7" not in args.skip,
        run_phase_8="8" not in args.skip,
        run_cross_layer="cross_layer" not in args.skip,
        run_cross_seed="cross_seed" not in args.skip,
        fdr_q=args.fdr_q,
        ablation_spectra=args.ablation_spectra,
        ablation_top_n=args.ablation_top_n,
        ablation_per_feature_top=args.ablation_per_feature_top,
        cross_layer_token_sample=args.cross_layer_tokens,
        device=args.device,
        batch_size=args.batch_size,
    )

    evaluator = Evaluator(config)
    evaluator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())