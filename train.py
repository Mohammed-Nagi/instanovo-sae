"""train.py -- sparse autoencoder training for InstaNovo interpretability.

Trains one SAE on the cached activations of a single encoder layer, reading the
per-layer files written by extract.py.

Architecture:
  - BatchTopK (Bussmann et al. 2024) during training: the top (k * batch)
    pre-activations are kept across the whole batch, giving k active features per
    token on average. Selecting across the batch rather than per token keeps
    rarely-winning features receiving gradient.
  - AuxK (Gao et al. 2024) reconstructs the main pass's residual from the
    next-best features, so features that lose the BatchTopK competition still
    receive gradient and do not go permanently dormant.
  - JumpReLU at inference with a single global threshold, calibrated to the
    BatchTopK selection boundary so the average L0 at inference matches k
    (Bussmann's BatchTopK->JumpReLU conversion).
  - Decoder unit-norm constraint (||W_dec[f]|| = 1) enforced by re-projection after
    every optimizer step, with optional tangent-space gradient projection.
  - 16x feature expansion (d_dict = 16 * d_model = 12,288 by default).

Activation loading:
  - cache_in_ram (default): the target layer's activations for the train split
    are loaded into RAM once, then every epoch is a true global shuffle with no
    further disk reads. Needs ~total_tokens * d_model * dtype bytes (~130 GB for
    the full nine-species dataset in bf16 -- use a high-memory instance).
  - --no-ram-cache: stream one chunk file per step instead (shuffles within and
    across chunks, not globally). For memory-constrained machines.

Multi-seed runs (--seed) write to seed-specific subdirectories so several seeds
coexist for cross-seed verification.

Output layout under --output-dir:
    layer_{L}/seed_{S}/
        checkpoint.pt          # SAECheckpoint (state + history + metrics)
        training_log.jsonl     # one row per logging interval
        config.json            # training configuration for reproducibility
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from schema import EXTRACT_SCHEMA_VERSION, SAE_SCHEMA_VERSION

LOG = logging.getLogger("train")

# Storage dtypes selectable on the CLI. float32 is the default and what the
# reported results use; bfloat16 halves activation memory at some cost in
# reconstruction fidelity.
DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Batches used by a mid-training (non-full) validation pass. Capped so periodic
# validation stays cheap relative to training; the final pass uses full=True.
QUICK_VALIDATION_BATCHES = 8


@dataclasses.dataclass
class TrainingConfig:
    """All hyperparameters for one SAE training run."""

    # I/O
    extract_dir: Path
    output_dir: Path
    target_layer: int

    # Architecture
    d_dict: int = 12_288               # number of features (16x d_model=768)
    k: int = 32                        # avg active features per token under BatchTopK
    k_aux: int = 512                   # aux features for dead-feature recovery
    alpha_aux: float = 1.0 / 32.0      # aux loss weight (Gao et al. 2024)

    # Decoder normalization
    normalize_decoder: bool = True             # enforce ||W_dec[f]|| = 1
    project_grad_to_tangent: bool = False      # subtract parallel grad component

    # Optimizer
    n_epochs: int = 3                  # cap; ~85M tokens/epoch reaches convergence by ~1.5-2
    batch_size: int = 8192             # tokens/batch; larger steadies the BatchTopK boundary
    lr: float = 2e-4
    lr_min_ratio: float = 0.1          # cosine decay floor (10% of peak)
    warmup_steps: int = 1000           # ~3% of total steps at the full-dataset scale
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    # JumpReLU threshold calibration
    threshold_ema_decay: float = 0.999  # decay for the provisional mid-training EMA
    threshold_calib_batches: int = 50   # batches for the authoritative post-training calibration

    # Validation / logging
    val_fraction: float = 0.02         # fraction of chunks held out for validation
    log_every: int = 100               # batches between log rows
    val_every: int = 1000              # batches between full validation passes

    # Reproducibility / parallelism
    seed: int = 0
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    # Activation loading
    cache_in_ram: bool = True          # load the layer into RAM once vs stream per chunk

    def output_subdir(self) -> Path:
        return self.output_dir / f"layer_{self.target_layer}" / f"seed_{self.seed}"

    def as_jsonable(self) -> dict:
        out = dataclasses.asdict(self)
        out["extract_dir"] = str(self.extract_dir)
        out["output_dir"] = str(self.output_dir)
        out["dtype"] = str(self.dtype).replace("torch.", "")
        return out


class SparseAutoencoder(nn.Module):
    """BatchTopK + AuxK at training, JumpReLU at inference, decoder unit-norm.

    Parameters:
      W_enc: [d_model, d_dict]  encoder direction per feature
      b_enc: [d_dict]           encoder bias
      W_dec: [d_dict, d_model]  decoder direction per feature, unit-norm per row
      b_dec: [d_model]          decoder bias (subtracted before encoding)
    """

    def __init__(
        self,
        d_model: int,
        d_dict: int,
        k: int,
        k_aux: int = 512,
        normalize_decoder: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_dict = d_dict
        self.k = k
        self.k_aux = k_aux
        self.normalize_decoder = normalize_decoder

        # Decoder initialization: unit-norm random directions.
        W_dec = torch.randn(d_dict, d_model)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec = nn.Parameter(W_dec)

        # Encoder initialization: tied to decoder (transpose).
        # The two diverge during training; this is just a sensible starting point.
        self.W_enc = nn.Parameter(W_dec.t().contiguous())
        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Global JumpReLU threshold (a buffer, not optimized). Set after training
        # by calibrate_threshold() to the BatchTopK selection boundary of the
        # converged model; a provisional EMA is tracked during training so
        # mid-training validation is meaningful. _threshold_count seeds that EMA
        # on the first batch.
        self.register_buffer("jumprelu_threshold", torch.zeros(()))
        self.register_buffer("_threshold_count", torch.zeros((), dtype=torch.long))

        # Running firing counts for dead-feature tracking. Reset once per epoch,
        # so these describe the current epoch, not the whole run.
        self.register_buffer("firing_count", torch.zeros(d_dict, dtype=torch.long))
        self.register_buffer("tokens_seen", torch.zeros(1, dtype=torch.long))

        # One-shot latch so a degenerate BatchTopK boundary warns once per run
        # rather than once per batch (see _batch_topk_mask).
        self._warned_degenerate_topk = False

    # --- Encoder / decoder primitives ----------------------------------------

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        """Compute encoder pre-activations: h = (x - b_dec) @ W_enc + b_enc."""
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def decode(self, features: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        """Reconstruct from features. add_bias=False omits b_dec, which is what
        the AuxK head needs since it targets a residual that already has b_dec
        removed (the main reconstruction subtracted it)."""
        out = features @ self.W_dec
        return out + self.b_dec if add_bias else out

    def _batch_topk_mask(self, h_relu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select the top (k * batch_size) pre-activations across the whole batch.

        Returns the selection mask and the boundary value (the smallest kept
        pre-activation), which calibrates the inference JumpReLU threshold.

        The boundary is None, and every positive pre-activation is kept, in two
        cases: a batch too small to fill k * batch_size, and a boundary of zero.
        The latter means fewer than k * batch_size pre-activations were positive
        at all, so topk reached into the ReLU's zeros. Selecting on `>= 0` would
        then mark EVERY feature as active -- reporting L0 = d_dict, hiding the
        collapse behind a zero dead-feature count, and feeding a 0.0 boundary
        into the threshold EMA that leaves JumpReLU gating on `h > 0` at
        inference. Falling back keeps the mask honest, so the dead-feature
        metrics show the collapse instead of masking it.
        """
        k_total = self.k * h_relu.size(0)
        flat = h_relu.reshape(-1)
        if k_total >= flat.numel():
            return h_relu > 0, None
        kth_value = torch.topk(flat, k_total, sorted=False).values.min()
        if kth_value <= 0:
            if not self._warned_degenerate_topk:
                LOG.warning(
                    "BatchTopK boundary hit zero: fewer than k*batch_size=%d "
                    "pre-activations are positive, so the model is close to "
                    "collapse. Keeping only positive pre-activations and "
                    "skipping the threshold EMA for such batches.",
                    k_total,
                )
                self._warned_degenerate_topk = True
            return h_relu > 0, None
        return h_relu >= kth_value, kth_value.detach()

    def _auxk_features(self, h_relu: torch.Tensor, main_mask: torch.Tensor) -> torch.Tensor:
        """Densify the top-k_aux pre-activations NOT selected by the main pass.

        These reconstruct the main pass's residual, so features that lost the
        BatchTopK competition keep receiving gradient instead of going dormant.
        """
        unselected = h_relu.masked_fill(main_mask, 0.0)
        k_aux_use = min(self.k_aux, self.d_dict)
        top_vals, top_idx = torch.topk(unselected, k_aux_use, dim=-1)
        features_aux = torch.zeros_like(h_relu)
        features_aux.scatter_(-1, top_idx, top_vals)
        return features_aux

    def forward_train(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """One training-mode forward pass producing main and aux reconstructions."""
        h = self.preactivations(x)
        h_relu = F.relu(h)

        main_mask, batch_threshold = self._batch_topk_mask(h_relu)
        features_main = h_relu * main_mask.to(h_relu.dtype)
        x_hat_main = self.decode(features_main)

        out: dict[str, torch.Tensor] = {
            "x_hat_main": x_hat_main,
            "features_main": features_main,
            "main_mask": main_mask,
            "h": h,
        }
        if batch_threshold is not None:
            out["batch_threshold"] = batch_threshold

        if self.k_aux > 0:
            features_aux = self._auxk_features(h_relu, main_mask)
            # Target the main-pass residual (detached so gradients don't flow into
            # the main reconstruction). Decode WITHOUT b_dec: the residual already
            # has b_dec removed, so re-adding it here would double-count the bias.
            out["x_hat_aux"] = self.decode(features_aux, add_bias=False)
            out["x_residual"] = (x - x_hat_main).detach()
            out["features_aux"] = features_aux

        return out

    @torch.no_grad()
    def forward_inference(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """One inference-mode forward pass: JumpReLU with the global threshold."""
        h = self.preactivations(x)
        gate = (h > self.jumprelu_threshold)
        features = h * gate.to(h.dtype)
        return {"x_hat": self.decode(features), "features": features, "h": h}

    # --- Decoder normalization ------------------------------------------------

    @torch.no_grad()
    def renormalize_decoder(self) -> None:
        """Re-project each W_dec row to unit L2 norm (called after optimizer step)."""
        if not self.normalize_decoder:
            return
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def project_grad_to_tangent_space(self) -> None:
        """Remove the component of W_dec.grad parallel to each W_dec row.

        Strictly Riemannian-style optimizer update on the unit-sphere manifold.
        Optional; without it, the re-projection alone is the standard approach
        and works well in practice.
        """
        if not self.normalize_decoder or self.W_dec.grad is None:
            return
        # Component of grad parallel to each unit decoder direction:
        # proj = <grad, dir> * dir
        dot = (self.W_dec.grad * self.W_dec).sum(dim=1, keepdim=True)
        self.W_dec.grad -= dot * self.W_dec

    # --- Running statistics ---------------------------------------------------

    @torch.no_grad()
    def update_threshold_ema(self, batch_threshold: torch.Tensor, decay: float) -> None:
        """EMA the per-batch BatchTopK boundary into a PROVISIONAL global threshold.

        This rolling estimate exists only so mid-training validation (which runs
        in JumpReLU/inference mode) reports a meaningful L0 and FVE while the
        model is still changing. It lags the model -- with a high decay it stays
        near the seed for the first ~1/(1-decay) batches -- so it is NOT the value
        that gets checkpointed. After training, calibrate_threshold() recomputes
        the authoritative threshold from the converged model.

        Tracking the boundary (the smallest kept value) rather than the mean
        selected pre-activation is what keeps inference L0 ~ k; the mean would sit
        above the boundary and make JumpReLU under-fire. Seeded on the first batch.
        """
        if int(self._threshold_count) == 0:
            self.jumprelu_threshold.copy_(batch_threshold)
        else:
            self.jumprelu_threshold.mul_(decay).add_(batch_threshold, alpha=1.0 - decay)
        self._threshold_count += 1

    @torch.no_grad()
    def update_firing_count(self, main_mask: torch.Tensor) -> None:
        """Track per-feature firing counts and total tokens seen."""
        self.firing_count += main_mask.sum(dim=0).long()
        self.tokens_seen += main_mask.size(0)

    @torch.no_grad()
    def reset_firing_count(self) -> None:
        self.firing_count.zero_()
        self.tokens_seen.zero_()

    @torch.no_grad()
    def dead_feature_count(self, min_firings: int = 1) -> int:
        """Number of features that fired fewer than min_firings times in the current window."""
        return int((self.firing_count < min_firings).sum().item())


def _load_layer_activations(path: Path) -> torch.Tensor:
    """Read one layer's [total_tokens, d_model] tensor from an
    acts_L{layer}_{chunk}.pt file. Reads the same payload extract.py
    writes via save_layer_activations (and that its load_layer_activations reads);
    duplicated here so the trainer needs no import of the model-dependent
    extraction module."""
    return torch.load(path, map_location="cpu", weights_only=False)["activations"]


class SAETokenDataset:
    """Yields shuffled [batch_size, d_model] activation batches for one layer.

    Two modes (see TrainingConfig.cache_in_ram):
      - cache_in_ram=True: load this split's per-layer files into one tensor on
        first iteration, then every epoch is a TRUE global shuffle in RAM with no
        further disk reads. Memory ~ total_tokens * d_model * dtype bytes.
      - cache_in_ram=False: stream one chunk file per step, shuffling within each
        chunk and across chunk order (not a global shuffle). For tight-RAM boxes.

    Reads the layout from extract.py: each manifest chunk entry carries a
    per-layer activation path and a meta path.
    """

    def __init__(
        self,
        extract_dir: Path,
        target_layer: int,
        batch_size: int,
        chunk_indices: list[int],   # which chunks to include (train or val split)
        shuffle: bool = True,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        cache_in_ram: bool = True,
    ):
        self.extract_dir = extract_dir
        self.target_layer = target_layer
        self.batch_size = batch_size
        self.chunk_indices = list(chunk_indices)
        self.shuffle = shuffle
        self.dtype = dtype
        self.cache_in_ram = cache_in_ram
        self.rng = torch.Generator().manual_seed(seed)
        self._cached: torch.Tensor | None = None

        manifest = self._read_manifest(extract_dir)
        self.act_paths, self.meta_paths = self._resolve_chunk_paths(
            extract_dir, manifest, target_layer,
        )
        self.chunk_token_counts = [
            int(torch.load(self.meta_paths[ci], map_location="cpu",
                           weights_only=False)["total_tokens"])
            for ci in self.chunk_indices
        ]

    @staticmethod
    def _read_manifest(extract_dir: Path) -> dict:
        """Load the extraction manifest, rejecting an incompatible schema.

        Stale chunks from an earlier extraction would silently pair the wrong
        activations with this run's metadata, so a mismatch is fatal.
        """
        manifest = json.loads((extract_dir / "manifest.json").read_text())
        if manifest["schema_version"] != EXTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"Extraction schema mismatch: manifest={manifest['schema_version']}, "
                f"trainer expects {EXTRACT_SCHEMA_VERSION}. Re-run extract.py."
            )
        return manifest

    @staticmethod
    def _resolve_chunk_paths(
        extract_dir: Path, manifest: dict, target_layer: int
    ) -> tuple[list[Path], list[Path]]:
        """Map every manifest chunk to its activation file for this layer and its
        meta file, failing early if the layer was never extracted."""
        layer_key = str(target_layer)
        chunks = manifest["chunks"]
        if not chunks or layer_key not in chunks[0]["activations"]:
            available = list(chunks[0]["activations"].keys()) if chunks else []
            raise KeyError(
                f"Layer {target_layer} has no activation files; available: {available}"
            )
        act_paths = [extract_dir / c["activations"][layer_key] for c in chunks]
        meta_paths = [extract_dir / c["meta"] for c in chunks]
        return act_paths, meta_paths

    def n_tokens(self) -> int:
        """Exact number of activation rows in this train/validation split."""
        return int(sum(self.chunk_token_counts))

    def steps_per_epoch(self) -> int:
        """Exact number of batches yielded by one pass over this split.

        In RAM-cache mode all chunks are concatenated before batching, so there
        is only one final partial batch. In streaming mode each chunk is batched
        independently, so each chunk can contribute its own final partial batch.
        """
        if not self.chunk_token_counts:
            return 0
        if self.cache_in_ram:
            total = self.n_tokens()
            return max(1, (total + self.batch_size - 1) // self.batch_size)
        return sum((n + self.batch_size - 1) // self.batch_size for n in self.chunk_token_counts)

    def _load_split_into_ram(self) -> torch.Tensor:
        """Concatenate this split's per-layer activations into one tensor.

        Sizes come from the (small) meta files so the output is preallocated and
        each activation file is freed right after copying -- peak memory is the
        full split plus one chunk, not twice the split.

        Each chunk's row count is checked against its own meta entry. Checking
        only the total would let two chunks drift in opposite directions and
        cancel out, which would misalign every activation after the first of
        them against the labels annotate.py wrote for it.
        """
        sizes = self.chunk_token_counts
        total = sum(sizes)
        buf: torch.Tensor | None = None
        off = 0
        for ci, n in zip(self.chunk_indices, sizes):
            acts = _load_layer_activations(self.act_paths[ci]).to(self.dtype)
            if acts.size(0) != n:
                raise ValueError(
                    f"Chunk {ci}: activation file has {acts.size(0)} rows but its "
                    f"meta records {n} tokens. The activations and metadata are "
                    "from different extraction runs; re-run extract.py."
                )
            if buf is None:
                buf = torch.empty(total, acts.size(1), dtype=self.dtype)
            buf[off:off + n] = acts
            off += n
            del acts
        assert buf is not None and off == total, (
            f"cached {off} tokens but meta reported {total}"
        )
        LOG.info(
            "Cached layer %d (%d chunks): %d tokens, %.2f GB in RAM",
            self.target_layer, len(self.chunk_indices), total,
            buf.element_size() * buf.nelement() / 1e9,
        )
        return buf

    def _iter_cached(self) -> Iterator[torch.Tensor]:
        """Batch the RAM-cached split under one global shuffle per epoch."""
        if self._cached is None:
            self._cached = self._load_split_into_ram()
        n = self._cached.size(0)
        order = (torch.randperm(n, generator=self.rng) if self.shuffle
                 else torch.arange(n))
        for start in range(0, n, self.batch_size):
            yield self._cached[order[start:start + self.batch_size]]

    def _iter_streaming(self) -> Iterator[torch.Tensor]:
        """Batch one chunk file at a time, shuffling within and across chunks."""
        chunk_order = list(self.chunk_indices)
        if self.shuffle:
            perm = torch.randperm(len(chunk_order), generator=self.rng).tolist()
            chunk_order = [chunk_order[i] for i in perm]
        for ci in chunk_order:
            acts = _load_layer_activations(self.act_paths[ci]).to(self.dtype)
            n = acts.size(0)
            if self.shuffle:
                acts = acts[torch.randperm(n, generator=self.rng)]
            for start in range(0, n, self.batch_size):
                yield acts[start:start + self.batch_size]

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self._iter_cached() if self.cache_in_ram else self._iter_streaming()


def cosine_with_warmup_lr(
    step: int,
    total_steps: int,
    lr_max: float,
    lr_min_ratio: float,
    warmup_steps: int,
) -> float:
    """Linear warmup -> cosine decay to lr_max * lr_min_ratio."""
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    lr_min = lr_max * lr_min_ratio
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def calibrate_threshold(
    model: SparseAutoencoder,
    dataset: SAETokenDataset,
    device: str,
    dtype: torch.dtype,
    n_batches: int,
) -> float:
    """Set the global JumpReLU threshold from the CONVERGED model.

    Runs the trained model over up to n_batches in BatchTopK mode, collects the
    per-batch selection boundary, and sets jumprelu_threshold to its mean
    (Bussmann's BatchTopK->JumpReLU conversion). Because it uses the final
    weights it is independent of the during-training EMA, which lags the model.
    Returns the threshold it set.
    """
    boundaries: list[float] = []
    for batch in dataset:
        out = model.forward_train(batch.to(device, dtype=dtype, non_blocking=True))
        if "batch_threshold" in out:
            boundaries.append(float(out["batch_threshold"]))
        if len(boundaries) >= n_batches:
            break
    if not boundaries:
        LOG.warning("Threshold calibration found no batches; leaving threshold unchanged.")
        return float(model.jumprelu_threshold)
    thr = sum(boundaries) / len(boundaries)
    model.jumprelu_threshold.fill_(thr)
    LOG.info(
        "Calibrated JumpReLU threshold = %.4f (mean BatchTopK boundary over %d batches)",
        thr, len(boundaries),
    )
    return thr


# --- Training setup -----------------------------------------------------------

def _seed_everything(seed: int) -> None:
    """Seed every RNG the run touches, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_chunks(manifest: dict, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Partition chunk indices into train and validation splits.

    Splitting by chunk rather than by token keeps a validation spectrum's tokens
    out of training entirely. At least one chunk is always held out, which
    requires at least two chunks total.
    """
    n_chunks = manifest["n_chunks"]
    if n_chunks < 2:
        raise ValueError(
            f"Need at least 2 extraction chunks to hold out a validation split, "
            f"got {n_chunks}. Lower --chunk-size or raise the spectra count "
            "(e.g. MAX_SPECTRA / SMOKE_TEST sizing) so extraction produces more "
            "than one chunk."
        )
    n_val_chunks = max(1, int(round(val_fraction * n_chunks)))
    rng = random.Random(seed)
    all_chunk_indices = list(range(n_chunks))
    rng.shuffle(all_chunk_indices)
    return sorted(all_chunk_indices[n_val_chunks:]), sorted(all_chunk_indices[:n_val_chunks])


def _infer_d_model(extract_dir: Path, manifest: dict, target_layer: int, chunk_idx: int) -> int:
    """Read the activation width from one chunk, so d_model never has to be
    passed in and can never disagree with the extracted files.

    Memory-mapped: only the tensor header is needed, and a production chunk is
    ~630 MB that would otherwise be read in full to learn one integer. Falls
    back to a normal load if the file predates zipfile serialisation.
    """
    chunk_entry = manifest["chunks"][chunk_idx]
    acts_path = extract_dir / chunk_entry["activations"][str(target_layer)]
    try:
        obj = torch.load(acts_path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError):
        obj = torch.load(acts_path, map_location="cpu", weights_only=False)
    return int(obj["activations"].size(1))


def _build_datasets(
    config: TrainingConfig, train_chunks: list[int], val_chunks: list[int]
) -> tuple[SAETokenDataset, SAETokenDataset]:
    """Build the train and validation loaders once, so any RAM cache they hold
    is populated a single time and reused across every epoch and validation."""
    def make(chunks: list[int], shuffle: bool) -> SAETokenDataset:
        return SAETokenDataset(
            extract_dir=config.extract_dir,
            target_layer=config.target_layer,
            batch_size=config.batch_size,
            chunk_indices=chunks,
            shuffle=shuffle,
            seed=config.seed,
            dtype=config.dtype,
            cache_in_ram=config.cache_in_ram,
        )

    return make(train_chunks, shuffle=True), make(val_chunks, shuffle=False)


def _compute_losses(
    model_out: dict[str, torch.Tensor],
    x: torch.Tensor,
    config: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Main reconstruction loss, AuxK loss, and their weighted total."""
    recon_loss = F.mse_loss(model_out["x_hat_main"], x)
    aux_loss = torch.tensor(0.0, device=config.device, dtype=config.dtype)
    if "x_hat_aux" in model_out:
        aux_loss = F.mse_loss(model_out["x_hat_aux"], model_out["x_residual"])
    return recon_loss, aux_loss, recon_loss + config.alpha_aux * aux_loss


def _optimizer_step(
    model: SparseAutoencoder,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    config: TrainingConfig,
) -> None:
    """Backward pass, optional decoder-gradient projection, clip, step, and the
    unit-norm re-projection that enforces ||W_dec[f]|| = 1."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if config.project_grad_to_tangent:
        model.project_grad_to_tangent_space()
    if config.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

    optimizer.step()
    model.renormalize_decoder()


class _EpochAccumulator:
    """Running per-epoch loss totals."""

    def __init__(self) -> None:
        self.main_loss = 0.0
        self.aux_loss = 0.0
        self.n_batches = 0

    def add(self, recon_loss: torch.Tensor, aux_loss: torch.Tensor) -> None:
        self.main_loss += float(recon_loss.detach())
        self.aux_loss += float(aux_loss.detach())
        self.n_batches += 1

    def mean_main(self) -> float:
        return self.main_loss / max(1, self.n_batches)

    def mean_aux(self) -> float:
        return self.aux_loss / max(1, self.n_batches)


def _run_epoch(
    epoch: int,
    step: int,
    model: SparseAutoencoder,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    train_dataset: SAETokenDataset,
    val_dataset: SAETokenDataset,
    total_steps: int,
    log_file,
    t0: float,
) -> tuple[int, _EpochAccumulator]:
    """Train for one pass over the train split. Returns the updated global step
    counter and the epoch's loss accumulator."""
    model.reset_firing_count()
    acc = _EpochAccumulator()

    for batch in train_dataset:
        x = batch.to(config.device, dtype=config.dtype, non_blocking=True)

        lr_now = cosine_with_warmup_lr(
            step, total_steps, config.lr, config.lr_min_ratio, config.warmup_steps,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        out = model.forward_train(x)
        recon_loss, aux_loss, loss = _compute_losses(out, x, config)
        _optimizer_step(model, optimizer, loss, config)

        # Firing counts and the provisional threshold EMA used only for
        # mid-training validation (calibrate_threshold sets the final threshold
        # after training). The EMA needs the BatchTopK boundary, which is absent
        # only for batches too small to fill k * batch_size.
        with torch.no_grad():
            model.update_firing_count(out["main_mask"])
            if "batch_threshold" in out:
                model.update_threshold_ema(
                    out["batch_threshold"], config.threshold_ema_decay,
                )

        acc.add(recon_loss, aux_loss)
        step += 1

        if step % config.log_every == 0:
            log_file.write(json.dumps({
                "step": step,
                "epoch": epoch,
                "lr": lr_now,
                "recon_loss": float(recon_loss.detach()),
                "aux_loss": float(aux_loss.detach()),
                "elapsed_s": time.time() - t0,
            }) + "\n")

        if step % config.val_every == 0:
            val_metrics = _validate(model, config, val_dataset)
            LOG.info(
                "Step %d (epoch %d): recon=%.4f aux=%.4f "
                "val_FVE_uncentered=%.4f val_FVE_centered=%.4f val_L0=%.1f dead=%d",
                step, epoch, recon_loss.item(), aux_loss.item(),
                val_metrics["fve_uncentered"], val_metrics["fve_centered"],
                val_metrics["l0_mean"], val_metrics["dead_features"],
            )

    return step, acc


def train_one_sae(config: TrainingConfig) -> dict:
    """Run one full SAE training session and write the checkpoint to disk."""
    out_dir = config.output_subdir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config.as_jsonable(), indent=2))

    log_path = out_dir / "training_log.jsonl"
    if log_path.exists():
        log_path.unlink()  # restart fresh -- checkpoints carry their own resume info
    log_file = open(log_path, "a", buffering=1)

    _seed_everything(config.seed)

    manifest = json.loads((config.extract_dir / "manifest.json").read_text())
    train_chunks, val_chunks = _split_chunks(manifest, config.val_fraction, config.seed)
    LOG.info(
        "Layer %d, seed %d: %d train chunks, %d val chunks",
        config.target_layer, config.seed, len(train_chunks), len(val_chunks),
    )

    d_model = _infer_d_model(
        config.extract_dir, manifest, config.target_layer, train_chunks[0],
    )
    LOG.info("d_model = %d, d_dict = %d, k = %d", d_model, config.d_dict, config.k)

    model = SparseAutoencoder(
        d_model=d_model,
        d_dict=config.d_dict,
        k=config.k,
        k_aux=config.k_aux,
        normalize_decoder=config.normalize_decoder,
    ).to(config.device, dtype=config.dtype)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    train_dataset, val_dataset = _build_datasets(config, train_chunks, val_chunks)

    # Exact planned step count for the LR scheduler. This uses per-chunk token
    # counts from meta files, so short final chunks and small smoke-test splits
    # do not distort cosine decay.
    train_tokens = train_dataset.n_tokens()
    steps_per_epoch = max(1, train_dataset.steps_per_epoch())
    total_steps = steps_per_epoch * config.n_epochs
    LOG.info(
        "Planned %d train tokens/epoch, %d steps/epoch, %d total steps",
        train_tokens, steps_per_epoch, total_steps,
    )

    step = 0
    training_history: list[dict] = []
    t0 = time.time()

    try:
        for epoch in range(config.n_epochs):
            step, acc = _run_epoch(
                epoch, step, model, optimizer, config,
                train_dataset, val_dataset, total_steps, log_file, t0,
            )

            val_metrics = _validate(model, config, val_dataset)
            epoch_summary = {
                "epoch": epoch,
                "train_recon_loss": acc.mean_main(),
                "train_aux_loss": acc.mean_aux(),
                "val_fve": val_metrics["fve"],
                "val_fve_uncentered": val_metrics["fve_uncentered"],
                "val_fve_centered": val_metrics["fve_centered"],
                "val_l0_mean": val_metrics["l0_mean"],
                "val_dead_features": val_metrics["dead_features"],
                "train_dead_features": model.dead_feature_count(),
                "elapsed_s": time.time() - t0,
            }
            training_history.append(epoch_summary)
            LOG.info("Epoch %d done: %s", epoch, epoch_summary)

        # Calibrate the JumpReLU threshold on the converged model, then final
        # validation + checkpoint. Calibration uses the train split (rng state no
        # longer matters post-training); final L0/FVE are measured on held-out val.
        calibrate_threshold(
            model, train_dataset, config.device, config.dtype, config.threshold_calib_batches,
        )
        final_metrics = _validate(model, config, val_dataset, full=True)
        _save_checkpoint(model, config, training_history, final_metrics, out_dir)

        LOG.info(
            "Training complete: FVE_uncentered=%.4f, FVE_centered=%.4f, "
            "L0=%.1f, dead=%d, %.0fs total",
            final_metrics["fve_uncentered"], final_metrics["fve_centered"],
            final_metrics["l0_mean"], final_metrics["dead_features"],
            time.time() - t0,
        )
        return final_metrics
    finally:
        log_file.close()


class _ValidationAccumulator:
    """Streaming accumulators for the validation metrics.

    Sums are kept over all tokens and dimensions so the metrics are exact for any
    batching, rather than an average of per-batch ratios.
    """

    def __init__(self, d_dict: int, device: str):
        self.sq_resid = 0.0
        self.sq_total = 0.0   # uncentered second moment: sum_{t,d} x^2
        self.colsum = None    # sum_t x, per hidden dimension, for centered FVE
        self.sum_l0 = 0.0
        self.n_tokens = 0
        self.n_batches = 0
        self.feature_fire = torch.zeros(d_dict, device=device)

    def add(self, x: torch.Tensor, x_hat: torch.Tensor, features: torch.Tensor) -> None:
        self.sq_resid += float(((x - x_hat) ** 2).sum().item())
        self.sq_total += float((x ** 2).sum().item())
        xs = x.sum(dim=0).to(torch.float64)
        self.colsum = xs if self.colsum is None else self.colsum + xs

        fired = (features > 0)
        self.sum_l0 += float(fired.float().sum().item())
        self.feature_fire += fired.float().sum(dim=0)
        self.n_tokens += x.size(0)
        self.n_batches += 1

    def metrics(self) -> dict:
        """Finalise into the metric dict, including both FVE conventions.

        `fve` is kept as the legacy uncentered fraction of variance explained:
        1 - SSE / sum(x^2). `fve_centered` is also reported for direct comparison
        to evaluate.py's phase_1_2["fve_overall"], which uses the centered total
        sum of squares around the per-dimension validation mean.
        """
        mean_energy = (
            float((self.colsum ** 2).sum().item()) / max(self.n_tokens, 1)
            if self.colsum is not None else 0.0
        )
        ss_tot_centered = max(self.sq_total - mean_energy, 1e-12)
        fve_uncentered = 1.0 - self.sq_resid / max(self.sq_total, 1e-12)

        return {
            "fve": fve_uncentered,
            "fve_uncentered": fve_uncentered,
            "fve_centered": 1.0 - self.sq_resid / ss_tot_centered,
            "fve_definition": "legacy_uncentered: 1 - SSE/sum(x^2)",
            "fve_centered_definition": "1 - SSE/sum((x - mean_validation_activation)^2)",
            "l0_mean": self.sum_l0 / max(self.n_tokens, 1),
            "dead_features": int((self.feature_fire == 0).sum().item()),
            "n_tokens": self.n_tokens,
        }


@torch.no_grad()
def _validate(
    model: SparseAutoencoder,
    config: TrainingConfig,
    val_dataset: SAETokenDataset,
    full: bool = False,
) -> dict:
    """Run inference-mode validation and return reconstruction + sparsity metrics.

    Takes a prebuilt loader so the (optionally cached) validation activations are
    loaded once and reused across every validation call. Mid-training calls are
    capped at QUICK_VALIDATION_BATCHES; pass full=True for the whole split.
    """
    model.eval()
    acc = _ValidationAccumulator(model.d_dict, config.device)

    for batch in val_dataset:
        x = batch.to(config.device, dtype=config.dtype, non_blocking=True)
        out = model.forward_inference(x)
        acc.add(x, out["x_hat"], out["features"])

        if not full and acc.n_batches >= QUICK_VALIDATION_BATCHES:
            break

    model.train()
    return acc.metrics()


def _save_checkpoint(
    model: SparseAutoencoder,
    config: TrainingConfig,
    training_history: list[dict],
    final_metrics: dict,
    out_dir: Path,
) -> None:
    """Write the final SAE state to checkpoint.pt."""
    ckpt = {
        "schema_version": SAE_SCHEMA_VERSION,
        "config": config.as_jsonable(),
        "target_layer": config.target_layer,
        "seed": config.seed,
        "d_model": model.d_model,
        "d_dict": model.d_dict,
        "k": model.k,
        "k_aux": model.k_aux,
        # Parameters.
        "W_enc": model.W_enc.detach().cpu(),
        "b_enc": model.b_enc.detach().cpu(),
        "W_dec": model.W_dec.detach().cpu(),
        "b_dec": model.b_dec.detach().cpu(),
        # Calibrated global JumpReLU threshold (scalar).
        "jumprelu_threshold": model.jumprelu_threshold.detach().cpu(),
        # Firing statistics for the FINAL EPOCH only -- _run_epoch resets these
        # each epoch so its dead-feature figure is per-epoch. The window is
        # recorded alongside them so the file is self-describing; evaluate.py
        # computes its own firing rates over the tokens it scores and does not
        # read these.
        "firing_count": model.firing_count.detach().cpu(),
        "tokens_seen": int(model.tokens_seen.item()),
        "firing_count_window": "final_epoch",
        # History + final metrics for downstream consumers.
        "training_history": training_history,
        "final_metrics": final_metrics,
        # Sanity check on the unit-norm constraint.
        "decoder_norm_min": float(model.W_dec.norm(dim=1).min().item()),
        "decoder_norm_max": float(model.W_dec.norm(dim=1).max().item()),
    }
    # Write to a temp file and atomically rename into place, so a process killed
    # mid-write (OOM-kill, preemption, Ctrl-C) never leaves a truncated
    # checkpoint.pt that a resumed run's existence-check would mistake for a
    # complete, loadable checkpoint.
    final_path = out_dir / "checkpoint.pt"
    tmp_path = out_dir / "checkpoint.pt.tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, final_path)
    LOG.info("Wrote checkpoint to %s", final_path)
    LOG.info(
        "  Decoder norm range after training: [%.6f, %.6f] (should be ~1.0)",
        ckpt["decoder_norm_min"], ckpt["decoder_norm_max"],
    )


def load_sae_from_checkpoint(
    checkpoint_path: Path,
    device: str = "cuda",
) -> SparseAutoencoder:
    """Reconstruct an SAE module from a checkpoint, ready for inference."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt["schema_version"] != SAE_SCHEMA_VERSION:
        raise ValueError(
            f"Schema mismatch: checkpoint={ckpt['schema_version']}, loader={SAE_SCHEMA_VERSION}"
        )

    # normalize_decoder only affects training (renormalize_decoder is never
    # called at inference), but carry the trained setting through so a
    # checkpoint reloaded to continue training keeps its own constraint.
    model = SparseAutoencoder(
        d_model=ckpt["d_model"],
        d_dict=ckpt["d_dict"],
        k=ckpt["k"],
        k_aux=ckpt.get("k_aux", 0),
        normalize_decoder=bool(ckpt.get("config", {}).get("normalize_decoder", True)),
    ).to(device)

    model.W_enc.data = ckpt["W_enc"].to(device)
    model.b_enc.data = ckpt["b_enc"].to(device)
    model.W_dec.data = ckpt["W_dec"].to(device)
    model.b_dec.data = ckpt["b_dec"].to(device)
    model.jumprelu_threshold = ckpt["jumprelu_threshold"].to(device)
    model.firing_count = ckpt["firing_count"].to(device)
    model.tokens_seen = torch.tensor([ckpt["tokens_seen"]], device=device, dtype=torch.long)
    model.eval()
    return model


# --- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--extract-dir", type=Path, required=True,
                   help="Directory with manifest.json from extract.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-layer", type=int, required=True,
                   help="Which encoder layer's activations to train on")

    p.add_argument("--d-dict", type=int, default=12_288)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--k-aux", type=int, default=512)
    p.add_argument("--alpha-aux", type=float, default=1.0 / 32.0)

    p.add_argument("--no-decoder-norm", action="store_true",
                   help="Disable ||W_dec[f]|| = 1 constraint (not recommended).")
    p.add_argument("--tangent-grad-projection", action="store_true",
                   help="Project W_dec grad onto tangent space before optimizer step.")

    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)

    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1000)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=list(DTYPE_MAP),
                   help="Compute/storage dtype for training. float32 is the "
                        "default and what reported results use; bfloat16 halves "
                        "activation memory at some cost in reconstruction "
                        "fidelity. Recorded in config.json either way.")
    p.add_argument("--no-ram-cache", action="store_true",
                   help="Stream chunk files instead of caching the layer in RAM "
                        "(slower, but for memory-constrained machines).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    config = TrainingConfig(
        extract_dir=args.extract_dir,
        output_dir=args.output_dir,
        target_layer=args.target_layer,
        d_dict=args.d_dict,
        k=args.k,
        k_aux=args.k_aux,
        alpha_aux=args.alpha_aux,
        normalize_decoder=not args.no_decoder_norm,
        project_grad_to_tangent=args.tangent_grad_projection,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min_ratio=args.lr_min_ratio,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        val_fraction=args.val_fraction,
        log_every=args.log_every,
        val_every=args.val_every,
        seed=args.seed,
        device=args.device,
        dtype=DTYPE_MAP[args.dtype],
        cache_in_ram=not args.no_ram_cache,
    )

    train_one_sae(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
