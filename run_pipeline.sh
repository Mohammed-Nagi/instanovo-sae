#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh -- SAE pipeline for layers {2, 4, 6, 8} of InstaNovo
# =============================================================================
#
# Four-script pipeline:
#   1. extract.py   -- one multi-layer forward pass over all spectra; writes
#                      per-layer activation files + ChunkMeta.
#                      REUSABLE: chunks are kept by default so future experiments
#                      (new SAE widths, seeds, layers) skip this expensive step.
#   2. annotate.py  -- one annotation pass; concept labels are layer-independent
#                      and dataset-fixed. REUSABLE: labels are always kept.
#   3. train.py     -- one run per (layer, seed); reads only its own layer's
#                      activation files from the extract dir.
#   4. evaluate.py  -- one run per (layer, seed); eight evaluation phases,
#                      including causal ablation (Phases 7/8) when the InstaNovo
#                      model is available.
#
# Flow:
#   prepare dataset (merge all HF splits into one parquet)
#     -> extract all layers in a single forward pass     [reusable, kept]
#     -> annotate spectra (layer-independent labels)     [reusable, kept]
#     -> for each layer: train SAE, unless SKIP_TRAIN=1 and checkpoints exist
#     -> for each layer: evaluate SAE
#          (cross-layer matching runs on the deepest/anchor layer)
#     -> cross-layer summary table
#
# Resume: every step skips if its sentinel output already exists.
#         Safe to kill and re-run at any point.
#
# -----------------------------------------------------------------------------
# WHERE THE TIME GOES, and the knobs that move it
# -----------------------------------------------------------------------------
#   Phase 8 (causal ablation) dominates everything else when enabled. Cost is
#   n_concepts x (1 group + N_RANDOM_CONTROLS + ABLATION_PER_FEATURE_TOP) model
#   passes over ABLATION_SPECTRA spectra, per layer. At the defaults that is
#   50 x 106 = 5,300 passes over 5,000 spectra = 26.5M spectrum-forwards per
#   layer. ABLATION_PER_FEATURE_TOP is 94% of that; halving it halves Phase 8.
#   This is why RUN_PHASE_8 defaults to 0.
#
#   SAE training I/O is the next lever. TRAIN_NO_RAM_CACHE=1 re-reads the whole
#   layer from disk every epoch (3x); =0 loads it once and shuffles in RAM. The
#   script now sizes this automatically against free memory -- see below.
#
#   Extraction is the largest one-off cost but runs once and is cached.
#
# Usage
#   MODEL_PATH=/path/to/instanovo.ckpt ./run_pipeline.sh
#   SMOKE_TEST=1 MODEL_PATH=... ./run_pipeline.sh          # fast end-to-end test
#   LAYERS_OVERRIDE="8" MODEL_PATH=... ./run_pipeline.sh   # single layer
#   OUTPUT_ROOT=/mnt/ssd/sae MODEL_PATH=... ./run_pipeline.sh
#   DATASET_PATH=/data/combined.parquet MODEL_PATH=... ./run_pipeline.sh  # skip merge
#   KEEP_CHUNKS=0 MODEL_PATH=... ./run_pipeline.sh         # free disk after run
#   SKIP_TRAIN=1 MODEL_PATH=... ./run_pipeline.sh          # evaluate existing checkpoints
#   RUN_PHASE_8=1 ABLATION_PER_FEATURE_TOP=20 MODEL_PATH=... ./run_pipeline.sh
#       # causal ablation at ~4x lower cost than the default per-feature depth
#   PHASE8_RESUME=1 SKIP_TRAIN=1 MODEL_PATH=... ./run_pipeline.sh
#       # reuse existing Phase 4 CSV/report cache and run only Phase 8
#
# Layers are independent: train/eval for different layers can run concurrently
# on separate GPUs by launching this script with LAYERS_OVERRIDE="<layer>" and
# a distinct DEVICE in separate shells, once extract + annotate have completed.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Layers to extract and train. Override: LAYERS_OVERRIDE="2 4" ./run_pipeline.sh
if [[ -n "${LAYERS_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    LAYERS=($LAYERS_OVERRIDE)
else
    LAYERS=(2 4 6 8)
fi

# Smoke runs default to their own output root. Sharing one with a production run
# mixes artefacts written under different settings -- extract.py now refuses such
# a resume outright (differing dtype / n_peaks), so an explicit split is clearer
# than a confusing hard failure.
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT:-./sae_pipeline_outputs_smoke}"
else
    OUTPUT_ROOT="${OUTPUT_ROOT:-./sae_pipeline_outputs}"
fi
PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
# Whether DEVICE came from the environment or from the default below. An explicit
# request is treated as strict (see the resolution block before the header); the
# default is free to follow the hardware.
DEVICE_EXPLICIT=0
[[ -n "${DEVICE:-}" ]] && DEVICE_EXPLICIT=1
DEVICE="${DEVICE:-cuda}"

# -----------------------------------------------------------------------------
# Dataset
# Full nine-species benchmark (all splits concatenated). If DATASET_PATH is
# already set (pointing at a pre-merged parquet), the merge step is skipped.
DATASET_HF_ID="${DATASET_HF_ID:-InstaDeepAI/ms_ninespecies_benchmark}"
# Kept OUTSIDE OUTPUT_ROOT so a smoke run reuses the merged parquet rather than
# re-downloading and re-merging 639k spectra into its own directory.
DATASET_DIR="${DATASET_DIR:-./sae_pipeline_outputs}"
COMBINED_DATASET="${COMBINED_DATASET:-$DATASET_DIR/combined_ninespecies.parquet}"
DATASET_PATH="${DATASET_PATH:-}"   # set externally to skip the merge step

# -----------------------------------------------------------------------------
# Model
MODEL_PATH="${MODEL_PATH:-./instanovo_v1.1.0.ckpt}"

# -----------------------------------------------------------------------------
# Extraction
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
# Remember whether the caller set these before the defaults below erase the
# distinction: the smoke block re-sizes them, and must not silently overrule an
# explicit request (same rule as DEVICE_EXPLICIT above).
CHUNK_SIZE_EXPLICIT="${CHUNK_SIZE:-}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# bfloat16 halves disk vs float32 with negligible precision loss for SAE training.
EXTRACT_DTYPE="${EXTRACT_DTYPE:-bfloat16}"
# Peaks kept per spectrum. Recorded in the extract manifest; evaluate.py reads it
# from there so the Phase 7/8 re-run loader rebuilds exactly the same spectra.
N_PEAKS="${N_PEAKS:-200}"

# -----------------------------------------------------------------------------
# Annotation
# Ion types: b ions, y ions, immonium (I), precursor (p).
# Internal fragments (ion type 'm') are enabled by default via --enable-internal
# inside annotate.py; pass --no-internal below to disable.
ION_TYPES="${ION_TYPES:-byIp}"
# 20 ppm is appropriate for high-resolution Orbitrap data. Use 'Da' mode and
# ~0.02 Da for lower-resolution instruments.
FRAGMENT_TOL="${FRAGMENT_TOL:-20.0}"
FRAGMENT_TOL_MODE="${FRAGMENT_TOL_MODE:-ppm}"
# Parallel annotation processes. Annotation is CPU-only, so the default uses
# every core; cap it on a shared node. Worker count never changes the output.
ANNOTATE_WORKERS="${ANNOTATE_WORKERS:-}"

# -----------------------------------------------------------------------------
# SAE architecture and training
D_MODEL=768                                    # InstaNovo encoder hidden size
EXPANSION_FACTOR="${EXPANSION_FACTOR:-16}"
D_DICT=$(( D_MODEL * EXPANSION_FACTOR ))       # 12,288 features at 16x
K="${K:-32}"                                   # avg active features / token (BatchTopK)
K_AUX="${K_AUX:-512}"                          # aux features for dead-feature recovery
ALPHA_AUX="${ALPHA_AUX:-0.03125}"              # aux loss weight (1/32, Gao et al. 2024)
LR="${LR:-2e-4}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
EPOCHS="${EPOCHS:-3}"                          # ~85M tokens/epoch -> convergence by ep 2
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-8192}"       # tokens/batch; larger steadies BatchTopK
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"  # 1 = require restored checkpoints; never train
# Dtype the trainer computes in, INDEPENDENT of EXTRACT_DTYPE: activations are
# cast on load, so a bfloat16 extract still occupies 4 bytes/element in a
# float32 RAM cache. choose_ram_cache sizes against THIS value -- sizing against
# the extract dtype under-counts the cache by 2x and invites an OOM mid-run.
TRAIN_DTYPE="${TRAIN_DTYPE:-float32}"

# RAM cache vs streaming. Unset (the default) means "decide automatically once
# the manifest is known" -- see choose_ram_cache below. Set to 1 to force
# streaming, 0 to force the RAM cache.
TRAIN_NO_RAM_CACHE="${TRAIN_NO_RAM_CACHE:-}"
# Fraction of free RAM the cached layer may occupy before streaming is chosen.
RAM_CACHE_SAFETY="${RAM_CACHE_SAFETY:-0.6}"

# -----------------------------------------------------------------------------
# Evaluation
FDR_Q="${FDR_Q:-0.05}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
# Phases 7 (loss recovered) and 8 (causal ablation) require InstaNovo forward
# passes. RUN_CAUSAL is the legacy switch for model-dependent evaluation:
# RUN_CAUSAL=0 disables both. By default, full runs keep Phase 7 enabled and
# skip Phase 8 because causal ablation is much more expensive.
RUN_CAUSAL="${RUN_CAUSAL:-1}"
RUN_PHASE_7_EXPLICIT="${RUN_PHASE_7:-}"   # see CHUNK_SIZE_EXPLICIT above
RUN_PHASE_7="${RUN_PHASE_7:-$RUN_CAUSAL}"
RUN_PHASE_8="${RUN_PHASE_8:-0}"
PHASE8_RESUME="${PHASE8_RESUME:-0}"          # 1 = force eval and skip all phases except 8
FORCE_EVAL="${FORCE_EVAL:-$PHASE8_RESUME}"   # 1 = ignore existing report.json sentinels
EVAL_SKIP_PHASES="${EVAL_SKIP_PHASES:-}"     # optional explicit evaluate.py --skip list
ABLATION_SPECTRA="${ABLATION_SPECTRA:-5000}"
ABLATION_TOP_N="${ABLATION_TOP_N:-10}"
# Single-feature ablations per concept. This is the dominant Phase 8 cost: each
# one is a full model pass over ABLATION_SPECTRA spectra, so the per-concept cost
# is 1 group + N_RANDOM_CONTROLS + this. At 20 that is 26 passes per concept
# rather than the 106 the old default of 100 implied -- a ~4x saving on the
# phase, for coarser per-feature resolution. The group-level causal metrics
# (selectivity, selectivity_z) are unaffected: they come from the group ablation.
ABLATION_PER_FEATURE_TOP="${ABLATION_PER_FEATURE_TOP:-20}"
CROSS_LAYER_TOKENS="${CROSS_LAYER_TOKENS:-100000}"

# -----------------------------------------------------------------------------
# Interpretation (Step 6) -- off by default: it is the only step that calls a
# paid external API, so it never runs unless asked for.
#
# It needs no GPU. The one SAE encode over INTERPRET_SAMPLE_CHUNKS chunks is a
# few minutes on CPU, and the rest is API latency. Run it on a CPU pod rather
# than queueing behind an accelerator.
#
# The key is read from a file, never from the environment of this script and
# never from the manifest, so it stays out of git and out of `ps`. Point
# INTERPRET_ENV_FILE at a file holding OPENAI_API_KEY=sk-... . See
# docs/aichor_deployment.md for placing one on the PVC.
RUN_INTERPRET="${RUN_INTERPRET:-0}"
INTERPRET_ENV_FILE="${INTERPRET_ENV_FILE:-}"
INTERPRET_STRATA="${INTERPRET_STRATA:-}"          # blank = interpret.py's default set
INTERPRET_N_PER_STRATUM="${INTERPRET_N_PER_STRATUM:-40}"
INTERPRET_SAMPLE_CHUNKS="${INTERPRET_SAMPLE_CHUNKS:-12}"
INTERPRET_MIN_FIRING_RATE="${INTERPRET_MIN_FIRING_RATE:-}"
INTERPRET_MODEL="${INTERPRET_MODEL:-}"            # blank = interpret.py's default
INTERPRET_MAX_TOKENS="${INTERPRET_MAX_TOKENS:-}"
INTERPRET_DEVICE="${INTERPRET_DEVICE:-auto}"      # auto = cuda if visible, else cpu
INTERPRET_DRY_RUN="${INTERPRET_DRY_RUN:-0}"       # 1 = build prompts, call nothing

# -----------------------------------------------------------------------------
# Disk management
# KEEP_CHUNKS=1 (default): extraction chunks are expensive to produce and are
# reusable across all future SAE experiments on this dataset. Only set to 0 if
# you are genuinely disk-constrained and do not plan to re-train.
KEEP_CHUNKS="${KEEP_CHUNKS:-1}"
# When KEEP_CHUNKS=0, retain this many chunks per layer instead of deleting them
# all. Cross-layer matching needs a token sample from EVERY layer at once, which
# a layer-at-a-time run cannot provide unless each layer leaves a sample behind.
# It reads chunks in manifest order and stops at CROSS_LAYER_TOKENS, so the first
# chunk or two is enough: at the defaults one chunk is ~108k tokens against a
# 100k requirement, and costs ~0.17 GB per layer. Set to 0 to delete everything
# and give up deferred cross-layer matching.
KEEP_CHUNK_SAMPLE="${KEEP_CHUNK_SAMPLE:-2}"
# Annotation labels are always kept -- they are small and layer-independent.
KEEP_ANNOTATION="${KEEP_ANNOTATION:-1}"
# `du -sh` over a multi-terabyte chunk tree can take minutes; set to 0 to skip.
REPORT_DISK_USAGE="${REPORT_DISK_USAGE:-1}"

# -----------------------------------------------------------------------------
# Smoke-test override
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    MAX_SPECTRA="${MAX_SPECTRA:-4096}"   # 0 = no cap; nonzero must be batch-aligned
    # Evaluation materialises a dense [tokens, d_dict] feature matrix for one
    # chunk at a time. At the production CHUNK_SIZE of 1024 with d_dict=12288
    # that is ~10 GB, which OOMs a typical development box, so the smoke test
    # uses smaller chunks. Must stay a multiple of EXTRACT_BATCH_SIZE.
    CHUNK_SIZE="${CHUNK_SIZE_EXPLICIT:-${SMOKE_CHUNK_SIZE:-128}}"
    EPOCHS=1
    SAE_BATCH_SIZE=1024
    EVAL_BATCH_SIZE=1024
    TRAIN_NO_RAM_CACHE="${TRAIN_NO_RAM_CACHE:-0}"
    # Warmup must be shorter than the run, or every step stays on the warmup
    # branch and the cosine decay is never exercised. At the smoke sizing above
    # one epoch is only a few hundred steps.
    WARMUP_STEPS="${SMOKE_WARMUP_STEPS:-25}"
    # Phase 7 IS exercised: it is one substitution pass, and it is the only step
    # that checks the encoder hook, the n_peaks agreement between extraction and
    # the re-run loader, and the flash-attention guard. Phase 8 stays off because
    # its cost is quadratic in concepts x features.
    RUN_CAUSAL="${SMOKE_RUN_CAUSAL:-1}"
    RUN_PHASE_7="${RUN_PHASE_7_EXPLICIT:-$RUN_CAUSAL}"
    RUN_PHASE_8="${SMOKE_RUN_PHASE_8:-0}"
    ABLATION_SPECTRA=256
    ABLATION_PER_FEATURE_TOP=5
    EXTRACT_DTYPE="float32"              # avoid bfloat16 issues on CPU-only smoke runs
    TRAIN_DTYPE="float32"
else
    MAX_SPECTRA="${MAX_SPECTRA:-0}"      # 0 = no cap (entire dataset)
fi

if [[ "$PHASE8_RESUME" == "1" ]]; then
    RUN_PHASE_7=0
    RUN_PHASE_8=1
    FORCE_EVAL=1
    EVAL_SKIP_PHASES="${EVAL_SKIP_PHASES:-1 2 3 4 5 6 7 cross_seed cross_layer}"
fi

# -----------------------------------------------------------------------------
# Setup -- paths and validation
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_PY="$SCRIPT_DIR/extract.py"
ANNOTATE_PY="$SCRIPT_DIR/annotate.py"
TRAIN_PY="$SCRIPT_DIR/train.py"
EVALUATE_PY="$SCRIPT_DIR/evaluate.py"
INTERPRET_PY="$SCRIPT_DIR/interpret.py"

for f in "$EXTRACT_PY" "$ANNOTATE_PY" "$TRAIN_PY" "$EVALUATE_PY"; do
    [[ -f "$f" ]] || { echo "ERROR: Missing pipeline script: $f" >&2; exit 1; }
done

# MODEL_PATH may be a local .ckpt path OR a pretrained model id (e.g.
# "instanovo-v1.1.0"), which instanovo_io resolves via InstaNovo.from_pretrained
# and downloads into InstaNovo's cache on first use. Only the path form is
# checked here, using the same path-vs-id rule instanovo_io applies.
case "$MODEL_PATH" in
    *.ckpt|*/*|*\\*)
        if [[ ! -f "$MODEL_PATH" ]]; then
            echo "ERROR: MODEL_PATH not found: $MODEL_PATH" >&2
            echo "       Set MODEL_PATH=/path/to/instanovo.ckpt, or use a model id" >&2
            echo "       such as MODEL_PATH=instanovo-v1.1.0 to download automatically." >&2
            exit 1
        fi
        ;;
    *)
        log_model_note="(pretrained id -- resolved and cached by InstaNovo on first use)"
        ;;
esac
if [[ "$MAX_SPECTRA" != "0" && $((MAX_SPECTRA % EXTRACT_BATCH_SIZE)) -ne 0 ]]; then
    echo "ERROR: MAX_SPECTRA ($MAX_SPECTRA) must be 0 or a multiple of EXTRACT_BATCH_SIZE ($EXTRACT_BATCH_SIZE)." >&2
    echo "       Extraction processes whole DataLoader batches; choose a batch-aligned cap." >&2
    exit 1
fi

# The sibling modules must be importable from SCRIPT_DIR.
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# Create and prove writability before anything expensive starts. On a cluster
# the output root is usually a mounted volume, and a volume owned by root is
# unwritable to an image that drops to an unprivileged user -- a failure that
# otherwise surfaces as a bare "mkdir: permission denied" partway through
# setup, after a queue wait. Name the uid and the offending path instead.
if ! mkdir -p "$OUTPUT_ROOT" 2>/dev/null || ! touch "$OUTPUT_ROOT/.write_probe" 2>/dev/null; then
    echo "ERROR: cannot write to OUTPUT_ROOT: $OUTPUT_ROOT" >&2
    echo "       running as uid=$(id -u) gid=$(id -g)" >&2
    parent="$OUTPUT_ROOT"
    while [[ ! -e "$parent" && "$parent" != "/" ]]; do parent="$(dirname "$parent")"; done
    echo "       nearest existing path: $parent" >&2
    ls -ld "$parent" 2>/dev/null | sed 's/^/         /' >&2
    echo "       If this is a mounted volume, it is owned by another user: set an" >&2
    echo "       fsGroup on the pod, or point OUTPUT_ROOT somewhere writable." >&2
    exit 1
fi
rm -f "$OUTPUT_ROOT/.write_probe"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
PIPELINE_LOG="$OUTPUT_ROOT/pipeline.log"

# Shared output locations.
EXTRACT_DIR="$OUTPUT_ROOT/extract"
ANNOTATION_DIR="$OUTPUT_ROOT/annotation"
# SAE training writes to $SAE_ROOT/layer_{L}/seed_{S}/checkpoint.pt
# Evaluation writes to  $SAE_ROOT/layer_{L}/seed_{S}/eval/report.json
# (evaluate.py appends layer/seed/eval automatically via output_subdir())
SAE_ROOT="$OUTPUT_ROOT/sae"
mkdir -p "$SAE_ROOT"

# Cross-process RAM-cache accounting for the documented pattern of running
# several invocations of this script concurrently against the same
# OUTPUT_ROOT (one per layer, on separate GPUs/DEVICE -- see choose_ram_cache
# below). Each invocation records how many bytes it has committed to the RAM
# cache here, so a sibling invocation's free-memory check can account for it.
RAM_RESERVATION_DIR="$OUTPUT_ROOT/.ram_reservations"
RAM_RESERVATION_FILE="$RAM_RESERVATION_DIR/$$"
trap 'rm -f "$RAM_RESERVATION_FILE" 2>/dev/null || true' EXIT

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
}

# run_step "description" "log_file" cmd [args...]
run_step() {
    local desc="$1"; shift
    local log_path="$1"; shift
    log "RUN  $desc"
    local t0
    t0=$(date +%s)
    if "$@" 2>&1 | tee "$log_path"; then
        log "OK   $desc  ($(($(date +%s) - t0))s)"
    else
        log "FAIL $desc -- see $log_path"
        exit 1
    fi
}

run_step_soft() {
    # run_step, but a failure returns 1 instead of ending the run. For work that
    # is an addition to an already-complete result, where aborting would strand
    # everything the run has not yet published.
    local desc="$1"; shift
    local log_path="$1"; shift
    log "RUN  $desc"
    local t0
    t0=$(date +%s)
    if "$@" 2>&1 | tee "$log_path"; then
        log "OK   $desc  ($(($(date +%s) - t0))s)"
        return 0
    fi
    log "WARN $desc failed -- see $log_path"
    return 1
}

sae_checkpoint_path() {   # $1 = layer
    echo "$SAE_ROOT/layer_$1/seed_${SEED}/checkpoint.pt"
}

eval_report_path() {       # $1 = layer
    # evaluate.py writes to output_dir/layer_{L}/seed_{S}/eval/ automatically.
    echo "$SAE_ROOT/layer_$1/seed_${SEED}/eval/report.json"
}

dir_size() {              # $1 = directory; cheap no-op when disabled
    if [[ "$REPORT_DISK_USAGE" != "1" ]]; then
        echo "(not measured)"
        return
    fi
    du -sh "$1" 2>/dev/null | awk '{print $1}' || echo "?"
}

# Decide between the RAM cache and streaming for SAE training.
#
# The cache loads one layer's activations into memory once, so every epoch is a
# true global shuffle with no further disk reads; streaming re-reads the layer
# from disk each epoch. With EPOCHS=3 that is 3 full reads of a ~100 GB layer
# instead of 1, so the cache is a large win whenever it fits. This sizes the
# layer from the manifest's token count and compares it against free RAM.
choose_ram_cache() {
    if [[ -n "$TRAIN_NO_RAM_CACHE" ]]; then
        log "  RAM cache      : explicit TRAIN_NO_RAM_CACHE=$TRAIN_NO_RAM_CACHE"
        return
    fi

    mkdir -p "$RAM_RESERVATION_DIR"

    # Sum bytes reserved by other still-running invocations of this script
    # against the same OUTPUT_ROOT, so this invocation doesn't independently
    # see the same "free" RAM a sibling already committed to its own cache and
    # collectively over-commit. A reservation file left behind by a process
    # that has since died (crash, kill -9) is stale and cleaned up here.
    local reserved_by_others=0 f pid bytes
    for f in "$RAM_RESERVATION_DIR"/*; do
        [[ -e "$f" ]] || continue
        pid="$(basename "$f")"
        [[ "$pid" == "$$" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            bytes="$(cat "$f" 2>/dev/null)"
            reserved_by_others=$(( reserved_by_others + ${bytes:-0} ))
        else
            rm -f "$f" 2>/dev/null || true
        fi
    done
    if (( reserved_by_others > 0 )); then
        log "  RAM cache      : $(( reserved_by_others / 1000000000 ))GB already reserved by other concurrent invocation(s)"
    fi

    local decision
    decision="$("$PYTHON" - "$EXTRACT_DIR/manifest.json" "$D_MODEL" "$RAM_CACHE_SAFETY" "$reserved_by_others" "$TRAIN_DTYPE" <<'PYEOF'
import json, sys
from pathlib import Path

manifest_path, d_model, safety, reserved, train_dtype = (
    Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]),
    sys.argv[5],
)
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}

try:
    manifest = json.loads(manifest_path.read_text())
    n_tokens = int(manifest["n_tokens"])
    # Size against the TRAINING dtype, not the manifest's storage dtype: the
    # loader casts on read, so a bfloat16 extract still fills a float32 buffer.
    needed = n_tokens * d_model * DTYPE_BYTES.get(train_dtype, 4)
except Exception as exc:                       # manifest unreadable -> stay safe
    print(f"1 0 unknown could-not-size-layer:{exc}")
    raise SystemExit(0)

# MemAvailable is the kernel's own estimate of what can be allocated without
# swapping, which is the right number here (MemFree ignores reclaimable cache).
# Bytes already reserved by concurrent sibling invocations are treated as
# unavailable to this one.
available = 0
try:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
except Exception:
    pass
free_for_us = max(0, available - reserved)

if available <= 0:
    print(f"1 {needed} free-ram-unknown")
elif needed <= free_for_us * safety:
    print(f"0 {needed} fits-in-{free_for_us/1e9:.0f}GB-available-after-other-reservations")
else:
    print(f"1 {needed} exceeds-{safety:.0%}-of-{free_for_us/1e9:.0f}GB-available-after-other-reservations")
PYEOF
)"
    TRAIN_NO_RAM_CACHE="${decision%% *}"
    local rest="${decision#* }"
    local needed_bytes="${rest%% *}"
    log "  RAM cache      : auto -> no_ram_cache=$TRAIN_NO_RAM_CACHE  (${rest#* })"

    # Commit this invocation's reservation so concurrent siblings see it.
    # Held for the lifetime of this process (cleaned up by the EXIT trap),
    # since training loops over every layer in $LAYERS sequentially and each
    # layer's cache is the same size, so the peak is one layer, not the sum.
    if [[ "$TRAIN_NO_RAM_CACHE" == "0" ]]; then
        echo "$needed_bytes" > "$RAM_RESERVATION_FILE"
    fi
}

# -----------------------------------------------------------------------------
# Device resolution
#
# The default follows the hardware: cuda when torch can actually see a GPU, cpu
# otherwise, so a laptop smoke test needs no extra env var. torch is the arbiter
# rather than nvidia-smi -- a driver can be present while torch is a CPU-only
# build, and it is torch that every stage actually calls.
#
# Setting DEVICE explicitly is strict: DEVICE=cuda on a machine with no usable
# GPU aborts instead of quietly falling back, because a CPU run over the full
# dataset is orders of magnitude slower and would burn a cluster allocation
# before anyone noticed. Unset DEVICE to auto-select, or pass DEVICE=cpu.
# -----------------------------------------------------------------------------

if [[ "$DEVICE" == cuda* ]]; then
    if ! "$PYTHON" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
            >/dev/null 2>&1; then
        if [[ "$DEVICE_EXPLICIT" == "1" ]]; then
            echo "ERROR: DEVICE=$DEVICE was requested, but torch.cuda.is_available() is False." >&2
            echo "       Unset DEVICE to auto-select a device, or pass DEVICE=cpu to force CPU." >&2
            exit 1
        fi
        DEVICE_FALLBACK_NOTE="  (auto: no usable CUDA device, fell back from cuda)"
        DEVICE=cpu
    fi
fi

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

log "================================================================"
log "SAE pipeline starting"
log "  Script dir      : $SCRIPT_DIR"
log "  Output root     : $OUTPUT_ROOT"
log "  Layers          : ${LAYERS[*]}  (seed $SEED)"
log "  Dataset         : ${DATASET_PATH:-$DATASET_HF_ID (all splits merged)}"
log "  Model           : $MODEL_PATH ${log_model_note:-}"
log "  Extract dtype   : $EXTRACT_DTYPE  (chunk $CHUNK_SIZE, batch $EXTRACT_BATCH_SIZE, n_peaks $N_PEAKS)"
log "  Annotate        : ion_types=$ION_TYPES, tol=${FRAGMENT_TOL} ${FRAGMENT_TOL_MODE}"
log "  SAE config      : d_dict=$D_DICT (${EXPANSION_FACTOR}x), k=$K, k_aux=$K_AUX"
log "                    lr=$LR (min_ratio=$LR_MIN_RATIO, warmup=$WARMUP_STEPS)"
log "                    epochs=$EPOCHS, batch=$SAE_BATCH_SIZE, grad_clip=$GRAD_CLIP"
log "                    skip_train=$SKIP_TRAIN"
log "  Eval            : FDR q=$FDR_Q, phase7=$RUN_PHASE_7, phase8=$RUN_PHASE_8"
log "                    ablation_spectra=$ABLATION_SPECTRA, per_feature_top=$ABLATION_PER_FEATURE_TOP"
log "                    force_eval=$FORCE_EVAL, phase8_resume=$PHASE8_RESUME, explicit_skip='${EVAL_SKIP_PHASES:-}'"
log "  Keep chunks     : $KEEP_CHUNKS  (annotation always kept)"
log "  Smoke / cap     : smoke=${SMOKE_TEST:-0}, max_spectra=$MAX_SPECTRA"
log "  Python          : $("$PYTHON" --version 2>&1)"
log "  GPU             : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none detected')"
log "  Device          : $DEVICE${DEVICE_FALLBACK_NOTE:-}"
log "================================================================"

# Loud and greppable: on a GPU node this line means something is wrong with the
# GPU, not with the request, and the run is about to be far slower than planned.
if [[ -n "${DEVICE_FALLBACK_NOTE:-}" ]]; then
    log "WARNING: no usable CUDA device detected -- running on CPU. Expect a large"
    log "         slowdown; pass DEVICE=cuda to make this a hard error instead."
fi

# -----------------------------------------------------------------------------
# Step 0: Prepare dataset -- merge all HF splits into one local parquet
# -----------------------------------------------------------------------------

if [[ -n "$DATASET_PATH" ]]; then
    log "Using pre-existing dataset: $DATASET_PATH (skipping merge)"
else
    DATASET_PATH="$COMBINED_DATASET"
    if [[ -f "$DATASET_PATH" ]]; then
        log "Merged dataset already exists: $DATASET_PATH (skipping merge)"
    else
        run_step "Merge all splits of $DATASET_HF_ID" \
            "$OUTPUT_ROOT/prepare_dataset.log" \
            "$PYTHON" - "$DATASET_PATH" "$DATASET_HF_ID" <<'PYEOF'
import sys
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

out = Path(sys.argv[1])
hf_id = sys.argv[2]
print(f"Loading every split of {hf_id} ...")
dd = load_dataset(hf_id)
sizes = {k: len(v) for k, v in dd.items()}
print("Splits found:", sizes)
combined = concatenate_datasets([dd[k] for k in dd.keys()])
print(f"Combined total: {len(combined):,} spectra across {len(sizes)} split(s)")
if "spectrum_id" not in combined.column_names:
    combined = combined.add_column("spectrum_id", [str(i) for i in range(len(combined))])
    print("Added synthetic spectrum_id column (row index)")
out.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(str(out))
print(f"Wrote merged dataset -> {out}")
PYEOF
    fi
fi

# -----------------------------------------------------------------------------
# Step 1: Extract activations -- single forward pass, all target layers
# -----------------------------------------------------------------------------
#
# This is the most expensive step. The result is stored permanently (KEEP_CHUNKS=1
# by default) so it can be reused across all future SAE experiments:
#   - different d_dict / k / seed values (re-run only train + evaluate)
#   - additional layers (re-run extract; per-chunk resume fills the new layer)
#   - evaluation-only reruns (extract + annotate are already done)
#
# Per-chunk resume is handled internally: if the manifest exists and individual
# chunk files are intact, only missing chunks are re-extracted. Running this step
# again after a crash is safe and efficient.

EXTRACT_ARGS=(
    --model-path   "$MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --output-dir   "$EXTRACT_DIR"
    --layers       "${LAYERS[@]}"
    --chunk-size   "$CHUNK_SIZE"
    --batch-size   "$EXTRACT_BATCH_SIZE"
    --num-workers  "$NUM_WORKERS"
    --n-peaks      "$N_PEAKS"
    --device       "$DEVICE"
    --dtype        "$EXTRACT_DTYPE"
)
if [[ "$MAX_SPECTRA" != "0" ]]; then
    EXTRACT_ARGS+=(--max-spectra "$MAX_SPECTRA")
fi

# Manifest/chunk disagreement. Before the resume-counting fix, extract.py's
# __iter__ did not yield chunks it skipped on resume, so extract_all counted only
# the NEW ones -- and _write_manifest lists chunks as range(n_chunks). A run that
# skipped chunks 0-9 and wrote 10-19 therefore recorded n_chunks=10 describing
# chunks 0-9 while carrying chunk 10-19's token count. Such a manifest is
# structurally valid and passes every schema check, so it is caught here by
# comparing it against what is actually on disk. The repair is cheap: delete
# manifest.json and re-run: per-chunk resume reuses every existing chunk and only
# the manifest is rebuilt (no re-extraction).
if [[ -f "$EXTRACT_DIR/manifest.json" && -d "$EXTRACT_DIR/chunks" ]]; then
    on_disk="$(find "$EXTRACT_DIR/chunks" -maxdepth 1 -name 'meta_*.pt' | wc -l | tr -d ' ')"
    listed="$("$PYTHON" -c "
import json, sys
print(json.load(open(sys.argv[1])).get('n_chunks', 0))
" "$EXTRACT_DIR/manifest.json" 2>/dev/null || echo 0)"
    if [[ "$on_disk" -gt 0 && "$listed" -gt 0 && "$on_disk" -ne "$listed" ]]; then
        log "FAIL Extract manifest lists $listed chunk(s) but $on_disk meta_*.pt files are on disk."
        log "     This manifest was written by a resumed run before the chunk-counting fix,"
        log "     so its chunk list, n_spectra and n_tokens describe the wrong chunks."
        log "     Delete $EXTRACT_DIR/manifest.json and re-run: the existing chunks are"
        log "     reused as-is and only the manifest is rebuilt."
        exit 1
    fi
fi

RUN_EXTRACT=1
if [[ -f "$EXTRACT_DIR/manifest.json" ]]; then
    missing_layers="$("$PYTHON" - "$EXTRACT_DIR/manifest.json" "${LAYERS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
requested = [str(x) for x in sys.argv[2:]]
manifest = json.loads(manifest_path.read_text())
chunks = manifest.get("chunks", [])

missing = []
for layer in requested:
    if not chunks or any(layer not in chunk.get("activations", {}) for chunk in chunks):
        missing.append(layer)

print(" ".join(missing))
PYEOF
)"
    if [[ -z "$missing_layers" ]]; then
        log "Extract manifest exists and contains requested layers (${LAYERS[*]}) -- skipping extraction"
        RUN_EXTRACT=0
    else
        log "Extract manifest exists but is missing requested layer(s): $missing_layers"
        log "Re-running extraction; per-chunk resume will keep intact chunks and fill missing layer files."
    fi
fi

if [[ "$RUN_EXTRACT" == "1" ]]; then
    # Disk preflight. Extraction is the one step that can consume hundreds of GB,
    # and it writes for hours before it would hit ENOSPC -- by which point the
    # GPU time is already spent. Estimating from the measured footprint of the
    # nine-species benchmark (~104 GB per layer in bf16 over 639,286 spectra),
    # scaled by this run's spectrum cap, dtype width and layer count.
    #
    # There is no way to ask AIchor how large a PVC is; free space on the mount
    # is the number that actually matters anyway, so read it directly.
    if [[ "${SKIP_DISK_CHECK:-0}" != "1" ]]; then
        avail_kb="$(df -Pk "$OUTPUT_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
        if [[ -z "$avail_kb" ]]; then
            log "WARNING: could not read free space for $OUTPUT_ROOT -- skipping the disk preflight"
        else
            bytes_per_elem=2
            [[ "$EXTRACT_DTYPE" == "float32" ]] && bytes_per_elem=4
            spectra="$MAX_SPECTRA"
            [[ "$spectra" == "0" ]] && spectra=639286
            need_gb="$("$PYTHON" -c "
import sys
spectra, n_layers, width = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
# 104 GB/layer at bf16 for 639,286 spectra, measured.
print(round(104.0 * (spectra / 639286.0) * (width / 2.0) * n_layers, 1))
" "$spectra" "${#LAYERS[@]}" "$bytes_per_elem")"
            avail_gb="$(awk -v k="$avail_kb" 'BEGIN {printf "%.1f", k / 1048576}')"
            per_layer_gb="$(awk -v n="$need_gb" -v l="${#LAYERS[@]}" 'BEGIN {printf "%.1f", n / l}')"
            log "Disk preflight   : ${avail_gb} GB free at $OUTPUT_ROOT, ~${need_gb} GB needed" \
                "(~${per_layer_gb} GB/layer x ${#LAYERS[@]})"
            if awk -v a="$avail_gb" -v n="$need_gb" 'BEGIN {exit !(a < n)}'; then
                # Say what would fit, not just that this does not. The PVC's size
                # is not visible from inside the job and the CLI cannot report it,
                # so this message is often the only way to find out how much room
                # there actually is -- make it enough to act on.
                fits="$(awk -v a="$avail_gb" -v p="$per_layer_gb" 'BEGIN {n=int(a/p); print (n<0?0:n)}')"
                echo "ERROR: not enough free space for extraction." >&2
                echo "       ${avail_gb} GB free, ~${need_gb} GB needed for ${#LAYERS[@]} layer(s)" >&2
                echo "       of $spectra spectra at $EXTRACT_DTYPE (~${per_layer_gb} GB per layer)." >&2
                if [[ "$fits" -ge 1 ]]; then
                    echo "       This volume holds about ${fits} layer(s) at a time. Run them in" >&2
                    echo "       batches, deleting each batch's activations before the next:" >&2
                    echo "         LAYERS_OVERRIDE=\"2\" KEEP_CHUNKS=0 ... ./run_pipeline.sh" >&2
                    echo "       Each batch repeats the extraction forward pass, so prefer the" >&2
                    echo "       largest batch that fits." >&2
                else
                    echo "       Not even one layer fits; the volume needs to be enlarged." >&2
                fi
                echo "       Or set SKIP_DISK_CHECK=1 to override this estimate." >&2
                exit 1
            fi
        fi
    fi

    run_step "Extract activations (layers ${LAYERS[*]}, dtype=$EXTRACT_DTYPE)" \
        "$OUTPUT_ROOT/extract.log" \
        "$PYTHON" "$EXTRACT_PY" "${EXTRACT_ARGS[@]}"
fi

# -----------------------------------------------------------------------------
# Step 2: Annotate spectra -- single pass, layer-independent concept labels
# -----------------------------------------------------------------------------
#
# Reads only ChunkMeta files (not activation tensors) -- much cheaper than
# extraction. Labels are fixed once the dataset and concept vocabulary are fixed,
# so they are always kept and reused across all layer/seed/SAE-width experiments.
#
# Fragment tolerance: 20 ppm for Orbitrap-class high-resolution data. A 0.5 Da
# window is far too wide at Orbitrap resolution and would match spurious peaks
# as b/y ions, corrupting the cleavage-site labels.

# The existence check alone is not enough: annotate.py's own schema check only
# runs once annotate.py is invoked, and skipping the step means it never is. A
# label set written under an older ANNOTATION_SCHEMA_VERSION would then be reused
# silently, which is exactly the stale-artefact failure the versions exist to
# prevent -- so validate before deciding to skip.
ANNOTATION_STATE="missing"
if [[ -f "$ANNOTATION_DIR/annotation_manifest.json" ]]; then
    ANNOTATION_STATE="$("$PYTHON" - "$ANNOTATION_DIR/annotation_manifest.json" "$SCRIPT_DIR" <<'PYEOF'
import json, sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])
from schema import ANNOTATION_SCHEMA_VERSION

try:
    found = int(json.loads(Path(sys.argv[1]).read_text())["schema_version"])
except Exception as exc:
    print(f"unreadable:{exc}")
    raise SystemExit(0)
print("current" if found == ANNOTATION_SCHEMA_VERSION
      else f"stale:{found}:{ANNOTATION_SCHEMA_VERSION}")
PYEOF
)"
fi

if [[ "$ANNOTATION_STATE" == "current" ]]; then
    log "Annotation manifest exists and matches the current schema -- skipping annotation"
elif [[ "$ANNOTATION_STATE" != "missing" ]]; then
    log "FAIL Annotation at $ANNOTATION_DIR is not usable by this build ($ANNOTATION_STATE)."
    log "     Concept labels have changed, so cached labels would give wrong results."
    log "     Delete $ANNOTATION_DIR and re-run to regenerate them."
    exit 1
else
    ANNOTATE_ARGS=(
        --extract-dir       "$EXTRACT_DIR"
        --output-dir        "$ANNOTATION_DIR"
        --ion-types         "$ION_TYPES"
        --fragment-tol      "$FRAGMENT_TOL"
        --fragment-tol-mode "$FRAGMENT_TOL_MODE"
    )
    if [[ -n "$ANNOTATE_WORKERS" ]]; then
        ANNOTATE_ARGS+=(--num-workers "$ANNOTATE_WORKERS")
    fi
    run_step "Annotate spectra (ion_types=$ION_TYPES, tol=${FRAGMENT_TOL} ${FRAGMENT_TOL_MODE})" \
        "$OUTPUT_ROOT/annotate.log" \
        "$PYTHON" "$ANNOTATE_PY" "${ANNOTATE_ARGS[@]}"
    # Internal fragments (ion type 'm') are enabled by default in annotate.py.
    # Add --no-internal above if you want to disable them.
fi

# -----------------------------------------------------------------------------
# Step 3: Train SAE -- one per layer, sequential
# -----------------------------------------------------------------------------
#
# Each run reads only its own layer's activation files (acts_L{N}_*.pt), so
# layers are independent and can be parallelised across GPUs by running this
# script with LAYERS_OVERRIDE="<layer>" and a distinct DEVICE in separate shells.

if [[ "$SKIP_TRAIN" != "1" ]]; then
    choose_ram_cache
fi

for LAYER in "${LAYERS[@]}"; do
    CKPT="$(sae_checkpoint_path "$LAYER")"
    if [[ -f "$CKPT" ]]; then
        log "SAE checkpoint exists for layer $LAYER -- skipping training"
        continue
    fi

    if [[ "$SKIP_TRAIN" == "1" ]]; then
        log "FAIL SAE checkpoint missing for layer $LAYER while SKIP_TRAIN=1 ($CKPT)"
        exit 1
    fi

    TRAIN_ARGS=(
        --extract-dir    "$EXTRACT_DIR"
        --output-dir     "$SAE_ROOT"
        --target-layer   "$LAYER"
        --seed           "$SEED"
        --d-dict         "$D_DICT"
        --k              "$K"
        --k-aux          "$K_AUX"
        --alpha-aux      "$ALPHA_AUX"
        --lr             "$LR"
        --lr-min-ratio   "$LR_MIN_RATIO"
        --warmup-steps   "$WARMUP_STEPS"
        --n-epochs       "$EPOCHS"
        --batch-size     "$SAE_BATCH_SIZE"
        --grad-clip-norm "$GRAD_CLIP"
        --dtype          "$TRAIN_DTYPE"
        --device         "$DEVICE"
    )
    if [[ "$TRAIN_NO_RAM_CACHE" == "1" ]]; then
        TRAIN_ARGS+=(--no-ram-cache)
    fi

    run_step "Train SAE (layer $LAYER, seed $SEED, $EPOCHS epoch(s))" \
        "$SAE_ROOT/train_layer${LAYER}_seed${SEED}.log" \
        "$PYTHON" "$TRAIN_PY" "${TRAIN_ARGS[@]}"
done

# -----------------------------------------------------------------------------
# Step 4: Evaluate SAE -- one per layer
# -----------------------------------------------------------------------------
#
# All eight phases run by default (Phases 7/8 require the InstaNovo model and
# are gated on RUN_PHASE_7 / RUN_PHASE_8). Cross-layer matching runs on the
# deepest requested anchor layer against all other layers -- it needs all SAE
# checkpoints to exist, which they do at this point in the script.
#
# evaluate.py appends layer_{L}/seed_{S}/eval/ to --output-dir automatically,
# so --output-dir must be SAE_ROOT (not a separate eval directory).
#
# Phases 7/8 rebuild their own DataLoader from the extract manifest's
# dataset_path -- the FULL dataset, even when extraction was capped. Processing
# 639k spectra to evaluate 4k costs ~18 minutes of mapping per layer and yields
# nothing, so a capped run gets a pre-sliced source instead. The slice must be
# the FIRST n rows in dataset order, because that is exactly what extract.py
# consumed (shuffle=False), and Phase 7/8 align per-spectrum CE against the
# per-spectrum prevalence taken from those chunks.

EVAL_SPECTRA_PATH=""
if [[ "$RUN_PHASE_7" == "1" || "$RUN_PHASE_8" == "1" ]]; then
    EXTRACTED_SPECTRA="$("$PYTHON" -c "
import json, sys
print(json.load(open(sys.argv[1]))['n_spectra'])
" "$EXTRACT_DIR/manifest.json" 2>/dev/null || echo 0)"
    TOTAL_SPECTRA="$("$PYTHON" -c "
import sys
import pyarrow.parquet as pq
try:
    print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
except Exception:
    print(0)
" "$DATASET_PATH" 2>/dev/null || echo 0)"

    if [[ "$EXTRACTED_SPECTRA" -gt 0 && "$TOTAL_SPECTRA" -gt "$EXTRACTED_SPECTRA" ]]; then
        EVAL_SPECTRA_PATH="$OUTPUT_ROOT/eval_spectra_first${EXTRACTED_SPECTRA}.parquet"
        if [[ -f "$EVAL_SPECTRA_PATH" ]]; then
            log "Capped eval spectra source already exists: $EVAL_SPECTRA_PATH"
        else
            run_step "Cap Phase 7/8 spectra source to the first $EXTRACTED_SPECTRA of $TOTAL_SPECTRA rows" \
                "$OUTPUT_ROOT/cap_eval_spectra.log" \
                "$PYTHON" - "$DATASET_PATH" "$EVAL_SPECTRA_PATH" "$EXTRACTED_SPECTRA" <<'PYEOF'
import sys
import pyarrow as pa
import pyarrow.parquet as pq

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
reader = pq.ParquetFile(src)
batches, taken = [], 0
for batch in reader.iter_batches(batch_size=min(n, 8192)):
    if taken >= n:
        break
    if taken + batch.num_rows > n:
        batch = batch.slice(0, n - taken)
    batches.append(batch)
    taken += batch.num_rows

if taken != n:
    raise SystemExit(f"source has {taken} rows, expected at least {n}")

pq.write_table(pa.Table.from_batches(batches), dst)
print(f"Wrote first {taken} rows of {src} -> {dst}")
PYEOF
        fi
    fi
fi

ANCHOR_LAYER="${LAYERS[0]}"
for LAYER in "${LAYERS[@]}"; do
    if (( LAYER > ANCHOR_LAYER )); then
        ANCHOR_LAYER="$LAYER"
    fi
done

for LAYER in "${LAYERS[@]}"; do
    REPORT="$(eval_report_path "$LAYER")"
    if [[ -f "$REPORT" && "$FORCE_EVAL" != "1" ]]; then
        log "Eval report exists for layer $LAYER -- skipping evaluation"
        continue
    elif [[ -f "$REPORT" && "$FORCE_EVAL" == "1" ]]; then
        log "Eval report exists for layer $LAYER, but FORCE_EVAL=1 -- re-running evaluation"
    fi

    CKPT="$(sae_checkpoint_path "$LAYER")"
    if [[ ! -f "$CKPT" ]]; then
        log "WARNING: SAE checkpoint missing for layer $LAYER ($CKPT) -- skipping eval"
        continue
    fi

    # Build the --skip list. EVAL_SKIP_PHASES is an explicit override for
    # resume/specialty runs, e.g. PHASE8_RESUME=1 uses the Phase 4 cache and
    # skips every other phase. Otherwise, keep the default full-eval policy.
    USING_EXPLICIT_SKIP=0
    SKIP_LIST=()
    if [[ -n "$EVAL_SKIP_PHASES" ]]; then
        USING_EXPLICIT_SKIP=1
        # shellcheck disable=SC2206
        SKIP_LIST=($EVAL_SKIP_PHASES)
    else
        # cross_seed: always skipped (single-seed run; add other seeds via
        #   --other-seed-checkpoint if you re-run with a different SEED).
        SKIP_LIST=(cross_seed)

        # Phases 7 + 8 require InstaNovo forward passes, but are independently gated
        # because Phase 8 is much more expensive than Phase 7.
        if [[ "$RUN_PHASE_7" != "1" ]]; then
            SKIP_LIST+=(7)
        fi
        if [[ "$RUN_PHASE_8" != "1" ]]; then
            SKIP_LIST+=(8)
        fi
    fi

    EVAL_ARGS=(
        --extract-dir             "$EXTRACT_DIR"
        --annotation-dir          "$ANNOTATION_DIR"
        --sae-checkpoint          "$CKPT"
        --output-dir              "$SAE_ROOT"    # evaluate.py appends layer/seed/eval
        --target-layer            "$LAYER"
        --seed                    "$SEED"
        --fdr-q                   "$FDR_Q"
        --batch-size              "$EVAL_BATCH_SIZE"
        --ablation-spectra        "$ABLATION_SPECTRA"
        --ablation-top-n          "$ABLATION_TOP_N"
        --ablation-per-feature-top "$ABLATION_PER_FEATURE_TOP"
        --cross-layer-tokens      "$CROSS_LAYER_TOKENS"
        --device                  "$DEVICE"
    )
    if [[ "$PHASE8_RESUME" == "1" ]]; then
        EVAL_ARGS+=(--phase4-cache-dir "$SAE_ROOT/layer_${LAYER}/seed_${SEED}/eval")
    fi

    # Phases 7/8: pass the model path so the evaluator can load InstaNovo.
    # The spectra source and n_peaks both default to the extract manifest, so
    # --spectra-path is only needed if the dataset file has moved.
    if [[ "$RUN_PHASE_7" == "1" || "$RUN_PHASE_8" == "1" ]]; then
        EVAL_ARGS+=(--instanovo-path "$MODEL_PATH")
        if [[ -n "$EVAL_SPECTRA_PATH" ]]; then
            EVAL_ARGS+=(--spectra-path "$EVAL_SPECTRA_PATH")
        fi
    fi

    # Cross-layer matching: only the anchor layer runs it, pointing at the other
    # layers' checkpoints. Non-anchor layers skip it. An explicit skip list is
    # left exactly as given.
    if [[ "$USING_EXPLICIT_SKIP" == "0" ]]; then
        if [[ "$LAYER" == "$ANCHOR_LAYER" && ${#LAYERS[@]} -ge 2 ]]; then
            for OL in "${LAYERS[@]}"; do
                [[ "$OL" == "$ANCHOR_LAYER" ]] && continue
                OTHER_CKPT="$(sae_checkpoint_path "$OL")"
                if [[ -f "$OTHER_CKPT" ]]; then
                    EVAL_ARGS+=(--other-layer-checkpoint "${OL}=${OTHER_CKPT}")
                else
                    log "WARNING: checkpoint for layer $OL not found; skipping it in cross-layer matching"
                fi
            done
        else
            SKIP_LIST+=(cross_layer)
        fi
    fi

    EVAL_ARGS+=(--skip "${SKIP_LIST[@]}")

    run_step "Evaluate SAE (layer $LAYER, seed $SEED)" \
        "$SAE_ROOT/eval_layer${LAYER}_seed${SEED}.log" \
        "$PYTHON" "$EVALUATE_PY" "${EVAL_ARGS[@]}"
done

# -----------------------------------------------------------------------------
# Step 5: Chunk management
# -----------------------------------------------------------------------------

if [[ "$KEEP_CHUNKS" == "1" ]]; then
    log "KEEP_CHUNKS=1 -- extract chunks retained for future experiments"
    log "  Location : $EXTRACT_DIR/chunks/"
    log "  Disk used: $(dir_size "$EXTRACT_DIR/chunks")"
    log "  To free disk later: rm -rf $EXTRACT_DIR/chunks"
    log "  (The manifest.json is kept for provenance even if chunks are deleted.)"
else
    if [[ -d "$EXTRACT_DIR/chunks" ]]; then
        # The manifest can cover layers beyond this invocation's $LAYERS (e.g.
        # a sibling invocation is concurrently training/evaluating another
        # layer against the same OUTPUT_ROOT, per the documented per-layer
        # concurrent-run pattern). Per-layer activation files
        # (acts_L{layer}_*.pt) are safe to remove selectively, but meta_*.pt
        # is layer-independent and still needed by every other layer's
        # evaluate.py run, so it must only be removed when nothing else in the
        # manifest still needs this chunk directory.
        other_layers="$("$PYTHON" - "$EXTRACT_DIR/manifest.json" "${LAYERS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
requested = {str(x) for x in sys.argv[2:]}
target = {str(x) for x in manifest.get("target_layers", [])}
print(" ".join(sorted(target - requested)))
PYEOF
)"
        # Chunk indices this invocation keeps as the cross-layer token sample.
        # Deleting a layer's activations entirely forecloses cross-layer matching
        # later, which a layer-at-a-time run defers by construction.
        keep_globs=()
        if [[ "$KEEP_CHUNK_SAMPLE" -gt 0 ]]; then
            for ((i = 0; i < KEEP_CHUNK_SAMPLE; i++)); do
                keep_globs+=("$(printf '%05d' "$i")")
            done
            # The sample has to cover CROSS_LAYER_TOKENS, which is read from the
            # first chunks in manifest order. Too few and the deferred cross-layer
            # run dies on a missing file rather than a clear error, since the
            # manifest still lists the chunks that were deleted.
            needed="$("$PYTHON" - "$EXTRACT_DIR/manifest.json" "$CROSS_LAYER_TOKENS" <<'PYEOF'
import json, math, sys
from pathlib import Path
try:
    m = json.loads(Path(sys.argv[1]).read_text())
    per_chunk = m["n_tokens"] / max(len(m["chunks"]), 1)
    print(max(1, math.ceil(int(sys.argv[2]) / per_chunk)))
except Exception:
    print(0)      # unknown; the caller then skips the check
PYEOF
)"
            if [[ "$needed" -gt 0 && "$KEEP_CHUNK_SAMPLE" -lt "$needed" ]]; then
                log "WARNING: KEEP_CHUNK_SAMPLE=$KEEP_CHUNK_SAMPLE is below the $needed chunk(s)"
                log "  CROSS_LAYER_TOKENS=$CROSS_LAYER_TOKENS needs; keeping $needed instead."
                keep_globs=()
                for ((i = 0; i < needed; i++)); do
                    keep_globs+=("$(printf '%05d' "$i")")
                done
            fi
        fi

        delete_layer_acts() {   # $1 = layer
            local L="$1" f base idx
            for f in "$EXTRACT_DIR"/chunks/acts_L${L}_*.pt; do
                [[ -e "$f" ]] || continue
                base="$(basename "$f")"
                idx="${base##*_}"; idx="${idx%.pt}"
                local keep=0
                for k in "${keep_globs[@]}"; do
                    [[ "$idx" == "$k" ]] && keep=1 && break
                done
                [[ "$keep" == "1" ]] || rm -f "$f"
            done
        }

        if [[ -n "$other_layers" ]]; then
            log "KEEP_CHUNKS=0, but the manifest also covers layer(s) $other_layers not"
            log "  requested by this invocation (LAYERS=${LAYERS[*]}) -- only deleting this"
            log "  invocation's own activation files; metadata and other layers' activations"
            log "  are kept for the invocation(s) still using them."
            for L in "${LAYERS[@]}"; do
                delete_layer_acts "$L"
            done
        elif [[ "$KEEP_CHUNK_SAMPLE" -gt 0 ]]; then
            log "KEEP_CHUNKS=0 -- deleting extract chunks, keeping the first"
            log "  $KEEP_CHUNK_SAMPLE chunk(s) per layer as a cross-layer token sample"
            for L in "${LAYERS[@]}"; do
                delete_layer_acts "$L"
            done
        else
            log "KEEP_CHUNKS=0 -- deleting extract chunks (freeing ~$(dir_size "$EXTRACT_DIR/chunks"))"
            rm -rf "$EXTRACT_DIR/chunks"
            log "  manifest.json kept for provenance"
        fi
        if [[ "$KEEP_CHUNK_SAMPLE" -gt 0 && -d "$EXTRACT_DIR/chunks" ]]; then
            log "  Retained for cross-layer: $(dir_size "$EXTRACT_DIR/chunks")"
        fi
    fi
fi

# Annotation labels: always kept (they are small and layer-independent).
if [[ -d "$ANNOTATION_DIR" ]]; then
    log "Annotation labels retained: $ANNOTATION_DIR  ($(dir_size "$ANNOTATION_DIR"))"
fi

# -----------------------------------------------------------------------------
# Cross-layer summary table
# -----------------------------------------------------------------------------

log ""
log "================================================================"
log "Pipeline complete -- cross-layer summary"
log "================================================================"

"$PYTHON" - "$SAE_ROOT" "$SEED" "${LAYERS[@]}" <<'PYEOF' 2>&1 | tee -a "$PIPELINE_LOG"
import json, sys
from pathlib import Path

sae_root = Path(sys.argv[1])
seed     = int(sys.argv[2])
layers   = [int(x) for x in sys.argv[3:]]

def fmt(d, key, spec, default="-"):
    if d is None:
        return default
    v = d.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return default
    try:
        return spec.format(v)
    except Exception:
        return str(v)

# Phase 8 per-concept causal metrics: mean selectivity_z across all non-diagnostic concepts.
def mean_selectivity_z(p8):
    if not p8:
        return None
    per_concept = p8.get("per_concept", {})
    vals = []
    for info in per_concept.values():
        if info.get("diagnostic"):
            continue
        causal = info.get("causal")
        if causal and causal.get("selectivity_z") == causal.get("selectivity_z"):  # not NaN
            vals.append(causal["selectivity_z"])
    return sum(vals) / len(vals) if vals else None

print()
hdr = (
    f'{"layer":<7} {"FVE":<8} {"L0":<6} {"dead%":<7} '
    f'{"Gini":<6} {"n_sig_pairs":<13} {"feat_w_conc":<13} '
    f'{"loss_rec":<10} {"top1_drop%":<11} {"mean_sel_z":<11}'
)
print(hdr)
print("-" * len(hdr))

for L in layers:
    rpt_path = sae_root / f"layer_{L}" / f"seed_{seed}" / "eval" / "report.json"
    if not rpt_path.exists():
        print(f"{L:<7} INCOMPLETE -- no report.json at {rpt_path}")
        continue
    try:
        r = json.loads(rpt_path.read_text())
    except Exception as e:
        print(f"{L:<7} ERROR reading report: {e}")
        continue

    p12 = r.get("phase_1_2", {})
    p4  = r.get("phase_4",   {})
    p7  = r.get("phase_7",   {})
    p8  = r.get("phase_8",   {})

    sel_z = mean_selectivity_z(p8)
    sel_z_str = f"{sel_z:.3f}" if sel_z is not None else "-"

    top1_drop = p7.get("top1_drop_pp") if p7 else None
    top1_str  = f"{top1_drop:.2f}" if top1_drop is not None else "-"

    print(
        f"{L:<7} "
        f"{fmt(p12, 'fve_overall',           '{:.4f}'):<8} "
        f"{fmt(p12, 'l0_mean',               '{:.1f}'):<6} "
        f"{fmt(p12, 'strict_dead_pct',       '{:.1f}'):<7} "
        f"{fmt(p12, 'firing_rate_gini',      '{:.3f}'):<6} "
        f"{fmt(p4,  'n_significant_pairs',   '{:,d}'):<13} "
        f"{fmt(p4,  'n_features_with_concept','{:,d}'):<13} "
        f"{fmt(p7,  'loss_recovered_vs_zero','{:.4f}'):<10} "
        f"{top1_str:<11} "
        f"{sel_z_str:<11}"
    )

print()
print("Column guide:")
print("  FVE          centred fraction of variance explained (R^2-like, higher = better)")
print("  L0           mean active features per encoder token (target ~ k)")
print("  dead%        features that never fired (lower = better)")
print("  Gini         firing-rate inequality (0=uniform, 1=one feature dominates)")
print("  n_sig_pairs  BH-significant (feature, concept) pairs at FDR q")
print("  feat_w_conc  features with >=1 significant concept")
print("  loss_rec     (CE_zero - CE_sae) / (CE_zero - CE_clean), Phase 7")
print("  top1_drop%   percentage-point drop in top-1 accuracy under SAE substitution")
print("  mean_sel_z   mean concept-selectivity z-score across non-diagnostic concepts, Phase 8")

print()
print("Output locations:")
for L in layers:
    base = sae_root / f"layer_{L}" / f"seed_{seed}" / "eval"
    if base.exists():
        n = sum(1 for f in base.rglob("*") if f.is_file())
        print(f"  Layer {L}: {base}/  ({n} files)")
    ckpt = sae_root / f"layer_{L}" / f"seed_{seed}" / "checkpoint.pt"
    if ckpt.exists():
        size_mb = ckpt.stat().st_size / 1e6
        print(f"          checkpoint: {ckpt}  ({size_mb:.0f} MB)")
PYEOF

# -----------------------------------------------------------------------------
# Smoke-test verification
# -----------------------------------------------------------------------------
#
# Exiting 0 only means no command crashed. These checks assert the artefacts are
# actually well-formed and mutually consistent, which is what makes the smoke
# test worth running before committing a GPU cluster to the full dataset.

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    log ""
    log "================================================================"
    log "Smoke-test verification"
    log "================================================================"
    # pipefail (set at the top) makes the pipeline report the Python exit status
    # rather than tee's, and `|| smoke_status=1` keeps `set -e` from exiting here
    # so the failure can be logged before the explicit exit below.
    smoke_status=0
    "$PYTHON" - "$OUTPUT_ROOT" "$SAE_ROOT" "$SEED" "$SCRIPT_DIR" "${LAYERS[@]}" <<'PYEOF' 2>&1 | tee -a "$PIPELINE_LOG" || smoke_status=1
import json, sys
from pathlib import Path

root, sae_root, seed, script_dir = (
    Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
)
layers = [int(x) for x in sys.argv[5:]]
sys.path.insert(0, script_dir)
from schema import ANNOTATION_SCHEMA_VERSION, EXTRACT_SCHEMA_VERSION, SAE_SCHEMA_VERSION

import torch

failures, checks = [], 0

def check(label, ok, detail=""):
    global checks
    checks += 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)

def finite(x):
    return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")

def num(x, spec):
    """Format x, or "n/a" if it is missing/None/non-numeric.

    check()'s detail argument is evaluated eagerly, before check() can decide
    the result -- so a bare f"{value:.1f}" on a missing metric raises and kills
    the whole verification pass in exactly the case the check exists to report.
    """
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "n/a"

# --- extraction ---
em_path = root / "extract" / "manifest.json"
check("extract manifest exists", em_path.exists(), str(em_path))
if em_path.exists():
    em = json.loads(em_path.read_text())
    check("extract schema current", em["schema_version"] == EXTRACT_SCHEMA_VERSION,
          f"found {em['schema_version']}")
    check("extract produced >= 2 chunks", em["n_chunks"] >= 2,
          f"{em['n_chunks']} chunks (train.py needs >=2 to hold out a val split)")
    check("every requested layer extracted",
          all(str(L) in em["chunks"][0]["activations"] for L in layers),
          f"layers {layers}")
    check("resume fingerprint written", (root / "extract" / "extract_config.json").exists())
    n_extract_tokens = em["n_tokens"]
    check("extract token count positive", n_extract_tokens > 0, f"{n_extract_tokens:,} tokens")
else:
    n_extract_tokens = None

# --- annotation ---
am_path = root / "annotation" / "annotation_manifest.json"
check("annotation manifest exists", am_path.exists())
if am_path.exists():
    am = json.loads(am_path.read_text())
    check("annotation schema current", am["schema_version"] == ANNOTATION_SCHEMA_VERSION,
          f"found {am['schema_version']}")
    if n_extract_tokens is not None:
        check("annotation token count matches extraction",
              am["n_tokens"] == n_extract_tokens,
              f"{am['n_tokens']:,} vs {n_extract_tokens:,}")
    check("concept registry populated", len(am["registry"]["names"]) > 0,
          f"{len(am['registry']['names'])} concepts")
    rates = am["base_rates"]
    check("base rates are probabilities",
          all(0.0 <= v <= 1.0 for v in rates.values()))
    # The fix that motivated schema v2: unmatched peaks carry no fragment charge.
    check("fragment-charge concepts are not noise-dominated",
          rates.get("is_fragment_charge_1", 0.0) < 0.90,
          f"is_fragment_charge_1 base rate {rates.get('is_fragment_charge_1', 0.0):.3f}")

# --- per layer: checkpoint + eval ---
for L in layers:
    base = sae_root / f"layer_{L}" / f"seed_{seed}"
    ckpt_path = base / "checkpoint.pt"
    check(f"layer {L}: checkpoint exists", ckpt_path.exists())
    if ckpt_path.exists():
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            check(f"layer {L}: SAE schema current",
                  ck.get("schema_version") == SAE_SCHEMA_VERSION,
                  f"found {ck.get('schema_version')}")
            dmin, dmax = ck.get("decoder_norm_min"), ck.get("decoder_norm_max")
            check(f"layer {L}: decoder rows unit-norm",
                  finite(dmin) and finite(dmax)
                  and abs(dmin - 1.0) < 1e-3 and abs(dmax - 1.0) < 1e-3,
                  f"[{num(dmin, '.6f')}, {num(dmax, '.6f')}]")
            fm = ck.get("final_metrics") or {}
            check(f"layer {L}: final FVE finite", finite(fm.get("fve_uncentered")),
                  f"{fm.get('fve_uncentered')}")
            check(f"layer {L}: inference L0 > 0", (fm.get("l0_mean") or 0) > 0,
                  f"L0={num(fm.get('l0_mean'), '.1f')} (k={ck.get('k')})")
            d_dict = ck.get("d_dict")
            check(f"layer {L}: not every feature dead",
                  finite(d_dict) and fm.get("dead_features", d_dict) < d_dict,
                  f"{fm.get('dead_features')}/{d_dict} dead")
            # Stored as a 0-dim torch tensor, so coerce rather than isinstance-test.
            try:
                thr = float(ck.get("jumprelu_threshold"))
            except (TypeError, ValueError):
                thr = None
            check(f"layer {L}: JumpReLU threshold calibrated",
                  thr is not None and thr > 0,
                  f"thr={num(thr, '.4f')}")
        except Exception as exc:
            # A malformed checkpoint is exactly what this block exists to catch,
            # so report it as a failed check rather than dying with a traceback
            # and skipping every remaining layer's checks.
            check(f"layer {L}: checkpoint readable", False, f"{type(exc).__name__}: {exc}")

    rpt_path = base / "eval" / "report.json"
    check(f"layer {L}: eval report exists", rpt_path.exists())
    if rpt_path.exists():
        r = json.loads(rpt_path.read_text())
        p12 = r.get("phase_1_2", {})
        check(f"layer {L}: phase 1+2 FVE finite", finite(p12.get("fve_overall")),
              f"{p12.get('fve_overall')}")
        check(f"layer {L}: phase 1+2 L0 > 0", (p12.get("l0_mean") or 0) > 0,
              f"L0={p12.get('l0_mean')}")
        if "phase_7" in r and r["phase_7"]:
            lr_ = r["phase_7"].get("loss_recovered_vs_zero")
            check(f"layer {L}: phase 7 loss_recovered finite", finite(lr_), f"{lr_}")
        csv_path = base / "eval" / "per_feature_stats.csv"
        check(f"layer {L}: per_feature_stats.csv exists", csv_path.exists())
        if csv_path.exists():
            header = csv_path.read_text().splitlines()[0].split(",")
            check(f"layer {L}: discovery columns present",
                  "unexplained_mass_fraction" in header and "unexplained_enrichment" in header,
                  ",".join(header))

print()
print(f"{checks - len(failures)}/{checks} checks passed")
if failures:
    print("FAILED: " + "; ".join(failures))
raise SystemExit(1 if failures else 0)
PYEOF

    if [[ "$smoke_status" == "0" ]]; then
        log "Smoke test PASSED"
    else
        log "Smoke test FAILED -- see the checks above"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Step 6: Interpret features -- one pass per layer, after evaluation
# -----------------------------------------------------------------------------
#
# Reads the eval directory Step 4 wrote and the extract/annotation chunks, and
# asks an LLM to describe a sample of features. Off unless RUN_INTERPRET=1.
#
# Every earlier step is idempotent, so an interpretation-only run is just this
# script over an output root that already has its artefacts:
#
#   RUN_INTERPRET=1 SKIP_TRAIN=1 KEEP_CHUNKS=1 \
#   INTERPRET_ENV_FILE=/app/fm_mount/secrets/openai.env \
#   OUTPUT_ROOT=... bash ./run_pipeline.sh
#
# Steps 0-5 find their outputs present and skip, leaving this and the publish.
# SKIP_TRAIN=1 matters: without it a missing checkpoint would silently start a
# multi-hour training run instead of failing.
#
# Failure here is deliberately not fatal. The descriptions are an addition to a
# finished evaluation, and the publish step below is what makes everything else
# retrievable -- losing a run's artefacts because an API key expired would be a
# far worse outcome than losing the descriptions.
# -----------------------------------------------------------------------------

if [[ "$RUN_INTERPRET" == "1" ]]; then
    if [[ ! -f "$INTERPRET_PY" ]]; then
        log "FAIL RUN_INTERPRET=1 but $INTERPRET_PY is missing"
        exit 1
    fi
    # A dry run builds prompts and calls nothing, so it needs no key. Checking
    # anyway would make the one submission that can validate the plumbing before
    # the key exists impossible to run.
    if [[ -n "$INTERPRET_ENV_FILE" && ! -f "$INTERPRET_ENV_FILE" ]]; then
        if [[ "$INTERPRET_DRY_RUN" == "1" ]]; then
            log "INTERPRET_ENV_FILE=$INTERPRET_ENV_FILE is absent, but INTERPRET_DRY_RUN=1 -- continuing"
            INTERPRET_ENV_FILE=""
        else
            log "FAIL INTERPRET_ENV_FILE=$INTERPRET_ENV_FILE does not exist"
            log "     Create it on the PVC with a single line: OPENAI_API_KEY=sk-..."
            exit 1
        fi
    fi

    for LAYER in "${LAYERS[@]}"; do
        INTERPRET_EVAL_DIR="$SAE_ROOT/layer_$LAYER/seed_${SEED}/eval"
        INTERPRET_CKPT="$(sae_checkpoint_path "$LAYER")"
        INTERPRET_OUT="$OUTPUT_ROOT/interpret/layer_$LAYER"

        if [[ ! -f "$INTERPRET_EVAL_DIR/per_feature_stats.csv" ]]; then
            log "WARNING: no per_feature_stats.csv for layer $LAYER -- skipping interpretation"
            continue
        fi
        if [[ ! -f "$INTERPRET_CKPT" ]]; then
            log "WARNING: no SAE checkpoint for layer $LAYER -- skipping interpretation"
            continue
        fi

        INTERPRET_ARGS=(
            --eval-dir        "$INTERPRET_EVAL_DIR"
            --extract-dir     "$EXTRACT_DIR"
            --annotation-dir  "$ANNOTATION_DIR"
            --sae-checkpoint  "$INTERPRET_CKPT"
            --output-dir      "$INTERPRET_OUT"
            --target-layer    "$LAYER"
            --seed            "$SEED"
            --device          "$INTERPRET_DEVICE"
            --n-per-stratum   "$INTERPRET_N_PER_STRATUM"
            --n-sample-chunks "$INTERPRET_SAMPLE_CHUNKS"
        )
        [[ -n "$INTERPRET_ENV_FILE" ]] && INTERPRET_ARGS+=(--env-file "$INTERPRET_ENV_FILE")
        [[ -n "$INTERPRET_MODEL" ]] && INTERPRET_ARGS+=(--model "$INTERPRET_MODEL")
        [[ -n "$INTERPRET_MAX_TOKENS" ]] && INTERPRET_ARGS+=(--max-tokens "$INTERPRET_MAX_TOKENS")
        [[ -n "$INTERPRET_MIN_FIRING_RATE" ]] && \
            INTERPRET_ARGS+=(--min-firing-rate "$INTERPRET_MIN_FIRING_RATE")
        # Unquoted on purpose: INTERPRET_STRATA is a space-separated list for nargs="+".
        [[ -n "$INTERPRET_STRATA" ]] && INTERPRET_ARGS+=(--strata $INTERPRET_STRATA)
        [[ "$INTERPRET_DRY_RUN" == "1" ]] && INTERPRET_ARGS+=(--dry-run)

        if run_step_soft "Interpret features (layer $LAYER, seed $SEED)" \
            "$OUTPUT_ROOT/interpret_layer${LAYER}.log" \
            "$PYTHON" "$INTERPRET_PY" "${INTERPRET_ARGS[@]}"; then
            log "  Descriptions: $INTERPRET_OUT/feature_descriptions.csv"
        else
            log "WARNING: interpretation failed for layer $LAYER -- continuing to publish"
        fi
    done
fi

# -----------------------------------------------------------------------------
# Publish durable artefacts to the AIchor output bucket
#
# AIchor does not mount its buckets. AICHOR_OUTPUT_PATH is an S3 URI reached
# through an S3 client using AWS_ENDPOINT_URL, while an attached PVC is the only
# real filesystem -- and AIchor's own docs call a PVC "a fast working cache, not
# a system of record". So OUTPUT_ROOT stays on the PVC, where the ~415 GB of
# activations can be written and re-read, and the small durable artefacts are
# copied to the bucket here. AIchor scopes AICHOR_OUTPUT_PATH to the experiment
# id, so each run's results land in their own subfolder.
#
# Chunks, label chunks and the merged parquet are deliberately NOT published:
# they are regenerable intermediates, and together they dwarf everything else.
#
# The whole step is skipped when AICHOR_OUTPUT_PATH is unset, so local and smoke
# runs are unaffected.
# -----------------------------------------------------------------------------

if [[ -n "${AICHOR_OUTPUT_PATH:-}" ]]; then
    run_step "Publish artefacts to AIchor output bucket" \
        "$OUTPUT_ROOT/publish.log" \
        "$PYTHON" - "$OUTPUT_ROOT" <<'PYEOF'
import fnmatch
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

root = Path(sys.argv[1])

# Small, durable, and expensive to reproduce. Anything not matched here stays on
# the PVC. Patterns are matched against POSIX paths relative to OUTPUT_ROOT.
INCLUDE = [
    "*.log",
    "extract/manifest.json",
    "extract/extract_config.json",
    "annotation/annotation_manifest.json",
    "annotation/concept_phi.pt",
    "sae/*.log",
    "sae/layer_*/seed_*/checkpoint.pt",
    "sae/layer_*/seed_*/config.json",
    "sae/layer_*/seed_*/training_log.jsonl",
    "sae/layer_*/seed_*/eval/*",
    "interpret/layer_*/*",
]

# PUBLISH_CHUNKS=1 additionally ships the activation chunks still on disk, so an
# analysis needing raw activations can run somewhere other than this cluster.
# Off by default because a full run holds ~104 GB per layer. It is practical only
# after KEEP_CHUNKS=0 has pruned a run to its cross-layer sample, which is why
# PUBLISH_MAX_GB refuses the job rather than quietly starting a 100 GB upload.
#
# Restricted to chunks whose activations survived. Metadata and label files are
# written for every chunk and are never pruned -- on the nine-species run that is
# ~2 GB of meta and ~4 GB of labels, none of it usable without the matching
# activations, and enough on its own to trip the size guard.
if os.environ.get("PUBLISH_CHUNKS") == "1":
    import re
    live = sorted({
        m.group(1)
        for p in (root / "extract" / "chunks").glob("acts_L*_*.pt")
        if (m := re.search(r"_(\d+)\.pt$", p.name))
    })
    print(f"PUBLISH_CHUNKS=1: {len(live)} chunk(s) still hold activations: "
          f"{', '.join(live) if len(live) <= 12 else ', '.join(live[:12]) + ', ...'}")
    for i in live:
        INCLUDE += [
            f"extract/chunks/acts_L*_{i}.pt",
            f"extract/chunks/meta_{i}.pt",
            f"annotation/labels/chunk_{i}.pt",
        ]

dest = os.environ["AICHOR_OUTPUT_PATH"]
if "s3://" not in dest:                      # AIchor may hand it over bare
    dest = f"s3://{dest}"
dest = dest.rstrip("/")

if "AWS_ENDPOINT_URL" not in os.environ:
    print("ERROR: AICHOR_OUTPUT_PATH is set but AWS_ENDPOINT_URL is not; "
          "cannot reach the bucket.", file=sys.stderr)
    raise SystemExit(1)

import s3fs

fs = s3fs.core.S3FileSystem(
    key=os.environ.get("AWS_ACCESS_KEY_ID"),
    secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    client_kwargs={"endpoint_url": os.environ["AWS_ENDPOINT_URL"]},
)

selected = []
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(root).as_posix()
    if any(fnmatch.fnmatch(rel, pat) for pat in INCLUDE):
        selected.append((path, rel))

if not selected:
    print(f"ERROR: nothing matched the publish patterns under {root}.", file=sys.stderr)
    raise SystemExit(1)

total = sum(p.stat().st_size for p, _ in selected)

# A wrong pattern here uploads for hours and bills for the transfer, so the size
# is checked before the first byte moves rather than discovered from the log.
max_gb = float(os.environ.get("PUBLISH_MAX_GB", "5"))
if total / 1e9 > max_gb:
    print(f"ERROR: the selection is {total / 1e9:.1f} GB, over PUBLISH_MAX_GB={max_gb} GB. "
          f"Nothing was uploaded. Raise PUBLISH_MAX_GB if this is intended, or narrow "
          f"what PUBLISH_CHUNKS matches.", file=sys.stderr)
    raise SystemExit(1)

print(f"Publishing {len(selected)} files ({total / 1e6:.1f} MB) to {dest}")

failed = []
for path, rel in selected:
    target = f"{dest}/{rel}"
    parsed = urlparse(target)
    key = f"{parsed.netloc}{parsed.path}"
    try:
        fs.put(str(path), key)
        print(f"  ok    {rel}")
    except Exception as exc:                 # noqa: BLE001 -- report every failure
        print(f"  FAIL  {rel}: {exc}")
        failed.append(rel)

if failed:
    print(f"\n{len(failed)}/{len(selected)} uploads failed. The artefacts are "
          f"still on the PVC under {root}; re-run this step to retry.",
          file=sys.stderr)
    raise SystemExit(1)

print(f"\nPublished {len(selected)} files to {dest}")
PYEOF
else
    log "AICHOR_OUTPUT_PATH not set -- skipping bucket publish (artefacts stay in $OUTPUT_ROOT)"
fi

log ""
log "All done. Full pipeline log: $PIPELINE_LOG"
