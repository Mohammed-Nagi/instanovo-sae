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

OUTPUT_ROOT="${OUTPUT_ROOT:-./sae_pipeline_outputs}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"

# -----------------------------------------------------------------------------
# Dataset
# Full nine-species benchmark (all splits concatenated). If DATASET_PATH is
# already set (pointing at a pre-merged parquet), the merge step is skipped.
DATASET_HF_ID="${DATASET_HF_ID:-InstaDeepAI/ms_ninespecies_benchmark}"
COMBINED_DATASET="${COMBINED_DATASET:-$OUTPUT_ROOT/combined_ninespecies.parquet}"
DATASET_PATH="${DATASET_PATH:-}"   # set externally to skip the merge step

# -----------------------------------------------------------------------------
# Model
MODEL_PATH="${MODEL_PATH:-./instanovo_v1.1.0.ckpt}"

# -----------------------------------------------------------------------------
# Extraction
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
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
RUN_PHASE_7="${RUN_PHASE_7:-$RUN_CAUSAL}"
RUN_PHASE_8="${RUN_PHASE_8:-0}"
PHASE8_RESUME="${PHASE8_RESUME:-0}"          # 1 = force eval and skip all phases except 8
FORCE_EVAL="${FORCE_EVAL:-$PHASE8_RESUME}"   # 1 = ignore existing report.json sentinels
EVAL_SKIP_PHASES="${EVAL_SKIP_PHASES:-}"     # optional explicit evaluate.py --skip list
ABLATION_SPECTRA="${ABLATION_SPECTRA:-5000}"
ABLATION_TOP_N="${ABLATION_TOP_N:-10}"
# Single-feature ablations per concept. This is the dominant Phase 8 cost: each
# one is a full model pass over ABLATION_SPECTRA spectra. The evaluate.py default
# is 100; lower it to trade per-feature resolution for wall-clock.
ABLATION_PER_FEATURE_TOP="${ABLATION_PER_FEATURE_TOP:-100}"
CROSS_LAYER_TOKENS="${CROSS_LAYER_TOKENS:-100000}"

# -----------------------------------------------------------------------------
# Disk management
# KEEP_CHUNKS=1 (default): extraction chunks are expensive to produce and are
# reusable across all future SAE experiments on this dataset. Only set to 0 if
# you are genuinely disk-constrained and do not plan to re-train.
KEEP_CHUNKS="${KEEP_CHUNKS:-1}"
# Annotation labels are always kept -- they are small and layer-independent.
KEEP_ANNOTATION="${KEEP_ANNOTATION:-1}"
# `du -sh` over a multi-terabyte chunk tree can take minutes; set to 0 to skip.
REPORT_DISK_USAGE="${REPORT_DISK_USAGE:-1}"

# -----------------------------------------------------------------------------
# Smoke-test override
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    MAX_SPECTRA="${MAX_SPECTRA:-4096}"   # 0 = no cap; nonzero must be batch-aligned
    EPOCHS=1
    SAE_BATCH_SIZE=1024
    EVAL_BATCH_SIZE=1024
    TRAIN_NO_RAM_CACHE="${TRAIN_NO_RAM_CACHE:-0}"
    RUN_CAUSAL=0                         # skip model-dependent phases in smoke test
    RUN_PHASE_7=0
    RUN_PHASE_8=0
    ABLATION_SPECTRA=256
    ABLATION_PER_FEATURE_TOP=5
    EXTRACT_DTYPE="float32"              # avoid bfloat16 issues on CPU-only smoke runs
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

mkdir -p "$OUTPUT_ROOT"
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

    local decision
    decision="$("$PYTHON" - "$EXTRACT_DIR/manifest.json" "$D_MODEL" "$RAM_CACHE_SAFETY" <<'PYEOF'
import json, sys
from pathlib import Path

manifest_path, d_model, safety = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}

try:
    manifest = json.loads(manifest_path.read_text())
    n_tokens = int(manifest["n_tokens"])
    dtype = str(manifest.get("config", {}).get("dtype", "float32"))
    needed = n_tokens * d_model * DTYPE_BYTES.get(dtype, 4)
except Exception as exc:                       # manifest unreadable -> stay safe
    print(f"1 unknown could-not-size-layer:{exc}")
    raise SystemExit(0)

# MemAvailable is the kernel's own estimate of what can be allocated without
# swapping, which is the right number here (MemFree ignores reclaimable cache).
available = 0
try:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
except Exception:
    pass

if available <= 0:
    print(f"1 {needed/1e9:.0f}GB free-ram-unknown")
elif needed <= available * safety:
    print(f"0 {needed/1e9:.0f}GB fits-in-{available/1e9:.0f}GB-available")
else:
    print(f"1 {needed/1e9:.0f}GB exceeds-{safety:.0%}-of-{available/1e9:.0f}GB-available")
PYEOF
)"
    TRAIN_NO_RAM_CACHE="${decision%% *}"
    log "  RAM cache      : auto -> no_ram_cache=$TRAIN_NO_RAM_CACHE  (${decision#* })"
}

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
log "================================================================"

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

if [[ -f "$ANNOTATION_DIR/annotation_manifest.json" ]]; then
    log "Annotation manifest exists -- skipping annotation"
else
    run_step "Annotate spectra (ion_types=$ION_TYPES, tol=${FRAGMENT_TOL} ${FRAGMENT_TOL_MODE})" \
        "$OUTPUT_ROOT/annotate.log" \
        "$PYTHON" "$ANNOTATE_PY" \
            --extract-dir          "$EXTRACT_DIR" \
            --output-dir           "$ANNOTATION_DIR" \
            --ion-types            "$ION_TYPES" \
            --fragment-tol         "$FRAGMENT_TOL" \
            --fragment-tol-mode    "$FRAGMENT_TOL_MODE"
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
        log "KEEP_CHUNKS=0 -- deleting extract chunks (freeing ~$(dir_size "$EXTRACT_DIR/chunks"))"
        rm -rf "$EXTRACT_DIR/chunks"
        log "  manifest.json kept for provenance"
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

log ""
log "All done. Full pipeline log: $PIPELINE_LOG"
