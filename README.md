# Sparse Autoencoder Interpretability Pipeline for InstaNovo

Sparse autoencoders are trained on the residual-stream activations of [InstaNovo](https://github.com/instadeepai/InstaNovo)'s encoder at four depths. Every spectral peak is labelled with chemical concepts derived from first-principles fragment-ion theory, allowing us to ask which learned features correspond to which chemical events — both correlationally and causally.

## Contents

| File | Role |
|---|---|
| `run_pipeline.sh` | Orchestration with resume and artefact reuse |
| `instanovo_io.py` | The single boundary against the InstaNovo API |
| `extract.py` | Multi-layer activation extraction (layers 2, 4, 6, 8) |
| `train.py` | Sparse autoencoder: BatchTopK training, AuxK recovery, JumpReLU inference |
| `annotate.py` | Per-peak fragment-ion annotation: 50 concepts, 14 families |
| `evaluate.py` | Eight-phase evaluation suite |
| `interpret.py` | LLM-assisted feature description, validated by held-out activation prediction |
| `schema.py` | On-disk schema versions shared across the pipeline |

The pipeline runs in four stages — extract, annotate, train, evaluate — orchestrated by `run_pipeline.sh`. Extraction and annotation are the expensive one-off stages and are reused across every layer, seed, and SAE width. Every step is idempotent and skips if its output already exists, so the pipeline is safe to interrupt and resume. `interpret.py` is a separate step run after evaluation.

InstaNovo itself is installed from PyPI and is not vendored here. Only four upstream symbols are used — `InstaNovo`, `TransformerDataProcessor`, `SpectrumDataFrame`, and `LEGACY_PTM_TO_UNIMOD` — all reached through `instanovo_io.py`.

Each stage records a schema version in the artefacts it writes, and each consumer rejects an artefact whose version it does not expect. Because the pipeline caches expensive intermediates and reuses them across runs, this turns a stale-artefact mistake into an error rather than silently wrong numbers.

## Setup

Requires Python 3.10–3.13. The upper bound matters: `instanovo` declares `<3.14`.

```bash
git clone <repository-url>
cd instanovo-sae

uv python install 3.13
uv sync --extra cu126     # GPU;  use --extra cpu for CPU-only
```

Verify the install:

```bash
uv run python -c "import instanovo_io, extract, annotate, train, evaluate; print('imports OK')"
```

`MODEL_PATH` accepts either a local `.ckpt` path or a pretrained model id, which InstaNovo resolves and caches on first use:

```bash
MODEL_PATH=instanovo-v1.1.0        # downloaded automatically
MODEL_PATH=/path/to/model.ckpt     # local checkpoint
```

Ids come from InstaNovo's `models.json` — `instanovo-v1.1.0` (used in our experiments), `instanovo-v1.2.0`, and `instanovo-phospho-v1.0.0`. Anything ending in `.ckpt` or containing a path separator is treated as a file; everything else as an id.

Copy `.env.example` to `.env` for the optional settings and, if you plan to run `interpret.py`, the API key. `.env` is gitignored and must never be committed.

## Running

`run_pipeline.sh` is a bash script. On Windows, use Git Bash or WSL.

Smoke test first — a few thousand spectra, end to end:

```bash
SMOKE_TEST=1 MODEL_PATH=instanovo-v1.1.0 ./run_pipeline.sh
```

A smoke run writes to its own `OUTPUT_ROOT` (`./sae_pipeline_outputs_smoke`) so it never mixes artefacts with a production run, and ends with a verification block that checks every layer: schema versions current, decoder rows unit-norm, FVE finite, `L0 > 0`, calibrated JumpReLU threshold, and the expected columns in `per_feature_stats.csv`. It exits non-zero if any check fails.

Full run over the nine-species benchmark:

```bash
MODEL_PATH=instanovo-v1.1.0 ./run_pipeline.sh
```

Common variants:

```bash
LAYERS_OVERRIDE="8"                  # single layer
DATASET_PATH=/data/combined.parquet  # skip the dataset merge
SKIP_TRAIN=1                         # evaluate existing checkpoints
OUTPUT_ROOT=/mnt/ssd/sae             # write artefacts elsewhere
KEEP_CHUNKS=0                        # delete this run's activations afterwards
ANNOTATE_WORKERS=8                   # parallel chunk annotation (default: all cores)
DEVICE=cpu                           # override the auto-detected device
```

`DEVICE` follows the hardware by default: `cuda` when torch can see a GPU, `cpu` otherwise. Setting it explicitly is strict — `DEVICE=cuda` on a machine with no usable GPU aborts rather than silently running on CPU, since that would be orders of magnitude slower over the full dataset.

Layers are independent. Once extraction and annotation are done, training and evaluation for different layers can run concurrently on separate GPUs by launching the script with `LAYERS_OVERRIDE="<layer>"` and a distinct `DEVICE` in separate shells.

### Evaluation phases

Phases 1–6 run from the cached chunks alone. Phases 7 and 8 need `MODEL_PATH`, and `evaluate.py` imports the model lazily so the earlier phases work without it.

Phase definitions in `evaluate.py` are ordered to follow the results section of the accompanying paper:

1. **1+2** reconstruction and sparsity
2. **6** threshold sweep
3. **5** dictionary geometry
4. **3** top-activating tokens
5. **4** feature–concept associations
6. **7** loss recovered
7. **cross-layer matching**
8. **8** causal ablation

Phase *numbers* are a stable cross-file contract — they key `report.json`, the `--skip` CLI, and the Phase 4 resume cache — so they stay fixed even though execution order differs: Phases 3 and 4 run as a single streaming pass, and Phase 8 and the cross-layer/cross-seed checks all consume Phase 4's output, so Phase 4 runs before them.

**Phase 8 is off by default** (`RUN_PHASE_8=0`) because it dominates the runtime — its cost is `n_concepts × (1 + controls + ABLATION_PER_FEATURE_TOP)` model passes over `ABLATION_SPECTRA` spectra, per layer. At the defaults that is 50 × 26 passes over 5,000 spectra:

```bash
RUN_PHASE_8=1 MODEL_PATH=... ./run_pipeline.sh
```

`ABLATION_PER_FEATURE_TOP` (default 20) is the dominant term; raising it buys per-feature resolution at a proportional cost in wall-clock and leaves the group-level causal metrics unchanged, since those come from the group ablation. Identical feature sets are ablated once and reused, so concepts sharing top features cost only one pass between them.

Progress is checkpointed after each concept to `eval/phase8_partial.json`. A run killed part-way resumes from there and produces exactly the results an uninterrupted run would have, since each concept's random draws are keyed on its own index rather than on how many ran before it. The file is removed once the phase completes, and ignored if the settings that define an ablation have changed.

### LLM-assisted interpretation

`interpret.py` adapts the automated feature-description pipeline of InterPLM (Simon and Zou, 2025) to this setting. It reads an existing evaluation directory and needs `OPENAI_API_KEY` in `.env` or the environment; no other stage needs an API key.

```bash
uv add openai

python interpret.py \
  --eval-dir        $OUTPUT_ROOT/sae/layer_2/seed_0/eval \
  --extract-dir     $OUTPUT_ROOT/extract \
  --annotation-dir  $OUTPUT_ROOT/annotation \
  --sae-checkpoint  $OUTPUT_ROOT/sae/layer_2/seed_0/checkpoint.pt \
  --output-dir      $OUTPUT_ROOT/interpret/layer_2 \
  --target-layer    2
```

`--dry-run` builds the prompts without calling the API. Four feature strata are interpreted, selected with `--strata`; each feature is assigned to exactly one, in this order:

| Stratum | Features | Purpose |
|---|---|---|
| `causal` | implicated by the Phase 8 ablations | Asks whether causally necessary features have describable structure |
| `unexplained` | activation mass concentrated on peaks the theory cannot label, ranked by `unexplained_mass_fraction` above a firing floor | Discovery set |
| `concept` | strongest BH-significant chemical association | Positive control: the chemistry is already known, so recovering it validates the pipeline |
| `unlabelled` | no significant concept at all, sampled uniformly | The features the evaluation is silent about |

`unexplained` and `unlabelled` sound alike but need not overlap: a feature whose best concept is `is_noise_peak` is excluded from `unlabelled` by construction, yet is the clearest possible unexplained-peak specialist. Both pools and their overlap are logged per run. The `concept` ranking skips features whose best concept is structural rather than chemical (`is_noise_peak`, `is_latent_token`), since neither validates inferring chemistry.

Concept labels are withheld from the prompt by default, so any chemistry the model names is inferred from the spectra alone; `--include-concept-labels` overrides this. Each description is scored the way InterPLM scores them: a held-out set of examples is shown without activations, the model predicts each one, and the Pearson correlation against the measured values is reported. The `causal` stratum requires Phase 8 to have run, and `unexplained` requires a Phase 4 that was not restored from cache.

## Scale

The nine-species benchmark contains 639,286 spectra, giving roughly 67.5 million encoder tokens per layer.

- **Disk:** ~104 GB per layer at the default `bfloat16` (~415 GB for four layers); double that for `float32`.
- **RAM:** SAE training loads one layer into memory when it fits, and streams from disk otherwise. `run_pipeline.sh` sizes this automatically against free memory; `TRAIN_NO_RAM_CACHE=0` or `=1` forces either mode. The cached mode gives a true global shuffle and avoids re-reading the layer once per epoch.
- **SAE:** `F = 12,288` features (16× expansion of `d_model = 768`), `k = 32`, 3 epochs.
- Extraction is the largest one-off cost, runs once, and is preserved by `KEEP_CHUNKS=1` (the default).

`N_PEAKS` (default 200) sets how many peaks each spectrum contributes. It is recorded in the extract manifest and read back by `evaluate.py`, so Phases 7–8 rebuild exactly the spectra extraction saw.

## Outputs

```
$OUTPUT_ROOT/
├── pipeline.log                     plus one log per stage, layer and seed
├── combined_ninespecies.parquet     merged dataset (unless DATASET_PATH is set)
├── eval_spectra_first{N}.parquet    capped Phase 7/8 source (capped runs only)
├── extract/
│   ├── manifest.json
│   ├── extract_config.json          resume fingerprint: dtype, n_peaks, dataset
│   └── chunks/                      acts_L{2,4,6,8}_{chunk}.pt, meta_{chunk}.pt
├── annotation/
│   ├── annotation_manifest.json     registry, base rates, vocabularies
│   ├── concept_phi.pt               concept–concept correlation matrix
│   └── labels/                      one LabelChunkData per chunk
├── sae/
│   └── layer_{L}/seed_{S}/
│       ├── checkpoint.pt
│       ├── training_log.jsonl
│       └── eval/
│           ├── report.json          all phase outputs
│           ├── per_feature_stats.csv   incl. unexplained-mass discovery columns
│           ├── feature_label_associations.csv
│           ├── top_activating_tokens.csv
│           ├── causal_ablation.csv
│           └── cross_layer_matches.csv
└── interpret/
    └── layer_{L}/
        ├── feature_descriptions.csv    stratum, summary, prediction r
        └── feature_descriptions.json   full descriptions and predictions
```

The run ends with a cross-layer summary table printed to the log. None of these artefacts is committed; see `.gitignore`.

## Citation

Citation details withheld for anonymous review.

Built on InstaNovo (Eloff et al., *Nature Machine Intelligence* 7:565–579, 2025) and evaluated on the nine-species benchmark (Wen and Noble, *Scientific Data* 11, 2024). The interpretation step follows InterPLM (Simon and Zou, *Nature Methods* 22:2107–2117, 2025).

## License

Apache-2.0, matching upstream InstaNovo. See `LICENSE`.
