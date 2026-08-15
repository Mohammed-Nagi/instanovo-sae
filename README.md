# Sparse Autoencoders for Interpretability of InstaNovo

Code for the MSc thesis *Sparse Autoencoders for Interpretability of InstaNovo* (Mohammed Esameldin Adam Nagi, AIMS South Africa / InstaDeep, 2026).

Sparse autoencoders are trained on the residual-stream activations of [InstaNovo](https://github.com/instadeepai/InstaNovo)'s encoder at four depths. Every spectral peak is labelled with chemical concepts derived from first-principles fragment-ion theory, allowing us to ask which learned features correspond to which chemical events — both correlationally and causally.

## Contents

| File | Role | Thesis |
|---|---|---|
| `extract.py` | Multi-layer activation extraction (layers 2, 4, 6, 8) | §3.2 |
| `annotate.py` | Per-peak fragment-ion annotation: 50 concepts, 14 families | §3.4 |
| `train.py` | Sparse autoencoder: BatchTopK training, AuxK recovery, JumpReLU inference | §3.3 |
| `evaluate.py` | Eight-phase evaluation suite | §3.5 |
| `interpret.py` | LLM-assisted feature description, validated by held-out activation prediction | §5.3.1 |
| `instanovo_io.py` | The single boundary against the InstaNovo API | §3.2 |
| `schema.py` | On-disk schema versions shared across the pipeline | — |
| `run_pipeline.sh` | Orchestration with resume and artefact reuse | Fig. 3.1 |

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

Ids come from InstaNovo's `models.json` — `instanovo-v1.1.0` (used for the thesis), `instanovo-v1.2.0`, and `instanovo-phospho-v1.0.0`. Anything ending in `.ckpt` or containing a path separator is treated as a file; everything else as an id.

Copy `.env.example` to `.env` for the optional settings and, if you plan to run `interpret.py`, the API key. `.env` is gitignored and must never be committed.

## Running

`run_pipeline.sh` is a bash script. On Windows, use Git Bash or WSL.

Smoke test first — a few thousand spectra, end to end, in minutes:

```bash
SMOKE_TEST=1 MODEL_PATH=instanovo-v1.1.0 ./run_pipeline.sh
```

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
KEEP_CHUNKS=0                        # delete activations after the run
```

Layers are independent. Once extraction and annotation are done, training and evaluation for different layers can run concurrently on separate GPUs by launching the script with `LAYERS_OVERRIDE="<layer>"` and a distinct `DEVICE` in separate shells.

### Evaluation phases

Phases 1–6 run from the cached chunks alone. Phases 7 and 8 need `MODEL_PATH`, and `evaluate.py` imports the model lazily so the earlier phases work without it.

Phase definitions in `evaluate.py` are ordered to follow the thesis results chapter:

| Phase | Thesis |
|---|---|
| 1+2 reconstruction and sparsity | §4.1.1, §4.2.1 |
| 6 threshold sweep | §4.1.2 (Figure 4.1) |
| 5 dictionary geometry | §4.2.2 (Table 4.3) |
| 3 top-activating tokens | §4.3 |
| 4 feature–concept associations | §4.3 (Tables 4.4–4.5) |
| 7 loss recovered | §4.4 (Table 4.6) |
| cross-layer matching | §4.5 (Table 4.7) |
| 8 causal ablation | §4.6 |

Execution order differs: Phase 8 and the cross-layer/cross-seed checks all consume Phase 4's output, so Phase 4 runs before them.

**Phase 8 is off by default** (`RUN_PHASE_8=0`) because it dominates the runtime — its cost is `n_concepts × (1 + controls + ABLATION_PER_FEATURE_TOP)` model passes over `ABLATION_SPECTRA` spectra, per layer. At the defaults that is 50 × 106 passes over 5,000 spectra. To enable it at a lower cost:

```bash
RUN_PHASE_8=1 ABLATION_PER_FEATURE_TOP=20 MODEL_PATH=... ./run_pipeline.sh
```

`ABLATION_PER_FEATURE_TOP` accounts for ~94% of Phase 8; lowering it trades per-feature resolution for wall-clock and leaves the group-level causal metrics unchanged.

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

`--dry-run` builds the prompts without calling the API. Three feature strata are interpreted, selected with `--strata`:

| Stratum | Features | Purpose |
|---|---|---|
| `concept` | strongest BH-significant concept association | Positive control: the chemistry is already known, so recovering it validates the pipeline |
| `unlabelled` | no significant concept at all | Discovery set: the features the registry cannot describe |
| `causal` | implicated by the Phase 8 ablations | Asks whether causally necessary features have describable structure |

Concept labels are withheld from the prompt by default, so any chemistry the model names is inferred from the spectra alone; `--include-concept-labels` overrides this. Each description is scored the way InterPLM scores them: a held-out set of examples is shown without activations, the model predicts each one, and the Pearson correlation against the measured values is reported. The `causal` stratum requires Phase 8 to have run.

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
├── pipeline.log
├── combined_ninespecies.parquet     merged dataset (unless DATASET_PATH is set)
├── extract/
│   ├── manifest.json
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
│           ├── per_feature_stats.csv
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

```bibtex
@mastersthesis{nagi2026sae,
  title  = {Sparse Autoencoders for Interpretability of InstaNovo},
  author = {Nagi, Mohammed Esameldin Adam},
  school = {African Institute for Mathematical Sciences (AIMS) South Africa},
  year   = {2026}
}
```

Built on InstaNovo (Eloff et al., *Nature Machine Intelligence* 7:565–579, 2025) and evaluated on the nine-species benchmark (Wen and Noble, *Scientific Data* 11, 2024). The interpretation step follows InterPLM (Simon and Zou, *Nature Methods* 22:2107–2117, 2025).

## License

Apache-2.0, matching upstream InstaNovo. See `LICENSE`.
