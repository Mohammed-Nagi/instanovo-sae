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
| `instanovo_io.py` | The single boundary against the InstaNovo API | §3.2 |
| `run_pipeline.sh` | Orchestration with resume and artefact reuse | Fig. 3.1 |

InstaNovo itself is installed from PyPI and is not vendored here. Only four upstream symbols are used — `InstaNovo`, `TransformerDataProcessor`, `SpectrumDataFrame`, and `LEGACY_PTM_TO_UNIMOD` — all reached through `instanovo_io.py`.

## Pipeline

```
extract.py  ──▶  one forward pass, all four layers        [reusable]
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
annotate.py                         train.py
concept labels     [reusable]       one SAE per layer
        │                               │
        └───────────────┬───────────────┘
                        ▼
                   evaluate.py
              eight evaluation phases
```

Extraction and annotation are the expensive stages and are reused across every layer, seed, and SAE width. Every step is idempotent and skips if its output already exists, so the pipeline is safe to interrupt and resume.

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

You also need an InstaNovo model checkpoint. Obtain it from the [InstaNovo project](https://github.com/instadeepai/InstaNovo) and point `MODEL_PATH` at it; the pipeline defaults to `./instanovo_v1.1.0.ckpt`.

## Running

`run_pipeline.sh` is a bash script. On Windows, use WSL or Git Bash.

Smoke test first — a few hundred spectra, end to end, in minutes:

```bash
SMOKE_TEST=1 MODEL_PATH=/path/to/instanovo_v1.1.0.ckpt ./run_pipeline.sh
```

Full run over the nine-species benchmark:

```bash
MODEL_PATH=/path/to/instanovo_v1.1.0.ckpt ./run_pipeline.sh
```

Common variants:

```bash
LAYERS_OVERRIDE="8"                  # single layer
DATASET_PATH=/data/combined.parquet  # skip the dataset merge
SKIP_TRAIN=1                         # evaluate existing checkpoints
OUTPUT_ROOT=/mnt/ssd/sae             # write artefacts elsewhere
KEEP_CHUNKS=0                        # delete activations after the run
```

### Which phases need the model

Phases 1–6 (reconstruction, sparsity, top-activating tokens, feature–concept associations, dictionary geometry, threshold sweep) run from cached chunks alone. Phases 7 and 8 (loss recovered, causal ablation) require `MODEL_PATH`. `evaluate.py` imports the model lazily so the earlier phases work without it.

## Scale

The nine-species benchmark contains 639,286 spectra, giving roughly 67.5 million encoder tokens per layer.

- Allow about 1.5 TB of SSD for activation chunks across four layers.
- SAE configuration: `F = 12,288` features (16× expansion of `d_model = 768`), `k = 32`, 3 epochs.
- Extraction dominates runtime and runs once; `KEEP_CHUNKS=1` (the default) preserves it.

## Outputs

```
$OUTPUT_ROOT/
├── extract/
│   ├── manifest.json
│   └── chunks/          acts_L{2,4,6,8}_{chunk}.pt, meta_{chunk}.pt
├── annotation/          per-token concept labels
└── sae/
    └── layer_{L}/seed_{S}/
        ├── checkpoint.pt
        └── eval/report.json
```

None of this is committed; see `.gitignore`.

## Citation

```bibtex
@mastersthesis{nagi2026sae,
  title  = {Sparse Autoencoders for Interpretability of InstaNovo},
  author = {Nagi, Mohammed Esameldin Adam},
  school = {African Institute for Mathematical Sciences (AIMS) South Africa},
  year   = {2026}
}
```

Built on InstaNovo (Eloff et al., *Nature Machine Intelligence* 7:565–579, 2025) and evaluated on the nine-species benchmark (Wen and Noble, *Scientific Data* 11, 2024).

## License

Apache-2.0, matching upstream InstaNovo. See `LICENSE`.
