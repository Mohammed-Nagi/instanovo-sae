# Sparse Autoencoders for Interpretability of InstaNovo

Code for the MSc thesis *Sparse Autoencoders for Interpretability of InstaNovo* (Nagi, AIMS South Africa / InstaDeep, 2026).

Trains sparse autoencoders on the residual-stream activations of [InstaNovo](https://github.com/instadeepai/InstaNovo)'s encoder at four depths, labels every spectral peak with chemical concepts derived from first-principles fragment-ion theory, and tests which learned features correspond to which chemical events — correlationally and causally.

---

## Pipeline

Four stages. The first two are expensive and reusable across every layer, seed, and SAE width.

```
                   ┌──────────────────────────────┐
  spectra ────────▶│ 1. extract.py                │  one forward pass,
                   │    layers 2, 4, 6, 8         │  all four layers at once
                   └──────────────┬───────────────┘
                                  │  activations + metadata   [REUSABLE]
                   ┌──────────────┴───────────────┐
                   │                              │
     ┌─────────────▼──────────────┐  ┌────────────▼─────────────┐
     │ 2. annotate.py             │  │ 3. train.py              │
     │    50 concepts, 14 families│  │    one SAE per layer     │
     │    metadata only, no acts  │  │    BatchTopK + AuxK      │
     └─────────────┬──────────────┘  └────────────┬─────────────┘
                   │  concept labels   [REUSABLE]  │  checkpoints
                   └──────────────┬───────────────┘
                   ┌──────────────▼───────────────┐
                   │ 4. evaluate.py               │
                   │    eight evaluation phases   │
                   └──────────────────────────────┘
```

| File | Role | Thesis |
|---|---|---|
| `extract.py` | Multi-layer activation extraction, chunked with resume | §3.2 |
| `annotate.py` | Per-peak fragment-ion annotation, 50 concepts / 14 families | §3.4 |
| `train.py` | SAE: BatchTopK training, AuxK recovery, JumpReLU inference | §3.3 |
| `evaluate.py` | Eight-phase evaluation suite | §3.5 |
| `instanovo_io.py` | The single boundary against the InstaNovo API | §3.2 |
| `run_pipeline.sh` | Orchestration, resume, artefact reuse | Fig. 3.1 |

---

## Install

Requires Python 3.10–3.13. Note the upper bound: `instanovo` declares `<3.14`.

```bash
git clone https://github.com/<you>/instanovo-sae.git
cd instanovo-sae

# GPU
uv sync --extra cu126

# CPU only (Phases 1-6 and annotation; extraction will be very slow)
uv sync --extra cpu
```

With pip instead of uv:

```bash
pip install -e ".[cu126]"
```

`instanovo` is installed from PyPI — this repository deliberately does not vendor it. The only upstream symbols used are `InstaNovo`, `TransformerDataProcessor`, `SpectrumDataFrame`, and `LEGACY_PTM_TO_UNIMOD`, all reached through `instanovo_io.py`.

### Model checkpoint

Download the InstaNovo v1.1.0 checkpoint from the [InstaNovo releases](https://github.com/instadeepai/InstaNovo) and point `MODEL_PATH` at it. The pipeline defaults to `./instanovo_v1.1.0.ckpt`.

---

## Run

Start with the smoke test — a few hundred spectra, end to end, in minutes:

```bash
SMOKE_TEST=1 MODEL_PATH=/path/to/instanovo_v1.1.0.ckpt ./run_pipeline.sh
```

Full run over the nine-species benchmark (all four layers):

```bash
MODEL_PATH=/path/to/instanovo_v1.1.0.ckpt ./run_pipeline.sh
```

Useful variants:

```bash
LAYERS_OVERRIDE="8"   MODEL_PATH=... ./run_pipeline.sh   # single layer
DATASET_PATH=/data/combined.parquet MODEL_PATH=... ./run_pipeline.sh   # skip the merge
SKIP_TRAIN=1          MODEL_PATH=... ./run_pipeline.sh   # evaluate existing checkpoints
OUTPUT_ROOT=/mnt/ssd/sae MODEL_PATH=... ./run_pipeline.sh
```

`run_pipeline.sh` is a bash script. On Windows use WSL or Git Bash.

### What needs the model, and what doesn't

| Phase | Needs `MODEL_PATH` |
|---|---|
| 1–2 Reconstruction, sparsity | No |
| 3 Top-activating tokens | No |
| 4 Feature–concept associations | No |
| 5 Dictionary geometry | No |
| 6 Threshold sweep | No |
| 7 Loss recovered | **Yes** |
| 8 Causal ablation | **Yes** |

Phases 1–6 run from cached chunks alone, so a collaborator can do substantial work without the checkpoint. `evaluate.py` imports `instanovo_io` lazily for exactly this reason.

---

## Scale

The nine-species benchmark is 639,286 spectra, giving roughly 67.5 million encoder tokens **per layer**.

- Disk: allow ~1.5 TB SSD for activation chunks across four layers.
- Extraction is the dominant cost and runs once. `KEEP_CHUNKS=1` (the default) preserves it; set `KEEP_CHUNKS=0` only if genuinely short on disk.
- Annotation is layer-independent and dataset-fixed. `KEEP_ANNOTATION=1` by default.
- SAE training: `F = 12,288` features (16× expansion of `d_model = 768`), `k = 32`, 3 epochs.

Every step is idempotent and skips if its sentinel output exists, so the pipeline is safe to kill and resume.

---

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

---

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
