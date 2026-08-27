# Sparse Autoencoders for De Novo Peptide Sequencing

Sparse autoencoders trained on the residual-stream activations of
[InstaNovo](https://github.com/instadeepai/InstaNovo)'s encoder at four depths (layers 2, 4, 6, 8).
Every spectral peak is labelled with chemical concepts derived from first-principles fragment-ion
theory rather than a reference database, so we can ask which learned features correspond to which
chemical events, correlationally and causally, and what the model attends to that the theory
cannot label.

Code for the paper of the same name. All results below are reproducible from a public model
checkpoint and the public nine-species benchmark.

## Results in this repository

| Path | Contents |
|---|---|
| `eval_reports/layer_{2,4,6,8}.json` | Every evaluation phase per layer, including causal ablation |
| `feature_descriptions/layer_{2,4,6,8}.csv` | LLM feature descriptions with held-out prediction scores |

Headline numbers, all from the runs these files record:

- **Reconstruction.** FVE 0.985 / 0.970 / 0.944 / 0.920 at layers 2 / 4 / 6 / 8, with over
  99.5% of sequencing loss preserved when the reconstruction is substituted into the forward pass.
- **Features beyond the vocabulary.** Only 38% of peaks match a theoretical fragment. Features
  responding to the rest are described *better* than the strongest concept-associated features at
  every depth: median held-out prediction *r* = 0.806 / 0.785 / 0.815 / 0.799 against
  0.648 / 0.569 / 0.646 / 0.616.
- **Depth organisation.** Mean peak layer per concept family runs monotonically from 2.0
  (peptide length, intensity) to 8.0 (fragment charge, cleavage specificity).
- **Association is not use.** Rank correlation between `F1dom` and ablation selectivity is
  +0.04 / +0.53 / +0.13 / +0.06 across the four depths. Only the structural positive control is
  causally significant at all of them.

## Layout

| File | Role |
|---|---|
| `run_pipeline.sh` | Orchestration: resume, artefact reuse, publishing |
| `instanovo_io.py` | The single boundary against the InstaNovo API |
| `extract.py` | Multi-layer activation extraction, one forward pass |
| `annotate.py` | Per-peak fragment-ion annotation: 50 concepts, 14 families |
| `train.py` | SAE: BatchTopK training, AuxK recovery, JumpReLU inference |
| `evaluate.py` | Eight-phase evaluation suite |
| `interpret.py` | LLM feature description, scored by held-out activation prediction |
| `schema.py` | On-disk schema versions shared across the pipeline |

Four sequential stages — extract, annotate, train, evaluate — plus `interpret.py` afterwards.
Every step is idempotent and skips when its output exists, so the pipeline is safe to interrupt.
Extraction and annotation are the expensive one-off stages and are reused across every layer,
seed and SAE width.

Each stage records a schema version in what it writes and each consumer rejects an unexpected
one, which turns a stale-artefact mistake into an error rather than silently wrong numbers.

InstaNovo is a PyPI dependency, not vendored. Only four upstream symbols are used, all through
`instanovo_io.py`.

## Setup

Python 3.10–3.13 (`instanovo` pins `<3.14`).

```bash
uv python install 3.13
uv sync --extra cu126          # GPU; --extra cpu for CPU-only
uv run python -c "import instanovo_io, extract, annotate, train, evaluate; print('imports OK')"
```

`MODEL_PATH` takes a local `.ckpt` or a pretrained id that InstaNovo resolves and caches
(`instanovo-v1.1.0` is what we used). Copy `.env.example` to `.env` for `OPENAI_API_KEY`, needed
only by `interpret.py`. `.env` is gitignored and must never be committed.

## Running

`run_pipeline.sh` is bash; on Windows use Git Bash or WSL.

```bash
SMOKE_TEST=1 MODEL_PATH=instanovo-v1.1.0 ./run_pipeline.sh   # end to end, a few thousand spectra
MODEL_PATH=instanovo-v1.1.0 ./run_pipeline.sh                # full nine-species run
```

The smoke run writes to its own `OUTPUT_ROOT` and ends with a verification block that fails
non-zero if any layer's schema, decoder norms, FVE, `L0` or threshold look wrong.

Frequently used overrides:

```bash
LAYERS_OVERRIDE="8"        # one layer; layers are independent once extraction is done
SKIP_TRAIN=1               # evaluate existing checkpoints, never train
OUTPUT_ROOT=/mnt/ssd/sae   # where artefacts land
KEEP_CHUNKS=0              # delete activations afterwards (KEEP_CHUNK_SAMPLE retains N/layer)
FORCE_EXTRACT=1            # re-extract even if the manifest looks complete
RUN_PHASE_8=1              # causal ablation (off by default; see below)
RUN_INTERPRET=1            # LLM description pass, needs INTERPRET_ENV_FILE or OPENAI_API_KEY
DEVICE=cpu                 # override the auto-detected device
```

`DEVICE` follows the hardware by default and is strict when set explicitly, so `DEVICE=cuda` on a
machine without a usable GPU aborts rather than silently running orders of magnitude slower.

### Evaluation phases

Phases 1–6 run from cached chunks alone; 7 and 8 need `MODEL_PATH`, and the model is imported
lazily so the rest work without it. Phase *numbers* are a stable cross-file contract keying
`report.json`, the `--skip` CLI and the Phase 4 resume cache, so they stay fixed even though
execution order differs: Phases 3 and 4 run as one streaming pass, and Phase 8 and the
cross-layer checks consume Phase 4's output.

**Phase 8 (causal ablation) is off by default.** Its cost is
`n_concepts × (1 + controls + ABLATION_PER_FEATURE_TOP)` model passes over `ABLATION_SPECTRA`
spectra per layer, which dominates the runtime. Run it alone against a finished output root:

```bash
PHASE8_RESUME=1 SKIP_TRAIN=1 ABLATION_PER_FEATURE_TOP=5 \
LAYERS_OVERRIDE="2" OUTPUT_ROOT=... ./run_pipeline.sh
```

`ABLATION_PER_FEATURE_TOP` is the dominant term. Targets are ranked by `F1dom`, so rank 1 is
always the concept's strongest detector; the reported results use 5. Progress is checkpointed per
concept to `eval/phase8_partial.json` and resumes to exactly what an uninterrupted run would
produce, since each concept's random draws are keyed on its own index.

### LLM-assisted interpretation

`interpret.py` adapts the description pipeline of InterPLM (Simon and Zou, 2025). It reads a
finished evaluation directory and needs `OPENAI_API_KEY`.

```bash
python interpret.py \
  --eval-dir       $OUTPUT_ROOT/sae/layer_2/seed_0/eval \
  --extract-dir    $OUTPUT_ROOT/extract \
  --annotation-dir $OUTPUT_ROOT/annotation \
  --sae-checkpoint $OUTPUT_ROOT/sae/layer_2/seed_0/checkpoint.pt \
  --output-dir     $OUTPUT_ROOT/interpret/layer_2 \
  --target-layer   2
```

`--dry-run` builds prompts without calling the API. Features are assigned to exactly one stratum,
in precedence order: `causal` (needs Phase 8), `unexplained` (the discovery set: activation mass
on peaks the theory cannot label), `concept` (positive control), `unlabelled`.

Concept labels are withheld from prompts by default, so any chemistry named is inferred from
spectra alone. Each description is scored by showing held-out examples without activations, asking
the model to predict each, and correlating against measured values. Results append to a resume log
as each feature completes, so an interrupted run does not pay twice.

Expect `unlabelled` to come back empty. With 12,288 features against 50 concepts, everything that
fires appreciably picks up some significant association, leaving that pool to the near-dead tail.
That is a property of the layer, not a misconfiguration.

## Scale

639,286 spectra, ~67.5M encoder tokens per layer. Extraction is ~104 GB per layer in bfloat16 and
is the largest one-off cost, so `KEEP_CHUNKS=1` is the default. SAE: `F = 12,288` (16× expansion
of `d_model = 768`), `k = 32`, 3 epochs. `N_PEAKS` (default 200) is recorded in the extract
manifest and read back by `evaluate.py`, so Phases 7–8 rebuild exactly the spectra extraction saw.

A manifest listing a chunk is not evidence the file exists — `KEEP_CHUNKS=0` prunes activations
and leaves the manifest intact — so anything selecting chunks stats the file first.

## Outputs

```
$OUTPUT_ROOT/
├── extract/      manifest.json, extract_config.json, chunks/
├── annotation/   annotation_manifest.json, concept_phi.pt, labels/
├── sae/layer_{L}/seed_{S}/
│   ├── checkpoint.pt, training_log.jsonl
│   └── eval/     report.json, per_feature_stats.csv,
│                 feature_label_associations.csv, top_activating_tokens.csv,
│                 causal_ablation.csv, cross_layer_matches.csv
└── interpret/layer_{L}/   feature_descriptions.{csv,json}
```

None of these is committed except the summaries listed under **Results** above; see `.gitignore`.

`Dockerfile.aichor` and `manifest.yaml` describe the cluster job used for the reported runs. The
PVC name in the manifest is redacted for review; substitute your own to use it.

## Citation

Withheld for anonymous review.

Built on InstaNovo (Eloff et al., *Nature Machine Intelligence* 7:565–579, 2025), evaluated on the
nine-species benchmark (Wen and Noble, *Scientific Data* 11, 2024). The interpretation step follows
InterPLM (Simon and Zou, *Nature Methods* 22:2107–2117, 2025).

## License

Apache-2.0, matching upstream InstaNovo. See `LICENSE`.
