# instanovo-sae

Sparse autoencoders on InstaNovo's encoder (layers 2, 4, 6, 8), a first-principles
fragment-ion annotator, an evaluation suite, and LLM feature descriptions.

## Goal

An **interpretability tool used alongside InstaNovo**: a Python API plus a
dashboard that lets InstaDeep scientists inspect a spectrum as the model
processes it — which SAE features fire on each peak at each layer, what they
mean, how well that meaning is validated, and what happens to the prediction
when a feature is ablated. Scope is the encoder; the tool explains what the
encoder extracted, not how the decoder chose each residue, and the UI says so.

The paper is on hold. Work proceeds in versions: **v1 is a small, solid,
working tool**; rigour and features expand afterwards. The task list, findings
and status live in `STATUS.md` (repo root) — read it at the start of every
session and update it at the end.

## Repo layout

```
src/                       one source root (run with PYTHONPATH=src)
├── instanovo_io.py        the only boundary against the InstaNovo API
├── extract.py, annotate.py, train.py, evaluate.py, interpret.py   research pipeline
├── schema.py              on-disk schema versions
├── analysis.py            post-hoc analysis over eval outputs           (v1, new)
├── build_dictionary.py    builds the feature dictionary                  (v1, new)
└── inspector/             the tool                                       (v1, new)
    ├── runtime.py         explain(spectrum), ablation
    ├── dictionary.py      loads the feature dictionary
    └── app.py             dashboard                                      (V1-6, not yet)
tests/                     pytest
paper_artifacts/           released report.json per layer, description CSVs (read-only)
run_pipeline.sh            pipeline orchestration
STATUS.md                  task list, findings, backlog, log
docs/                      paper material (gitignored, local only)
```

## Local data (gitignored)

One combined root, `sae_pipeline_outputs/`; InstaNovo checkpoint at
`checkpoints/instanovo_v1.1.0.ckpt`.

```
sae_pipeline_outputs/
├── extract/       manifest.json is a consolidated 13-chunk manifest (n_chunks=13,
│                  the chunks on disk, 8.2 GB); the annotation manifest lists 625.
│                  ChunkStream joins a consolidated manifest by chunk id.
│                  meta, acts_L2, acts_L4: 13 chunks; acts_L6, acts_L8: 12
├── annotation/    labels/ for the same 13 chunks
├── sae/layer_{2,4,6,8}/seed_0/
│   ├── checkpoint.pt
│   └── eval/      report.json, per_feature_stats.csv, ...
│                  causal_ablation.csv only L2/L4 (numbers also in report.json)
├── interpret/layer_{L}/<run_tag>/    descriptions: blind_v1 (dictionary), unblinded_v2
│                                      (comparison only); L8 runs incomplete
├── interpret/layer_{L}/archive/      superseded runs (unblinded_v1), provenance only;
│                                      nothing reads it (schema.check_run_tag)
├── analysis/                          outputs of analysis.py
└── dictionary/layer_{L}.parquet       the contract between pipeline and tool
```

Anything needing the full corpus (Phase 4, full Phase 8) waits for the cluster.

## Rules

- No new top-level directories, output roots or parallel versions of existing
  modules without asking. Edit modules in place; new behaviour goes behind a flag.
- Every generated file goes under `sae_pipeline_outputs/<stage>/`.
- Every number that reaches a report or the UI is computed by code in this repo.
- The tool depends on the pipeline only through checkpoints, `instanovo_io`,
  `annotate`, and the dictionary — not through eval internals.
- Stay inside the current v1 task. Ideas for later go into the backlog in
  `STATUS.md`, not into code.
- Machine: 15.4 GB RAM (often a few GB free), 6 cores, no usable GPU. Load one
  chunk at a time; never materialise a full [n_tokens, 12288] matrix.
- Work on `main`, one commit per finished task. `docs/` is gitignored and stays
  local. `CLAUDE.md` and `STATUS.md` are committed and the repo is public: never
  write keys, tokens, personal paths or private notes into them.
- Causal effects are reported as effect sizes with their provenance (estimated
  by attribution patching vs verified by exact ablation; SAE-patched vs
  error-preserving; n spectra; CI if any) — no binary labels or thresholds
  without uncertainty. Features never measured are "untested", never "not
  important".
- Keep the existing style: idempotent stages, schema versions, seeded RNGs,
  loud errors over silent fallbacks. Python 3.10–3.13, `uv`.
- Run tests with `PYTHONPATH=src uv run python -m pytest tests`
  (`uv run pytest` fails on this Windows setup with a trampoline error).
- Never commit `.env`, checkpoints or anything under `sae_pipeline_outputs/`.
