# Evaluation reports

`report.json` from `evaluate.py`, one per encoder layer, seed 0, produced by the
nine-species run described in the paper: 639,286 spectra, 67,503,495 encoder
tokens per layer, SAEs with `d_dict = 12,288` and `k = 32` trained for 3 epochs.

These are the complete phase outputs, checked in so the reported numbers can be
verified without re-running the pipeline. The artefacts they summarise, SAE
checkpoints and the per-feature CSVs, are far larger and are not in the
repository; the pipeline regenerates them from a public model checkpoint and a
public benchmark.

| file | layer |
|---|---|
| `layer_2.json` | 2 |
| `layer_4.json` | 4 |
| `layer_6.json` | 6 |
| `layer_8.json` | 8 |

## Reading one

```python
import json
r = json.load(open("eval_reports/layer_2.json"))

r["phase_1_2"]["fve_overall"]              # centred fraction of variance explained
r["phase_1_2"]["l0_mean"]                  # mean active features per token
r["phase_7"]["loss_recovered_vs_zero"]     # task performance surviving substitution
r["phase_7"]["clean_ce_alignment"]         # per-spectrum alignment check
r["phase_4"]["n_significant_pairs"]        # BH-significant (feature, concept) pairs
r["phase_4"]["unexplained_peaks"]          # peaks the fragment theory cannot label
r["cross_layer"]                           # anchor layer only
```

Phase numbers are the stable identifiers used across the codebase, so
`phase_1_2` is reconstruction and sparsity, `phase_7` is task preservation, and
so on; see the paper's evaluation section for the full list.

`phase_8` (causal ablation) is absent where a run was launched with
`RUN_PHASE_8=0`. It is expensive enough to be run as a separate pass over the
cached Phase 4 output.
