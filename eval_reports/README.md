# Evaluation reports

`report.json` from `evaluate.py`, one per encoder layer, seed 0, produced by the
nine-species run described in the paper: 639,286 spectra, 67,503,495 encoder
tokens per layer, SAEs with `d_dict = 12,288` and `k = 32` trained for 3 epochs.

Complete phase outputs, checked in so the reported numbers can be verified
without re-running the pipeline. The artefacts they summarise, SAE checkpoints
and the per-feature CSVs, are far larger and are not in the repository; the
pipeline regenerates them from a public model checkpoint and a public benchmark.

## Reading one

```python
import json
r = json.load(open("eval_reports/layer_2.json"))

r["phase_1_2"]["fve_overall"]              # centred fraction of variance explained
r["phase_1_2"]["l0_mean"]                  # mean active features per token
r["phase_7"]["loss_recovered_vs_zero"]     # task performance surviving substitution
r["phase_4"]["n_significant_pairs"]        # BH-significant (feature, concept) pairs
r["phase_4"]["unexplained_peaks"]          # peaks the fragment theory cannot label
r["phase_8"]["per_concept"]                # causal ablation, per concept
```

Phase numbers are the stable identifiers used across the codebase: `phase_1_2` is
reconstruction and sparsity, `phase_7` task preservation, `phase_8` causal
ablation. See the paper's evaluation section for the full list.

## Two asymmetries worth knowing

All four files carry phases 1 through 8.

`cross_layer` appears only in `layer_4.json`, which matched against layer 2
within the run that produced both. Layers 6 and 8 were run separately, and
`cross_layer_matching` reads every layer from a single extract directory, so
layers in separate output roots cannot be compared.

Causal ablation used `ABLATION_PER_FEATURE_TOP=5` rather than the default 20.
Targets are ranked by `F1dom`, so each concept's strongest detector is always
included; the reduction trades depth of per-feature attribution for runtime and
leaves the group-level metrics unchanged.

Layers 2 and 4 come from one run of the pair; layers 6 and 8 from separate
single-layer runs. Reconstruction fidelity declines monotonically with depth
across all four, as the paper reports.

For the LLM feature descriptions and their held-out prediction scores, see
`../feature_descriptions/`.
