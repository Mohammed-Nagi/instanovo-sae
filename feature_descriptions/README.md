# LLM feature descriptions

Output of `interpret.py`, one file per encoder layer, from the run reported in the
paper. A language model was shown each feature's activating tokens rendered as
physical observables, with no concept labels, and asked what predicts activation.
Each description was then scored by presenting held-out examples *without*
activations, asking the model to predict each, and correlating against measured
values.

Model: `gpt-4o-2024-08-06`. 632 API calls, 1,247,511 prompt and 94,630 completion
tokens for all four layers. Every layer sampled the same twelve activation chunks,
evenly spaced across the corpus, so the four are directly comparable.

## Columns

| column | meaning |
|---|---|
| `stratum` | `concept` (strongest chemical association) or `unexplained` (the discovery set) |
| `pearson_r` | held-out activation prediction, the score that validates the description |
| `holdout_coverage` | fraction of held-out examples the model actually answered |
| `max_f1_dom`, `best_concept` | the correlational association, for reference |
| `unexplained_mass_fraction` | share of peak-token activation mass on peaks the theory cannot label |
| `summary` | one-sentence description |

## Headline

Median `pearson_r` by stratum:

| layer | concept | discovery |
|---|---|---|
| 2 | 0.648 | **0.806** |
| 4 | 0.569 | **0.785** |
| 6 | 0.646 | **0.815** |
| 8 | 0.616 | **0.799** |

Features responding to peaks the fragment theory cannot label are described better
than those with the strongest chemical associations, at every depth, and only the
discovery stratum exceeds the 0.72 median InterPLM reports for protein features.
Holdout coverage is 1.00 throughout, so no correlation is computed over a
self-selected subset.

Two caveats stated in the paper apply here. Held-out prediction measures predictive
adequacy, not chemical correctness: a description can predict activations well
while naming the wrong chemistry, and none of these has been checked by a
proteomics specialist. The strata are also not matched on rule complexity, since
discovery features are predominantly narrow mass detectors and a fixed-`m/z` rule
is easier to apply than the context-dependent patterns concept features require.

The `unlabelled` stratum is absent by design: with 12,288 features against 50
concepts, everything that fires appreciably picks up some significant association,
leaving that pool to the near-dead tail.
