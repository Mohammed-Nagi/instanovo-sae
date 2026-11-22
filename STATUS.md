# Status

Update this file at the end of every session: tick tasks, add a line to the log,
move new ideas to the backlog.

## v1 — a small, solid, working tool

Done when: a scientist can load a benchmark spectrum in the dashboard, see
InstaNovo's prediction, inspect the active features on any peak at any layer
with honest labels and evidence indicators (association, causal and
description coverage, with their effect sizes), and ablate a feature to see the effect
on each predicted residue.

Open, in order: V1-4b, V1-4c, V1-5, V1-6, V1-2c (when API credit is
available), V1-7. Done: V1-1, V1-2a, V1-2b (layers 2/4/6), V1-3, V1-4a.

- [x] **V1-1 Hub inventory** — `src/analysis.py`, output `analysis/hubs.csv`.
      From existing per-feature ablations in each layer's report.json: rank by
      mean ΔCE, flag features > 10× the layer median, list the concept groups
      each falls in. Check against Finding 2. Also recompute the F1dom-vs-
      selectivity rank correlation per layer (Finding 3), one function, no more.
      (The flag was removed in V1-3: `hubs.csv` is a ranked list with
      `ratio_to_median`.)
- [x] **V1-2a Blinded interpretation: code** — `src/interpret.py` gets
      `--blind` (no `matched_ion`, neighbour labels, `peptide_proforma`;
      `token_kind` keeps the latent marker; own preamble), a spectrum-level
      split, random example ids and row order, a per-feature RNG keyed on
      (seed, layer, feature) via sha256, detection AUROC, the
      shuffled-description baseline (`--n-shuffles`, default 3; donors are other
      features in the same layer and run, never the feature itself;
      `description_beats_baseline` = r > p95 of the layer's pooled shuffled r),
      `--run-tag` (required) with a `run_config.json` resume guard,
      `--chunk-indices` (fails if any is missing), and `--dry-run` (prints the
      prompts and a cost estimate, writes nothing). `evaluate.ChunkStream` reads
      the consolidated local manifest. `build_dictionary` fills the baseline
      columns from a blind run. `analysis --descriptions` produces the
      comparison table. `run_pipeline.sh` is not touched (backlog).
- [x] **V1-2b Blinded interpretation: runs** (`900a295`) — `unblinded_v2` and
      `blind_v1` on the same 80 features per layer (`--strata concept
      unexplained`, `--chunk-indices 0 52 … 572`, `gpt-4o-2024-08-06`,
      `--n-shuffles 3`), layers 2, 4 and 6; layer 8 is V1-2c. Constant
      predictions now score r = 0 (NaN only for < 3 predictions or constant
      measured), rescored from the logs with no API calls. Dictionary: blinded
      descriptions only (`blind_v1`, layers 2/4/6; layer 8 has none, pending);
      unblinded_v2 is kept for the comparison only; unblinded_v1 moved to
      `interpret/layer_L/archive/` as provenance, which nothing reads.
      Results (median r blind / unblinded_v2; Δr = paired median, bootstrap
      95% CI; null p95 blind / unblinded; beats baseline blind / unblinded):

      | L | stratum | r | Δr [95% CI] | AUROC | null p95 | beats |
      |---|---|---|---|---|---|---|
      | 2 | concept | 0.673 / 0.513 | +0.082 [+0.032, +0.180] | 0.855 / 0.769 | 0.798 / 0.739 | 14 / 14 of 40 |
      | 2 | unexplained | 0.775 / 0.800 | −0.011 [−0.059, +0.011] | 0.950 / 0.909 | 0.798 / 0.739 | 16 / 25 |
      | 4 | concept | 0.606 / 0.620 | −0.011 [−0.114, +0.035] | 0.762 / 0.804 | 0.688 / 0.780 | 16 / 11 |
      | 4 | unexplained | 0.810 / 0.797 | +0.000 [−0.022, +0.025] | 0.929 / 0.913 | 0.688 / 0.780 | 29 / 23 |
      | 6 | concept | 0.676 / 0.555 | +0.018 [−0.028, +0.113] | 0.792 / 0.729 | 0.584 / 0.555 | 22 / 20 |
      | 6 | unexplained | 0.793 / 0.806 | +0.007 [−0.017, +0.050] | 0.891 / 0.891 | 0.584 / 0.555 | 33 / 33 |

      Notes:
      (a) Blinding leaves r unchanged overall. The layer-2 concept difference
          (+0.082) is the only CI excluding zero across 6 comparisons, so it
          isn't robust to multiple comparisons.
      (b) beats_baseline uses a per-layer pooled null, which ignores
          per-feature difficulty; a per-feature null would need more shuffles
          per feature.
      (c) Unexplained > concept remains confounded with ease of prediction
          under random negatives; hard negatives are in the backlog.
      (d) Same-stratum donors raise the null's upper tail at layer 6 (both
          runs) and layer 4 blind, consistent with feature redundancy; layer 2
          is mixed and layer 4 unblinded runs the other way.
      (e) 41–53% of shuffled predictions are constant and are now counted as
          r = 0, so the shuffled median is 0 and donor comparisons rest on p95.
- [ ] **V1-2c Layer 8 descriptions** — Complete layer 8 descriptions (~$3.84):
      check gpt-4o-2024-08-06 is still available, rerun run_one.sh 8
      unblinded_v2 then 8 blind_v1, recompute the tables and rebuild the layer
      8 dictionary. (Resume logs are intact: unblinded_v2 has 48/80 features,
      blind_v1 none. `run_one.sh` was a session script; its command is in
      V1-2b's settings above. Rebuild with `build_dictionary --layers 8
      --blinded-run blind_v1`; tables with `analysis --descriptions blind_v1
      unblinded_v2`, all four layers.)
- [x] **V1-3 Feature dictionary** — `src/build_dictionary.py`, output
      `dictionary/layer_{L}.parquet`. One row per live feature: firing rate,
      top 3 concept associations (F1dom, precision, recall, lift),
      `latent_token_fraction`, `unexplained_mass_fraction`.
      Causal fields = Phase 8 mean ΔCE where measured, provenance "SAE-patched,
      mean over 5,000 spectra, no CI", else "untested". No hub tiers, flags or
      thresholds. Descriptions: a blinded one takes precedence, the unblinded
      one is kept as a secondary field and always carries "written with access
      to annotation labels and the true peptide". Coverage is separate booleans
      (`has_significant_association`, `causal_measured`, `description_blinded`,
      `description_beats_baseline`; the last stays null until V1-2 defines it,
      never true for unblinded) — no combined score or trust level. Works with
      partial descriptions. Existing run moves to `interpret/layer_L/unblinded_v1/`.
      (Since then: V1-2a defined `description_beats_baseline`, and V1-2b moved
      unblinded_v1 to `interpret/layer_L/archive/`.)
- [x] **V1-4a Runtime forward pass** (`2a3166c`) — `src/inspector/runtime.py`,
      `dictionary.py`. Input: a raw spectrum (m/z array, intensity array,
      precursor m/z, charge), preprocessed by InstaNovo's own pipeline via
      `instanovo_io`, not the extraction's processed peaks. Preprocessing uses
      spectrum_utils (`use_spectrum_utils=True`, a parameter recorded in
      provenance): the standard input for the v1.1.0 checkpoint, matching
      extraction and InstaNovo 1.1.0's own training and prediction (in 1.1.0
      `SpectrumDataset._process_peaks` always used spectrum_utils); only
      InstaNovo 1.2.x's `predict.py` uses False. `explain(spectrum,
      attribution=False)`: beam-search prediction (max_length from the model
      config) plus per-residue log-probs from a teacher-forced pass on the
      predicted peptide; layers 2/4/6/8 encoded through the SAEs; top-k active
      features per peak per layer with dictionary lookup; peaks annotated
      against the predicted peptide (reuse `annotate.py`); per-layer SAE fit
      using the Phase 1–2 FVE definition. JSON-serialisable output with a
      versioned schema and bounded size; no network calls. Key test: for a
      spectrum present in a local chunk, explain()'s layer activations match
      the stored activations (within dtype tolerance). Also: the teacher-forced
      CE matches Phase 7 for that spectrum; the schema round-trips. Measure
      timings on 3 spectra. Footnote: on ~20 test spectra, how often the
      predicted peptide differs between `use_spectrum_utils` True and False.
      Results (chunk 0 spectra 0, 4 deamidated, 6 oxidised; CPU):
      processed peaks equal the stored ones exactly; activations vs stored
      bf16 within max |Δ|/(|a|+rms) 2.9e-3–3.4e-3 at every layer (bf16 half-ULP
      2⁻⁸ = 3.9e-3; test tolerance rtol 2⁻⁷, atol 2⁻⁷·rms), top-8 feature
      overlap 0.985–1.000; CE matches the cached Phase 7 CE to ≤ 7.4e-6, but
      only when teacher-forcing the bare sequence (Finding 5). Timings
      (`python -m inspector.runtime time`, HF test rows 4506–4508 = chunk 520
      rows 0–2): cold load 3.3 s (OS-cached files); first explain 13.4 s, of
      which beam search ~12 s is one-off warm-up; then 3.3 s and 2.7 s
      (beam search 2.7/2.2 s, teacher-forced pass 0.15/0.12 s, SAE + lookup
      0.27/0.21 s, annotation 0.03/0.02 s, serialise ≤ 0.05 s). JSON 0.78 /
      1.92 / 1.76 MB for 35 / 152 / 114 peaks, 85% of it dictionary rows
      (~1,400 distinct features × ~1.15 KB); computed bound 9.74 MB at k = 8.
      Beam vs teacher-forced per-residue log p agree to ≤ 1e-6. Footnote:
      6 of 20 test predictions (rows 4506–4525) differ between
      `use_spectrum_utils` True and False, 5 only by I↔L, 1 a different
      peptide.
- [ ] **V1-4b Attribution patching and verify** — first step: slim the
      Explanation JSON as `EXPLAIN_SCHEMA_VERSION` 2. Each listed feature
      carries a per-feature summary instead of its full dictionary row (V1-4a
      measured 0.8–1.9 MB per spectrum, 85% full rows); the dashboard reads
      full rows from the dictionary it already loads. Then per-spectrum
      attribution patching: zero-ablation estimate −a·(d·∂m/∂x) from a normal backward pass,
      so the SAE error term is untouched; metric m = log p of each predicted
      residue under teacher forcing. Exact error-preserving ablation (subtract
      a·d, rerun) verifies on demand and returns Δ log p per residue. Effects
      are split by token type (latent vs peak), reported per token and in total.
      The per-peak view uses peak tokens only; latent-token features get a
      separate spectrum-summary panel. First measure the timings (AtP pass,
      exact ablation) on a few spectra. Tests on one spectrum.
- [ ] **V1-4c Audit case study** — pick 3–5 confidently wrong predictions
      from the V1-4a accuracy run (seed 0); use the inspector to form a
      hypothesis for each failure; test it by exact ablation; record what
      held and what didn't. Candidates feed the V1-5 demo set.
- [ ] **V1-5 Data for the tool** — feature examples; demo set from the HF test
      split. Verified by Finding 7 from code and logs: chunk 520's spectrum
      IDs start at row 532,480, and the SAE validation chunks are 17, 116, 135,
      141, 230, 292, 316, 397, 448, 462, 489, 495. Calibrate the per-spectrum SAE-fit indicator: the
      per-spectrum centred FVE is not comparable to the corpus FVE in the
      eval reports, so compute its distribution per layer over local
      spectra and show a spectrum's value as a percentile of that reference
      (store the reference where the tool may read it, e.g. dictionary
      metadata).
- [ ] **V1-6 Dashboard** — `src/inspector/app.py`, Streamlit.
      Spectrum plot with peaks coloured by annotation, layer selector, per-peak
      feature panel, feature card (labels, evidence indicators, top examples), ablate button
      showing per-residue change, an "encoder-only" scope note.
- [ ] **V1-7 README** — Update README claims once all work is finished.
      Still stale (README line numbers as of 2026-11-22):
      - 10: "Code for the paper of the same name". The paper is on hold, and
        the repo's goal is now the tool.
      - 17–18: the results table lists the released descriptions without
        saying they were written with labels and the true peptide in the
        prompt (only said at 150–152).
      - 22–23: "over 99.5% of sequencing loss preserved". Phase 7
        teacher-forced the bare sequence (Finding 5).
      - 24–27: the discovery-vs-concept claim, r 0.806 vs 0.648, from
        label-leaking descriptions; the blinded result is in V1-2b.
      - 28–29: depth organisation rests on argmax over near-ties (Finding 3).
      - 33–34: "Only the structural positive control is causally significant
        at all of them" is unsupported (backlog, layer-6 sel_z = 1.97).
      - 38–47: the layout table lacks `analysis.py`, `build_dictionary.py`,
        `inspector/`, `tests/`, `STATUS.md`, `CLAUDE.md`. 49: "Four sequential
        stages … plus interpret.py" omits analysis, the dictionary and the tool.
      - 57–58: "Only four upstream symbols"; there are eight
        (`instanovo_io.py` docstring).
      - 70–71: `MODEL_PATH` setup omits the repo default
        `checkpoints/instanovo_v1.1.0.ckpt`; `.env` lives at the repo root.
      - 95: `RUN_INTERPRET=1` fails until Step 6 passes `--run-tag` (backlog).
      - 178–190: the outputs tree lacks `analysis/`, `dictionary/`, `logs/`,
        `interpret/layer_L/archive/`.
      - 194–195 and 199: "redacted for review" and "Withheld for anonymous
        review" are stale now that the author is named (pyproject).
      - No section on the inspector tool or dashboard. It must state which
        InstaNovo version the SAEs explain (v1.1.0; backlog item on v1.2.0).

## v1.1

- [ ] **V1.1-1 Global importance** — AtP over ~500 local spectra for all
      features at all four layers. Verify the top 20 per layer by exact
      ablation with bootstrap CIs over spectra. Report AtP-vs-exact agreement
      separately for latent and peak features. Cross-check against Phase 8.
      Write the results to the dictionary. If agreement is poor, switch the
      global run to integrated gradients.
- [ ] **V1.1-2 Descriptions at scale** — pilot a cheaper model on the V1-2
      features against gpt-4o, then describe features in priority order (hubs,
      then by firing rate) with the Batch API; rebuild the dictionary.
- [ ] **V1.1-3 Error-detection test (pre-specified, for the paper)** (planned
      as E1 in v1, moved here 2026-11-22: the tool's v1 does not depend on
      its result). Every choice below is fixed before running; any later
      change is logged as a deviation with its reason.
      - Question. Δ₁: do SAE features add information about InstaNovo's
        errors beyond InstaDeep's own calibration features? Δ₂: is any gain
        due to learned SAE structure rather than any sparse code of the
        encoder?
      - Data: 2,000 HF test rows (yeast, per the HF dataset card; there is no
        species column), drawn with `np.random.default_rng(1)`, excluding the
        200 V1-4a accuracy rows (`default_rng(0)`, n = 200). Decoder:
        InstaNovo 1.2.2 BeamSearchDecoder with the v1.1.0 checkpoint (as in
        V1-4a); line references are to the installed 1.2.2 package. 5 beams,
        `Inspector` defaults (float64, max_length 40, `use_spectrum_utils=True`,
        n_peaks 200), recorded. On this draw: 1,773 peptide groups (bare
        sequence, I=L; at most 10 spectra per group), 490 with a modified true
        peptide, 103 with a true peptide longer than 40. Expected ~38.5% wrong
        (V1-4a 77/200, Wilson 95% [32.0%, 45.4%]), i.e. ~770 wrong.
      - Labels: wrong = not exact (prediction vs `modified_sequence` in
        InstaNovo tokens, I=L, one helper shared with `accuracy_check`). True
        peptides longer than 40 stay in as wrong and are reported. Empty
        predictions (no beam completed) are excluded and counted.
      - Baseline B: Winnow's computable features (Mabona et al. 2025, arXiv
        2509.24952): sequence confidence, signed precursor ppm error, beam
        margin, ion match rate, ion match intensity fraction (our annotator on
        the predicted peptide). No retention time (not in the data). Why: it
        is InstaDeep's own published calibrator, so not a straw man, and
        margin and ion matches are its most important features. Beam margin =
        log p(beam 0) − log p(beam 1), in the decoder's own ranking order;
        negative values allowed; NaN when fewer than two beams complete.
        Checked: `instanovo_io.beam_search` already passes
        `return_beam=num_beams > 1` (`src/instanovo_io.py:320`), so
        `BeamSearchDecoder.decode` returns `predictions_log_probability_beam_{i}`
        (`inference/beam_search.py:469-486`). Beams are ranked by
        (meets_precursor, log p) (`beam_search.py:542`), so beam 1 can
        outscore beam 0; slots beyond the completed beams are padded with
        −inf (`beam_search.py:532-549`), which becomes NaN.
        `Inspector.predict` keeps only beam 0 today and needs extending.
      - SAE features S: latent-token code and max over peak tokens at layers
        2/4/6/8 (2 × 12,288 × 4 columns). Within each training fold, keep the
        32 with the largest absolute class-mean difference (Kantamneni et al.
        2025, ICML). Why: a published, fixed-size selection rule; the list is
        short enough to inspect and feeds the audit case study (V1-4c).
      - Control S_rand: per layer, a seeded random encoder of the same shape
        with the same top-32 per-token sparsity, summarised and selected
        identically. Why: random SAEs score close to trained ones on sparse
        probing (Korznikov et al. 2026, arXiv 2602.14111), and a random code
        also retains most raw-activation information, so this one control
        replaces the raw-activation block.
      - Model: `HistGradientBoostingClassifier` with defaults, identical for
        B, B+S and B+S_rand. Why: Winnow reports a non-monotonic mass-error
        effect; trees are scale-free (no scaling or rare-feature rules) and
        need no stacking or nested tuning.
      - Evaluation: `StratifiedGroupKFold(5, shuffle=True, random_state=0)`,
        grouped by bare sequence with I=L, the same folds for all models.
        AUROC (positive = wrong) on pooled out-of-fold predictions. Paired
        bootstrap over peptide groups (10,000, seed 0). Report
        Δ₁ = AUROC(B+S) − AUROC(B) and Δ₂ = AUROC(B+S) − AUROC(B+S_rand) with
        95% CIs, on all spectra and on the precursor-matching subset (≤ 20 ppm,
        min over isotopes 0–1, as in V1-4a; same predictions, since trees can
        condition on precursor error). Also report the selected S features,
        how often each is chosen across folds, and their dictionary
        descriptions.
      - Caveats stated with the results: encoder features can only signal
        spectrum difficulty (the encoder never sees the prediction);
        in-distribution only (yeast); every test spectrum was SAE training
        data, unsupervised (Finding 7).
      - Considered and dropped: the raw-activation block, stacking with
        nested CV, L2 logistic regression on 98k columns, extra baseline
        features (length, charge, peak count, has-modification), per-layer
        models.
      - Cost: featurisation ~2.5–4.5 s per spectrum (beam search 2.2–4.2 s in
        V1-4a), ~1.5–2.5 h for 2,000 on this machine, single process, peak RAM
        ~3 GB.

## Behaviour changes

Changes to existing pipeline modules, each meant to preserve behaviour, with the
tests that show it (`tests/test_inspector_runtime.py`).

- V1-4a, `instanovo_io.make_dataloader`: builds its TransformerDataProcessor
  through the new `make_processor` (same arguments; `use_spectrum_utils=True`,
  the processor default extraction ran with). Shown by
  `test_make_dataloader_reproduces_extraction` (first 32 spectra of chunk 0:
  processed peaks equal the stored ones exactly, per-token CE equals the cached
  baseline to 1e-4) and `test_activations_match_stored_chunk`. New, additive:
  `preprocess_spectrum`, `beam_search`, `encoder_forward`,
  `residue_delta_mass`; two new upstream symbols at the boundary
  (`BeamSearchDecoder`, `ANNOTATED_COLUMN`). 2026-11-22, additive:
  `greedy_search` and `instanovo_metrics` for the accuracy check (upstream
  `GreedyDecoder`, `Metrics`; eight symbols in all).
- V1-4a, `annotate._annotate_spectrum`: its body moved into the public
  `annotate_spectrum` (which also returns the per-peak annotation dicts);
  `_annotate_spectrum` is now a thin caller. Shown by
  `test_chunk_annotator_unchanged_by_refactor` (`annotate_chunk` on the first
  64 spectra of chunk 0 reproduces the stored labels and ion types exactly) and
  `test_annotation_via_tokens_reproduces_stored_labels`.
- V1-4a, `annotate.run_spectrum_utils` / `annotate_spectrum`: new `strict`
  flag, default False (the batch annotator's all-noise-with-warning behaviour
  is unchanged); the inspector passes True so a failed annotation raises.
  Shown by `test_strict_annotation_raises_instead_of_all_noise`.

## Backlog (after v1; do not start early)

Tool:
- Warm up beam search at Inspector load: the first call costs ~12 s extra (V1-4a)
- Corpus builder: run InstaNovo + the SAEs on an arbitrary spectra file (mgf/mzML/parquet) to produce feature examples, so the dictionary can be rebuilt on other datasets without the extraction pipeline

Pipeline:
- `run_pipeline.sh` Step 6 does not pass `--run-tag`, which is now required, so it fails loudly. Add `INTERPRET_RUN_TAG`, `INTERPRET_BLIND`, `INTERPRET_CHUNK_INDICES` and `INTERPRET_N_SHUFFLES`. The publish pattern `interpret/layer_*/*` already matches run-tag subdirectories (fnmatch `*` crosses `/`). Also fix the success log line (`run_pipeline.sh:1494`), which still prints the pre-run-tag path `$INTERPRET_OUT/feature_descriptions.csv`.
- Residue-to-peak attribution (gradient × input / integrated gradients, peak deletion)
- Failure diagnostics: features that predict wrong residues; I/L, deamidation, K/Q cases
- Four-way cross-layer matching on the 12 shared local chunks
- `extract._to_proforma` writes N-terminal mods without ProForma's hyphen, which spectrum_utils rejects, so such spectra are annotated all-noise (Finding 6). 0 of the 13,312 local spectra are affected; count over the full corpus and fix with a schema bump if any are

Rigour (CPU):
- README claim "Only the structural positive control is causally significant at all of them" is unsupported (layer-6 positive control sel_z = 1.97)
- Hub inventory only sees the top-5 F1dom detectors per concept (121–162 features per layer); a hub outside those lists is invisible until the per-feature scan (cluster)
- Per-(concept, feature) pair correlation of F1dom vs selectivity (n=175 per layer; gives +0.19/−0.02/+0.08/+0.03) as a complement to the concept-level one
- Context dependence per feature (peak's own m/z + intensity vs neighbours)
- Linear-probe baseline per concept
- Decision-tree baseline for descriptions
- Revisit the description evaluation protocol before any publication (known limits: small example sets, single draw, random negatives).
- Hard negatives for description scoring: held-out zeros are random tokens, so r and AUROC partly measure active-vs-easy-zero separation; add near-miss negatives (e.g. similar m/z or co-firing features) so unexplained vs concept is not confounded with ease of prediction (V1-2b note c)
- Per-feature shuffled null (more shuffles per feature) instead of the per-layer pooled p95 (V1-2b note b)
- Peptide-level describe/held-out split: the V1-2 split is by spectrum, and the same peptide can still appear on both sides in different spectra; precursor m/z gives it away even blind
- Depth ordering with an F1dom-weighted peak layer and tie margins
- Annotator: immonium ions of modified residues, isotope peaks

Rigour (cluster/GPU):
- Phase 8 rerun: error-preserving ablation, per-feature scan, ~50 controls, bootstrap CIs
- Phase 7/8 rerun with `modified_sequence` targets (Finding 5): report how much the layer-level CE, loss recovered and per-feature ΔCE move; then drop the caveat from the dictionary's `causal_provenance` (`build_dictionary.PHASE8_TARGET_CAVEAT`)
- Random-init InstaNovo control
- Seed stability (2–3 more seeds per layer)
- Phase 4 on the updated annotations
- Check InstaNovo v1.1 training data vs the nine-species benchmark
- If the SAEs are retrained, exclude the test split (chunks ≥ 515) (Finding 7)
- InstaNovo v1.2.0 (instanovo-v1.2.0.ckpt) is now the default release model; the SAEs explain v1.1.0. At the next retraining, decide whether to retrain on v1.2.0 (check its preprocessing, since 1.2.x predict passes use_spectrum_utils=False), excluding the test split (Finding 7). The tool and dashboard must state the model version.

Paper: revisit once v1 and the rigour items are done.

## Findings (November 2026 review)

The SAEs are sound; do not retrain. All headline numbers in the released reports
reproduce exactly. Problems are downstream.

**1. LLM descriptions leak labels (critical; addressed in V1-2a/V1-2b, and
the dictionary uses blinded runs only).** `format_example_table` in
`interpret.py` puts `matched_ion` (the annotation result; 7 of 50 concepts), a
latent-token marker, neighbour labels (`mz@int/label`) and `peptide_proforma`
(the ground-truth peptide) in both the description and prediction prompts. Only
the optional `concepts` column is withheld. 53/160 concept-stratum summaries
mention immonium and 72 mention b/y ions; 128/160 discovery summaries say
"noise", and that stratum is selected on annotator-noise peaks. The describe/
held-out split is random by example (peptides repeat; held-out peaks can appear
as neighbours), and there is no baseline. With ~8 random zeros among ~25
examples, r is largely active-vs-zero separation. Discovery masses probably survive
(peak m/z is a legitimate input) but need showing. unblinded_v1 also had an
example_id side channel: `merge_global_top` appended the globally strongest
activations with the highest ids, so a high id could signal a strong example.
Measured on unblinded_v1's held-out sets the channel is weak (mean Spearman ρ
of id vs activation −0.003 / +0.033 / +0.043 / −0.005 at L2/4/6/8), since few
global-top rows fall in a 12-of-625 sample; in a synthetic test where merged
rows are present it is ρ ≈ +0.5. Fixed in V1-2a by a seeded id permutation and
random row order.

**2. Hub features dominate the group ablations (high).** Single-feature ΔCE is
usually ~0.001–0.003, but: L2 F7522 +0.100, F11907 +0.067, F12077 +0.033;
L4 F11376 +0.057; L6 F9282 +0.034, F10689 +0.023; L8 F2481 +0.018. Table 7's
extremes are hub membership (`has_neutral_loss` L2 −77.95 and
`top_decile_intensity` L4 −35.77 contain F11907; `is_y_ion` L4 +6.57 contains
F11376). The thesis's y-ion L2 result was F7522 alone. σ from 5 controls is
~1e-4, so one hub gives |z| of 10–270. F11376 (L4) matches F12077 (0.85) and
F7522 (0.82) at L2. Phase 8 ablates in the SAE-patched model (reconstruction
error dropped), which is consistent for the paper but not what the tool needs.

**3. Unsupported claims (medium).** README rank correlations
(+0.04/+0.53/+0.13/+0.06) have no code; recomputation gives roughly
+0.23/−0.15/−0.05/+0.17 (README corrected in V1-3). Immonium null holds at L2 only (L4 sel_z −6.0). The
causal description stratum is absent. 143.118 as acetyl-lysine is unexplained
(no acetylation in the labels); 199.18 recurs in 11 discovery descriptions.
Lead only: 143.118 → 157.134 is +CH₂, 157.134 → 173.129 is +O.
Depth ordering rests on argmax over near-ties (deamidation is flat but counts
as "layer 2"; "ion series identity" includes immonium, a single-peak property). Layer-6 positive control
sel_z = 1.97.

**4. Annotation gaps (medium).** No immonium ions for modified residues;
`max_isotope=0` sends isotope peaks to "noise".

**5. Phase 7/8 teacher-forced the bare sequence (medium; found in V1-4a).**
The CE cached at extraction, which Phase 7's loader matched to 5e-6, equals
teacher forcing on the dataset's bare `sequence` column (≤ 7.4e-6), not on
`modified_sequence`: the loader feeds `sequence` to the processor, and the HF
parquet's `sequence` is unmodified. For a modified spectrum the CE on the true
peptide differs by up to 4.17 (deamidated, chunk 0 spectrum 4) and 2.27
(oxidised, spectrum 6) nats per token. Of Phase 8's 5,000 spectra (chunks
0–4), the local chunks 0–1 hold 2,048, of which 208 (10.2%; 132/1,024 and
76/1,024) carry a modification, so ce_clean, loss recovered and Phase 8 ΔCE
include wrong targets for them. The dictionary's `causal_provenance` says so
(2026-11-22). Encoder activations and annotations are unaffected
(the encoder never sees the peptide; annotation used the ProForma string). The
inspector teacher-forces the predicted peptide with its modifications.

**6. N-terminal ProForma (low locally; found in V1-4a).** `_to_proforma` turns
"(+42.01)PEP…" into "[+42.01]PEP…"; spectrum_utils needs "[+42.01]-PEP…" and
fails, which the annotator turns into an all-noise spectrum with a warning. No
local spectrum has an N-terminal mod; InstaNovo does predict them (e.g.
`[UNIMOD:385]` on chunk 0 spectrum 4), so the inspector writes the hyphen.

**7. Every test-split chunk (515–624; 515 is mixed) was SAE training data at
all four layers, for 3 epochs (found 2026-11-22).** `run_pipeline.sh:584-588`
merges the HF splits in order (`concatenate_datasets([dd[k] for k in
dd.keys()])`), and
`sae_pipeline_outputs/logs/run1_layers2-4/prepare_dataset.log:48-49` gives
train 499,402, validation 28,572, test 111,312, so the test split is merged
rows 527,974–639,285. `extract.py:137` writes 1,024 spectra per chunk, in
dataset order (checked at `extract.py:559`); the extract logs
(`sae_pipeline_outputs/logs/run*/extract.log`) end "Wrote final chunk 624
(n_spectra=310)" (624 · 1,024 + 310 = 639,286) and
`sae_pipeline_outputs/extract/manifest_run{1,3,4}.json` have `n_chunks` 625.
Chunk 515 holds rows 527,360–528,383 (410 of them test); chunks 516–624 are
all test. `train.py:775-776` splits the manifest's chunks with `_split_chunks`
(`train.py:596-600`: round(0.02 · 625) = 12 held out by `random.Random(seed)`;
`sae_pipeline_outputs/sae/layer_L/seed_0/config.json`: `val_fraction` 0.02,
seed 0, `n_epochs` 3). Every
`sae_pipeline_outputs/logs/run*/train_layer{2,4,6,8}_seed0.log` starts "613
train chunks, 12 val chunks", and
recomputing the split gives val chunks 17, 116, 135, 141, 230, 292, 316, 397,
448, 462, 489, 495, all below 515. The exposure is unsupervised (activations
only, no labels or peptides), but no test spectrum is unseen by the SAEs, so
any result on the test split must state it.

## Log

- 2026-11-20 — Review of paper, thesis and repo; findings above.
- 2026-11-21 — Session 0: environment and data verified; nothing changed.
- 2026-11-21 — V1-1 done (`14a7654`): `src/analysis.py`, `tests/test_analysis.py` (16 pass), outputs `analysis/hubs.csv` and `analysis/f1dom_selectivity.csv`.
  Finding 2 confirmed (all named hubs flagged, same ΔCE; top hub per layer F7522/F11376/F11907/F2481; has_neutral_loss L2, top_decile_intensity L4 contain F11907, is_y_ion L4 contains F11376).
  Finding 3 reproduced with concept-level definition (best ablated F1dom vs group selectivity, n=35, 15 NaN dropped): +0.230/−0.148/−0.052/+0.168; all p>0.18. README numbers not reproduced.
  Hub threshold uses median |ΔCE| (signed L8 median is negative) → 11/13/7/10 hubs at L2/4/6/8.
  Run tests with `PYTHONPATH=src uv run python -m pytest tests` (`uv run pytest` fails on Windows: trampoline error).
- 2026-11-21 — STATUS revised: V1-2 adds detection AUROC; V1-3 spec rewritten (no hub tiers or trust level); V1-4 (since split into V1-4a/V1-4b) gets per-spectrum AtP + exact error-preserving ablation; new V1.1-1 (global AtP importance; then called V1-4b).
- 2026-11-21 — V1-3 done (`f6fa694`): `src/build_dictionary.py`, `tests/test_build_dictionary.py` (42 tests pass in total), `dictionary/layer_{2,4,6,8}.parquet`.
  Live features 6,962 / 7,474 / 7,811 / 8,732; Phase 8 delta-CE measured for 121 / 141 / 141 / 162 (rest "untested"); 80 descriptions per layer, all `unblinded_v1` with the label-access provenance.
  Existing descriptions moved to `interpret/layer_L/unblinded_v1/` (nothing in `src/` read the old path; `run_pipeline.sh` still writes the old path, see V1-2).
  Hub threshold and flag removed from `analysis.py`; `hubs.csv` is a ranked list (columns `hub_threshold`, `is_hub` gone). Finding 2 features still rank 1-3.
  README fixed: correlations now +0.23/-0.15/-0.05/+0.17 (test pins README against the code); label-leak sentence corrected.
  Notes: recall needs concept token counts, recovered exactly from stored F1 (agree to 4e-5, float32 rounding), the build fails if they disagree. `phase8_delta_ce` is null (not NaN) when untested. A feature only in a group ablation is untested individually. Added `unblinded_description_provenance` column so the provenance travels with the secondary text. `description_shuffled_null_r` / `description_beats_baseline` are all null until V1-2.
  Still unsupported in the README: "Only the structural positive control is causally significant at all of them" (backlog), and the "described better than concept features" claim (r = 0.806 vs 0.648) rests on label-leaking descriptions (Finding 1); not edited here.
- 2026-11-21 — V1-2a done (code + dry runs; no API calls). 84 tests pass (new `tests/test_interpret.py`, `tests/test_chunk_stream.py`).
  Pre-checks: `--strata concept unexplained` reproduces unblinded_v1's 80 features per layer exactly (default strata now add `causal`, since Phase 8 is local). Chunks: v1 used 0, 52, …, 572; locally L6/L8 spread to the same, L2/L4 (13 chunks) would pick chunk 1 and drop 572, hence `--chunk-indices`. Key: `OPENAI_API_KEY` in root `.env` (not exported; interpret.py defaults to `src/.env`, so pass `--env-file .env`).
  Blocker fixed: local `extract/manifest.json` is a consolidated 13-chunk manifest; `ChunkStream` required equal chunk counts and indexed by position. It now joins a consolidated subset by chunk id and skips chunks lacking the layer; the full layout is unchanged (tested).
  Cost-estimate constants calibrated on unblinded_v1: 2.64 chars/prompt token, 296 completion tokens/feature, 860-char descriptions.
  Dry runs (80 features prepared per layer, `--n-shuffles 3`, 400 calls per layer-run): unblinded_v2 $2.45 / $2.42 / $2.40 / $2.39, blind_v1 $2.09 / $2.06 / $2.05 / $2.04 at L2/4/6/8 → ≈ $17.90 total for 3,200 calls (gpt-4o-2024-08-06 at $2.50 / $10 per M). Blind and unblinded prompts show identical describe and held-out rows (ids, m/z, order) on every layer. Dry run ≈ 5 min per layer-run (encode).
  STATUS restructured: v1 = V1-2, V1-4 (now V1-4a/V1-4b), V1-5 (the dashboard, now V1-6); global importance and descriptions at scale (now V1.1-1, V1.1-2) moved to v1.1. CLAUDE.md extract/ line corrected.
- 2026-11-21 — V1-2b done for layers 2/4/6 (`900a295`); 100 tests pass. Spend $14.22 of the $20 cap (each finished run within 3% of its dry-run estimate); the OpenAI credits ran out during layer 8 (all failures `429 credit_balance_exhausted`), so layer 8 moved to V1-2c (~$3.84 more).
  Shuffled-null fix: 98/107/113/119/110/127 of 240 shuffled r per run (L2/L4/L6 × unblinded_v2/blind_v1) were NaN from constant predictions, all with full held-out coverage, so reclassified exactly to 0; no feature's own r was NaN. Null p95 0.785→0.739, 0.856→0.798, 0.811→0.780, 0.804→0.688, 0.713→0.555, 0.678→0.584; beats baseline (all 80) 35→39, 19→30, 26→34, 28→45, 37→53, 49→55. One held-out set (L2, both runs) has no zeros, so its AUROC is undefined. Resume logs byte-identical after rescoring; the pre-fix CSVs were backed up only to a session temp dir.
  Dictionary rebuilt with blinded descriptions only: 80 described at L2/L4/L6 (beats baseline 30/45/55), 0 at L8. unblinded_v1 moved to `interpret/layer_L/archive/unblinded_v1/`; `schema.check_run_tag` rejects `archive` and path-like tags in interpret, analysis and build_dictionary. `build_dictionary --unblinded-run` now defaults to none.
  Hard-negative and per-feature-null items added to the backlog.
- 2026-11-21 — STATUS restructured: V1-4 split into V1-4a (forward pass) and V1-4b (attribution patching and verify); E1 added after V1-4b; V1-5 is now Data for the tool and the dashboard V1-6; v1.1 items renamed V1.1-1/V1.1-2 (text unchanged) and old-ID references updated. Demo-data question answered: the HF cache holds all four splits (train 499,402, validation 28,572, test 111,312; merged in that order, spectrum_id = row), and chunk 520 rows 0–1 equal HF test rows 4,506–4,507 (peptide, precursor m/z, charge); V1-5 keeps the formal check.
- 2026-11-21 — V1-4a done (`2a3166c`): `src/inspector/{runtime,dictionary}.py`, `tests/test_inspector_runtime.py`; 123 tests pass (23 new). Results and timings under V1-4a; refactors under Behaviour changes. Findings 5 (Phase 7/8 CE used bare-sequence targets) and 6 (N-terminal ProForma) added. `use_spectrum_utils=True` is the v1.1.0 checkpoint's standard input (InstaNovo 1.1.0 always used spectrum_utils; only 1.2.x's predict.py passes False), a recorded parameter, not a deviation. The repo checkpoint is truncated; tests and timings used `~/.cache/instanovo/instanovo-v1.1.0.ckpt` (backlog). InstaNovo on the first 40 chunk-0 spectra: 18 correct up to I/L; wrong predictions can miss the precursor by > 10⁴ ppm (beam search keeps them), which the output reports as `precursor_mz_error_ppm`. Beam search uses `max_length` from the checkpoint config (40).
- 2026-11-22 — V1-4a follow-ups. (1) Dictionary `causal_provenance` for measured features now ends "targets ignore modifications for ~10% of spectra (Finding 5)" (`0522bd4`): ~10%, not 13%, because 13% was chunk 0 alone; over the local part of Phase 8's spectra (chunks 0–1) 208/2,048 = 10.2% are modified. All four layers rebuilt, counts unchanged; Phase 7/8 rerun moved to the GPU backlog. (2) Accuracy check, `python -m inspector.runtime accuracy --parquet <HF test> --n 200 --seed 0` (200 random yeast test spectra, 48 with a modified true peptide): our 5-beam BeamSearchDecoder 123/200 exact (modified_sequence in InstaNovo tokens, I=L) = 0.615 [Wilson 95% CI 0.546, 0.680]; InstaNovo's own peptide recall (Metrics, mass-matched residues) 0.635, aa recall 0.646; 167/200 meet the precursor (min |ppm| over isotope 0–1 ≤ 20, InstaNovo's filter_precursor_ppm) and every exact prediction is among them; exact 106/152 unmodified vs 17/48 modified; median 4.2 s/spectrum. Greedy (InstaNovo 1.1.0's predict.py default, num_beams 1, before InstaNovo+ refinement): 0.570 [0.501, 0.637], recall 0.590, 142/200 meet precursor, 1.0 s. Published: InstaNovo v1.1 yeast peptide accuracy 0.618 at full coverage, knapsack beam search, zero-shot (instadeepai.github.io/InstaNovo/explanation/performance; I/L treatment not stated; InstaNovo+ refined 0.645). Our setup is in line with it, so no stop. Not run: knapsack beam search (chart 40,000,001 × 33 bools plus a pure-Python DFS over up to 40M masses; too much for this machine). Settings vs 1.1.0: 1.1.0 ships its own older BeamSearchDecoder; we run 1.2.2's with the 1.1.0 weights; max_length 40 in both; 1.1.0 preprocesses with spectrum_utils (`SpectrumDataset._process_peaks(use_sus_preprocess=True)`), as we do. Some true peptides exceed max_length 40 and cannot be predicted. (3) `checkpoints/instanovo_v1.1.0.ckpt` replaced with the full release asset: 1,136,412,059 bytes (GitHub's listed size; no published digest), sha256 c75d1f80fcf912031ed348ffdef4894acc09fa8142d13fec2c269a2f48df564b, pinned in `test_default_checkpoint_is_the_v1_1_0_release`. CLI and `Inspector` default to it, anchored at the repo root; tests no longer fall back to InstaNovo's cache. (4) Slim JSON (schema 2) is V1-4b's first step. 127 tests pass.
- 2026-11-22 — STATUS moved from `docs/STATUS.md` to the repo root; `STATUS.md` and `CLAUDE.md` are now committed (repo is public; checked for secrets), `docs/` stays gitignored. E1 moved ahead of V1-4b and specified (strong baseline, raw-activation control, peptide-grouped CV, paired AUROC differences). SAE-fit calibration added to V1-5; the backlog's corpus-FVE item folded into it.
- 2026-11-22 — E1 planned, then moved to v1.1 as V1.1-3 (pre-specified, for the paper; the tool's v1 does not depend on it) with a revised design: Winnow baseline, top-32 class-mean-difference selection, random-encoder control S_rand in place of the raw-activation block, gradient boosting. Beam margin is available from the decoder (`return_beam`, already passed). Finding 7 added (every test-split chunk was SAE training data, with file and log evidence); V1-4c (audit case study) added after V1-4b; two cluster backlog items (exclude the test split at retraining; v1.2.0 is now InstaNovo's default model). Housekeeping: CLAUDE.md and STATUS.md tracked, `docs/` stays ignored (`4cea931`). No code changed.
