# Status

Update this file at the end of every session: tick tasks, add a line to the log,
move new ideas to the backlog.

## v1 — a small, solid, working tool

Done when: a scientist can load a benchmark spectrum in the dashboard, see
InstaNovo's prediction, inspect the active features on any peak at any layer
with honest labels and evidence indicators (association, causal and
description coverage, with their effect sizes), and ablate a feature to see the effect
on each predicted residue.

Open, in order: V1-5, V1-6, V1-2c (when API credit is
available), V1-7. Done: V1-1, V1-2a, V1-2b (layers 2/4/6), V1-3, V1-3b, V1-4a, V1-4b, V1-4c.

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
- [x] **V1-3b Concept reliability** (`1ffa8ea`, `31338d9`) — make the dictionary honest about the
      concepts Finding 9 found unreliable, without rebuilding labels. No API,
      GPU or cluster.
      Part A, reliability table (`analysis.concept_reliability`, written by
      `python -m analysis --root sae_pipeline_outputs --label-audit` as
      `analysis/label_audit/concept_reliability.csv`; the audit is unchanged
      except two new columns of `concepts.csv`, `immonium_fix_lost` and
      `immonium_fix_gained`: the concept positives lost and gained on each
      absent-residue immonium peak when such immonium candidates are dropped
      and the next candidate wins). Rerun: reproduction exact on the same 200
      spectra; every other audit file and every existing `concepts.csv` column
      byte-identical; `immonium_fix_lost` equals `immonium_absent_residue` on
      all covers_*.
      What is stored, checked before building: `feature_label_associations.csv`
      holds every BH-significant (feature, concept) pair and no other
      (`evaluate.py:2969-2980`; BH over all 12,288 × 50 pairs at q 0.05,
      `:1169-1171`, `:173`): 283,454 / 316,812 / 331,398 / 397,842 rows at
      L2/4/6/8, all 50 concepts present. `per_feature_stats.csv` keeps only
      each feature's best significant concept (`:3010-3049`). F1dom > 0 implies
      enrichment (`:281-286`), so the dictionary's top 3 (significant,
      F1dom > 0) cannot hold a depleted pair. Phase 4 streamed all 625 cluster
      chunks: its ChunkStream has no `chunk_indices` (`:2735-2745`, `:439-440`)
      and report.json `n_tokens` 67,503,495 equals the 625-chunk manifests
      (`extract/manifest_run{1,3,4}.json`). The chance fractions come from the
      audit's 200 local spectra (13 chunks), a property of each concept's labels;
      both sources are recorded in the dictionary metadata.
      Record per concept: dependence on the peptide, chance fraction with its
      bootstrap 95% CI, positives that a bug adds (internal end + absent-residue
      immonium; wrong labels) and positives the immonium bug hides (a y ion with
      a loss outranked by an immonium ion without one: the labelled positives
      are still right, so reported, not counted), the share of positives on
      composition-determined ions, display name, reasons and notes.
      Display rule (`analysis.DISPLAY_RULE`): chance fraction (point estimate)
      ≤ 0.2, no bug-added positives, peptide-dependent. It is a display rule, not
      a test. Why 0.2: at most about one in five positives expected from a random
      same-composition sequence; the cut Finding 9 used; it falls in the gap
      between the b/y position and cleavage concepts (0.09–0.18) and internal
      fragments, losses and most covers_* (0.23–0.64). Concepts near it (e.g.
      cleaves_C_to_Arg 0.203 [0.104, 0.364], covers_C 0.190 [0.140, 0.259]) can
      fall either side by chance, so the chance fraction and CI are always
      shown.
      Finding 9's "reliable" list checked against the bugs: is_y_ion, is_b_ion,
      the three position_* and cleaves_N_to_Pro / C_to_Asp / C_to_Glu have no
      bug-added positives; is_y_ion and position_Cterm each have one hidden y
      positive (1 / 2,846 and 1 / 1,292). Touched: covers_R 56 + 4 of 1,467
      (4.1%), covers_C 7 of 163 (4.3%), ion_contains_cam_cys 7 of 162 (4.3%);
      they fail the rule. is_I_ion and is_first_ion are re-documented as m/z
      windows (display names; the keys stay, since they are in the annotation
      vocabulary, the Phase 4 CSVs, `concept_phi.pt` and the description
      prompts). 8 concepts meet the rule:

      | concept | dependence | chance [95% CI] | bug adds (int. end + imm.) / positives | hidden | determined | rule |
      |---|---|---|---|---|---|---|
      | is_b_ion | dep. | 0.177 [0.161, 0.196] | 0 |  | 2% | yes |
      | is_y_ion | dep. | 0.109 [0.101, 0.117] | 0 | 1 | 14% | yes |
      | is_I_ion | m/z window | — | 0 + 398 / 706 (56.4%) |  | 100% |  |
      | is_internal_fragment | dep. | 0.535 [0.503, 0.570] | 0 | 1 | 0% |  |
      | is_precursor_related | comp.-determined | — | 0 |  | 100% |  |
      | is_matched_peak | dep. | 0.307 [0.291, 0.323] | 0 + 396 / 8020 (4.9%) |  | 15% |  |
      | is_noise_peak | complement | 1.327 [1.290, 1.367] | 0 | 396 | 0% |  |
      | is_latent_token | structural | — | 0 |  | — |  |
      | is_H2O_loss | dep. | 0.379 [0.355, 0.406] | 0 + 1 / 1601 (0.1%) | 1 | 9% |  |
      | is_NH3_loss | dep. | 0.432 [0.385, 0.484] | 0 + 9 / 989 (0.9%) | 1 | 16% |  |
      | has_neutral_loss | dep. | 0.398 [0.371, 0.427] | 0 + 10 / 2590 (0.4%) | 2 | 12% |  |
      | is_fragment_charge_1 | dep. | 0.286 [0.273, 0.301] | 0 + 397 / 7666 (5.2%) |  | 15% |  |
      | is_fragment_charge_2 | dep. | 0.643 [0.564, 0.732] | 0 | 1 | 0% |  |
      | precursor_charge_2 | not label | — | 0 |  | — |  |
      | precursor_charge_3 | not label | — | 0 |  | — |  |
      | precursor_charge_4 | not label | — | 0 |  | — |  |
      | position_Nterm | dep. | 0.155 [0.141, 0.169] | 0 |  | 0% | yes |
      | position_middle | dep. | 0.092 [0.083, 0.101] | 0 |  | 0% | yes |
      | position_Cterm | dep. | 0.183 [0.168, 0.200] | 0 | 1 | 33% | yes |
      | mz_low | not label | — | 0 |  | — |  |
      | mz_mid | not label | — | 0 |  | — |  |
      | mz_high | not label | — | 0 |  | — |  |
      | top_decile_intensity | not label | — | 0 |  | — |  |
      | is_first_ion | m/z window | 0.830 [0.576, 1.441] | 0 | 1 | 95% |  |
      | is_last_ion | dep. | 0.222 [0.188, 0.265] | 0 |  | 18% |  |
      | cleaves_C_to_Lys | dep. | 0.241 [0.180, 0.326] | 0 |  | 21% |  |
      | cleaves_C_to_Arg | dep. | 0.203 [0.104, 0.364] | 0 |  | 0% |  |
      | cleaves_N_to_Pro | dep. | 0.101 [0.082, 0.123] | 0 |  | 0% | yes |
      | cleaves_C_to_Asp | dep. | 0.108 [0.090, 0.130] | 0 |  | 6% | yes |
      | cleaves_C_to_Glu | dep. | 0.124 [0.104, 0.149] | 0 |  | 9% | yes |
      | covers_K | dep. | 0.228 [0.206, 0.250] | 166 + 4 / 2565 (6.6%) |  | 13% |  |
      | covers_R | dep. | 0.172 [0.152, 0.196] | 56 + 4 / 1467 (4.1%) | 1 | 10% |  |
      | covers_W | dep. | 0.229 [0.189, 0.281] | 33 + 31 / 401 (16.0%) |  | 15% |  |
      | covers_F | dep. | 0.298 [0.266, 0.337] | 101 + 87 / 1348 (13.9%) |  | 13% |  |
      | covers_Y | dep. | 0.277 [0.245, 0.313] | 91 + 100 / 1294 (14.8%) |  | 14% |  |
      | covers_P | dep. | 0.255 [0.226, 0.288] | 139 + 0 / 1731 (8.0%) | 1 | 2% |  |
      | covers_M | dep. | 0.305 [0.255, 0.371] | 37 + 22 / 702 (8.4%) |  | 9% |  |
      | covers_C | dep. | 0.190 [0.140, 0.259] | 7 + 0 / 163 (4.3%) |  | 2% |  |
      | covers_D | dep. | 0.280 [0.255, 0.307] | 79 + 1 / 2288 (3.5%) |  | 2% |  |
      | covers_E | dep. | 0.261 [0.243, 0.281] | 136 + 19 / 2941 (5.3%) |  | 4% |  |
      | covers_N | dep. | 0.275 [0.246, 0.305] | 73 + 2 / 1778 (4.2%) |  | 2% |  |
      | peptide_length_short | not label | — | 0 |  | — |  |
      | peptide_length_medium | not label | — | 0 |  | — |  |
      | peptide_length_long | not label | — | 0 |  | — |  |
      | ion_contains_oxidation | dep. | 0.498 [0.261, 0.814] | 0 |  | 3% |  |
      | ion_contains_cam_cys | dep. | 0.191 [0.141, 0.263] | 7 + 0 / 162 (4.3%) |  | 2% |  |
      | ion_contains_deamidation | dep. | 0.321 [0.231, 0.446] | 7 + 0 / 132 (5.3%) |  | 2% |  |
      | spectrum_contains_oxidation | not label | — | 0 |  | — |  |
      | spectrum_contains_cam_cys | not label | — | 0 |  | — |  |
      | spectrum_contains_deamidation | not label | — | 0 |  | — |  |

      dep. = peptide-dependent; m/z window = peptide-independent; comp.-determined
      = fixed by the precursor m/z; complement = of is_matched_peak (decoys match
      fewer peaks, so its chance fraction exceeds 1); determined = share of
      positives on immonium, precursor, y1 or b_{L−1} ions. The deamidation
      concepts carry a note: part of the local deamidated labels fit a ¹³C
      misassignment better (Finding 8(d)). Chance fractions exclude
      composition-determined ions, so is_first_ion's 0.830 is its small b1
      remainder.
      Part B, dictionary (schema 2; `build_dictionary` reads the reliability
      table, required, default `analysis/label_audit/concept_reliability.csv`,
      `--concept-reliability` to override). Existing columns unchanged; beside
      each top concept `chance_fraction_k` and `meets_display_rule_k`; new
      `n_reliable_associations` and `reliable_concept_k` / `reliable_f1_dom_k` /
      `reliable_precision_k` / `reliable_recall_k` / `reliable_lift_k` /
      `reliable_chance_fraction_k` (k = 1–3: the same ranking as the top 3,
      restricted to the 8 concepts meeting the rule; null `reliable_concept_1`
      = "no reliable association recorded"). Metadata: `concept_reliability`
      (the 50 records as JSON, with display names), `display_rule`,
      `concept_reliability_source` (path, sha256) and `association_source`.
      `inspector.dictionary`: summaries add `top_concept_chance_fraction`,
      `top_concept_meets_display_rule`, `reliable_concept` and
      `reliable_concept_f1_dom` (Explanation schema 3); `concept_reliability()`
      and `display_name()` read the metadata. Rebuilt all four layers
      (`--layers 2 4 6 --blinded-run blind_v1`, then `--layers 8`): every
      pre-existing column identical to the schema-1 files; metadata differs only
      in `schema_version` plus the new keys. Live features with a reliable
      association 4,078 / 4,278 / 4,455 / 5,491 of 6,962 / 7,474 / 7,811 / 8,732
      at L2/4/6/8; top-3 entries meeting the rule 2,239 / 2,543 / 2,986 / 4,347.
      These associations are weak (Finding 10). `python -m analysis --root
      sae_pipeline_outputs --reliable-associations` lists the features whose best
      reliable F1dom is ≥ 0.1 (a listing cut set for later use, demo set and
      feature cards; not a reliability claim): `analysis/reliable_associations.csv`
      (44 / 47 / 62 / 97 rows; reliable concepts, causal and description
      fields) and `reliable_associations_summary.json`. Of the 250 listed, 145
      have a Phase 8 ΔCE and 45 a description; the six ≥ 0.3 are is_y_ion
      except L8 F5107 (cleaves_N_to_Pro), and include the Finding 2 hubs L4
      F11376 and L6 F9282 / F10689.
- [x] **V1-4a Runtime forward pass** (`2a3166c`) — `src/inspector/runtime.py`,
      `dictionary.py`. Input: a raw spectrum (m/z array, intensity array,
      precursor m/z, charge), preprocessed by InstaNovo's own pipeline via
      `instanovo_io`, not the extraction's processed peaks. Preprocessing uses
      spectrum_utils (`use_spectrum_utils=True`, a parameter recorded in
      provenance): the standard input for the v1.1.0 checkpoint, matching
      extraction and InstaNovo 1.1.0's own training and prediction (in 1.1.0
      `SpectrumDataset._process_peaks` always used spectrum_utils); only
      InstaNovo 1.2.x's `predict.py` uses False. `explain(spectrum,
      attribution=False)` (since V1-4b `latent_ablation=False`): beam-search prediction (max_length from the model
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
- [x] **V1-4b Slim JSON and exact ablation** (revised 2026-11-24: no
      attribution patching in the tool; the AtP study moved to V1.1-1, where
      the planning preview is recorded). First step: slim the Explanation JSON
      as `EXPLAIN_SCHEMA_VERSION` 2. Each listed feature carries a per-feature
      summary instead of its full dictionary row (V1-4a measured 0.8–1.9 MB per
      spectrum, 85% full rows); the dashboard reads full rows from the
      dictionary it already loads, checked by the dictionary file's hash.
      Then exact error-preserving ablation (subtract a·W_dec[f] from the layer
      output, rerun teacher-forced; the SAE error term is kept) in two scopes:
      local = the feature at one token only; global = the feature at every
      token where it fires. Both report Δ log p per residue and in total
      (residues + EOS), labelled with their scope. Per-peak view: local
      ablation of every active feature at the selected token, for the selected
      layer (not just the top-k listed), sorted by |delta_total|, activation
      shown. Latent panel: the same at token 0, labelled "latent token only";
      no ranking proxy and no top-N cut. Global stays available through
      `ablate` (e.g. a feature card's "remove everywhere"). `ablate_token`
      also runs one joint ablation (all active features at the token removed
      together) and returns `joint_delta_total`, per-residue joint deltas and
      `sum_of_singles`, labelled "joint = what the SAE-explained content at
      this token contributes; a joint effect far from the sum of singles
      indicates redundancy or interaction". `ablation_context` takes an
      optional target peptide (default: the prediction) to teacher-force
      instead; results record the target used, and the provenance string
      follows it (for V1-4c's Δ[log p(pred) − log p(true)]). UI caveat: a
      single-feature effect understates the importance of redundant features;
      small means "removing this alone changes little". Timings for both
      scopes and per-click times for both panels on 3 spectra. Tests on one
      spectrum. For the record: the SAEs' decoder rows have unit norm
      (`train.py:17`; the checkpoints record 0.99999976–1.0000002 at all four
      layers), so a ranking by activation × decoder norm would equal ranking
      by activation.
      What exact ablation measures, and its limits: it removes only the
      SAE-captured component a·W_dec[f]. SAE shrinkage can leave part of a
      concept's direction in the error term, which is kept, so effects are
      lower bounds for the concept. Single-feature edits can produce activation
      combinations the model rarely sees. A feature may be one part of a split
      concept, or a concept may be absorbed into another feature, so a small
      single-feature effect does not mean the concept is unimportant.
      Done: `inspector/runtime.py` (`ablation_context`, `ablate`,
      `ablate_token`, `explain(latent_ablation=)`, `feature_row`; the per-call
      container is `AblationReport`), `inspector/dictionary.py`
      (`SUMMARY_FIELDS`, `summary`, `sha256_16`, `causal_provenance`). Joint
      ablation keeps b_dec (the SAE's "nothing active" state), so at the token
      x′ − b_dec = x − x̂, matching the single ablations; the test asserts
      that. `latent_ablation` defaults to False so `explain()` costs what it
      did. Results (`python -m inspector.runtime time --ablation`, HF test rows
      4506–4508 = 35 / 152 / 114 peaks, CPU, 6 threads):
      JSON without latent ablation 325 / 871 / 766 KB (V1-4a 0.78 / 1.92 /
      1.76 MB, so 55–58% smaller); features block 77% / 66%, ~406 B per
      summary (mostly key names; the planning estimate of ~200 B was low),
      so the next lever, not taken, is a compact encoding. With latent ablation
      421 / 891 / 785 KB. L0 at the latent token 25–33 / 5–9 / 4–9 per layer.
      Timings: one ablation (either scope) 0.10–0.11 / 0.20–0.22 /
      0.17–0.19 s; ablation context 0.19 / 0.34 / 0.27 s; per-peak click (all
      active features at one token, one layer, plus the joint) 1.0–1.7 /
      3.8–6.5 / 2.5–5.0 s (L0 31–62, about 0.1 s per feature, batched 32 + 1);
      latent click, one layer, 0.76–1.04 / 0.71–1.14 / 0.44–0.83 s; all four
      latent layers 3.8 / 4.0 / 3.0 s, so `explain(latent_ablation=True)` takes
      5.2–6.8 s warm. Interactive: single ablations, the latent panel, and the
      per-peak view on small spectra; per-peak clicks on ~150-peak spectra take
      4–6.5 s (borderline). Exhaustive ablation stays offline (V1-5). Joint vs
      sum of singles differ widely, e.g. row 4506 L2 max-L0 peak: joint −3.39 vs
      singles −0.69; L8 latent: −1.36 vs −0.10; row 4508 L8 latent: +0.79 vs
      +1.93, consistent with redundancy and interaction. Caveat: removing
      every feature at a token is a large edit that can produce states the
      model rarely sees; the joint figure is a reference scale, not a clean
      total. 144 tests pass (14 new).
- [x] **V1-4c Audit case study** — pick 3–5 confidently wrong predictions
      from the V1-4a accuracy run (seed 0); use the inspector to form a
      hypothesis for each failure; test it by exact ablation; record what
      held and what didn't. Metric: Δ[log p(pred) − log p(true)], from two
      contexts, `ablation_context(spec, exp)` and
      `ablation_context(spec, exp, target=modified_sequence)` (V1-4b), so it
      shows whether removing a feature moves the model towards the truth.
      Candidates feed the V1-5 demo set.
      Part A, case selection (2026-11-24, `python -m inspector.runtime audit-cases
      --parquet <HF test> --n 200 --seed 0`; outputs `analysis/v1_4c/`:
      predictions.csv, candidates.csv, selection.json, explanations/, case_views/).
      Fixed rule (`runtime.AUDIT_SELECTION_RULE`): beam prediction wrong (I=L,
      modified_sequence) and meeting the precursor (min |ppm| over isotopes 0–1
      ≤ 20); ranked by mean teacher-forced log p over the predicted residues
      (EOS excluded); top 5, then the next-ranked "evidence for true" cases until
      2 selected cases are. Each candidate's ≤ 200 processed peaks (the model's
      input) are annotated against both peptides (`annotate_spectrum`, strict,
      20 ppm); a distinguishing peak is labelled a fragment (b, y, immonium,
      internal; not precursor or noise) under one peptide and not the other, one
      label per peak. Classes: amb = none; true = ≥ 1 peak a fragment only under
      the true peptide; pred = distinguishing peaks only under the prediction
      (reported as possible label errors, not model failures). Region = the span
      between the longest common prefix and suffix (I=L, no overlap); type in
      order: length change, modification (same base residues), swap (same
      multiset), substitution, other (mixed). Reproduction check: the regenerated
      beam predictions give exact 123/200 and 167 meeting the precursor (V1-4a: 123,
      167), so 44 candidates; beam search 912 s. Selected: 19475, 51238, 109208, 29350, 60182, 29977
      (top 5 hold one "evidence for true", 109208; 29977 added at rank 6;
      shortfall 0). Of 44 candidates: 7 amb, 34 true, 3 pred; 18 substitution, 14
      length change, 5 swap, 4 other, 3 modification. A
      region Δm near −0.98 Da is a deamidation difference that passes the precursor
      filter via isotope offset 1. The console table at the end of the run failed
      to print (cp1252 cannot encode "Δ"; fixed) after every output file had been
      written, so the run was not repeated.

      | # | row | mean log p | type | Δm (Da) | class | pred-only (b/y) | true-only (b/y) | sel | prediction [region] / true [region] |
      |---|---|---|---|---|---|---|---|---|---|
      | 1 | 19475 | -0.012 | substitution | +0.000 | amb | 0 (0) | 0 (0) | top 5 | VNN[UNIMOD:7]AVVTVPAYFNDSQR [N[UNIMOD:7]] / VNDAVVTVPAYFNDSQR [D] |
      | 2 | 51238 | -0.019 | swap | +0.000 | pred | 5 (0) | 0 (0) | top 5 | VLSTIDANFADKYDEVK [VL] / IVSTIDANFADKYDEVK [IV] |
      | 3 | 109208 | -0.021 | modification | -0.984 | true | 25 (11) | 6 (3) | top 5 | TLSSVTQTTVEVDSLFDGEDFESSLTR [Q] / TISSVTQ[UNIMOD:7]TTVEVDSIFDGEDFESSITR [Q[UNIMOD:7]] |
      | 4 | 29350 | -0.025 | substitution | -0.000 | amb | 0 (0) | 0 (0) | top 5 | TGNLMAGHWVAISGAAGGLGSLAVQYAK [TG] / SANIMAGHWVAISGAAGGIGSIAVQYAK [SA] |
      | 5 | 60182 | -0.031 | substitution | +0.000 | amb | 0 (0) | 0 (0) | top 5 | DVTFIDDC[UNIMOD:4]VGPEVEAAVK [D] / DVTFIN[UNIMOD:7]DC[UNIMOD:4]VGPEVEAAVK [N[UNIMOD:7]] |
      | 6 | 29977 | -0.035 | substitution | +0.000 | true | 0 (0) | 1 (0) | added | WETALGPLATNAVTSLSLR [AV] / WETAIGPIATNGITSISIR [GI] |
      | 7 | 74557 | -0.040 | substitution | -0.984 | true | 9 (9) | 1 (1) |  | PEAAAAGAAAVAAAER [AAER] / PEAAAAGAAAVAN[UNIMOD:7]QGK [N[UNIMOD:7]QGK] |
      | 8 | 594 | -0.053 | swap | +0.000 | amb | 0 (0) | 0 (0) |  | LVAGGAAQNATR [AT] / IVAGGAAQNTAR [TA] |
      | 9 | 46143 | -0.055 | swap | +0.000 | true | 3 (3) | 9 (0) |  | VVDLVEHVAAK [AK] / VVDIVEHVAKA [KA] |
      | 10 | 43804 | -0.068 | substitution | -0.036 | true | 2 (0) | 19 (13) |  | DSQLEETQAHYTNLYVK [Q] / DSQIEETKAHYTNIYVK [K] |
      | 11 | 56090 | -0.073 | substitution | -0.011 | true | 26 (22) | 31 (25) |  | EPSTSVQFLSNLANK [EPSTSVQFLSNLANK] / NAFIGNTDINISTVR [NAFIGNTDINISTVR] |
      | 12 | 40875 | -0.075 | substitution | +0.000 | amb | 0 (0) | 0 (0) |  | MEKN[UNIMOD:7]STTGGSSVMLDGVK [N[UNIMOD:7]] / MEKDSTTGGSSVMIDGVK [D] |
      | 13 | 8647 | -0.075 | other | -0.984 | true | 18 (16) | 6 (5) |  | YKVDYPISLNDAQRK [NDAQRK] / YKVDYPISIN[UNIMOD:7]DAQKR [N[UNIMOD:7]DAQKR] |
      | 14 | 47026 | -0.077 | length change | +0.018 | true | 15 (12) | 3 (0) |  | [UNIMOD:5]WAVATSGTR [[UNIMOD:5]WAVATSGT] / YHVASVC[UNIMOD:4]R [YHVASVC[UNIMOD:4]] |
      | 15 | 84997 | -0.084 | substitution | -0.011 | true | 1 (0) | 1 (1) |  | AQLPAGSDGLMIDYSHGNSLEDFR [DGLMIDYSHGNSLE] / AQIPAGSN[UNIMOD:7]GIMIDYSHGNSNKDFR [N[UNIMOD:7]GIMIDYSHGNSNK] |
      | 16 | 71941 | -0.086 | modification | +0.000 | true | 3 (0) | 1 (0) |  | MLELM[UNIMOD:35]LPVFDAPQNLVEQAK [MLELM[UNIMOD:35]] / M[UNIMOD:35]IEIMIPVFDAPQNIVEQAK [M[UNIMOD:35]IEIM] |
      | 17 | 31366 | -0.095 | other | -0.984 | true | 7 (2) | 1 (1) |  | PANLPWGSSNVDIADISTGVFKELDTAQK [NLPWGSSNVDIADI] / PAN[UNIMOD:7]IPWGSSNVDIAIDSTGVFKEIDTAQK [N[UNIMOD:7]IPWGSSNVDIAID] |
      | 18 | 74658 | -0.100 | length change | +0.000 | amb | 0 (0) | 0 (0) |  | AAGGHSGGTIQVST [AGGHSGG] / AGAGHSNTIQVST [GAGHSN] |
      | 19 | 90372 | -0.105 | substitution | +0.000 | pred | 2 (0) | 0 (0) |  | TTPTLFDQDVLK [DQ] / TTPTIFENDVIK [EN] |
      | 20 | 109037 | -0.112 | substitution | -0.000 | true | 2 (1) | 3 (1) |  | VYEPILLVGIDKFTVNDLR [TVN] / VYEPIIIVGIDKFSNIDIR [SNI] |
      | 21 | 97297 | -0.118 | length change | +0.000 | true | 0 (0) | 3 (0) |  | VNQAIALLTLQR [Q] / VNQAIAIITIGAR [GA] |
      | 22 | 110855 | -0.129 | substitution | +0.000 | amb | 0 (0) | 0 (0) |  | GVPGVLPPSGELVFDVELVDVK [GE] / GVPGVIPPSADIVFDVEIVDVK [AD] |
      | 23 | 58900 | -0.160 | substitution | -0.000 | true | 0 (0) | 4 (1) |  | VGDNHEDENLSPAQAAELVGEDLSR [VGDNHE] / AEQGEHDENISPAQAAEIVGEDISR [AEQGEH] |
      | 24 | 33227 | -0.165 | swap | +0.000 | true | 2 (0) | 1 (0) |  | GMALNLEPGQVGIVLFSGDIVRK [SGDIVR] / GMAINIEPGQVGIVIFGSDRIVK [GSDRIV] |
      | 25 | 98900 | -0.168 | other | -0.025 | true | 5 (2) | 11 (5) |  | DLYANNVM[UNIMOD:35]SGGTTMFPGLAER [DLYANN] / EIYGN[UNIMOD:7]IVM[UNIMOD:35]SGGTTMFPGIAER [EIYGN[UNIMOD:7]I] |
      | 26 | 45595 | -0.169 | substitution | -0.995 | true | 20 (13) | 9 (2) |  | MSASDPESILDILEEPKQVK [ESI] / MSASDPN[UNIMOD:7]SKIDIIEEPKQVK [N[UNIMOD:7]SK] |
      | 27 | 63550 | -0.199 | length change | -0.000 | true | 1 (0) | 6 (2) |  | LIGAPNSSESVFEVTYFK [NSSE] / IIGAPSEGGSSVFEVTYFK [SEGGS] |
      | 28 | 81095 | -0.203 | substitution | -0.000 | true | 4 (2) | 4 (1) |  | SVENPFLPTNLER [TNL] / SVENPFIPAKEER [AKE] |
      | 29 | 39783 | -0.217 | other | -0.995 | true | 9 (8) | 8 (7) |  | SWSGDNEAHWEK [NEAHWEK] / SWSGDN[UNIMOD:7]EAHRWT [N[UNIMOD:7]EAHRWT] |
      | 30 | 72165 | -0.250 | substitution | -0.006 | true | 9 (1) | 12 (4) |  | AASTVPFVTWSVDSLFDGEDFESSLTR [AASTVPFVTWS] / TISSVTQTTVEVDSIFDGEDFESSITR [TISSVTQTTVE] |
      | 31 | 41739 | -0.258 | substitution | -0.992 | true | 17 (12) | 7 (6) |  | LMSDQTYLK [LMSDQTYLK] / TIYGDIDFR [TIYGDIDFR] |
      | 32 | 55965 | -0.289 | swap | +0.000 | pred | 1 (1) | 0 (0) |  | KSSETADLSELLPEEIEER [KSSETA] / SKASETDISEIIPEEIEER [SKASET] |
      | 33 | 106541 | -0.311 | substitution | -0.958 | true | 30 (18) | 41 (31) |  | QNVANPSLLIYAIC[UNIMOD:4]NSFGFGGVNTSLLFK [QNVANPSLLIYAIC[UNIMOD:4]NSFGFGGVNTSLLFK] / GQNVANPTAM[UNIMOD:35]IISSTIMINHIGINEYATR [GQNVANPTAM[UNIMOD:35]IISSTIMINHIGINEYATR] |
      | 34 | 42191 | -0.320 | modification | -0.984 | true | 7 (6) | 4 (4) |  | SLMDKVTNVR [N] / SIMDKVTN[UNIMOD:7]VR [N[UNIMOD:7]] |
      | 35 | 93419 | -0.331 | length change | -1.009 | true | 18 (9) | 45 (27) |  | ESDPYSHALVAQGVSQVVEVGTLR [ESDPYSHALVAQGVSQVVEVGTLR] / YFSESDSVIVAQGVSGAVVDKGSIK [YFSESDSVIVAQGVSGAVVDKGSIK] |
      | 36 | 99547 | -0.345 | length change | +0.000 | true | 5 (1) | 22 (18) |  | [UNIMOD:5]ELPGAVIIDQM[UNIMOD:35]QGM[UNIMOD:35]FNPEAHYFGTGR [[UNIMOD:5]ELPGAVIIDQM[UNIMOD:35]QGM[UNIMOD:35]F] / EIPGAVIIDQYNNM[UNIMOD:35]MNPEAHYFGTGR [EIPGAVIIDQYNNM[UNIMOD:35]M] |
      | 37 | 40105 | -0.470 | length change | -0.956 | true | 10 (4) | 24 (16) |  | EVVPNNASPGEVTLVDLKADVTGVEDANVK [GEVTLVDLKADVTGVEDANV] / EVVPNNASPASATGARPVSVSNGAANTEREK [ASATGARPVSVSNGAANTERE] |
      | 38 | 56794 | -0.496 | length change | +0.032 | true | 25 (10) | 31 (19) |  | AEAFGTHETTILREFKEDLEQFLK [EAFGTHETTILREFKEDLEQFL] / AAFGN[UNIMOD:7]C[UNIMOD:4]HGIYAGDIAIRPEIIAEHQK [AFGN[UNIMOD:7]C[UNIMOD:4]HGIYAGDIAIRPEIIAEHQ] |
      | 39 | 105288 | -0.650 | length change | +0.049 | true | 18 (5) | 18 (7) |  | [UNIMOD:385]QATGGQAPSSSPTSLTFSLSTLGSDPIDPTSK [[UNIMOD:385]QATGGQAPSSSPTSLTFSL] / QATGGQAFPQMVFDHWSTIGSDPIDPTSK [QATGGQAFPQMVFDHW] |
      | 40 | 2456 | -0.776 | length change | -1.029 | true | 0 (0) | 2 (2) |  | GGGEEDEDDPGSR [GGGEEDEDDPGS] / ADWGC[UNIMOD:4]AEDPSGR [ADWGC[UNIMOD:4]AEDPSG] |
      | 41 | 37383 | -0.780 | length change | -0.044 | true | 0 (0) | 3 (1) |  | [UNIMOD:385]GDGYDNYGGYGSEEDSYGSGGSR [[UNIMOD:385]GDGYDNYGGYGSEEDSYGSGGS] / NSYGSSNYGSSNNDDSYGSSNR [NSYGSSNYGSSNNDDSYGSSN] |
      | 42 | 42651 | -0.972 | length change | -0.044 | true | 0 (0) | 35 (22) |  | [UNIMOD:385]CVTITDGVQHHATDDVEDAAPETK [[UNIMOD:385]CVTITDGVQ] / ISITGGFSHHHATDDVEDAAPETK [ISITGGFSH] |
      | 43 | 95938 | -1.071 | length change | -0.019 | true | 10 (1) | 53 (28) |  | [UNIMOD:385]AGGNISGIC[UNIMOD:4]QDSDSNISGVHC[UNIMOD:4]M[UNIMOD:35]TTQELDELANSK [[UNIMOD:385]AGGNISGIC[UNIMOD:4]QDSDSNI] / FINNTYENPFM[UNIMOD:35]NASGVHC[UNIMOD:4]M[UNIMOD:35]TTQEIDEIANSK [FINNTYENPFM[UNIMOD:35]NA] |
      | 44 | 1636 | -1.274 | length change | -1.027 | true | 13 (3) | 15 (8) |  | [UNIMOD:385]AIVHIAQAGQC[UNIMOD:4]GNQIGAK [[UNIMOD:385]AIVHIAQAGQC[UNIMOD:4]GNQIGA] / KGFGEKASEAIKPDSQK [KGFGEKASEAIKPDSQ] |

      Part A, pre-registered hypotheses (2026-11-24; written from explain()
      output only, top_k 8, no ablation, via `analysis/v1_4c/case_views/`;
      machine-readable copy `analysis/v1_4c/hypotheses.json`, sha256
      2177039b13515fdf46d8f07de6d1c3542d92f7829c8b4ee73cca9120963e84a9). Nothing
      after this commit may change them. Each names features at one layer as
      (feature, token); token 0 = latent, token j + 1 = peak j. Sign: + = the set
      pushed the model towards its wrong answer (E > 0, defined below). The
      first draft (H1–H6) was amended at review before this commit: H1's
      confound and low motivation, H3's narrowed claim, the H2/H6
      internal-fragment caveat and H7 (a deviation, below) were added.
      Layer choice (recorded because "most specific" was not defined): no
      criterion was fixed in advance. For H1, H2, H5, H6 the layer is where, on
      the peaks named, the features whose top concept matched the claim
      (covers_N, is_internal_fragment, covers_D, is_I_ion) had their largest
      activation in the case view, which was L8 each time; H3, H4 and H7 were
      then put at L8 too, the last encoder layer (`n_layers` 9, layers 0–8),
      whose output the decoder reads. A post-hoc choice, stated as such.
      - H1, row 19475 (ambiguous; N[UNIMOD:7] vs D at residue 2, isobaric). L8
        {F511, F9744, F3490} at token 9 (b2 peak, m/z 214.118); predicted +.
        Low motivation. Claim: the default N[UNIMOD:7] comes from covers_N
        features on the b2 fragment VN next to the region. Evidence: b2 and b3
        have the same mass under both peptides, no distinguishing peak; region
        log p −0.129; F511 act 12.38 (covers_N, F1dom 0.024), F9744 6.61
        (0.010), F3490 6.04 (0.152, Phase 8 ΔCE +0.0022); no L8 descriptions.
        Confound: b2 = VN covers residue 1's N, which is correct under both
        peptides, so covers_N on this peak need not be evidence about the
        region residue.
      - H2, row 51238 (evidence favours prediction; VL vs IV = LV, a swap). L8
        {F5902, F3969} at tokens 75 and 81, F12279 at token 75 (peaks 74 / 80,
        m2:6+-H2O / m2:6+); predicted +. Claim: the VL order rests on
        internal-fragment peaks only the prediction explains, and the L8
        internal-fragment features on them push towards VL (possible label
        error). Evidence: all 5 distinguishing peaks are internal fragments
        under the prediction (m2:4, m2:6, m2:14), none under the true peptide;
        b2 identical; F5902 13.71 / 11.01 (is_internal_fragment, F1dom 0.070),
        F3969 5.88 / 7.05 (0.217, Phase 8 ΔCE −0.0032), F12279 6.45 (0.138).
        Caveat: the evidence rests only on internal-fragment matches, whose
        chance rate was not measured when this was written (check (c) below:
        decoys match 3–13 internal fragments in this spectrum).
      - H3, row 109208 (evidence for true; Q vs Q[UNIMOD:7], region Δm −0.984 Da,
        the prediction meets the precursor via isotope offset 1). L8 F2369 at
        token 155, F3879 at 171, F5569 at 108, F7859 at 195; predicted +.
        Claim (narrowed at review): the representations of these pred-only
        b-ion peaks push towards the prediction (unmodified Q). Features chosen
        by rule: the top L8 activation on each of the four most intense
        pred-only b/y peaks (154 b10+-H2O, intensity rank 11; 170 b11+, 20; 107
        b7+, 22; 194 b14+-H2O, 27; true-only peaks rank 58–197): F2369 17.21
        (mz_mid, 0.072), F3879 16.17 (mz_high, 0.627, ΔCE −0.0081), F5569 15.33
        (is_internal_fragment, 0.055), F7859 18.50 (mz_high, 0.452, ΔCE
        −0.0054). The data lean to the prediction (25 pred-only, 11 b/y; 6
        true-only, 3 b/y; and 5 of the 6 true-only peaks are 13C peaks of
        predicted fragments, check (a)), so this is also a possible label
        error. Side note: the true-only b9+ peak (m/z 920.458, 0.982 above the
        pred b9+ at 919.476) is encoded at L8 mainly by F7398 (18.32,
        is_NH3_loss).
      - H4, row 29350 (ambiguous; TG vs SA at residues 0–1, isobaric, no
        b/y peak in or at the region's edges). L8 {F1276, F2212, F7444} at
        token 0; predicted +. Claim: with no fragment evidence, the last-decoded
        N-terminal pair follows the spectrum-specific latent code, which
        supports TG. Named: the L8 latent top-k features other than F46, F4646,
        F4770, F5019 (in the L8 latent top-k of all six original cases): F1276
        4.59 (F1dom 0.284), F2212 1.73 (0.204), F7444 1.40 (0.179). Residue log p
        T −0.001, G −0.218. The least motivated hypothesis.
      - H5, row 60182 (ambiguous; D vs N[UNIMOD:7] at residue 5, isobaric). L8
        {F11636, F12167} at token 151 (y13, the most intense peak) and F11636
        at token 149 (y13+-H2O); predicted +. Claim: the default D comes from
        covers_D features on the y13 fragments, which contain the region
        residue. Evidence: b5, y12, y13 have the same masses under both
        peptides; region log p −0.458; F11636 9.66 / 5.19 (covers_D, 0.175,
        ΔCE +0.0003), F12167 5.45 (0.150). y13 also covers the D at residue 6
        under both peptides.
      - H6, row 29977 (evidence for true; AV vs GI = GL at residues 11–12,
        isobaric). L8 F6206 at token 19; predicted −. Claim: the one true-only
        peak is represented and pushes towards the true peptide, too weakly to
        overturn the default. Evidence: peak 18 (m/z 215.139, intensity rank
        104 of 117) is internal m13:15 under the true peptide, noise under the
        prediction; y6, y8 identical; region log p A −0.0001, V −0.550; F6206
        12.10 (is_I_ion, 0.137) is the only L8 top-k entry on that peak whose top
        concept is not an m/z bin. Caveat: the evidence rests only on one
        internal-fragment match; decoys match 5–15 internal fragments in this
        spectrum (check (c)), so one match is within chance.
      - H7, row 43804 (evidence for true; Q vs K at residue 7, region Δm
        −0.036 Da, precursor 17.4 ppm). L8 F9707 at tokens 124 and 119, F3971
        at 123, F12249 at 112; predicted −. Deviation: added at review, after
        seeing the candidate table (rank 10, the only top-ranked candidate with
        clear b/y evidence for the true peptide) and before any ablation;
        written from its case view (explain() only). Claim: the
        representations of the true-only y-ion peaks push towards the true
        peptide (K). Evidence: 19 true-only peaks (13 b/y) vs 2 pred-only (0
        b/y); the y10–y15 and b9–b14 ions containing residue 7 match only K,
        since 0.036 Da is ~20–30 ppm at m/z 1,000–1,800, just outside the
        tolerance for the prediction. Region log p (Q) −0.333; EOS log p −5.70.
        Features by H3's rule (top L8 activation on each of the four most
        intense true-only b/y peaks): peak 123 y13+ (intensity rank 2): F9707
        13.79 (covers_E, F1dom 0.206, ΔCE +0.0003); 122 y13+-H2O (8): F3971
        14.83 (mz_high, 0.287, ΔCE −0.0013); 118 y12+ (11): F9707 14.37; 111
        y11+ (13): F12249 16.65 (mz_mid, 0.081).

      Descriptive checks before part B (no ablation; `python -m
      inspector.runtime audit-checks --parquet <HF test> --isotope-rows 109208
      74557 8647 31366 45595 39783 42191 --decoy-rows 51238 29977`, output
      `analysis/v1_4c/descriptive_checks.json`, sha256
      8f444cfe54d4b0f905e1002d1eccf355d0c828ed5be8d42759dc34e5bbdec94f; code:
      `runtime.audit_descriptive_checks` and the `max_isotope` flag of
      `annotate.run_spectrum_utils`, committed in `f4933d3`):
      (a) The 7 candidates whose label carries a deamidation and whose
          prediction meets the precursor only via isotope offset 1, re-annotated
          under the prediction with max_isotope=1. True-only peaks that become
          13C isotope peaks of predicted fragments, and b/y counts:

          | # | row | ppm iso 0 / 1 | pred-only (b/y) | true-only (b/y) | true-only → pred 13C (b/y) |
          |---|---|---|---|---|---|
          | 3 | 109208 | −339.6 / −1.3 | 25 (11) | 6 (3) | 5 (3) |
          | 7 | 74557 | −719.1 / −1.7 | 9 (9) | 1 (1) | 0 (0) |
          | 13 | 8647 | −551.3 / +2.1 | 18 (16) | 6 (5) | 4 (4) |
          | 17 | 31366 | −330.7 / −4.6 | 7 (2) | 1 (1) | 1 (1) |
          | 26 | 45595 | −453.2 / −3.7 | 20 (13) | 9 (2) | 6 (2) |
          | 29 | 39783 | −700.8 / −7.7 | 9 (8) | 8 (7) | 3 (3) |
          | 34 | 42191 | −861.2 / +0.3 | 7 (6) | 4 (4) | 4 (4) |

          In all: 23 of 35 true-only peaks (17 of 23 true-only b/y) are 13C
          peaks of predicted fragments; b/y favour the prediction in 5 rows,
          rows ranked 17 (2 vs 1) and 29 (8 vs 7) are near-ties. Above ~967 Da a
          13C peak (+1.00336) lies within 20 ppm of the +0.98402 deamidation
          shift; misassigning 13C peaks of amidated peptides as deamidation is
          a known artifact, so these labels may be wrong.
      (b) Row 72165 (rank 30)'s true peptide is row 109208 (rank 3)'s true
          peptide unmodified.
      (c) Internal-fragment chance matches: 20 same-composition decoys of the
          prediction (C-terminal residue fixed, the rest shuffled with their
          modifications, `default_rng([0, row])`), one label per peak. Row
          51238 (H2, 179 peaks): peaks labelled internal, prediction 21, true
          peptide 16, decoys mean 9.3 [3, 13]; any fragment 54 / 49 / decoys 19.6
          [14, 25]. Row 29977 (H6, 117 peaks): internal 16 / 17 / decoys 8.9
          [5, 15]; any fragment 48 / 49 / decoys 14.4 [10, 20]. So 5–8% of
          peaks match an internal fragment of a random same-composition
          sequence at 20 ppm.
      (d) Control pools (rule below, from the SAE features at each named
          token): L0 at token / named feature's activation rank / pool size.
          H1: F511 26/2/2 (underpowered), F9744 26/4/8, F3490 26/5/9; 49
          distinct sets. H2: F5902@75 34/1/1 (underpowered), F5902@81 37/1/4,
          F3969@75 34/5/10, F3969@81 37/4/10, F12279 34/3/10; 1,080 sets. H3:
          F2369 43/1/3, F3879 50/1/1 (underpowered), F5569 50/1/3, F7859 49/1/0
          (underpowered); 0 sets. H4: L0 7 at the latent token (confirmed), so
          the pool of each named feature is exactly F46, F4646, F4770, F5019
          (all ≥ 0.5×); 4 sets. H5: F11636@151 41/2/9, F12167 41/8/10,
          F11636@149 36/8/10; 540 sets. H6: L0 1 at token 19 (F6206 is the only
          active L8 feature there), pool 0; 0 sets. H7: F9707@124 36/1/3,
          F3971 30/1/2 (underpowered), F9707@119 38/1/4, F12249 37/1/2
          (underpowered); 48 sets. Under the outcome rule below, H3, H4 and H6
          are therefore undetermined by construction; H1, H2, H5 and H7 can be
          decided. H3's and H7's named features are rank 1 at their tokens by
          construction (top activation), so all their pool members are weaker.

      Review notes (2026-11-25, after pre-registration; the hypotheses above are
      unchanged; checked against `analysis/v1_4c/explanations/` and
      `dictionary/layer_8.parquet`):
      - H1: the top L8 feature on peak 8 (token 9, b2) is F1330 (17.61, top
        concept cleaves_C_to_Asp, F1dom 0.109). It bears on the region residue
        (D in the true peptide) and was not named; the named covers_N features
        (F511 12.38, F9744 6.61, F3490 6.04) fit the hypothesis's story.
      - H6: F6206 is not specific to peak 18. It also fires, alone or nearly
        alone, on unrelated peaks in other spectra (row 51238 peak 23: L0 2,
        F6206 8.55 with F8596 5.05; row 109208 peaks 99, 134, 189: L0 1). In
        each of the 7 case-view spectra it is the only active L8 feature on
        6–15 peaks, mostly noise-labelled (29977: 8 including peak 18); firing
        rate 8.9%.
      - H4: F1276 also appears in the L8 latent top-k of rows 51238 (1.59),
        60182 (1.42) and 109208 (1.63), so it is not specific to row 29350.
      - General: L8 concept labels are mostly m/z bins, with F1dom < 0.3 and no
        descriptions. m/z bins are the top concept of 59% of the 7,632 L8
        top-k entries shown in the 7 case views (45% of their 1,031 distinct
        features; 31% of all 8,732 L8 features); F1dom < 0.3 for 97% of those
        features (99.6% of L8); no L8 feature has a description (V1-2c).
      - Finding 8 records which top candidates are not model failures; Finding
        9 audits the labels these hypotheses rest on (immonium and
        internal-fragment labels, covers_*).

      Amendment 1 (2026-11-25, before part B; the hypotheses, the case
      selection and the part B protocol are unchanged; hypotheses.json sha256
      still 2177039b…e84a9): evidence classes redefined on distinguishing b/y
      peaks only, with the same rule (true = ≥ 1 true-only b/y peak; pred =
      distinguishing b/y peaks only under the prediction; amb = none). Reason:
      internal-fragment matches are at chance level (chance fraction 0.54) and
      immonium labels are peptide-independent (Finding 9), so they cannot count
      as evidence. Computed from the stored counts (`n_pred_only_by`,
      `n_true_only_by`; `runtime.backbone_evidence_classes`, `python -m
      inspector.runtime audit-evidence-by`, output
      `analysis/v1_4c/evidence_classes_by.csv`; candidates.csv untouched, and
      its stored classes are reproduced from its counts). Limit: those counts
      take a peak that is b/y under one peptide and noise or precursor under the
      other; a peak that is b/y under one and internal or immonium under the
      other is not counted (counting it would need re-annotation, not done).
      Classes 34 true / 7 amb / 3 pred → 28 / 13 / 3. 8 of 44 change:

      | rank | row | sel | all fragments | b/y only | pred-only (b/y) | true-only (b/y) |
      |---|---|---|---|---|---|---|
      | 2 | 51238 | top 5 | pred | amb | 5 (0) | 0 (0) |
      | 6 | 29977 | added | true | amb | 0 (0) | 1 (0) |
      | 9 | 46143 |  | true | pred | 3 (3) | 9 (0) |
      | 14 | 47026 |  | true | pred | 15 (12) | 3 (0) |
      | 16 | 71941 |  | true | amb | 3 (0) | 1 (0) |
      | 19 | 90372 |  | pred | amb | 2 (0) | 0 (0) |
      | 21 | 97297 |  | true | amb | 0 (0) | 3 (0) |
      | 24 | 33227 |  | true | amb | 2 (0) | 1 (0) |

      Consequences, recorded and not acted on: among the selected cases H2's row
      (51238) becomes ambiguous (its evidence was internal fragments only) and
      H6's row (29977) ambiguous (one internal peak). Only row 109208 (H3, a
      probable ¹³C artefact, Finding 8(a)) stays "evidence for true", so the
      selection rule's "2 selected cases for true" would not be met under the
      amended classes; the selection is not redone. H7's row (43804, rank 10)
      stays "evidence for true" (13 true-only b/y vs 0).

      Amendment 2 (2026-11-25, before any part B ablation on the selected
      cases; the hypotheses, the registered controls and the outcome rule are
      unchanged; hypotheses.json sha256 still 2177039b…e84a9):
      - H7 secondary control, peak-matched (reported beside the registered
        control; it does not change H7's outcome label). Reason: H7's named
        features are rank 1 at their tokens (check (d)), so every registered
        pool member is weaker than the feature it replaces. Pool of a named
        (feature, token): the 10 non-distinguishing b/y peaks nearest in
        intensity rank to the named peak, the named tokens excluded.
        Non-distinguishing b/y = ion type b or y under both the prediction and
        the true peptide in the existing annotation (`annotate_spectrum`,
        strict, 20 ppm; neutral losses and charges included, as the named peaks
        include y13+-H2O). Intensity rank as in the case views (1 = most
        intense among the processed peaks, ties by peak index); ties in rank
        distance go to the higher intensity, as in `control_pool`; peaks with
        no active L8 feature are dropped before the nearest-10 cut. A draw takes
        one peak per named item from its pool, no peak twice, and uses that
        peak's top-activation L8 feature at that peak's token (it may be a named
        feature, at a different token). 20 draws with
        `np.random.default_rng([1, row])`, each ablated exactly as the
        hypothesis set (joint, local, L8, both contexts). Reported: every draw's
        E and E / M(intact), Σ activation(control) / Σ activation(named), the
        number of distinct draws and the percentile (same formula as the
        registered control).
      - Dropped: H8 and a noise-peak check, discussed while planning part B but
        never registered, are dropped: both relied on is_noise_peak, the
        complement of the matched-peak labels Finding 9 found contaminated.
      - Fixed here because the registration left them open (both controls):
        (i) sampling: each named item's member is drawn uniformly from its full
        pool, in hypotheses.json item order; a draw that repeats a (feature,
        token) pair (secondary: a peak) is redrawn whole (rejection sampling,
        at most 10,000 attempts per draw, else the run raises); the 20 draws
        are independent, so a set may recur across draws. (ii) "Distinct
        control draws" in the outcome rule = the number of possible distinct
        sets (`n_distinct_control_sets`, as check (d) used for "undetermined by
        construction"); the number of distinct sets among the 20 realised draws
        is reported beside it. (iii) A percentile tie is |E_control − E| ≤ 1e-6
        (the batching-noise level); the hypothesis set and its controls are
        ablated in the same batch, against the same baseline.

      Part B protocol (fixed with the hypotheses):
      - Contexts: ctx_p = `ablation_context(spec, exp)` and
        ctx_t = `ablation_context(spec, exp, target=T)`, with `exp` = a fresh
        `explain()` that must reproduce the saved explanation's prediction.
        T = the prediction's tokens outside the region plus the true region
        with I mapped to L: pred[:start] + L(true[start:end_true]) +
        pred[end_pred:]. In all seven cases the prediction's region contains
        no I, so this equals setting I/L to match the prediction. T replaces
        the raw `modified_sequence` because `ablation_context` does not
        canonicalise I/L (`runtime.py:871` maps only modification notation
        through InstaNovo's `residue_remapping`, which has no I/L entry) and I
        and L are distinct InstaNovo tokens, while the accuracy metric and the
        region use I = L; the true peptides of H3, H4, H6 and H7 differ from
        the prediction by I/L outside the region (so "same C-terminal suffix"
        holds only after mapping), and the true regions of H2 (IV) and H6 (GI)
        contain I.
      - Metric: M = Σ log p(pred region | ctx_p) − Σ log p(L(true region) | ctx_t),
        per-residue log p in residue order over the regions in the candidate
        table (an empty region adds 0; EOS is not in the region). InstaNovo
        decodes C→N and T shares the prediction's tokens outside the region,
        so both regions are conditioned on the same C-terminal suffix. Effect
        of a set S: E(S) = M(intact) − M(S removed), the negative of the raw
        ablation delta of M. E > 0: S pushed the model towards its wrong
        answer.
      - Ablation: exact, error-preserving (SAE error kept), joint removal of the
        named set, local at the named tokens, one layer, in both contexts.
        Needs a small extension (`ablate` runs single features, `ablate_token`
        all features at one token): a joint ablation of an arbitrary set of
        (feature, token) pairs at one layer, built in part B with a test that it
        equals `ablate` for a single pair and `ablate_token`'s joint for a full
        token.
      - Reported per hypothesis: E and E / M(intact); per-residue deltas in both
        contexts; whole-sequence Δ (residues + EOS) in both; each named
        feature's single-feature E; the joint ablation of all active features at
        each named token as a reference scale (with the V1-4b caveat: a large
        edit, not a clean total); provenance (exact, error-preserving, local,
        targets as above, n = 1 spectrum, no CI).
      - Control: 20 draws with `np.random.default_rng([0, row])`. Pool of a named
        (feature, token) (`runtime.control_pool`): the non-named active features
        at that token and layer with activation ≥ 0.5 × the named feature's (a
        judgement call), of which the 10 nearest in activation rank (ties to
        the higher activation); fewer than 3 qualifying → that item's control
        is marked "underpowered". A draw takes one feature per named item
        uniformly from its pool, with no (feature, token) pair twice in a draw,
        and is ablated exactly as the hypothesis set. For every draw, report
        Σ activation(control) / Σ activation(named); decoder rows have unit
        norm, so this is the ratio of the removed norms. Report every control E
        and E / M(intact), and the number of distinct draws. Percentile =
        (#controls with E below the hypothesis's + ½ #ties) / 20, taken in the
        predicted direction (signs flipped for a predicted −).
      - Outcome (pre-registered): undetermined = fewer than 10 distinct
        control draws (including any item with an empty pool); otherwise
        supported = predicted sign and beyond all 20 controls in that
        direction; weak = predicted sign but not beyond all controls; not
        supported = wrong sign. "Weak" needs only the predicted sign (~50% by
        chance). If the controls are exchangeable with the hypothesis set,
        each test has a 1/21 chance of "supported" by chance, so across the
        four decidable tests (H1, H2, H5, H7) about 0.2 false "supported"
        results are expected (0.3 had all six original tests been decidable).
        Always reported with the numbers (n = 1 spectrum each; 20 draws give
        5% resolution) and the underpowered items. E / M(intact) is reported
        alongside, for scale only; it does not change the outcome.
      - Caveats stated with the results: every test spectrum was SAE training
        data (Finding 7); L8 has no descriptions (V1-2c), so the dictionary
        evidence above is concept associations only, all weak (F1dom ≤ 0.63);
        the encoder does not see the peptide; H2, H3 (and the deamidation
        labels in check (a)) may be label errors; the layer choice was post hoc.

      Part B results (2026-11-25, `python -m inspector.runtime audit-part-b
      --parquet <HF test>`, 272 s on CPU against a 5–10 min estimate; code
      `800558a`). Outputs in `analysis/v1_4c/`, with sha256:
      - part_b.json `ca24461c9bf06b865d91dcd9b82b5888e0b1cc268765e17974020b40d72f46d9`
      - part_b_summary.csv `5d391c258b497a5453a85880aa43fb632eed974350e838f4568fd2f9e749702a`
      - part_b_controls.csv `e6b09a55d0972f4d86c8bedf8d2b71724c808a346d9a2b9f5ef69119297bbf2a`
      - part_b_token_sweep.csv `e5a950ef386b557fc11628d4102a5547ae02cc8be5d75ccb7139bf85dd648b56`

      Run as registered, with the following checks (the run stops if any fails):
      - hypotheses.json matched the registered sha256.
      - For all 7 rows, the fresh explain() reproduced the saved prediction,
        its per-residue log p (to 1e-5) and the candidates.csv region.
      - T round-tripped through the tokenizer, and ctx_t saw the same L8 features.
      - Every recomputed control pool and set count equalled
        descriptive_checks.json.

      H6's set E equals its single-feature E (one item). H7's secondary batch
      reproduced the set's E exactly. Provenance for every number below:
      exact error-preserving ablation, local at the named tokens, L8, joint
      removal of the named set; contexts ctx_p and ctx_t as registered;
      n = 1 spectrum per hypothesis; no CI.

      | hyp | row | class (Amendment 1) | sign | M(intact) | E | E / M | control percentile | secondary (H7) | outcome |
      |---|---|---|---|---|---|---|---|---|---|
      | H1 | 19475 | amb | + | 2.743 | −1.043 | −0.380 | 0.45 | | not supported |
      | H2 | 51238 | amb | + | 2.498 | −0.005 | −0.002 | 0.10 | | not supported |
      | H3 | 109208 | true | + | 7.138 | +0.096 | +0.013 | – | | undetermined by construction (registered) |
      | H4 | 29350 | amb | + | 1.439 | −0.093 | −0.064 | – | | undetermined by construction (registered) |
      | H5 | 60182 | amb | + | 0.544 | +0.779 | +1.431 | 1.00 | | supported |
      | H6 | 29977 | amb | − | 1.479 | +0.232 | +0.157 | – | | undetermined by construction (registered) |
      | H7 | 43804 | true | − | 0.933 | +0.351 | +0.377 | 0.10 | 0.20 | not supported |

      M(intact) > 0 in every case, so E / M is well defined. Percentiles are
      taken in the predicted direction. Distinct control draws among the 20
      (possible sets): H1 17 (49), H2 20 (1,080), H5 20 (540), H7 15 (48);
      H7 secondary 20 (330). Underpowered items: H1 F511@9, H2 F5902@75, H7
      F3971@123 and F12249@112. Control removed-norm ratios Σact(control) /
      Σact(named): H1 0.73–1.37, H2 0.60–0.80, H5 0.70–1.34, H7 0.65–0.85;
      H7 secondary 0.80–1.27.

      How to read the result set: about 0.19 false "supported" results are
      expected across the four decidable tests (4/21) if the controls are
      exchangeable with the named sets. One test came out "supported". With
      n = 1 spectrum each and 5% control resolution, it is one data point,
      not a finding about the model. The cases are mostly not model failures
      (Finding 8):
      - H5's case is notation-equivalent.
      - H3's case is a probable ¹³C artefact.
      - H1's and H4's cases cannot be resolved from the spectrum.
      - Under Amendment 1, H2's and H6's cases have no distinguishing b/y peak.

      The concept evidence the hypotheses cite (covers_N/D/E,
      is_internal_fragment, is_I_ion, m/z bins) is weak or contaminated
      (Findings 9, 10). L8 has no descriptions, the layer choice was post hoc,
      and every test spectrum was SAE training data (Finding 7). Single-feature
      effects show what removing that feature alone does. The joint ablation of
      a whole token is a large edit, quoted only as a reference scale (V1-4b).

      Per case (E > 0: the named set pushed the model towards its wrong answer):
      - H1, row 19475. InstaNovo wrote VN**N[UNIMOD:7]**… where the label has
        VN**D**…, the same composition (Finding 8(b)). The tool suggested that
        covers_N features on the b2 peak (VN) favour the N[UNIMOD:7] reading.
        Removing them lowered log p of D in ctx_t by 1.02 and barely changed
        log p of N[UNIMOD:7] (+0.02). So E = −1.04: the set supported D, the
        opposite of the claim. Single E: F511 −0.74, F9744 −0.30, F3490 −0.13.
        But every control drawn at the same peak also had negative E (−1.23 to
        −0.52, percentile 0.45). Removing any strong feature on that b2 peak
        lowers log p(D) about as much, so the effect belongs to the peak, not
        to the named features. Joint at the token: −1.79 (sum of singles −4.10).
        Step (c): F511 and F9744 rank 2 and 4 of 26 by |effect on M|, F3490
        rank 11.
      - H2, row 51238. InstaNovo wrote **VL**STI… for **IV**STI… (a swap). The
        tool suggested that internal-fragment features on two peaks only the
        prediction explains push towards VL; those matches are at chance level
        (Finding 9). E = −0.005 (E / M −0.002), so removing the set changes
        little. The single effects are of mixed sign and
        ≤ 0.05; the controls run −0.02 to +0.08 (percentile 0.10). Joint per
        token: −0.18 and −0.07.
      - H3, row 109208. InstaNovo wrote Q where the label has Q[UNIMOD:7], a
        probable ¹³C-as-deamidation label artefact (Finding 8(a)). The tool
        suggested that the top features on the four most intense pred-only
        b-ion peaks favour unmodified Q. E = +0.096 (E / M 0.013): a small push
        in the claimed direction, mostly from F5569 (+0.063). The set's
        largest whole-sequence effect is elsewhere: it lowers log p of the
        residue at index 9 (V) by 0.67 in ctx_p. Undetermined by construction:
        every pool had 0–3 members, as check (d) found.
      - H4, row 29350. InstaNovo wrote **TG**NL… for **SA**NI…, isobaric with
        no distinguishing peak. The tool suggested that three latent-token
        features favour TG. E = −0.093 (the set leaned slightly towards SA),
        mostly from F1276 (−0.089). Undetermined by construction (4 possible
        control sets). The joint over all 7 latent features is +0.165.
      - H5, row 60182. InstaNovo wrote DVTFI**D**DC… where the label has
        DVTFI**N[UNIMOD:7]**DC…, the same molecule written differently
        (Finding 8(b)). InstaNovo's own metric counts it correct. The tool
        suggested that covers_D features on the most intense peak (y13) favour
        writing D. Removing the three items lowered log p(D) by 0.36 and
        raised log p(N[UNIMOD:7]) by 0.42. So E = +0.78, E / M 1.43: under
        teacher forcing, the region's preference turns towards N[UNIMOD:7]
        (M from +0.54 to −0.23; this is not a re-decode). All 20 controls had
        smaller E (−1.61 to +0.24), so the outcome is "supported". Most of the
        effect is F11636 on y13 alone (+0.75; step (c) rank 2 of 41). F12167
        does little (−0.03, rank 30 of 41), and F11636 on y13-H2O adds +0.16.
        The joint at y13 is +0.003 (sum of singles −0.12), so other features
        at that peak offset it. What this measures is a naming choice between
        two notations of one molecule, not a sequencing error. covers_D's
        chance fraction is 0.28 (Finding 9), so the claim that F11636 encodes
        D content is not established by the dictionary.
      - H6, row 29977. InstaNovo wrote …TN**AV**TS… for …TN**GI**TS…,
        isobaric. The one true-only peak is an internal-fragment match within
        chance (Amendment 1: ambiguous). The tool suggested that F6206 on that
        peak, the only active L8 feature there, pushes towards the truth.
        E = +0.23: removing it raised log p of GL in ctx_t by 0.18, so it
        pushed the other way, towards AV. Undetermined by construction (no
        control pool), so no outcome. F6206 is a generic lone-firing feature
        (review note), so this says little.
      - H7, row 43804. InstaNovo wrote …EET**Q**AH… where the label has
        …EET**K**AH…. Q and K differ by 0.036 Da, and 13 b/y peaks match only
        K. This is the one candidate with clear b/y evidence for the true
        peptide. The tool suggested that the top features on the four most
        intense true-only b/y peaks push towards K. Removing all four lowered
        log p(Q) by 0.11 and raised log p(K) by 0.24. So E = +0.35: as a set
        they pushed towards Q, the wrong answer, and the outcome is "not
        supported" (percentile 0.10 in the predicted direction; secondary
        peak-matched control 0.20). The items disagree:
        - F9707 on y13+ (−0.21) and y12+ (−0.14) pushed towards K, as
          claimed.
        - F12249 on y11+ (+0.71) pushed towards Q and dominates the set.
        - F3971 on y13+-H2O did little (+0.03).

        Step (c): F9707@124 and F12249@112 rank 1 and 2 of 36–37 by |effect
        on M|; F3971 and F9707@119 rank 6. Removing the set also raises the
        EOS log p (intact −5.70) by 0.05 in ctx_p and 0.22 in ctx_t; EOS is
        not in M.

      Step (c), each named feature's rank by |effect on M| among all active L8
      features at its token (activation rank in brackets):
      - H1: 2, 4, 11 of 26 (2, 4, 5)
      - H2: 20/34 (1), 3/37 (1), 5/34 (5), 1/37 (4), 25/34 (3)
      - H3: 8/43, 38/50, 3/50, 1/49 (all 1)
      - H4: 3, 6, 7 of 7 (4, 6, 7)
      - H5: 2/41 (2), 30/41 (8), 5/36 (8)
      - H6: 1/1
      - H7: 1/36, 6/30, 6/38, 2/37 (all 1)

      Being among the top-k by activation and having a large effect on M
      overlap only partly.

      Exploratory (not pre-registered), H7, row 43804. The effect on M of every
      active L8 feature at the 13 true-only b/y peaks (482 pairs). The peaks
      are 87, 88 (y14++ with losses), 98 b9+, 105 b10+, 110 y10+, 111 y11+,
      116 y12+-H2O, 118 y12+, 120 b13+, 122 y13+-H2O, 123 y13+, 124 b14+ and
      130 y15+; the full list is in part_b_token_sweep.csv.

      E_f is the effect on M of removing one feature at one peak (local,
      exact, error-preserving, both contexts). E_f > 0 means the feature
      pushed towards Q, the wrong answer; E_f < 0 means it pushed towards K.
      Token = peak + 1. Rank = activation rank among the features active at
      that peak. "–" = no chance fraction, because m/z bins are not derived
      from peak labels.

      Pushing towards Q (E_f > 0):

      | feature | token | peak (true label) | activation | rank | E_f | top concept (F1dom, chance) | reliable concept (F1dom, chance) |
      |---|---|---|---|---|---|---|---|
      | F7761 | 111 | y10+ | 9.07 | 4/47 | +2.548 | mz_mid (0.058, –) | position_middle (0.029, 0.092) |
      | F3879 | 111 | y10+ | 12.38 | 1/47 | +1.451 | mz_high (0.627, –) | position_middle (0.102, 0.092) |
      | F12249 (named, H7) | 112 | y11+ | 16.65 | 1/37 | +0.711 | mz_mid (0.081, –) | is_b_ion (0.011, 0.177) |
      | F884 | 111 | y10+ | 4.92 | 11/47 | +0.601 | position_middle (0.099, 0.092) | position_middle (0.099, 0.092) |
      | F2483 | 112 | y11+ | 6.22 | 7/37 | +0.576 | mz_high (0.488, –) | is_y_ion (0.087, 0.109) |
      | F7859 | 112 | y11+ | 13.36 | 2/37 | +0.507 | mz_high (0.452, –) | is_y_ion (0.083, 0.109) |
      | F9750 | 112 | y11+ | 7.61 | 4/37 | +0.396 | mz_high (0.454, –) | is_y_ion (0.259, 0.109) |
      | F2134 | 111 | y10+ | 2.09 | 32/47 | +0.366 | mz_mid (0.070, –) | position_middle (0.009, 0.092) |
      | F3093 | 112 | y11+ | 4.38 | 10/37 | +0.324 | is_y_ion (0.140, 0.109) | is_y_ion (0.140, 0.109) |
      | F10566 | 112 | y11+ | 3.24 | 18/37 | +0.267 | cleaves_C_to_Asp (0.272, 0.108) | cleaves_C_to_Asp (0.272, 0.108) |

      Pushing towards K (E_f < 0):

      | feature | token | peak (true label) | activation | rank | E_f | top concept (F1dom, chance) | reliable concept (F1dom, chance) |
      |---|---|---|---|---|---|---|---|
      | F8109 | 112 | y11+ | 7.15 | 5/37 | −2.203 | mz_high (0.224, –) | is_y_ion (0.123, 0.109) |
      | F3422 | 111 | y10+ | 5.96 | 6/47 | −1.849 | covers_K (0.039, 0.228) | cleaves_N_to_Pro (0.014, 0.101) |
      | F8109 | 111 | y10+ | 5.55 | 8/47 | −1.512 | mz_high (0.224, –) | is_y_ion (0.123, 0.109) |
      | F7859 | 111 | y10+ | 5.39 | 9/47 | −0.728 | mz_high (0.452, –) | is_y_ion (0.083, 0.109) |
      | F3993 | 111 | y10+ | 3.35 | 17/47 | −0.520 | mz_mid (0.081, –) | cleaves_N_to_Pro (0.050, 0.101) |
      | F8109 | 117 | y12+-H2O | 6.67 | 5/47 | −0.446 | mz_high (0.224, –) | is_y_ion (0.123, 0.109) |
      | F8109 | 119 | y12+ | 5.88 | 8/38 | −0.341 | mz_high (0.224, –) | is_y_ion (0.123, 0.109) |
      | F5800 | 112 | y11+ | 2.90 | 21/37 | −0.327 | mz_high (0.484, –) | is_y_ion (0.081, 0.109) |
      | F7859 | 119 | y12+ | 13.96 | 2/38 | −0.302 | mz_high (0.452, –) | is_y_ion (0.083, 0.109) |
      | F9750 | 119 | y12+ | 4.92 | 10/38 | −0.245 | mz_high (0.454, –) | is_y_ion (0.259, 0.109) |

      Removing every active feature at a peak together (joint, a reference
      scale; V1-4b caveat) gives a much smaller net effect: y10+ +0.23,
      y11+ +0.27, y12+ −0.60, y13+ −0.18. The other nine peaks lie between
      −0.07 and +0.17.

      What it suggests (exploratory):
      - Effects are concentrated on y10–y12. All 20 listed pairs sit there.
        y10 and y11 are the smallest y ions that contain residue 7, the K/Q
        position, so they are where the 0.036 Da difference first shows. The
        b ions and the larger y ions have small net effects. The encoder does
        not ignore the evidence for K; this is where the choice is shaped.
      - Mixed signs and cancellation. Single effects are large and opposed
        (+2.55, −2.20, −1.85) but net to about ±0.2–0.6 per peak, which
        suggests redundancy or cancellation; any one feature's single effect
        overstates its role. At y10 and y11 the parts pushing towards Q
        outweigh those pushing towards K (net +0.23, +0.27); at y12 and y13
        the net is towards K. So the encoding of the evidence peaks does not
        come down cleanly on K, and at the two most direct peaks it tips
        slightly towards Q. From the raw m/z, these y peaks lie 0–6 ppm from
        K and 18–28 ppm from Q (y10 −6.1 / +23.3, y11 +0.8 / +28.0, y12
        −3.3 / +21.5, y13 −1.0 / +21.8 ppm). One untested possibility: the
        model may not resolve m/z that finely, so these peaks look consistent
        with both residues and something else tips the choice to Q. The
        backlog's peak-shift test addresses this.
      - A feature's effect depends on the peak. F7859 pushes towards Q at y11
        (+0.51) but towards K at y10 (−0.73) and y12 (−0.30); F9750 does the
        same between y11 (+0.40) and y12 (−0.25). Only F8109 pushes
        consistently, towards K at four y peaks. These are not "Q detectors"
        or "K detectors".
      - The dictionary cannot say what these features encode. Most top
        concepts are m/z bins, and the reliable concepts (is_y_ion,
        position_middle, cleaves_N_to_Pro, cleaves_C_to_Asp) have F1dom
        ≤ 0.27. F3422's covers_K has F1dom 0.039 and is a contaminated
        concept (chance fraction 0.23, Finding 9), so it is not evidence.
      - Activation as a screen, a first look. The activation ranks of the
        20 largest effects are 1, 1, 2, 2, 4, 4, 5, 5, 6, 7, 8, 8, 9, 10,
        10, 11, 17, 18, 21, 32. The three largest |E_f| (+2.55, −2.20,
        −1.85) rank 4–6. 8 of the 20 lie outside the top 8 that explain()
        lists (|E_f| 0.25–0.73), the furthest at rank 32 of 47. So a top-8
        activation cut would keep the largest effects here but miss
        mid-sized ones. Step (c) points the same way: named features ranked
        1 by activation ranged from 1 to 38 by |effect on M|.

      Limits: one spectrum. The lists are the extreme end of 482 pairs, so
      their top values will tend to be overstated. They compare two
      teacher-forced targets; nothing was re-decoded. They cover the encoder
      only, so they do not explain the decoder's final choice (EOS log p
      −5.70 is unexplained too). The SAEs were trained on this spectrum
      (Finding 7).

      What held and what didn't:
      - Of the four decidable tests, one is "supported" (H5) and three are
        "not supported" (H1 and H7 with the wrong sign, H2 with E ≈ 0). About
        0.19 false "supported" results were expected by chance.
      - H5's supported effect concerns a choice between two notations of one
        molecule. In the one case with a real b/y-supported error (H7), the
        tool's top-activation features did not, as a set, push towards the
        truth: one item pushed hard the other way.
      - In H1 the named features mattered, but no more than any strong feature
        on the same peak. The control is what showed this.
      - Lesson for the tool: top-k by activation and a concept label are not a
        guide to causal role. The per-peak ablation panel (V1-4b), not the
        concept column, is what a user should read, as Finding 10 already
        implies.

      V1-5 demo set:
      - H7 (row 43804). The one genuine, spectrum-resolvable error. It shows
        mixed-sign features on the true-only y peaks, a set effect dominated
        by one item, and joint vs single differences. It is a good showcase of
        why the dashboard leads with measured effects.
      - H1 (row 19475). Shows why controls matter: a large single effect that
        is a property of the peak, not of the named features. The case itself
        is notation-equivalent, so label it as such.
      - H5 (row 60182). Usable only labelled as a notation choice (D vs
        N[UNIMOD:7]) with a clear measured effect, not as a model error.
      - Not suited: H2 and H3 (possible label errors, ¹³C artefact), H4 and H6
        (undetermined, no spectrum evidence).
- [ ] **V1-5 Data for the tool** — pre-registration and spec (2026-11-26;
      approved plan, nothing implemented or run yet). Two separate parts:
      (1) data for a small demo in V1-6: explanations, feature examples for the
      features active in the demo set, and a calibrated per-spectrum SAE-fit
      indicator. The demo set gets no precomputed ablation; the dashboard
      computes ablations live (V1-4b: 1–6.5 s per click). (2) A measurement of
      activation as a screen for causal effect, on a seeded random sample with
      exhaustive local ablation; the demo set is not used for it. No API, GPU or
      cluster. Feature examples are the primary evidence of what a feature does
      (Finding 10). Every test-split spectrum was SAE training data (Finding 7;
      verified from code and logs: chunk 520's spectrum IDs start at row
      532,480, and the SAE validation chunks are 17, 116, 135, 141, 230, 292,
      316, 397, 448, 462, 489, 495); every output carries that note. HF test
      row r = merged row 527,974 + r, so chunk = (527,974 + r) // 1024; the
      local test chunks are 520 (rows 4506–5529) and 572 (rows 57754–58777).
      Nothing after this commit may change the demo rule, the random-sample
      rule or the screen definitions; any change is logged as a deviation with
      its reason.

      Decisions:
      - D1, demo set, 10 HF test rows:
        - Group A (V1-4c cases): H7 row 43804 (136 peaks), the one
          b/y-resolvable error. H1 row 19475 (71 peaks), labelled
          notation-equivalent; the controls showed its effect belongs to the
          peak. H5 row 60182 (168 peaks), labelled "notation choice: D vs
          N[UNIMOD:7], same molecule; InstaNovo's metric counts it correct".
        - Group B (correct predictions): the V1-4a seed-0 rows that are exact,
          meet the precursor and lie in chunks 520/572 (4552, 5398, 5406,
          58409, 58427), top 3 by beam log p: 5398, 5406, 58409 (35, 135, 141
          peaks).
        - Group C (reliable associations), one per layer: the feature in
          `analysis/reliable_associations.csv` with the highest best reliable
          F1dom (L2 F7522, L4 F11376, L6 F9282, L8 F5107; the L2 and L4 picks
          are Finding 2 hubs). Take the chunk 520/572 spectrum where that
          feature's peak-token activation is highest. If that spectrum is
          already chosen or its beam does not complete, take the next one.
      - D2, examples: K = 20 per feature, at most one per spectrum. A
        feature's examples are its top 20 spectra, ranked by the spectrum's
        maximum activation; each is shown at its argmax token. They are drawn
        from all local chunks (13 for L2/L4, 12 for L6/L8; chunk 1 has no
        L6/L8), for the union of features active at any token of the demo
        spectra only.
      - D3, example context: the label peptide and its annotation, marked
        "database-search label, not InstaNovo's prediction; benchmark labels
        can be wrong (Finding 8)". A resumable `--with-predictions` flag adds
        beam predictions; the examples pass prints the referenced-spectrum
        count and its beam cost (2.2–4.2 s each) so that can be decided later.
      - D4, outputs (no new output root): `analysis/v1_5/` (demo set,
        explanations, screen sample, ablations, screen results);
        `dictionary/examples/layer_{L}.parquet` + `dictionary/examples/
        spectra.parquet`; `dictionary/fit_reference.{json,csv}`. The fit
        reference is a separate file, not dictionary metadata. Embedding it
        would bump the dictionary schema, and the Inspector refuses
        old-schema dictionaries, which would force a second streaming pass.

      Random-sample rule (screen sample):
      - Population: the 2,048 HF test rows in chunks 520/572.
      - Order: `np.random.default_rng(0).permutation` of that list, ascending
        row order before permuting. Wrong and correct predictions alike; no
        conditioning on correctness, size or content, no hand selection.
      - Walk the order. Rows whose beam does not complete are skipped and
        counted.
      - The sample is the longest prefix whose predicted local-ablation cost is
        ≤ 9 h.
      - Frozen cost model, used for the cut and never recalibrated, so the
        sample is deterministic:
        - Cost per spectrum = Σ over layers 2/4/6/8 of (active pairs + tokens
          with L0 > 0) × c(n), with pairs exact from explain() and n = number
          of processed peaks.
        - c(n) = 0.006 + 0.00068·n s per ablated row, batched 32 + 1. It is
          fitted to V1-4b's per-peak clicks (~0.03 s per feature at 35 peaks,
          0.10–0.12 s at 152).
      - Measured timing (a one-batch calibration) is reported only.
      - Overlap with demo group B/C rows is allowed and reported.
      - Expected size: the model gives ~22 min per spectrum at these chunks'
        means (108 and 137 peaks per spectrum; ~30.4 active pairs per token in
        the V1-4c explanations), so roughly 20–25 spectra. The rule, not this
        expectation, fixes the size.

      Screen definitions:
      - Unit: a peak token (tokens 1..n) of one screen-sample spectrum at one
        layer. The latent token is ablated too, but excluded from (a)–(c) and
        reported descriptively.
      - Effect e = `delta_total` (Δ total log p of the prediction, residues +
        EOS) of the local, exact, error-preserving single-feature ablation,
        teacher-forced on the prediction. Activation = the feature's SAE
        activation at that token.
      - (a) Spearman ρ(activation, |e|) over the peak's active features
        (average ranks for ties), peaks with L0 ≥ 3. Layer statistic: the mean
        of per-peak ρ (median also reported).
      - (b) Screening recall = |T3 ∩ A8| / 3. T3 = the peak's top 3 by |e|;
        A8 = its top 8 by activation (explain()'s `top_k`); ties are broken by
        feature id. Only peaks with L0 > 8 (otherwise trivially 1; the number
        excluded is reported). Layer statistic: the mean over peaks. Reported
        beside it, both threshold-free: the chance level mean(8 / L0), and
        mass recall Σ_{T3∩A8} |e| / Σ_{T3} |e|.
      - (c) Variance split of |e|, per spectrum s and layer, around the
        spectrum's own mean: SS_between = Σ_p n_p (ē_p − ē_s)², SS_within =
        Σ_p Σ_f (|e| − ē_p)². Layer statistic: Σ_s SS_between / Σ_s SS_total.
        The exchangeable-null reference Σ_s (P_s − 1) / Σ_s (N_s − 1) is
        reported beside it (P_s peaks, N_s pairs). Secondary: the same on
        log10(|e| + 1e-6), since |e| is heavy-tailed; 1e-6 is the
        batching-noise level (`TIE_TOLERANCE`).
      - CIs: bootstrap over spectra (the resampling unit; peaks within a
        spectrum are correlated). 10,000 resamples, `np.random.default_rng(0)`,
        percentile 95%. Each statistic is a ratio of per-spectrum sums, so
        resampling is cheap.
      - Alongside, labelled exploratory and recomputed by code from
        `analysis/v1_4c/part_b_token_sweep.csv`: V1-4c's first look (8 of H7's
        20 largest effects outside the top 8 by activation; the 3 largest rank
        4–6, inside it). A different metric (effect on M across two contexts),
        13 peaks, L8 only, one spectrum: not comparable to (b).
      - Also reported, descriptively: the sample's exact-match count against
        the label.
      - Caveats stated with the results: all sample spectra were SAE training
        data (Finding 7); single-feature effects understate redundant features
        (V1-4b); n ≈ 20–25 spectra from two chunks.

      Steps and outputs (runtime estimates):
      1. Demo set: `python -m analysis --demo-set`, pipeline side, ~5 min.
         - Reads `analysis/v1_4c/predictions.csv`,
           `analysis/reliable_associations.csv`, and chunks 520/572 via
           `evaluate.ChunkStream(chunk_indices=[520, 572],
           feature_subset=[f])`, one layer at a time.
         - Checks each spectrum_id against its HF row.
         - Output `analysis/v1_5/demo_set.json` (rule, rows, group, label,
           reason).
      2. Explain and screen-sample estimate (~2–3 min), then **stop and show
         the estimate before step 4**.
         - `python -m inspector.runtime demo-explain`: a fresh explain() per
           demo row; H1/H5/H7 must reproduce the saved V1-4c prediction and
           log p. Writes `analysis/v1_5/explanations/` and
           `analysis/v1_5/demo_features.json` (per layer, the union of
           features active at any token, not only the top 8).
         - `python -m inspector.runtime screen-sample`: applies the
           random-sample rule. Writes `analysis/v1_5/screen/screen_sample.json`
           (rows; per row peaks, pairs and predicted cost; skipped rows;
           overlap with the demo set; total predicted hours; the measured
           one-batch timing) and `analysis/v1_5/screen/explanations/`.
      3. Feature examples and fit reference, one streaming pass:
         `python -m build_dictionary --examples --example-features
         analysis/v1_5/demo_features.json`. ~20–45 min (≈ 57 TFLOP per layer
         on CPU; the first chunk is timed and extrapolated); re-annotation
         ≤ 7 min; RAM < 2 GB. Per layer and chunk:
         - `ChunkStream.iter_activations()` + `iter_metadata_annotations()`.
         - `sae.forward_inference` in batches of whole spectra (~4k tokens),
           columns sliced to the demo features inside the batch. Never a full
           [n_tokens, 12288] matrix.
         - Per-spectrum centred FVE: SSE, Σx² and Σ_t x accumulated per
           spectrum across batches in float64, with the `runtime.fve`
           definition (a test asserts equality; the pipeline does not import
           the tool).
         - Per-spectrum max and argmax token per demo feature ([1024, F] per
           chunk), merged into `evaluate.TopKAccumulator`. It is extended with
           an optional `aux` provenance tensor; default None = unchanged.
         - Re-annotation of the referenced spectra against the label
           (`annotate.annotate_spectrum`, strict).

         Provenance per example: chunk, spectrum id, token, activation, layer,
         SAE checkpoint sha256_16, "activations from stored bf16 extraction",
         the Finding 7 note, the D3 label note, and the Finding 9 caveats for
         the annotation. Fit reference: 1,001 quantiles per layer of the
         per-spectrum centred FVE, plus n spectra (13,312 / 13,312 / 12,288 /
         12,288), chunks, the bf16 note, and "reference spectra were SAE
         training data (Finding 7); an unseen spectrum may sit lower". Can run
         before or after step 4.
      4. Exhaustive local ablation on the screen sample, overnight, after the
         step-2 estimate is approved: `python -m inspector.runtime
         screen-ablate`.
         - Per row: a fresh explain() (must match step 2), `ablation_context`
           on the prediction, then per layer `ablate_token` at every token
           with L0 > 0: every active pair plus the per-token joint. No global
           scope.
         - Each (row, layer) is written atomically as a parquet file in
           `analysis/v1_5/screen/ablation/`; existing files are skipped
           (resume).
         - Refuses if the estimate exceeds `--budget-hours` 9 unless
           `--force`. ≤ 9 h by construction (±30% model error; resumable if it
           overruns); RAM < 2 GB.
         - Provenance: exact, error-preserving, local, prediction target,
           n = 1 spectrum per row, no CI.
      5. Screen statistics: `python -m analysis --activation-screen`, < 1 min.
         Writes `analysis/v1_5/screen/activation_screen.json` (layer summaries
         with CIs, the V1-4c exploratory numbers) and
         `activation_screen_peaks.csv` (per peak).

      Storage estimate:

      | output | size |
      |---|---|
      | screen ablations (~20–25 spectra × ~15k pairs, per-residue deltas, joints) | 30–40 MB |
      | explanations (10 demo + ~25 screen) | ~20 MB |
      | dictionary/examples (K = 20) | ≤ 39 MB |
      | dictionary/fit_reference | ~1 MB |
      | screen results | < 1 MB |

      Sizing notes. Examples: 12–20k (layer, feature) pairs × 20 rows,
      7–12 MB. The spectrum context table is deduplicated (processed peaks
      with labels, precursor, label peptide), ≤ 13,312 spectra × ~2 KB.

      Files edited (no new modules or top-level dirs):
      - `src/analysis.py`: `--demo-set`, `--activation-screen` (pure stats
        functions + CLI).
      - `src/inspector/runtime.py`: `demo-explain`, `screen-sample`,
        `screen-ablate`; helpers `active_features`, `ablation_cost`,
        `random_sample` (pure), `exhaustive_local` (on `ablate_token`, which
        stays unchanged).
      - `src/build_dictionary.py`: `--examples` pass (torch imported lazily);
        the normal build is unchanged.
      - `src/evaluate.py`: `TopKAccumulator(aux=None)`, additive.
      - `src/inspector/dictionary.py`: `FitReference` (`percentile(layer,
        fve)`, loud error if the file is missing) and a `FeatureExamples`
        loader with a schema check, for V1-6. explain() is unchanged
        (Explanation schema stays 3).
      - `src/schema.py`: `EXAMPLES_SCHEMA_VERSION`,
        `FIT_REFERENCE_SCHEMA_VERSION`, `SCREEN_ABLATION_SCHEMA_VERSION` = 1.
      - `STATUS.md` (results, Behaviour changes, Log) and `CLAUDE.md` (data
        layout).

      Tests:
      - `test_build_dictionary.py`, on a toy chunk with a tiny SAE:
        - Streamed FVE equals `runtime.fve` per spectrum, for any batch size,
          with spectra straddling batches.
        - Examples equal a brute-force top-K, one per spectrum, with the right
          argmax tokens.
        - Quantiles and the metadata schema.
      - `test_chunk_stream.py`: `TopKAccumulator` aux provenance; the default
        output is identical.
      - `test_analysis.py`:
        - The demo rule on toys (ranking, collision goes to the next
          spectrum).
        - Screen statistics on hand-computed toys: ρ, recall with the L0 ≤ 8
          exclusion, mass recall, chance level, variance split, null
          reference.
        - Bootstrap determinism.
      - `test_inspector_runtime.py`:
        - `random_sample`: deterministic, the prefix stops at the budget,
          no-beam rows are skipped and counted.
        - `exhaustive_local` on the test spectrum for one layer equals
          `ablate_token`.
        - Resume skips files already written; the budget check refuses.
        - Demo features = the union of active features.
        - `FitReference.percentile`: bounds, monotone, missing file raises.

      Verification:
      1. Group B/C demo rows reproduce the stored activations (V1-4a
         tolerance); H1/H5/H7 reproduce the V1-4c predictions.
      2. explain()'s fve_centred vs the reference value for the same local
         spectra: max |Δ| reported (bf16 vs float32).
      3. Three random pairs recomputed with `ablate` match the stored rows to
         1e-6.
      4. Measured vs predicted runtime; on-disk sizes vs the storage table.
      5. The V1-4c exploratory numbers are reproduced (8 of 20, ranks 4–6).
      6. The full test suite passes.

      Commits:
      1. `V1-5 pre-registration` (this entry).
      2. `V1-5 code: feature examples and fit reference`.
      3. `V1-5 code: demo set, screen sample, exhaustive local ablation,
         activation screen`.
      4. `V1-5 results`: rows, runtime, storage, output sha256, the screen
         table with CIs and the V1-4c exploratory numbers alongside;
         CLAUDE.md layout; Log line.

      Sequence: commit 1 → code (commits 2–3) → steps 1–2 → stop for the
      step-2 estimate → step 4 overnight → steps 3 and 5 → commit 4.

      Deviations (logged after commit 1):
      - Dev 1 (2026-11-27, approved before use), verification 1, peak
        check: demo-explain compared the recomputed processed peaks to the
        chunk exactly (as `test_activations_match_stored_chunk` does). For
        group C row 58193 (C-L2, chunk 572) m/z and the peak count (36) are
        identical, but 32 of 36 intensities differ at float32 rounding: max
        5.96e-08 absolute, 1.43e-07 relative. The check is now m/z exact,
        intensities within 1e-6 relative (`PEAK_INTENSITY_RTOL`); the
        activation tolerance (V1-4a: rtol 2^-7, atol 2^-7 · rms) is
        unchanged. Row 58193 then passes it at every layer (max
        |Δ|/(|a| + rms) 0.0030–0.0032, as the other rows). The demo rule,
        rows and every other row are unchanged; the other six local demo
        rows match exactly.
- [ ] **V1-6 Dashboard** — `src/inspector/app.py`, Streamlit.
      Spectrum plot with peaks coloured by annotation, layer selector, per-peak
      feature panel, feature card. The card leads with the feature's examples
      (V1-5) and measured effects (exact ablation, with scope and provenance),
      then the blinded description with its score. Concept associations are
      secondary (Finding 10): each is always shown with its F1dom and chance
      fraction (CI from the dictionary metadata, `concept_reliability()`),
      under the display name for re-documented concepts (is_I_ion, is_first_ion
      as m/z windows, `display_name()`), with the reliable associations
      (`reliable_*`) beside the top 3; an "encoder-only" scope note; and the InstaNovo version the SAEs explain
      (v1.1.0). Ablation, built on V1-4b: per-peak panel = `ablate_token` on
      the clicked peak for the selected layer (every active feature, sorted by
      |delta_total|, activation shown, joint vs sum of singles; 1–6.5 s per
      click, so show progress); latent panel = `ablate_token(ctx, L, 0)`,
      labelled "latent token only"; feature card "remove everywhere" =
      `ablate(..., "global")`. Every effect shows its scope, provenance and the
      single-feature caveat. Cache the Inspector and the AblationContext per
      spectrum; read full rows with `feature_row` (fails on a dictionary hash
      mismatch).
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
      AtP-vs-exact agreement study (moved here from V1-4b, 2026-11-24):
      candidate estimators plain AtP, integrated gradients and AtP* (Kramár et
      al. 2024); exact error-preserving ablation is the reference; report rank
      correlation, sign agreement and top-k overlap separately for latent and
      peak tokens, with bootstrap CIs over spectra.
      First evidence (2026-11-24, read-only planning script, not in the repo):
      HF test rows 4506–4508 (= chunk 520 rows 0–2; predictions
      GQPLTQADLSNPTR, SEYDRGVSTFSPEGR, LIGDAAKDQAAINPR). Per layer, AtP's top
      10 features by |Δ̂ sequence log p| plus 10 random other active features
      (`default_rng(0)`), each ablated exactly at every token where it fires
      (error-preserving, teacher-forced on the prediction). Spearman ρ, AtP vs
      exact, L2/L4/L6/L8: row 4506 0.76/0.84/0.68/0.55; 4507
      0.46/0.98/0.76/0.91; 4508 0.23/0.51/0.98/0.80. Sign agreement within
      AtP's top 10: 0.6–1.0. Underestimates (AtP vs exact): −0.223 vs −4.609
      (4506 L2, 21×); −0.113 vs −1.580 (4507 L8, 14×); −0.140 vs −1.356 (4507
      L4, 10×); −0.132 vs −1.111 (4507 L6, 8×); −0.226 vs −1.372 (4506 L2,
      6×). Sign errors among AtP's top 3: +0.930 vs −2.886 (4508 L4), +0.423 vs
      −0.893 (4508 L2), +1.512 vs −0.660 (4508 L8); near zero: −0.202 vs
      +0.031 (4506 L6), +0.550 vs −0.026 (4506 L8). Random features: the
      largest exact effect was L2 F647 on row 4506 (AtP rank 30 of 497: −0.041
      vs exact −0.202), twice the median exact |Δ| of AtP's top 10 at that
      layer (0.097). In 5 of 12 (spectrum, layer) cells the largest random
      feature exceeded the smallest exact |Δ| in AtP's top 10 (all ≤ 0.074
      except F647). Two random features had a sign error (4506 L6 F8887, rank
      13: −0.083 vs +0.069; 4508 L4 F5326, rank 116: +0.056 vs −0.020). Below
      rank ~100, AtP ≈ exact (|Δ| ≤ 0.03). A biased sample (top 10 + 10 random
      of 500–1,250 active features per layer): a sanity check, not a
      calibration. Notes for the study: per-residue gradients via n+1 backward
      passes on one forward took 1.0–2.0 s, faster than one backward over n+1
      batch copies (1.4–5.6 s); grad-path vs inference-path log p differ by
      ≤ 1.9e-6. Rows 4506–4508 were SAE training data (Finding 7).
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
      - Control S_rand: to be fixed before V1.1-3 runs. Preferred: a
        frozen-decoder SAE per Korznikov et al. 2026 (random decoder frozen,
        rest trained), trained alongside any GPU retraining. Fallback if no
        GPU: the SAE's own initialisation, untrained, which is a weaker
        control; a positive Δ₂ against it would show only that the SAE beats a
        random sparse code. Fallback recipe (`src/train.py`):
        `_seed_everything(seed)` (`train.py:572-578`, called at `:773`), then
        `SparseAutoencoder(d_model=768, d_dict=12288, k=32)` (`:787-793`). No
        torch RNG is drawn in between (`_split_chunks` uses Python's `random`,
        `:581-600`), so on CPU with the same torch version this should
        reproduce seed 0's starting weights; they were never saved, so this
        cannot be checked. Init (`SparseAutoencoder.__init__`, `:158-167`):
        W_dec rows are unit-normalised `torch.randn(d_dict, d_model)`,
        W_enc = W_dec.T (tied), b_enc = b_dec = 0. Input preprocessing: none;
        the SAE reads raw layer outputs, cast from the stored bf16 to float32
        (`:469`, `:508`), with no centring or scaling; the only offset is b_dec,
        subtracted in `preactivations` (`:188-190`) and 0 at init. The init
        JumpReLU threshold is 0 (`:174`), which would keep every positive
        pre-activation (about half the dictionary); match the trained SAEs'
        sparsity with `calibrate_threshold` (`:536-567`, the mean BatchTopK
        boundary at k = 32) on the same training chunks. Summarised and
        selected identically to S. Why: random SAEs score close to trained
        ones on sparse probing (Korznikov et al. 2026, arXiv 2602.14111), and
        a random code also retains most raw-activation information, so this
        one control replaces the raw-activation block.
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

Changes to existing pipeline modules, each meant to preserve behaviour unless
stated, with the tests that show it (V1-4a's in `tests/test_inspector_runtime.py`).

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
- 2026-11-22, `build_dictionary.build_layer`: a description run without a
  `blind` column now raises instead of being read as unblinded. Only pre-V1-2
  runs lacked the column; they are archived and unreadable
  (`schema.check_run_tag`), and all eight live runs (blind_v1, unblinded_v2 ×
  4 layers) carry it, so the dictionary build is unchanged. The test fixture now
  writes `blind` for every run. Shown by `test_run_without_blind_column_raises`
  and the existing dictionary tests.
- 2026-11-22, `interpret.py`: the default `--env-file` is the repo root's `.env`
  (beside `.env.example`, where the README and the real key put it), not
  `src/.env`; error and help text say so. Nothing relied on the old default:
  no `src/.env` exists, the Docker/AIchor ignore files ship only `src/*.py`, and
  `run_pipeline.sh` passes `--env-file` from `INTERPRET_ENV_FILE`. Shown by
  `test_default_env_file_is_the_repo_root_dotenv`.
- V1-4b, `inspector/runtime.py`: `explain()` emits Explanation schema 2.
  `features[L][f]` is a summary (`dictionary.SUMMARY_KEYS` plus
  `n_tokens_active`, `max_activation`, `fires_on_latent`) instead of the full
  dictionary row; `provenance.dictionary[L]` carries the file's sha256 and the
  layer's causal provenance (replacing `dictionary_schema` and
  `dictionary_blinded_run`); the `attribution` key is replaced by `ablation`.
  Schema 1 is rejected; none was ever written to disk.
  `explain(attribution=…)`, which only raised `NotImplementedError` (its one
  caller was the test), is replaced by `latent_ablation=False`.
  `_capture_encoder_layers` gets `deltas=None`, unchanged when None, shown by
  `test_activations_match_stored_chunk`, `test_teacher_forced_ce_matches_phase_7`
  and `test_zero_delta_is_identity`. No pipeline module changed.
- V1-4c (`f4933d3`), `annotate.run_spectrum_utils`: new `max_isotope` flag,
  default 0 (the pipeline's setting, previously hard-coded). Shown by
  `test_run_spectrum_utils_max_isotope_flag` and the stored-label tests.
- 2026-11-25 (Finding 9), `annotate.py`, additive except one refactor:
  `_annotation_to_dict` now takes the first of `_annotation_candidates` (the
  same parse and `_candidate_sort_key` order, moved into a function that keeps
  every candidate); `run_spectrum_utils` gets `return_candidates=False`;
  new pure helpers `select_annotation`, `is_composition_determined`,
  `proforma_residue_tokens`, `annotation_inputs_from_tokens`,
  `proforma_neutral_mass`, `proforma_mz`, `precursor_ppm` (label audit only).
  `peptide_decoys` moved here from `inspector/runtime.py` unchanged (runtime
  re-exports it) so the pipeline does not import the tool. Shown by
  `test_chunk_annotator_unchanged_by_refactor`,
  `test_annotation_via_tokens_reproduces_stored_labels`, the audit's own
  reproduction check (200 spectra over 13 chunks, exact) and
  `test_annotation_to_dict_is_first_candidate`. `analysis.py` gets
  `--deamidation-precursor` and `--label-audit`, which import torch lazily;
  its existing modes are unchanged.
- V1-3b, `analysis.py` label audit: `peptide_label_metrics` also records the
  concept changes on absent-residue immonium peaks under an immonium filter
  (`con|immfix_lost|c`, `con|immfix_gained|c`); `concepts.csv` gains
  `immonium_fix_lost` and `immonium_fix_gained`, and the audit writes
  `concept_reliability.csv` (`concept_reliability`). Every other output and
  column is unchanged (rerun compared byte for byte). Shown by
  `test_immonium_fix_counts_toy`, `test_concept_reliability_display_rule`,
  `test_concept_reliability_requires_every_concept` and the existing audit tests.
- V1-3b, `build_dictionary.py`: dictionary schema 2 (reliability columns and
  metadata added; existing columns unchanged, shown on the rebuilt local files
  and by the existing dictionary tests). The concept reliability table is a new
  required input (the build fails without it, or if an associated concept has
  no record). `top_concepts` gets an optional concept filter and column prefix
  (defaults unchanged). Schema-1 dictionaries are rejected by the tool until
  rebuilt. Shown by the new `test_build_dictionary.py` reliability tests.
- V1-3b, `inspector/dictionary.py` and `runtime.py`: Explanation schema 3 (four
  summary keys added, `json_size_bound` covers them); schema 2 is rejected. The
  saved V1-4c `analysis/v1_4c/explanations/*.json` stay schema 2 on disk;
  nothing reads them through `from_json`, and part B must read their prediction
  as raw JSON. `feature_row` on them now fails by design (dictionary hash
  changed). Shown by `test_dictionary_summary_hash_and_causal_provenance` and
  the explain() schema test.
- V1-3b, `analysis.py`: new `--reliable-associations` mode
  (`reliable_association_list`, `load_dictionary_rows`); existing modes
  unchanged. Shown by `test_reliable_association_list_cut_order_and_summary`.
- V1-4c amendment 1, `inspector/runtime.py`: new pure
  `backbone_evidence_classes` and the `audit-evidence-by` subcommand, which
  reads candidates.csv only and loads no model. `--parquet` is no longer an
  argparse requirement; every other command still fails without it (now via
  `p.error`). Shown by `test_backbone_evidence_classes_toy_and_stored_counts`.
- V1-4c part B (`800558a`), `inspector/runtime.py`, additive: `set_delta`,
  `Inspector.ablate_sets` / `ablate_set` (joint local ablation of an arbitrary
  set of (feature, token) pairs at one layer, reported as an `AblationReport`
  whose `joint` carries the pairs and `SET_NOTE`), the part B helpers
  (`saved_prediction`, `audit_target_tokens`, `audit_effect`, `draw_controls`,
  `control_percentile`, `audit_outcome`, `intensity_ranks`,
  `peak_matched_pool`) and the `audit-part-b` subcommand. Existing functions
  are unchanged. Shown by `test_ablate_set_single_pair_equals_ablate`,
  `test_ablate_set_full_token_equals_ablate_token_joint`,
  `test_ablate_set_is_error_preserving_at_the_named_tokens_only` and nine
  more.

## Backlog (after v1; do not start early)

High priority after V1-6:
- Pre-registered peak-shift test for H7 (row 43804): shift the true-only y10–y15 peaks by ±0.036 Da (towards an exact Q match / further away), recompute M with no SAE involvement, to test whether InstaNovo resolves the ~20 ppm K/Q difference. Fix the shifts, peaks and prediction before running. (From the V1-4c exploratory sweep: the y peaks lie 0–6 ppm from K and 18–28 ppm from Q.)

Tool:
- Warm up beam search at Inspector load: the first call costs ~12 s extra (V1-4a)
- Re-decode under ablation (beam search with the ablation hook, ~2–4 s): V1-4b reports Δ log p of the original residues and teacher-forced argmax flips, not a new prediction
- Faster per-peak clicks on large spectra (4–6.5 s at ~150 peaks, V1-4b): rerun only from layer L onward with the layer input cached, or a compact summary encoding for the JSON size
- Precomputed demo-set ablations (V1-5 computes demo ablations live in the dashboard; precompute only for a static public demo)
- Exhaustive ablation against the true target (V1-5 uses the prediction only)
- Corpus-maximum reference for feature-example activations (Phase 3 `top_activating_tokens.csv`, 625 chunks) beside the local top 20
- Quantile-sampled feature examples (not only the top 20), and examples for all features (V1-5: demo-set features only)
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
- Check which residue pairs InstaNovo's mass metric merges (< 0.1 Da apart in
  its residue set, beyond the notation-equivalent N[UNIMOD:7]/D and
  Q[UNIMOD:7]/E), computed from the residue masses; Q/K (0.036 Da) is one
  (Finding 8(b))

Rigour (cluster/GPU):
- Phase 8 rerun: error-preserving ablation, per-feature scan, ~50 controls, bootstrap CIs
- Phase 7/8 rerun with `modified_sequence` targets (Finding 5): report how much the layer-level CE, loss recovered and per-feature ΔCE move; then drop the caveat from the dictionary's `causal_provenance` (`build_dictionary.PHASE8_TARGET_CAVEAT`)
- Random-init InstaNovo control
- Seed stability (2–3 more seeds per layer)
- Full label rebuild (Findings 9 and 10; V1-3b only documents the current
  labels), recomputed on the original 625-chunk data: annotation → Phase 4 →
  dictionary → V1-2 strata → concept reliability table. Fixes and decisions:
  restrict immonium ions to residues in the peptide (spectrum_utils generates
  all 23), and decide on immonium ions of modified residues; make `IonGeometry`
  treat spectrum_utils' internal end as exclusive; neutral losses on b/y only;
  decide whether internal fragments are labelled at all (chance fraction 0.54;
  NH3 losses on internal 0.92); ¹³C isotope peaks (excess 8% of noise above
  m/z 1,000); 2+ fragments (charges are capped at z − 1, the spectrum_utils
  default, not passed explicitly, so charge-2 precursors get none); a ions
  (excess 2.8% of noise). Rerun the label audit on the new labels.
- Decoy null for F1dom: per concept, the F1dom each feature reaches against the
  same concept computed from decoy-peptide labels (same-composition decoys as
  in the label audit), so an association can be compared with what chance
  matches alone produce
- Check InstaNovo v1.1 training data vs the nine-species benchmark
- If the SAEs are retrained, exclude the test split (chunks ≥ 515) (Finding 7)
- InstaNovo v1.2.0 (instanovo-v1.2.0.ckpt) is now the default release model; the SAEs explain v1.1.0. At the next retraining, decide whether to retrain on v1.2.0 (check its preprocessing, since 1.2.x predict passes use_spectrum_utils=False), excluding the test split (Finding 7). The tool and dashboard must state the model version.

Paper: revisit once v1 and the rigour items are done. Before any publication,
revisit every concept-level result: the F1dom analyses (Finding 3 correlations,
depth ordering, Phase 4 tables), the V1-2 concept stratum (mostly m/z detectors,
Finding 9) and the Phase 8 concept groups. They rest on labels with two bugs and
chance-dominated classes (Findings 9 and 10), and need the label rebuild and the
F1dom decoy null above.

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

**8. The V1-4c top candidates are mostly not model failures (descriptive,
found 2026-11-25).** From the V1-4c part A outputs (`analysis/v1_4c/`), no
ablation.
(a) The 7 candidates whose label carries a deamidation and whose prediction
meets the precursor only at isotope offset 1 (ranks 3, 7, 13, 17, 26, 29, 34)
are probable ¹³C-as-deamidation label artefacts (a known database-search
issue): 23 of their 35 true-only peaks (17 of 23 true-only b/y) are ¹³C peaks
of predicted fragments, and b/y favour the prediction in 5 of 7 (V1-4c check
(a)). They are not proven label errors.
(b) Notation-equivalent predictions: N[UNIMOD:7] and D, and Q[UNIMOD:7] and E,
have the same composition. 3 of the 44 candidates are equal to their true
peptide after N[UNIMOD:7]→D, Q[UNIMOD:7]→E, I→L: ranks 1, 5 and 12
(`runtime.notation_equivalent`; `python -m inspector.runtime audit-notation`,
output `analysis/v1_4c/notation_equivalent.json`). Exact-match accuracy counts
them wrong; InstaNovo's mass-based peptide metric (`Metrics.compute_precision_recall`
on each pair) counts them right. That metric gives 127 of the 200 predictions
(V1-4a: 127) against 123 exact: the 3 notation-equivalent pairs plus row 43804
(rank 10, H7; Q vs K, 0.036 Da, inside the metric's 0.1 Da per-residue
tolerance), so InstaNovo's own metric counts H7's case as correct. Q and K are
different residues all the same, and high-resolution data separate them: their
0.036 Da difference (candidates.csv region Δm −0.036) is ~20–36 ppm for a
singly charged fragment at m/z 1,000–1,800, against the 20 ppm fragment
tolerance and an instrument accuracy of 0.93 ppm MAD at the precursor (8(d)),
and in H7's spectrum the y10–y15 and b9–b14 ions containing the residue match
only K. The metric's 0.1 Da per-residue tolerance merges such pairs, so its
accuracy overstates correctness for them (backlog: list the residue pairs it
merges).
(c) The confidence-ranked top 6 hold no clear model failure: two
notation-equivalent (ranks 1, 5), one probable label artefact (3), two
unresolvable from the spectrum (4: no distinguishing peak; 6: one true-only
internal-fragment peak, within check (c)'s chance range of 5–15), and one where
the evidence favours the prediction (2). H7 (rank 10) is the first candidate
with b/y evidence clearly for the true peptide (13 true-only b/y vs 0); ranks 7
and 9 have b/y favouring the prediction (9 vs 1, 3 vs 0) and 8 is ambiguous.
(d) Corpus level, over the 13 local chunks (13,312 spectra; `python -m analysis
--root sae_pipeline_outputs --deamidation-precursor`, outputs
`analysis/label_audit/deamidation_precursor{.csv,_summary.json}`; ppm by
InstaNovo's definition, 20 ppm). 813 labels carry a deamidation (773 one, 40
two or more). (i) None of them, and none of the 10,776 unmodified labels, meets
the precursor only at isotope offset 1, so the labels contain no isotope-offset
matches at all. (ii) 758 of 813 become offset-1 matches when one deamidation is
removed; above ~967 Da this follows from the tolerance alone (the ¹³C step
1.00335 and the deamidation 0.98402 differ by 19.3 mDa) and does not single out
artefacts. (iii) Instrument accuracy from the unmodified labels (10,768 meeting
offset 0): median +0.14 ppm, MAD 0.93 ppm (robust σ 1.38 ppm), while the
19.3 mDa gap is 22.0 / 15.5 / 9.9 / 6.7 ppm at the median mass of the bins
< 967 / 967–1500 / 1500–2500 / ≥ 2500 Da, so the two explanations are
resolvable at every mass. Per label, D = |ppm(deamidated, offset 0) − median| −
|ppm(amidated, offset 1) − median| (D > 0: the ¹³C explanation fits better).
D in units of the gap is bimodal: 473 labels at −1.25…−0.75, 96 / 43 / 55 in
between, 146 at +0.75…+1.25. Within 3 robust σ (4.13 ppm) of the median:

| mass (Da) | n | deamidated only | ¹³C only | both | neither |
|---|---|---|---|---|---|
| < 967 | 36 | 31 | 0 | 0 | 5 |
| 967–1500 | 234 | 191 | 22 | 0 | 21 |
| 1500–2500 | 406 | 246 | 112 | 0 | 48 |
| ≥ 2500 | 137 | 56 | 48 | 19 | 14 |
| all | 813 | 524 | 182 | 19 | 88 |

So 182 of 813 deamidated labels (22%; 165 of 773 with one deamidation, 17 of 40
with two or more) are consistent with a ¹³C misassignment and not with
deamidation at the instrument's accuracy, and 524 (64%) are consistent with
deamidation only. The ¹³C-only share rises with mass (0%, 9%, 28%, 35%), as
expected if the assignment could only be made where the gap fell inside the
search's precursor tolerance. This says which explanation the precursor mass
supports, not which label is wrong.

**9. Label quality: two labelling bugs, and chance matches that contaminate
about half of the label-derived concepts (descriptive, found 2026-11-25).**
`python -m analysis --root sae_pipeline_outputs --label-audit` (outputs
`analysis/label_audit/`; 56 s on this machine: reproduction 7 s, annotation
28 s, metrics 20 s). 200 local spectra, `default_rng(0)`, 15–16 per chunk
over the 13 chunks, 22,682 peaks, re-annotated from the stored processed peaks.
(0) Reproduction: `annotate.annotate_spectrum` with the pipeline settings
reproduces the stored labels, ion types, indices, charges and loss ids exactly
for all 200. Annotator settings: 20 ppm; b, y, immonium, precursor and internal
fragments (b-type, ≥ 2 residues, never residue 1 or L); H2O and NH3 losses, one
at a time, on every ion type; fragment charges 1…max(1, z − 1) (the
spectrum_utils default `max_ion_charge`, never passed; immonium charge 1);
`max_isotope` 0; one label per peak by `_candidate_sort_key`: monoisotopic >
no loss > b/y > a/c/x/z > immonium > internal > precursor > lower charge >
smaller |ppm|. Pipeline labels in the sample: b 1,489, y 2,846, internal 2,931,
immonium 706, precursor 48, noise 14,662.
Labelling bugs (confirmed with toy inputs, `tests/test_label_audit.py`):
- **Immonium labels ignore the peptide.** spectrum_utils generates immonium
  ions for all 23 residue letters (including J, O, U) whatever the peptide, so
  `is_I_ion` is a fixed set of m/z values, the same for every peptide; 398 of
  the 706 immonium-labelled peaks (56%) name a residue the peptide lacks
  (I/L/J counted as one). Through `IonGeometry.covers_any_residue` these also
  set covers_<residue> for an absent residue: covers_Y 100 of 1,294
  positives, covers_F 87 / 1,348, covers_W 31 / 401, covers_M 22 / 702,
  covers_E 19 / 2,941, others ≤ 4.
- **Internal fragments cover one residue too many.** spectrum_utils labels
  residues s…e−1 as `m{s}:{e}` (end exclusive: H6's m13:15, m/z 215.139, is
  "IT", residues 13–14), but `IonGeometry.covers` treats the end as inclusive,
  so covers_* and ion_contains_* of an internal ion include the next residue,
  up to the C-terminal residue that internal fragments never contain.
  Positives due only to that residue: covers_K 166 of 2,565, covers_P 139 /
  1,731, covers_E 136 / 2,941, covers_F 101, covers_Y 91, covers_D 79,
  covers_N 73, covers_R 56, covers_M 37, covers_W 33, covers_C 7;
  ion_contains_deamidation 7 / 132, ion_contains_cam_cys 7 / 162.
  Together the two bugs account for 3.5–16% of each covers_* concept's
  positives (W 16%, Y 15%, F 14%, M 8%, P 8%, K 7%, E 5%, N 4%, R 4%, C 4%,
  D 3.5%).
(1) Chance matches: each spectrum against its true peptide and 20
same-composition decoys (`annotate.peptide_decoys`, C-terminal residue fixed;
195 spectra, 5 short peptides cannot give 20). Chance fraction = mean decoy
count / true count, bootstrap 95% CI over spectra (2,000). Immonium,
precursor, y1, b_{L−1} and a_{L−1} winners are excluded from both sides: their
m/z is the same for every decoy (y1 1.96 / spectrum, immonium 3.53, precursor
0.24, b_{L−1} 0.14). Per spectrum (full table `per_class.csv`):

| class | loss | charge | true | decoys | chance fraction [95% CI] |
|---|---|---|---|---|---|
| y | none | 1 | 8.23 | 0.57 | 0.069 [0.062, 0.076] |
| y | all | all | 12.42 | 1.35 | 0.109 [0.101, 0.117] |
| y | H2O / NH3 | all | 2.04 / 1.73 | 0.40 / 0.32 | 0.19 / 0.18 |
| y | all | 2 | 0.65 | 0.17 | 0.27 [0.21, 0.34] |
| b | none | 1 | 4.14 | 0.72 | 0.174 [0.155, 0.195] |
| b | all | all | 7.43 | 1.32 | 0.177 [0.161, 0.196] |
| b | H2O / NH3 | all | 2.06 / 1.08 | 0.32 / 0.22 | 0.16 / 0.20 |
| b | all | 2 | 0.31 | 0.14 | 0.45 [0.32, 0.67] |
| internal | none | 1 | 9.98 | 4.31 | 0.43 [0.40, 0.47] |
| internal | all | all | 15.00 | 8.03 | 0.54 [0.50, 0.57] |
| internal | H2O / NH3 | all | 3.32 / 1.39 | 2.09 / 1.28 | 0.63 / 0.92 |
| internal | all | 2 | 0.72 | 0.77 | 1.06 [0.88, 1.35] |
| a (a ions on) | all | all | 3.28 | 1.22 | 0.37 [0.34, 0.42] |
| b + y + internal | all | all | 34.85 | 10.70 | 0.31 [0.29, 0.32] |

(2) Isotope peaks in "noise": with max_isotope 1, 5.4% of the 14,411 noise
peaks become ¹³C peaks of fragments against 3.7% for decoys, excess 1.7%
[1.2, 2.2]; requiring the same ion's monoisotopic peak to be labelled, 2.5% vs
0.4%, excess 2.1% [1.7, 2.4]. By m/z: < 300 excess 0.1% [−0.2, 0.5]; 300–1,000
1.0% [0.4, 1.6]; ≥ 1,000 8.3% [6.3, 10.4] (with monoisotopic partner 7.5%).
The excess is b and y (y 284 vs decoys 108, b 144 vs 97); internal isotope
matches are at chance (334 vs 491).
(3) Precedence: 407 of 8,020 matched peaks (5.1%; decoys 3.0%) match more than
one class. Of 1,524 peaks with a b candidate, 35 are labelled otherwise: 17
internal and 18 y; 26 of the 35 because the b candidate carries a loss and the
winner does not (loss is ranked above ion class), 9 on charge or ppm against a
y. Of 2,918 peaks with a y candidate, 72 are labelled otherwise (52 internal,
19 b, 1 immonium; 68 by the loss rule).
(4) Coverage (observed / theoretical indices, y1 and b_{L−1} excluded,
`coverage.csv`): charge 1 without losses b 0.35 [0.32, 0.38] vs y 0.70 [0.66,
0.73], decoys 0.06 / 0.05, chance fraction b 0.17, y 0.07; with losses b 0.40,
y 0.71. Charge 2 (66 spectra with z ≥ 3): b 0.03, y 0.08 without losses
(chance fraction 0.37 / 0.17). a ions: 539 of the noise peaks (3.7%; decoys
1.0%, excess 2.8% [2.3, 3.3]) become a ions, 181 of them a2 (excess 0.9%);
internal → a 49 of 2,925 (1.7% vs 1.3%, excess 0.4% [−0.2, 1.0]). Under the
current concept code an a ion gets no position or cover concepts.
(5) Concepts (`concepts.csv`; chance fraction over non-determined peaks; F1dom
not recomputed, dictionary not rebuilt):
- Label-derived and reliable (chance fraction ≤ 0.2, CI mostly below): is_y_ion
  0.11, position_middle 0.09, cleaves_N_to_Pro 0.10, cleaves_C_to_Asp 0.11,
  cleaves_C_to_Glu 0.12, position_Nterm 0.16, is_b_ion 0.18 [0.16, 0.20],
  position_Cterm 0.18 (420 of its 1,292 positives are y1 or b_{L−1}), covers_R 0.17,
  covers_C 0.19 [0.14, 0.26], ion_contains_cam_cys 0.19 [0.14, 0.26].
- Contaminated by chance (> 0.2): is_internal_fragment 0.54, is_fragment_charge_2
  0.64, is_NH3_loss 0.43, has_neutral_loss 0.40, is_H2O_loss 0.38,
  ion_contains_oxidation 0.50 [0.26, 0.81], ion_contains_deamidation 0.32,
  is_matched_peak 0.31, covers_M 0.31, covers_F 0.30, covers_D 0.28,
  covers_Y 0.28, covers_N 0.28, covers_E 0.26, covers_P 0.26, cleaves_C_to_Lys
  0.24, covers_W 0.23, covers_K 0.23, is_last_ion 0.22, is_fragment_charge_1
  0.29, cleaves_C_to_Arg 0.20 [0.10, 0.36]. covers_* and ion_contains_* also
  carry the bug positives above.
- Peptide- or composition-determined: is_I_ion (peptide-independent, 56%
  absent residues), is_precursor_related, and is_first_ion, 392 of whose 412
  positives are y1 (two m/z values for tryptic K/R peptides; the b1 remainder
  is too small to judge, chance 0.83 [0.58, 1.44]). These are m/z detectors
  in practice.
- Not label-derived: the mz bins, top_decile_intensity, precursor charge,
  peptide length, spectrum_contains_* (unaffected).
- Change under (2), isotope peaks labelled: is_y_ion +286 (+10%; decoys +108),
  is_b_ion +146 (+10%; +97), is_internal_fragment +335 (decoys +491), covers_*
  +11–17%, most of it matched by decoys (+8–17%). Under (4), a ions on: is_matched_peak +550 (decoys +184),
  is_internal_fragment −49, is_I_ion −46 (a1 coincides with the immonium of
  residue 1), has_neutral_loss +228 (decoys +88).
Results that rest on the affected concepts:
- Dictionary: the top concept is covers_* for 535 / 626 / 633 / 679 features
  at L2/4/6/8, ion_contains_* for 360 / 356 / 279 / 230, is_I_ion for 290 /
  363 / 362 / 325, and is_internal_fragment for 82 / 132 / 161 / 299; their
  F1dom and associations use the affected labels.
- V1-4c case views: peak labels and concepts come from the same annotator.
  H6's evidence is F6206's top concept is_I_ion (F1dom 0.137), a
  peptide-independent label, and one internal-fragment match within chance; H2
  rests on is_internal_fragment features and internal-fragment peaks (chance
  0.54); H1 and H5 name covers_N / covers_D features and H7 a covers_E feature
  (chance 0.26–0.28 plus 3.5–5% bug positives). The candidates' evidence
  classes count internal (and immonium) peaks as fragments. The hypotheses are
  unchanged.
- V1-2 concept stratum (selected by F1dom over chemical concepts; 40 per layer):
  top concept is_I_ion for 14 / 9 / 8 and is_first_ion for 9 / 12 / 7 at
  L2/4/6; with the mz bins and intensity (17 / 16 / 22), at most one feature
  per layer has a label-derived concept with chance ≤ 0.2. The "positive
  control" is mostly m/z detectors.
- Phase 4 and the evaluation reports use the same labels (not re-examined here).

**10. After the display rule, concept associations are negligible for nearly
all features (descriptive, found 2026-11-25, V1-3b).** Only 8 of the 50
concepts meet the display rule (chance fraction ≤ 0.2, no bug-added positives,
peptide-dependent): is_b_ion, is_y_ion, the three position_* and
cleaves_N_to_Pro / C_to_Asp / C_to_Glu. Over the rebuilt dictionaries (`python
-m analysis --root sae_pipeline_outputs --reliable-associations`,
`analysis/reliable_associations_summary.json`), 4,078 / 4,278 / 4,455 / 5,491
live features at L2/4/6/8 have a BH-significant association with one of them,
but the median of their best reliable F1dom is 0.0012 / 0.0014 / 0.0013 /
0.0015. It reaches 0.1 for 44 / 47 / 62 / 97 features (listed in
`analysis/reliable_associations.csv`) and 0.3 for 0 / 2 / 2 / 2. Of the
listed features' best reliable concepts, 69 are position_Cterm, 56 is_y_ion, 44
position_middle and 35 is_b_ion; the six at ≥ 0.3 include the Finding 2 hubs L4
F11376 and L6 F9282 / F10689. So concept labels cannot carry the tool's
interpretation of features. Measured effects (exact ablation, V1-4b), feature
examples (V1-5) and blinded descriptions must; concept associations are
secondary evidence, shown with their F1dom and chance fraction.

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
- 2026-11-22 — Public-facing fixes (`c1b7c35`), then a staleness audit: stale docs and comments fixed and the V1-7 README note extended (`b1ef982`); `build_dictionary` raises on a description run without a `blind` column and `interpret.py` defaults `--env-file` to the repo-root `.env` (see Behaviour changes). Rebuilding layers 2/4/6 in memory with `blind_v1` reproduces the stored dictionaries exactly. `extract.load_layer_activations` (unused) kept; `run_pipeline.sh` Step 6 stays in the backlog. 130 tests pass. No API calls.
- 2026-11-24 — S_rand in V1.1-3 fixed before it runs (`992b3b4`): frozen-decoder SAE preferred, the untrained initialisation as fallback, with train.py lines cited. V1-4b planned, then revised (`8d6793b`) after a read-only preview showed attribution patching unreliable at the top of the ranking (large effects underestimated up to 21×, sign errors): the tool uses exact error-preserving ablation only (local and global scopes, per-peak and latent panels over every active feature, a joint ablation per token, an optional target peptide). The AtP study and the preview's numbers moved to V1.1-1; offline exhaustive ablation for the demo set added to V1-5. No code changed.
- 2026-11-24 — V1-4b done (`80e2653`): Explanation schema 2 (summaries plus dictionary hash), exact error-preserving ablation (local/global scopes, `ablate_token` over every active feature at a token plus a joint ablation, optional target peptide), `explain(latent_ablation=)`. Results and timings under V1-4b; behaviour changes recorded; two backlog items (re-decode under ablation, faster per-peak clicks). 144 tests pass.
- 2026-11-24 — CLAUDE.md and STATUS.md brought up to date after V1-4b: CLAUDE.md goal, layout, dependency and causal-effect rules; V1-4a notes the renamed flag; V1-4c gets its metric and contexts; V1-6 spec built on the V1-4b API. No code changed.
- 2026-11-24 — V1-4b commits pushed (`992b3b4`..`f8eb7df`); the joint-ablation caveat added to V1-4b's results (the joint figure is a reference scale, not a clean total). No code changed.
- 2026-11-24 — V1-4c part A, case selection: `runtime.py` gets the `audit-cases` subcommand and pure helpers (`differing_region`, `mismatch_type`, `distinguishing_peaks`, `evidence_class`, `mean_residue_log_prob`, `select_audit_cases`, `case_view`), additive; the CLI's `main` now returns a subcommand's exit code (the existing ones return 0 as before). The regenerated predictions reproduce V1-4a (123 exact, 167 meeting the precursor); 44 candidates, 6 selected; table under V1-4c. No ablation calls. 167 tests pass (23 new).
- 2026-11-24 — V1-4c part A, pre-registration, amended at review before commit: hypotheses H1–H7 (H7 a logged deviation; all at layer 8, a post-hoc choice now recorded) and the part-B protocol under V1-4c (ctx_t target = the prediction outside the region plus the I→L-mapped true region, since ablation_context does not canonicalise I/L; control pools with a 0.5× activation floor and an underpowered mark; removed-norm ratios; "undetermined" below 10 distinct control draws, which H3, H4 and H6 hit by construction). Descriptive checks (no ablation) under V1-4c: 23 of 35 true-only peaks of the 7 isotope-offset deamidation candidates are 13C peaks of predicted fragments; decoys match 5–8% of peaks as internal fragments. hypotheses.json sha256 2177039b…e84a9. Not in this commit (STATUS.md only, as asked): the check code (`runtime.audit_descriptive_checks`, `control_pool`, `n_distinct_control_sets`, `peptide_decoys`, the `audit-checks` subcommand), the additive `max_isotope` flag of `annotate.run_spectrum_utils` (default 0, unchanged behaviour) and 4 tests; they stay in the working tree for review and go into the repo with part B. No ablation calls.
- 2026-11-25 — Committed the V1-4c descriptive-check code and the `max_isotope` flag (`f4933d3`, 171 tests). Finding 8 (descriptive) and review notes under V1-4c (`82ed2f6`): ranks 1, 5, 12 are notation-equivalent; InstaNovo's mass metric (127/200) also counts H7's Q/K case correct; corpus-level, 182 of 813 local deamidated labels (22%) fit a ¹³C misassignment and not deamidation at the instrument's accuracy (MAD 0.93 ppm). Label-quality audit (Finding 9, `analysis --label-audit`, 56 s; 8(d) pass 6 s): stored labels reproduce exactly on 200 spectra; two labelling bugs (immonium for absent residues, 56% of immonium labels; internal-fragment end off by one), chance fractions per class and concept; annotator decisions moved to the backlog. CLAUDE.md's /doctor trim was not committed; restoring it was blocked by the permission classifier and is left to the user. No ablation, API calls or rebuilds. 191 tests pass.
- 2026-11-25 — V1-3b done: concept reliability table (`1ffa8ea`; the label audit also counts what an immonium filter would change, `concepts.csv` gains two columns, the rest byte-identical) and dictionary schema 2 with reliability beside every association and `reliable_*` associations (`31338d9`; all four layers rebuilt, existing columns identical; Explanation schema 3). Only 8 of 50 concepts meet the display rule; Finding 9's "reliable" covers_R, covers_C and ion_contains_cam_cys carry 4.1–4.3% bug-added positives, and the immonium bug also hides one y positive each from is_y_ion and position_Cterm, so the rule counts bug-added positives only (hidden ones are reported). Finding 10: reliable associations are negligible for nearly all features (median best F1dom 0.0012–0.0015); `analysis/reliable_associations.csv` lists the 250 at ≥ 0.1; V1-5 and V1-6 specs now lead with examples and measured effects. V1-4c amendment 1 (`0a87fee`): evidence classes on b/y peaks only, 8 of 44 candidates change, hypotheses unchanged. Finding 8(b) addendum (Q/K merged by the mass metric); backlog: full label rebuild, F1dom decoy null, merged residue pairs; paper note on concept-level results. No API, GPU or cluster. 203 tests pass.
- 2026-11-25 — V1-4c part B done. Amendment 2 committed before any ablation on the selected cases (`16ec5a2`): H7's peak-matched secondary control, the sampling rule (rejection, at most 10,000 attempts), "distinct control draws" = possible sets, ties at 1e-6; H8 and the noise-peak check dropped. Code (`800558a`): `ablate_set` / `ablate_sets` (joint ablation of arbitrary (feature, token) sets), the part B helpers and `audit-part-b`; 215 tests pass (12 new). Run 272 s, every reproduction check held. Outcomes: H5 supported (a notation choice, D vs N[UNIMOD:7]), H1, H2 and H7 not supported, H3, H4 and H6 undetermined by construction; ~0.19 false "supported" expected across the 4 decidable tests. Step (c), the H7 exploratory sweep and the V1-5 demo picks (H7, H1, H5 with labels) recorded under V1-4c. No API, GPU or cluster.
- 2026-11-25 — V1-4c exploratory H7 sweep recorded in full under V1-4c (two top-10 tables, per-peak joints, interpretation, activation as a screen: 8 of the 20 largest effects lie outside the top 8 by activation, limits). Backlog: pre-registered H7 peak-shift test (y10–y15 shifted by ±0.036 Da, no SAE), high priority after V1-6. No code or runs.
- 2026-11-26 — V1-5 pre-registration and spec committed under V1-5 (no code, no runs): decisions D1–D4 (10-row demo set incl. H5 labelled a notation choice; K = 20 examples, one per spectrum; label-peptide context; outputs in `analysis/v1_5/` and `dictionary/`), the seeded random-sample rule for the screen (chunks 520/572, `default_rng(0)`, longest prefix ≤ 9 h under a frozen cost model), screen definitions (a)–(c) with bootstrap over spectra, steps 1–5, files, tests, commits. Demo ablations are live in the dashboard; no global exhaustive scope. Four backlog items added.
