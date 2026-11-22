"""Tests for interpret.py: blinding, the spectrum-level split, example ids, scoring.

Run from the repo root: ``PYTHONPATH=src uv run python -m pytest tests``.

No API calls and no chunks on disk. Records are built by hand in the shape
_collect_activations produces, so build_examples, merge_global_top and
_make_example run on them unchanged. Annotation-derived values carry sentinel
strings (``ZZ...``) that cannot occur in a legitimate blind prompt, so a leak
shows up as a substring match.
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

import interpret as it

ION_SENTINEL = "ZZIONTYPE"
PEPTIDE_SENTINEL = "ZZPEPTIDE"
CONCEPT_SENTINEL = "ZZCONCEPT"
SENTINELS = (ION_SENTINEL, PEPTIDE_SENTINEL, CONCEPT_SENTINEL)
ION_VOCAB = {0: f"noise_{ION_SENTINEL}", 1: f"y_{ION_SENTINEL}", 2: f"b_{ION_SENTINEL}"}
CONCEPTS = [f"is_y_{CONCEPT_SENTINEL}", f"is_b_{CONCEPT_SENTINEL}"]

PEAKS_PER_SPECTRUM = 20          # plus one latent token at position 0


def _record(chunk_idx: int, n_spectra: int, offset: int, seed: int) -> dict:
    """One chunk record in the shape _collect_activations builds."""
    g = np.random.default_rng(seed)
    per = PEAKS_PER_SPECTRUM + 1
    n = n_spectra * per
    t2s = np.repeat(np.arange(n_spectra), per)
    t2p = np.tile(np.arange(per), n_spectra)
    labels = torch.zeros(n, len(CONCEPTS), dtype=torch.bool)
    labels[:, 0] = torch.from_numpy(g.random(n) < 0.3)
    return {
        "chunk_idx": chunk_idx, "offset": offset, "n_tokens": n, "labels": labels,
        "token_to_spectrum": t2s, "token_to_position": t2p,
        "ion_type_ids": g.integers(0, 3, n),
        "peak_mzs": np.where(t2p == 0, 0.0, g.uniform(100, 1500, n)),
        "peak_intensities": np.where(t2p == 0, 0.0, g.uniform(1, 1000, n)),
        "peptides": [f"PEP[+15.99]{PEPTIDE_SENTINEL}{i}" for i in range(n_spectra)],
        "precursor_mzs": g.uniform(400, 1200, n_spectra),
        "precursor_charges": g.integers(2, 4, n_spectra),
    }


def _records(seed: int = 0) -> list[dict]:
    first = _record(0, 60, 0, seed)
    second = _record(52, 60, first["n_tokens"], seed + 1)
    return [first, second]


def _activations(records: list[dict], seed: int, sparsity: float = 0.15) -> np.ndarray:
    """Activations over the concatenated records: mostly zero, the rest uniform in (0, 1]."""
    g = np.random.default_rng(seed)
    n = sum(r["n_tokens"] for r in records)
    acts = g.uniform(0.01, 1.0, n) * (g.random(n) < sparsity)
    return acts.astype(np.float32)


def _global_top(records: list[dict], acts: np.ndarray, seed: int, k: int = 4):
    """Phase 3 top-K rows: tokens in the sample, with activations above anything sampled."""
    g = np.random.default_rng(seed)
    rec = records[1]
    zero_tokens = np.flatnonzero(acts[rec["offset"]:rec["offset"] + rec["n_tokens"]] == 0)
    tokens = g.choice(zero_tokens, size=k, replace=False)
    peak = float(acts.max())
    return [(peak * (2.0 + i), rec["chunk_idx"], int(t)) for i, t in enumerate(tokens)]


def _prepare(fi: int, seed: int = 0, layer: int = 2, records=None, acts=None, top=None):
    records = records if records is not None else _records()
    acts = acts if acts is not None else _activations(records, seed=1000 + fi)
    top = top if top is not None else _global_top(records, acts, seed=fi)
    return it.prepare_feature(fi, acts, top, records, ION_VOCAB, CONCEPTS, seed, layer)


def _config(tmp_path: Path, blind: bool, **kw) -> it.InterpretConfig:
    return it.InterpretConfig(
        eval_dir=tmp_path, extract_dir=tmp_path, annotation_dir=tmp_path,
        sae_checkpoint=tmp_path / "c.pt", output_dir=tmp_path / "run", target_layer=2,
        run_tag="t", blind=blind, **kw,
    )


# --- blinding -----------------------------------------------------------------


@pytest.fixture
def split():
    out = _prepare(7)
    assert out is not None
    return out


def _table_cells(prompt: str) -> tuple[list[str], list[list[str]]]:
    """Header and rows of the CSV table at the end of a prompt."""
    lines = prompt.rstrip("\n").splitlines()
    start = max(i for i, line in enumerate(lines) if line.startswith("example_id,"))
    header = lines[start].split(",")
    rows = [line.split(",") for line in lines[start + 1:]]
    return header, rows


def _assert_blind(prompt: str, with_activation: bool) -> None:
    for sentinel in SENTINELS:
        assert sentinel not in prompt
    for column in ("matched_ion", "peptide_proforma", "concepts"):
        assert column not in prompt
    header, rows = _table_cells(prompt)
    expected = list(it.BLIND_COLUMNS)
    if with_activation:
        expected.insert(1, "activation")
    assert header == expected
    kind_col, neighbour_col = header.index("token_kind"), header.index("nearby_peaks")
    for row in rows:
        assert len(row) == len(header)
        assert row[kind_col] in ("peak", it.LATENT_MARKER)
        for cell in row[neighbour_col].split():
            assert re.fullmatch(r"\d+\.\d+@\d+\.\d+", cell), cell


def test_blind_prompts_carry_no_annotation(split):
    describe, holdout = split
    _assert_blind(it.build_description_prompt(describe, False, blind=True), with_activation=True)
    _assert_blind(it.build_prediction_prompt("some description", holdout, False, blind=True),
                  with_activation=False)


def test_blind_preamble_mentions_no_label_column():
    for word in ("matched_ion", "peptide_proforma", "noise", "ProForma"):
        assert word not in it.BLIND_PREAMBLE


def test_unblinded_prompts_do_show_the_sentinels(split):
    """Control: the sentinels reach an unblinded prompt, so the blind test can see a leak."""
    describe, _ = split
    prompt = it.build_description_prompt(describe, include_concepts=True, blind=False)
    for sentinel in SENTINELS:
        assert sentinel in prompt


def test_latent_marker_survives_blinding():
    records = _records()
    rec = records[0]
    latent = it._make_example(rec, 0, 0.5, 0, ION_VOCAB, CONCEPTS)
    peak = it._make_example(rec, 3, 0.2, 1, ION_VOCAB, CONCEPTS)
    table = it.format_example_table([latent, peak], include_activation=False,
                                    include_concepts=False, blind=True)
    assert it.LATENT_MARKER in table
    assert table.splitlines()[2].split(",")[1] == "peak"


def test_blind_with_concepts_is_refused(split):
    with pytest.raises(ValueError, match="annotation"):
        it.format_example_table(split[0], include_activation=True, include_concepts=True,
                                blind=True)


def test_interpret_feature_dry_run_prompts_are_blind(tmp_path, split):
    config = _config(tmp_path, blind=True, dry_run=True)
    result = it.interpret_feature(None, config, 7, "concept", *split)
    _assert_blind(result["description_prompt"], with_activation=True)
    _assert_blind(result["prediction_prompt"], with_activation=False)


class _FakeClient:
    """Stands in for OpenAI: records every prompt and answers in the expected formats."""

    def __init__(self):
        self.prompts: list[str] = []
        self.chat = self
        self.completions = self

    def create(self, model, messages, **_kw):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if "predicted_activation" in prompt:
            _header, rows = _table_cells(prompt)
            text = "example_id,predicted_activation\n" + "\n".join(
                f"{row[0]},{(int(row[0]) % 7) / 7:.2f}" for row in rows)
        else:
            text = "DESCRIPTION: The activation patterns are characterized by: x\n" \
                   "SUMMARY: The feature activates on x"
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
        choice = type("C", (), {"message": type("M", (), {"content": text})()})()
        return type("R", (), {"choices": [choice], "usage": usage, "model": model})()


def test_every_api_prompt_in_a_blind_run_is_blind(tmp_path):
    """Description, prediction and shuffled-baseline calls, through the real call path."""
    client = _FakeClient()
    config = _config(tmp_path, blind=True, n_shuffles=2)
    config.output_dir.mkdir(parents=True)
    results = []
    for fi in (3, 5, 9, 11):
        describe, holdout = _prepare(fi)
        results.append(it.interpret_feature(client, config, fi, "concept", describe, holdout))
    it.run_shuffled_baseline(client, config, results)
    assert len(client.prompts) == 4 * 2 + 4 * 2
    for prompt in client.prompts:
        _assert_blind(prompt, with_activation="'activation' column" in prompt)
    assert all(len(r["shuffled"]["r_values"]) == 2 for r in results)


# --- split, ids and row order ---------------------------------------------------


def test_no_spectrum_on_both_sides_and_sizes_balanced():
    for fi in range(30):
        out = _prepare(fi)
        if out is None:
            continue
        describe, holdout = out
        spectra = lambda xs: {(e.chunk_idx, e.spectrum_in_chunk) for e in xs}
        assert not spectra(describe) & spectra(holdout)
        assert abs(len(describe) - len(holdout)) <= 2
        ids = [e.example_id for e in describe + holdout]
        assert sorted(ids) == list(range(len(ids)))


def test_split_keeps_a_multi_example_spectrum_together():
    base = {"activation": 0.5, "chunk_idx": 0, "token_in_chunk": 0, "is_latent": False,
            "peak_mz": 1.0, "peak_intensity": 1.0, "ion_type": "y", "peak_rank": 1,
            "n_peaks": 5, "peptide": "P", "precursor_mz": 1.0, "precursor_charge": 2,
            "neighbours": [], "concepts": []}
    examples = [it.TokenExample(example_id=i, spectrum_in_chunk=i // 3, **base) for i in range(12)]
    describe, holdout = it.split_by_spectrum(examples, random.Random(0))
    by_side = {e.spectrum_in_chunk: side for side, xs in (("d", describe), ("h", holdout))
               for e in xs}
    for e in examples:
        assert by_side[e.spectrum_in_chunk] == ("d" if e in describe else "h")


def _holdout_rho(fi: int, reassign: bool) -> tuple[float, bool, int]:
    """Spearman(example_id, activation) on the held-out half; whether the top example has the top id."""
    records = _records()
    acts = _activations(records, seed=1000 + fi)
    top = _global_top(records, acts, seed=fi)
    rng = it.feature_rng(0, 2, fi)
    peak = it.feature_peak(acts, top)
    examples = it.build_examples(acts, peak, records, ION_VOCAB, CONCEPTS, rng)
    examples = it.merge_global_top(examples, top, peak, records, ION_VOCAB, CONCEPTS)
    if reassign:
        examples = it.reassign_example_ids(examples, rng)
    _describe, holdout = it.split_by_spectrum(examples, rng)
    ids = [e.example_id for e in holdout]
    acts_h = [e.activation for e in holdout]
    top_has_max_id = ids[int(np.argmax(acts_h))] == max(ids)
    return float(spearmanr(ids, acts_h).statistic), top_has_max_id, len(holdout)


def test_held_out_example_id_is_uncorrelated_with_activation():
    """The merged global-top rows are the strongest activations; their ids must not say so."""
    n = 200
    stats = [_holdout_rho(fi, reassign=True) for fi in range(n)]
    rhos = [s[0] for s in stats if not math.isnan(s[0])]
    assert abs(np.mean(rhos)) < 0.05
    # The strongest held-out example holds the largest id about 1/n_holdout of the time.
    hits = sum(s[1] for s in stats)
    p = np.mean([1 / s[2] for s in stats])
    assert hits <= n * p + 3 * math.sqrt(n * p * (1 - p))


def test_without_reassignment_the_id_side_channel_is_visible():
    """Control: the same measurement catches merge_global_top's appended ids."""
    n = 200
    stats = [_holdout_rho(fi, reassign=False) for fi in range(n)]
    rhos = [s[0] for s in stats if not math.isnan(s[0])]
    assert np.mean(rhos) > 0.1
    # Fails the bound the test above enforces.
    hits = sum(s[1] for s in stats)
    p = np.mean([1 / s[2] for s in stats])
    assert hits > n * p + 3 * math.sqrt(n * p * (1 - p))


def test_held_out_table_rows_are_not_sorted(split):
    _describe, holdout = split
    prompt = it.build_prediction_prompt("d", holdout, False, blind=True)
    _header, rows = _table_cells(prompt)
    ids = [int(r[0]) for r in rows]
    assert ids != sorted(ids) and ids != sorted(ids, reverse=True)
    acts = [e.activation for e in holdout]
    assert acts != sorted(acts) and acts != sorted(acts, reverse=True)


# --- determinism -------------------------------------------------------------------


def _signature(split) -> list:
    return [[(e.example_id, e.chunk_idx, e.token_in_chunk, e.activation) for e in side]
            for side in split]


def test_feature_rng_is_keyed_and_stable():
    draw = lambda *key: it.feature_rng(*key).random()
    assert draw(0, 2, 5) == draw(0, 2, 5)
    assert len({draw(0, 2, 5), draw(0, 4, 5), draw(0, 2, 6), draw(1, 2, 5)}) == 4
    assert it.feature_rng(0, 2, 5, "shuffle").random() != draw(0, 2, 5)


def test_resumed_run_prepares_the_same_examples_as_an_uninterrupted_one():
    records = _records()
    features = [3, 8, 13, 21]
    acts = {fi: _activations(records, seed=1000 + fi) for fi in features}
    tops = {fi: _global_top(records, acts[fi], seed=fi) for fi in features}

    def run(todo):
        return {fi: _prepare(fi, records=records, acts=acts[fi], top=tops[fi]) for fi in todo}

    uninterrupted = run(features)
    resumed = run([fi for fi in features if fi not in {3, 8}])   # 3 and 8 already completed
    for fi in (13, 21):
        assert resumed[fi] is not None
        assert _signature(resumed[fi]) == _signature(uninterrupted[fi])
        # and the rendered tables, row order included
        assert it.build_prediction_prompt("d", resumed[fi][1], False) == \
            it.build_prediction_prompt("d", uninterrupted[fi][1], False)


def test_blind_and_unblinded_runs_see_the_same_examples(tmp_path):
    """The two runs share sampling and split, so only the rendering differs."""
    a, b = _prepare(4), _prepare(4)
    assert _signature(a) == _signature(b)


# --- scoring and baseline --------------------------------------------------------


def test_detection_auroc():
    assert it.detection_auroc([(0.9, 1.0), (0.8, 0.5), (0.1, 0.0), (0.0, 0.0)]) == 1.0
    assert it.detection_auroc([(0.0, 1.0), (0.9, 0.0)]) == 0.0
    assert it.detection_auroc([(0.5, 1.0), (0.5, 0.0)]) == 0.5
    assert it.detection_auroc([(0.9, 1.0), (0.5, 0.0), (0.9, 0.0)]) == pytest.approx(0.75)
    assert math.isnan(it.detection_auroc([(0.9, 1.0), (0.2, 0.3)]))
    assert math.isnan(it.detection_auroc([(0.9, 0.0), (0.2, 0.0)]))


def test_constant_prediction_scores_zero_and_nan_is_reserved_for_untestable_pairs():
    varying = [0.0, 0.4, 0.9, 0.0]
    assert it.pearson_r([(0.3, m) for m in varying]) == 0.0
    assert it.pearson_r([(0.3, 0.0), (0.3, 0.5), (0.3, 1.0)]) == 0.0    # exactly 3
    assert math.isnan(it.pearson_r([(0.1, 0.0), (0.9, 0.5)]))           # < 3 predictions
    assert math.isnan(it.pearson_r([(0.1, 0.5), (0.4, 0.5), (0.9, 0.5)]))  # constant measured
    assert math.isnan(it.pearson_r([(0.3, 0.5)] * 4))                    # both constant
    assert it.pearson_r([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]) == pytest.approx(1.0)
    # A constant prediction scores AUROC 0.5 (all ties), unchanged by the rule.
    assert it.detection_auroc([(0.3, m) for m in varying]) == 0.5


class _SometimesConstantClient(_FakeClient):
    """Every other prediction call answers one value for every row."""

    def __init__(self):
        super().__init__()
        self.n_predict = 0

    def create(self, model, messages, **kw):
        reply = super().create(model, messages, **kw)
        if "predicted_activation" in messages[0]["content"]:
            self.n_predict += 1
            if self.n_predict % 2:
                _header, rows = _table_cells(messages[0]["content"])
                reply.choices[0].message.content = "example_id,predicted_activation\n" + \
                    "\n".join(f"{row[0]},0.30" for row in rows)
        return reply


def _as_legacy_log(path: Path, field: str) -> int:
    """Rewrite a resume log the way the pre-rule code wrote it: constant-prediction r as NaN."""
    lines, n = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if field == "pearson_r" and rec["pearson_r"] == 0.0:
            rec["pearson_r"], n = float("nan"), n + 1
        if field == "r_values":
            n += sum(v == 0.0 for v in rec["r_values"])
            rec["r_values"] = [float("nan") if v == 0.0 else v for v in rec["r_values"]]
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def test_rescore_recomputes_legacy_logs_without_api_calls(tmp_path):
    """Logs written under the NaN rule rescore to what a fresh run under the 0 rule gives."""
    client = _SometimesConstantClient()
    config = _config(tmp_path, blind=True, n_shuffles=3)
    config.output_dir.mkdir(parents=True)
    fresh = []
    for fi in (3, 5, 9, 11, 13):
        result = it.interpret_feature(client, config, fi, "concept", *_prepare(fi))
        fresh.append(result)
        it.append_completed(config.output_dir, result)
    it.run_shuffled_baseline(client, config, fresh)
    fresh_threshold = it.apply_baseline(fresh)
    n_calls = len(client.prompts)
    assert any(r["pearson_r"] == 0.0 for r in fresh)
    assert any(v == 0.0 for r in fresh for v in r["shuffled"]["r_values"])

    n_own = _as_legacy_log(it._resume_path(config.output_dir), "pearson_r")
    n_null = _as_legacy_log(it._shuffled_resume_path(config.output_dir), "r_values")
    assert n_own and n_null

    loaded = it.load_completed(config.output_dir)
    assert sum(math.isnan(r["pearson_r"]) for r in loaded.values()) == n_own
    assert sum(it.rescore_result(r) for r in loaded.values()) == n_own
    rescored = [loaded[fi] for fi in sorted(loaded)]
    it.run_shuffled_baseline(None, config, rescored)       # client=None: logs only
    assert len(client.prompts) == n_calls
    assert it.apply_baseline(rescored) == pytest.approx(fresh_threshold)
    by_fi = {r["feature_idx"]: r for r in fresh}
    for r in rescored:
        f = by_fi[r["feature_idx"]]
        assert r["pearson_r"] == pytest.approx(f["pearson_r"])
        assert r["detection_auroc"] == pytest.approx(f["detection_auroc"])
        assert r["shuffled"]["r_values"] == pytest.approx(f["shuffled"]["r_values"])
        assert r["description_beats_baseline"] == f["description_beats_baseline"]


def test_rescore_without_a_shuffled_record_refuses_to_call(tmp_path):
    client = _FakeClient()
    config = _config(tmp_path, blind=True, n_shuffles=1)
    config.output_dir.mkdir(parents=True)
    results = [it.interpret_feature(client, config, fi, "concept", *_prepare(fi))
               for fi in (3, 5, 9)]
    with pytest.raises(RuntimeError, match="--rescore"):
        it.run_shuffled_baseline(None, config, results)


def test_rescore_shuffled_keeps_nan_it_cannot_decide(split):
    _describe, holdout = split
    measured = [e.activation for e in holdout]
    largest_tie = max(measured.count(m) for m in set(measured))
    n = len(holdout)
    record = {"feature_idx": 1, "donors": [2, 3, 4, 5],
              "r_values": [float("nan"), float("nan"), float("nan"), 0.42],
              "n_predicted": [n, 2, largest_tie, n]}
    assert it.rescore_shuffled(record, holdout) == 1
    assert record["r_values"][0] == 0.0                   # full coverage: constant prediction
    assert math.isnan(record["r_values"][1])              # < 3 predictions
    assert math.isnan(record["r_values"][2])              # subset could be all-equal measured
    assert record["r_values"][3] == 0.42                  # finite r untouched
    flat = [dataclasses.replace(e, activation=0.5) for e in holdout]
    nan_record = {"feature_idx": 1, "donors": [2], "r_values": [float("nan")],
                  "n_predicted": [n]}
    assert it.rescore_shuffled(nan_record, flat) == 0     # constant measured stays NaN


def test_baseline_threshold_and_beats():
    null = [float(v) for v in range(101)] + [float("nan")]
    assert it.baseline_threshold(null) == pytest.approx(95.0)
    assert math.isnan(it.baseline_threshold([float("nan")]))
    assert it.beats_baseline(95.1, 95.0) is True
    assert it.beats_baseline(95.0, 95.0) is False
    assert it.beats_baseline(float("nan"), 95.0) is None
    assert it.beats_baseline(0.5, float("nan")) is None


def test_apply_baseline_needs_every_described_feature():
    def rec(fi, r, shuffled):
        out = {"feature_idx": fi, "pearson_r": r, "model_served": "m"}
        if shuffled is not None:
            out["shuffled"] = {"r_values": shuffled}
        return out

    full = [rec(1, 0.9, [0.1, 0.2]), rec(2, 0.15, [0.0, 0.3])]
    threshold = it.apply_baseline(full)
    assert threshold == pytest.approx(np.percentile([0.1, 0.2, 0.0, 0.3], 95))
    assert [r["description_beats_baseline"] for r in full] == [True, False]
    partial = [rec(1, 0.9, [0.1]), rec(2, 0.15, None)]
    assert math.isnan(it.apply_baseline(partial))
    assert all(r["description_beats_baseline"] is None for r in partial)


def test_donors_exclude_self_are_deterministic_and_leave_other_rngs_alone():
    pool = list(range(10, 90))
    main = random.Random(0)
    before = main.getstate()
    for fi in pool:
        donors = it.draw_donors(fi, pool, 3, seed=0, layer=2)
        assert fi not in donors and len(set(donors)) == 3 and set(donors) <= set(pool)
        assert donors == it.draw_donors(fi, list(reversed(pool)), 3, seed=0, layer=2)
    assert main.getstate() == before
    assert it.draw_donors(10, pool, 3, 0, 2) != it.draw_donors(10, pool, 3, 1, 2)
    with pytest.raises(ValueError):
        it.draw_donors(1, [1, 2], 3, 0, 2)


def test_shuffled_resume_with_changed_donors_raises(tmp_path):
    client = _FakeClient()
    config = _config(tmp_path, blind=True, n_shuffles=1)
    config.output_dir.mkdir(parents=True)
    results = [it.interpret_feature(client, config, fi, "concept", *_prepare(fi))
               for fi in (3, 5, 9)]
    it.run_shuffled_baseline(client, config, results)
    n_calls = len(client.prompts)
    it.run_shuffled_baseline(client, config, results)          # resume: nothing re-asked
    assert len(client.prompts) == n_calls
    with pytest.raises(RuntimeError, match="donors"):
        it.run_shuffled_baseline(client, config, results[:2])  # pool changed


# --- chunks and run config --------------------------------------------------------------


def test_explicit_chunk_indices_must_all_exist():
    available = [0, 1, 52, 104]
    assert it.resolve_chunk_indices(available, (104, 0, 52), 12) == [0, 52, 104]
    with pytest.raises(ValueError, match="572"):
        it.resolve_chunk_indices(available, (0, 572), 12)
    assert it.resolve_chunk_indices(available, None, 2) == [0, 52]


def test_run_config_mismatch_raises(tmp_path):
    config = _config(tmp_path, blind=True)
    it.check_run_config(config.output_dir, config.resume_key([0, 52]))
    it.check_run_config(config.output_dir, config.resume_key([0, 52]))   # same: fine
    stored = json.loads((config.output_dir / "run_config.json").read_text())
    assert stored["chunk_indices_used"] == [0, 52] and stored["blind"] is True
    with pytest.raises(ValueError, match="blind"):
        it.check_run_config(config.output_dir,
                            dataclasses.replace(config, blind=False).resume_key([0, 52]))
    with pytest.raises(ValueError, match="chunk_indices_used"):
        it.check_run_config(config.output_dir, config.resume_key([0, 1]))


def test_holdout_examples_round_trip_through_json(split):
    _describe, holdout = split
    stored = json.loads(json.dumps([dataclasses.asdict(e) for e in holdout]))
    rebuilt = [it._example_from_dict(d) for d in stored]
    assert it.build_prediction_prompt("d", rebuilt, False, blind=True) == \
        it.build_prediction_prompt("d", holdout, False, blind=True)


@pytest.mark.parametrize("tag", ["archive", "archive/unblinded_v1"])
def test_interpret_refuses_an_archive_run_tag(monkeypatch, tag):
    monkeypatch.setattr("sys.argv", [
        "interpret.py", "--eval-dir", "e", "--extract-dir", "x", "--annotation-dir", "a",
        "--sae-checkpoint", "c.pt", "--output-dir", "o", "--target-layer", "2",
        "--run-tag", tag])
    assert it.main() == 2


def test_default_env_file_is_the_repo_root_dotenv():
    # .env.example and the README put the key in the repo root's .env, not src/.
    repo_root = Path(__file__).resolve().parents[1]
    assert it.DEFAULT_ENV_FILE == repo_root / ".env"
    assert (it.DEFAULT_ENV_FILE.parent / ".env.example").is_file()
