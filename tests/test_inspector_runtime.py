"""Tests for inspector.runtime (explain) and inspector.dictionary.

Run from the repo root: ``PYTHONPATH=src uv run python -m pytest tests``.

Pure tests run anywhere. Data-backed tests need the local InstaNovo checkpoint,
the chunks and SAEs under ``sae_pipeline_outputs/`` and the HF nine-species
cache, and skip without them. The checkpoint is $INSTANOVO_CKPT, else
``checkpoints/instanovo_v1.1.0.ckpt`` (the runtime's default).

Ground truth for the data-backed tests is chunk 0, whose spectra are the first
1,024 rows of the merged corpus (HF train shard 0, rows 0..1023): all four
layers' activations are on disk, and chunk 0 lies within the 5,000 spectra
Phase 7 evaluated. Phase 7's loader CE agreed with the baseline CE cached in
the chunk meta to 5e-6, so the cached CE stands in for Phase 7's per spectrum.
Each HF row is checked against the chunk's peptide, precursor m/z and charge
before it is used.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import annotate
from inspector.dictionary import SUMMARY_FIELDS, SUMMARY_KEYS, FeatureDictionary
from inspector.runtime import (
    DEFAULT_CHECKPOINT,
    LATENT_LABEL,
    SPECTRUM_SUMMARY_KEYS,
    AblationContext,
    AblationReport,
    Explanation,
    Inspector,
    Spectrum,
    ablation_delta,
    ablation_positions,
    ablation_provenance,
    decoder_to_residue_order,
    feature_row,
    fve,
    joint_delta,
    peak_label,
    read_raw_spectra,
    top_k_features,
)
from schema import DICTIONARY_SCHEMA_VERSION, EXPLAIN_SCHEMA_VERSION

REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / "sae_pipeline_outputs"
CHUNKS = LOCAL / "extract" / "chunks"
LAYERS = (2, 4, 6, 8)
HF_SNAPSHOTS = (Path.home() / ".cache" / "huggingface" / "hub"
                / "datasets--InstaDeepAI--ms_ninespecies_benchmark" / "snapshots")


def _checkpoint() -> Path | None:
    if os.environ.get("INSTANOVO_CKPT"):
        return Path(os.environ["INSTANOVO_CKPT"])
    return DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT.is_file() else None


def _hf_train_shard() -> Path | None:
    hits = sorted(HF_SNAPSHOTS.glob("*/data/train-00000-of-00002.parquet"))
    return hits[0] if hits else None


CKPT, HF_TRAIN0 = _checkpoint(), _hf_train_shard()
needs_data = pytest.mark.skipif(
    CKPT is None or HF_TRAIN0 is None or not (CHUNKS / "meta_00000.pt").is_file()
    or not (LOCAL / "dictionary" / "layer_8.parquet").is_file(),
    reason="needs the InstaNovo checkpoint, chunk 0, the dictionaries and the HF nine-species cache",
)


@contextlib.contextmanager
def no_network():
    """Make any socket connection or DNS lookup raise."""
    def refuse(*_a, **_k):
        raise AssertionError("network access attempted")
    saved = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)
    socket.socket.connect = socket.socket.connect_ex = refuse
    socket.create_connection = socket.getaddrinfo = refuse
    try:
        yield
    finally:
        (socket.socket.connect, socket.socket.connect_ex,
         socket.create_connection, socket.getaddrinfo) = saved


# --- Pure helpers ----------------------------------------------------------------

def test_fve_matches_phase_1_2_definition():
    from evaluate import _ReconSparsityAccumulator
    g = torch.Generator().manual_seed(0)
    x = torch.randn(37, 16, generator=g) + 3.0          # non-zero mean: centring matters
    x_hat = x + 0.3 * torch.randn(37, 16, generator=g)
    acc = _ReconSparsityAccumulator(d_dict=4)
    acc.add_batch(x, x_hat, torch.zeros(37, 4))

    class _S:
        d_model, d_dict = 16, 4
    m = acc.metrics(_S())
    centred, uncentred = fve(x, x_hat)
    # evaluate sums squares in float32 before converting; fve() sums in float64.
    assert centred == pytest.approx(m["fve_overall"], rel=1e-6)
    assert uncentred == pytest.approx(m["fve_uncentered"], rel=1e-6)
    assert centred < uncentred


def test_decoder_to_residue_order():
    # Decoder order for PEPK is K, P, E, P, EOS.
    per_residue, eos = decoder_to_residue_order([-0.4, -0.3, -0.2, -0.1, -0.05], 4)
    assert per_residue == [-0.1, -0.2, -0.3, -0.4] and eos == -0.05
    with pytest.raises(ValueError):
        decoder_to_residue_order([0.0, 0.0], 2)


def test_top_k_features_drops_inactive_and_orders():
    feats = torch.tensor([[0.0, 2.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [5.0, 4.0, 3.0, 2.0]])
    assert top_k_features(feats, 3) == [[(1, 2.0), (3, 1.0)], [], [(0, 5.0), (1, 4.0), (2, 3.0)]]


@pytest.mark.parametrize("ann,label", [
    ({"ion_type": "y", "ion_index": 7, "charge": 1, "neutral_loss": ""}, "y7+"),
    ({"ion_type": "b", "ion_index": 3, "charge": 2, "neutral_loss": "H2O"}, "b3++-H2O"),
    ({"ion_type": "I", "ion_index": "K", "charge": 1, "neutral_loss": ""}, "I(K)+"),
    ({"ion_type": "internal", "ion_index": (2, 5), "charge": 1, "neutral_loss": ""}, "m2:5+"),
    ({"ion_type": "noise", "ion_index": None, "charge": 0, "neutral_loss": ""}, "noise"),
])
def test_peak_label(ann, label):
    assert peak_label(ann) == label


def test_spectrum_validates():
    with pytest.raises(ValueError):
        Spectrum([100.0, 200.0], [1.0], 500.0, 2)
    with pytest.raises(ValueError):
        Spectrum([100.0], [1.0], 500.0, 0)


def test_annotation_defaults_match_the_recorded_run():
    manifest = LOCAL / "annotation" / "annotation_manifest.json"
    if not manifest.is_file():
        pytest.skip("no local annotation manifest")
    recorded = json.loads(manifest.read_text())["config"]
    cfg = annotate.AnnotationConfig(extract_dir=Path(), output_dir=Path())
    for key in ("ion_types", "enable_internal", "fragment_tol_mass", "fragment_tol_mode"):
        assert getattr(cfg, key) == recorded[key], key


def test_strict_annotation_raises_instead_of_all_noise():
    cfg = annotate.AnnotationConfig(extract_dir=Path(), output_dir=Path())
    args = ("[+42.01]PEPTIDEK", torch.tensor([200.0, 300.0]), torch.ones(2), 500.0, 2,
            cfg.ion_types, cfg.fragment_tol_mass, cfg.fragment_tol_mode, cfg.enable_internal)
    assert all(a["ion_type"] == "noise" for a in annotate.run_spectrum_utils(*args))
    with pytest.raises(ValueError, match="could not annotate"):
        annotate.run_spectrum_utils(*args, strict=True)


# --- Dictionary loader -------------------------------------------------------------

def _write_dict(root: Path, layer: int, version: str, meta_layer: str) -> None:
    (root / "dictionary").mkdir(parents=True, exist_ok=True)
    table = pa.table({"layer": pa.array([layer], pa.int32()), "feature_idx": pa.array([7], pa.int32()),
                      "description_r": [float("nan")], "description": ["x"]})
    table = table.replace_schema_metadata({"schema_version": version, "layer": meta_layer})
    pq.write_table(table, root / "dictionary" / f"layer_{layer}.parquet")


def test_dictionary_loads_and_cleans_nan(tmp_path):
    _write_dict(tmp_path, 2, str(DICTIONARY_SCHEMA_VERSION), "2")
    d = FeatureDictionary.load(tmp_path, 2)
    assert list(d) == [7] and 7 in d and 8 not in d
    assert d.lookup(7) == {"feature_idx": 7, "description_r": None, "description": "x"}
    assert d.lookup(8) is None


@pytest.mark.parametrize("version,meta_layer", [("999", "2"), (str(DICTIONARY_SCHEMA_VERSION), "4")])
def test_dictionary_rejects_wrong_schema_or_layer(tmp_path, version, meta_layer):
    _write_dict(tmp_path, 2, version, meta_layer)
    with pytest.raises(ValueError):
        FeatureDictionary.load(tmp_path, 2)


def test_summary_fields_exist_in_dictionary_schema():
    from build_dictionary import _SCHEMA
    assert set(SUMMARY_FIELDS.values()) | {"description", "causal_provenance"} <= set(_SCHEMA.names)


def _write_summary_dict(root: Path, provenances: list[str]) -> Path:
    n = len(provenances)
    (root / "dictionary").mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "layer": pa.array([2] * n, pa.int32()), "feature_idx": pa.array(list(range(7, 7 + n)), pa.int32()),
        "firing_rate": [0.01] * n, "concept_1": ["is_y_ion"] * n, "f1_dom_1": [0.5] * n,
        "has_significant_association": [True] * n, "phase8_delta_ce": [None] * n,
        "description_blinded": [True] * n, "description_r": [float("nan")] * n,
        "description_beats_baseline": [None] * n, "latent_token_fraction": [None] * n,
        "description": ["fires on y ions"] + [None] * (n - 1), "causal_provenance": provenances,
    }).replace_schema_metadata({"schema_version": str(DICTIONARY_SCHEMA_VERSION), "layer": "2"})
    path = root / "dictionary" / "layer_2.parquet"
    pq.write_table(table, path)
    return path


def test_dictionary_summary_hash_and_causal_provenance(tmp_path):
    path = _write_summary_dict(tmp_path, ["untested", "SAE-patched, 5 spectra"])
    d = FeatureDictionary.load(tmp_path, 2)
    assert d.sha256_16 == hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    s = d.summary(7)
    assert tuple(s) == SUMMARY_KEYS
    assert s["in_dictionary"] and s["has_description"] and s["top_concept"] == "is_y_ion"
    assert s["causal_delta_ce"] is None and s["description_r"] is None       # untested / NaN -> None
    assert not d.summary(8)["has_description"]
    absent = d.summary(99)
    assert absent["in_dictionary"] is False and all(absent[k] is None for k in SUMMARY_KEYS[1:])
    assert d.causal_provenance() == "SAE-patched, 5 spectra"

    _write_summary_dict(tmp_path, ["SAE-patched, 5 spectra", "SAE-patched, 6 spectra"])
    with pytest.raises(ValueError):
        FeatureDictionary.load(tmp_path, 2).causal_provenance()


# --- Ablation helpers (no model) ---------------------------------------------------

def test_ablation_positions_and_delta():
    feats = torch.tensor([[0.0, 2.0, 0.0], [1.5, 0.0, 0.0], [0.5, 3.0, 0.0]])
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert ablation_positions(feats, 0, "global") == [1, 2]
    assert ablation_positions(feats, 1, "local", 2) == [2]
    for f, scope, pos in [(1, "local", 1), (2, "global", None), (0, "local", None),
                          (0, "local", 3), (0, "both", None), (0, "global", 1)]:
        with pytest.raises(ValueError):
            ablation_positions(feats, f, scope, pos)
    assert torch.equal(ablation_delta(feats, w, [(0, [1, 2])]),
                       torch.tensor([[0.0, 0.0], [-1.5, 0.0], [-0.5, 0.0]]))
    # Joint at token 2 removes 0.5 * w0 + 3.0 * w1 = x_hat[2] - b_dec.
    assert torch.equal(joint_delta(feats, w, 2), torch.tensor([[0.0, 0.0], [0.0, 0.0], [-0.5, -3.0]]))


def test_ablation_provenance_follows_target():
    assert "teacher-forced on the predicted peptide" in ablation_provenance("prediction")
    assert "teacher-forced on the given target peptide" in ablation_provenance("given")
    with pytest.raises(ValueError):
        ablation_provenance("true")


def _fake_ablation_setup():
    """An Inspector shell whose 'model' gives each ablation a delta log p equal,
    at every decoder position, to the sum of the delta's entries."""
    feats = torch.tensor([[0.0, 2.0, 0.0], [1.5, 0.0, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 0.0]])
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    insp = object.__new__(Inspector)
    insp.saes = {2: types.SimpleNamespace(W_dec=w)}
    insp._run_ablations = lambda ctx, layer, deltas: [
        ([float(d.sum())] * (ctx.n_residues + 1), [bool(d.sum() < -2)] * (ctx.n_residues + 1))
        for d in deltas]
    ctx = AblationContext("toy", {}, {"kind": "prediction", "peptide": "PEK", "tokens": ["P", "E", "K"]},
                          {2: feats}, torch.zeros(4))
    return insp, ctx


def test_ablate_token_covers_every_active_feature_sorted():
    insp, ctx = _fake_ablation_setup()
    rep = insp.ablate_token(ctx, 2, 2, with_summaries=False)
    # Token 2: feature 0 (0.5 -> delta sum -0.5) and feature 1 (3.0 -> -3.0); 4 decoder positions.
    assert [r["feature"] for r in rep.results] == [1, 0]                 # sorted by |delta_total|
    assert [r["activation"] for r in rep.results] == [3.0, 0.5]
    assert [r["delta_total"] for r in rep.results] == [-12.0, -2.0]
    assert rep.results[0]["delta_log_prob"] == [-3.0] * 3 and rep.results[0]["delta_eos"] == -3.0
    assert rep.results[0]["argmax_changed"] == [True] * 3 and not rep.results[1]["eos_argmax_changed"]
    assert rep.joint["n_features"] == 2 and rep.joint_delta_total == -14.0 and rep.sum_of_singles == -14.0
    assert rep.label == "peak 1 only" and rep.scope == "local" and rep.position == 2
    assert "predicted peptide" in rep.provenance and rep.joint_note.startswith("joint = ")
    assert tuple(rep.to_dict(include_summaries=False)) == AblationReport.KEYS

    latent = insp.ablate_token(ctx, 2, 0, with_summaries=False)
    assert latent.label == LATENT_LABEL and [r["feature"] for r in latent.results] == [1]
    empty = insp.ablate_token(ctx, 2, 3, with_summaries=False)
    assert empty.results == [] and empty.joint is None and empty.sum_of_singles is None
    with pytest.raises(ValueError):
        insp.ablate_token(ctx, 2, 4)

    glob = insp.ablate(ctx, 2, [0], "global", with_summaries=False)
    assert glob.results[0]["positions"] == [1, 2] and glob.results[0]["activation"] == 1.5
    assert glob.joint is None and glob.scope_note.startswith("global")


# --- Data-backed ------------------------------------------------------------------

@pytest.fixture(scope="module")
def inspector():
    if CKPT is None:
        pytest.skip("no InstaNovo checkpoint")
    with no_network():
        return Inspector(CKPT, LOCAL)


@pytest.fixture(scope="module")
def chunk0():
    return torch.load(CHUNKS / "meta_00000.pt", map_location="cpu", weights_only=False)


def _chunk0_spectra(chunk0) -> list[int]:
    """Spectrum 0 (unmodified), the first deamidated and the first oxidised spectrum."""
    pf = chunk0["proforma_strings"]
    assert "[" not in pf[0]
    return [0, next(i for i, p in enumerate(pf) if "[+0.98]" in p),
            next(i for i, p in enumerate(pf) if "[+15.99]" in p)]


def _raw_spectrum(chunk0, s: int) -> tuple[Spectrum, str]:
    """HF train row s (= merged corpus row s, chunk 0) and its modified_sequence,
    after checking it is the spectrum chunk 0 holds at row s."""
    spec, rec = _raw_spectrum_record(chunk0, s)
    return spec, rec["modified_sequence"]


def _raw_spectrum_record(chunk0, s: int) -> tuple[Spectrum, dict]:
    rec = pq.read_table(HF_TRAIN0).slice(s, 1).to_pylist()[0]
    assert chunk0["spectrum_ids"][s] == str(s)
    assert rec["sequence"] == chunk0["peptides"][s]
    assert rec["precursor_charge"] == int(chunk0["precursor_charges"][s])
    assert abs(rec["precursor_mz"] - float(chunk0["precursor_mzs"][s])) < 1e-3
    spec = Spectrum(rec["mz_array"], rec["intensity_array"], rec["precursor_mz"],
                    rec["precursor_charge"], spectrum_id=str(s))
    return spec, rec


def _stored_rows(chunk0, s: int) -> torch.Tensor:
    rows = (chunk0["token_to_spectrum"] == s).nonzero().flatten()
    assert chunk0["token_to_position"][rows].tolist() == list(range(rows.numel()))
    return rows


@needs_data
def test_activations_match_stored_chunk(inspector, chunk0):
    """Key test: explain()'s forward pass reproduces the stored activations.

    (a) preprocessing: processed peaks equal the stored ones exactly;
    (b) per layer: fp32 activations vs the stored bfloat16 rows within two bf16
        half-ULPs (rtol 2**-7) plus atol 2**-7 * rms, which also covers CPU-vs-GPU
        float32 differences.
    """
    for s in _chunk0_spectra(chunk0):
        spec, modseq = _raw_spectrum(chunk0, s)
        fwd = inspector.forward(spec, modseq)
        assert torch.equal(fwd.peaks_mz, chunk0["mz_arrays"][s])
        assert torch.equal(fwd.peaks_intensity, chunk0["intensity_arrays"][s])
        rows = _stored_rows(chunk0, s)
        for L in LAYERS:
            stored = torch.load(CHUNKS / f"acts_L{L}_00000.pt", map_location="cpu",
                                weights_only=False)["activations"][rows].float()
            ours = fwd.activations[L]
            rms = float(stored.pow(2).mean().sqrt())
            err = (ours - stored).abs() / (stored.abs() + rms)
            sae = inspector.saes[L]
            with torch.inference_mode():
                f_ours = sae.forward_inference(ours)["features"]
                f_stored = sae.forward_inference(stored)["features"]
            k = inspector.top_k
            agree = [len({f for f, _ in a} & {f for f, _ in b}) / max(len(b), 1)
                     for a, b in zip(top_k_features(f_ours, k), top_k_features(f_stored, k), strict=True)]
            print(f"spectrum {s} L{L}: max |d|/(|a|+rms) {float(err.max()):.2e}, "
                  f"median {float(err.median()):.2e}; top-{k} overlap {sum(agree) / len(agree):.3f}")
            torch.testing.assert_close(ours, stored, rtol=2 ** -7, atol=2 ** -7 * rms)


@needs_data
def test_teacher_forced_ce_matches_phase_7(inspector, chunk0):
    """Per-token CE equals the CE cached at extraction (Phase 7's code path).

    Phase 7 and the extraction cache teacher-forced the dataset's bare
    `sequence` column: for a modified peptide they scored the unmodified
    residues. So the match is on the bare sequence; for a modified spectrum the
    CE on the true modified peptide differs (pinned here so a change shows).
    explain()'s per-residue log-probs are checked against the runtime's own
    teacher-forced pass on the true modified peptide when the beam finds it.
    """
    for s in _chunk0_spectra(chunk0):
        spec, rec = _raw_spectrum_record(chunk0, s)
        mask = chunk0["baseline_decoder_mask"][s]
        cached = chunk0["baseline_ce"][s][mask].float()
        ce_bare = inspector.forward(spec, rec["sequence"]).ce
        print(f"spectrum {s} ({rec['modified_sequence']}): max |CE(bare) - cached| "
              f"{float((ce_bare - cached).abs().max()):.2e}")
        torch.testing.assert_close(ce_bare, cached, rtol=0, atol=1e-4)

        ce_true = inspector.forward(spec, rec["modified_sequence"]).ce
        if rec["modified_sequence"] != rec["sequence"]:
            gap = float((ce_true - cached).abs().max())
            print(f"spectrum {s}: max |CE(modified) - cached| {gap:.2e}")
            assert gap > 0.01

        exp = inspector.explain(spec)
        truth = [inspector.residue_set.residue_remapping.get(t, t)
                 for t in inspector.residue_set.tokenize(rec["modified_sequence"])]
        print(f"spectrum {s}: predicted {exp.prediction and exp.prediction['peptide']}")
        if exp.prediction and exp.prediction["tokens"] == truth:
            per_residue, eos = decoder_to_residue_order((-ce_true).tolist(), len(truth))
            got = [r["log_prob_tf"] for r in exp.prediction["residues"]]
            assert got == pytest.approx(per_residue, rel=1e-5, abs=1e-5)
            assert exp.prediction["eos_log_prob_tf"] == pytest.approx(eos, rel=1e-5, abs=1e-5)


@needs_data
def test_annotation_via_tokens_reproduces_stored_labels(inspector, chunk0):
    """annotate_spectrum on the true peptide, given in InstaNovo tokens as the
    runtime gives a prediction, reproduces the stored labels row by row."""
    from inspector.runtime import peptide_for_annotation
    labels = torch.load(LOCAL / "annotation" / "labels" / "chunk_00000.pt",
                        map_location="cpu", weights_only=False)
    for s in _chunk0_spectra(chunk0):
        _spec, modseq = _raw_spectrum(chunk0, s)
        tokens = [inspector.residue_set.residue_remapping.get(t, t)
                  for t in inspector.residue_set.tokenize(modseq)]
        bare, proforma, mods = peptide_for_annotation(tokens, inspector.residue_set)
        assert bare == chunk0["peptides"][s]
        _anns, tok = annotate.annotate_spectrum(
            bare, proforma, mods, chunk0["mz_arrays"][s], chunk0["intensity_arrays"][s],
            float(chunk0["precursor_mzs"][s]), int(chunk0["precursor_charges"][s]),
            inspector.annotation_config, inspector.registry, strict=True,
        )
        rows = _stored_rows(chunk0, s)
        assert torch.equal(tok.labels, labels["token_labels"][rows])
        assert torch.equal(tok.ion_type, labels["ion_type_ids"][rows])
        assert torch.equal(tok.ion_index, labels["ion_indices"][rows])
        assert torch.equal(tok.fragment_charge, labels["fragment_charges"][rows])
        assert torch.equal(tok.neutral_loss_id, labels["neutral_loss_ids"][rows])


@needs_data
def test_chunk_annotator_unchanged_by_refactor(inspector, chunk0):
    """annotate_chunk (via the refactored _annotate_spectrum) still writes the stored labels."""
    labels = torch.load(LOCAL / "annotation" / "labels" / "chunk_00000.pt",
                        map_location="cpu", weights_only=False)
    n = 64
    sub = {k: (v[:n] if isinstance(v, list) else v) for k, v in chunk0.items()}
    sub.update(n_spectra=n, precursor_charges=chunk0["precursor_charges"][:n],
               precursor_mzs=chunk0["precursor_mzs"][:n])
    rows = (chunk0["token_to_spectrum"] < n).nonzero().flatten()
    sub.update(total_tokens=rows.numel(), token_to_spectrum=chunk0["token_to_spectrum"][rows],
               token_to_position=chunk0["token_to_position"][rows])
    out = annotate.annotate_chunk(sub, inspector.annotation_config, inspector.registry)
    assert torch.equal(out.token_labels, labels["token_labels"][rows])
    assert torch.equal(out.ion_type_ids, labels["ion_type_ids"][rows])


def _n_terminal_spectrum(inspector) -> tuple[list[str], Spectrum]:
    """Synthetic acetylated peptide: b ions of [UNIMOD:1]PEPTIDEK at charge 1."""
    tokens = ["[UNIMOD:1]", "P", "E", "P", "T", "I", "D", "E", "K"]
    rs, proton = inspector.residue_set, 1.007276
    acc, mz = 0.0, []
    for tok in tokens[:-1]:
        acc += rs.get_mass(tok)
        if tok[0].isalpha():
            mz.append(acc + proton)
    return tokens, Spectrum(mz, [1.0] * len(mz), rs.get_sequence_mass(tokens, 2), 2)


@needs_data
def test_n_terminal_modification_annotates(inspector):
    from inspector.runtime import peptide_for_annotation
    tokens, spec = _n_terminal_spectrum(inspector)
    bare, proforma, mods = peptide_for_annotation(tokens, inspector.residue_set)
    assert bare == "PEPTIDEK" and proforma.startswith("[+42.010565]-P")
    assert mods == [{"position": 0, "mod_name": "+42.010565", "unimod_id": 1}]
    anns, _tok = annotate.annotate_spectrum(
        bare, proforma, mods, torch.tensor(spec.mz), torch.tensor(spec.intensity),
        spec.precursor_mz, 2, inspector.annotation_config, inspector.registry, strict=True,
    )
    assert [a["ion_type"] + str(a["ion_index"]) for a in anns] == [f"b{i}" for i in range(1, 8)]


@needs_data
def test_explain_makes_no_network_calls(inspector, chunk0):
    spec, _ = _raw_spectrum(chunk0, _chunk0_spectra(chunk0)[1])
    with no_network():
        exp = inspector.explain(spec)
    assert exp.prediction is not None
    assert any(p["annotation"]["ion_type"] != "noise" for p in exp.peaks)


@needs_data
def test_schema_round_trips_and_is_bounded(inspector, chunk0):
    spec, _ = _raw_spectrum(chunk0, 0)
    exp = inspector.explain(spec)
    text = exp.to_json()
    back = Explanation.from_json(text)
    assert back == exp and back.to_json() == text
    assert len(text) <= inspector.json_size_bound
    print(f"json {len(text):,} bytes for {exp.input['n_processed_peaks']} peaks; "
          f"bound {inspector.json_size_bound:,}")

    data = json.loads(text)
    assert data["schema_version"] == EXPLAIN_SCHEMA_VERSION == 2 and data["ablation"] is None
    assert set(data["layers"]) == {str(L) for L in LAYERS}
    assert all(len(t["top"]) <= inspector.top_k for b in data["layers"].values() for t in b["tokens"])
    assert data["provenance"]["preprocessing"]["use_spectrum_utils"] is True
    assert data["provenance"]["decoding"]["max_length"] == int(inspector.config["max_length"])
    for L in LAYERS:
        assert data["provenance"]["dictionary"][str(L)]["sha256_16"] == inspector.dictionaries[L].sha256_16
        for summary in data["features"][str(L)].values():
            assert list(summary) == [*SUMMARY_KEYS, *SPECTRUM_SUMMARY_KEYS]
            assert summary["n_tokens_active"] >= 1 and summary["max_activation"] > 0

    schema_1 = {k: v for k, v in data.items() if k != "ablation"} | {"attribution": None, "schema_version": 1}
    for bad in ({**data, "schema_version": EXPLAIN_SCHEMA_VERSION + 1},
                {k: v for k, v in data.items() if k != "features"}, schema_1):
        with pytest.raises(ValueError):
            Explanation.from_dict(bad)


@needs_data
def test_read_raw_spectra_rows(chunk0):
    specs = read_raw_spectra(HF_TRAIN0, [0, 1])
    assert [s.spectrum_id for s in specs] == ["train-00000-of-00002:0", "train-00000-of-00002:1"]
    assert specs[1].precursor_charge == int(chunk0["precursor_charges"][1])


@needs_data
def test_make_dataloader_reproduces_extraction(inspector, chunk0):
    """The refactored make_dataloader (processor via make_processor) still gives
    extraction's peaks and Phase 7's CE for the first 32 spectra of chunk 0."""
    import polars as pl

    import instanovo_io
    n = 32
    df = pl.from_arrow(pq.read_table(HF_TRAIN0).slice(0, n))
    sdf = instanovo_io.SpectrumDataFrame.from_polars(df, is_annotated=True)
    loader = instanovo_io.make_dataloader(sdf, inspector.residue_set, batch_size=n, num_workers=0)
    batch = next(iter(loader))
    for s in range(n):
        valid = ~batch["spectra_mask"][s]
        assert torch.equal(batch["spectra"][s, valid, 0], chunk0["mz_arrays"][s])
        assert torch.equal(batch["spectra"][s, valid, 1], chunk0["intensity_arrays"][s])
    with torch.inference_mode():
        logits = instanovo_io.model_forward_logits(inspector.model, batch, "cpu")
        ce, _top1, _targets, valid = instanovo_io.per_token_ce_and_top1(logits, batch["peptides"])
    width = min(ce.size(1), chunk0["baseline_ce"].size(1))
    assert torch.equal(valid[:, :width], chunk0["baseline_decoder_mask"][:n, :width])
    torch.testing.assert_close(ce[:, :width], chunk0["baseline_ce"][:n, :width].float(), rtol=0, atol=1e-4)


# The v1.1.0 release asset: GitHub lists this size; it publishes no digest, so
# the sha256 is the one measured on the downloaded file (2026-11-22).
V1_1_0_BYTES = 1_136_412_059
V1_1_0_SHA256 = "c75d1f80fcf912031ed348ffdef4894acc09fa8142d13fec2c269a2f48df564b"


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.is_file(), reason="no checkpoint at the default path")
def test_default_checkpoint_is_the_v1_1_0_release():
    assert DEFAULT_CHECKPOINT.stat().st_size == V1_1_0_BYTES
    h = hashlib.sha256()
    with open(DEFAULT_CHECKPOINT, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    assert h.hexdigest() == V1_1_0_SHA256


def test_normalise_peptide_maps_only_isoleucine_tokens():
    from inspector.runtime import normalise_peptide
    assert normalise_peptide(["I", "L", "M[UNIMOD:35]", "[UNIMOD:1]"]) == ["L", "L", "M[UNIMOD:35]", "[UNIMOD:1]"]


def test_wilson_interval():
    from scipy.stats import binomtest

    from inspector.runtime import wilson_interval
    for k, n in ((120, 200), (0, 10), (37, 41)):
        ref = binomtest(k, n).proportion_ci(method="wilson")
        assert wilson_interval(k, n) == pytest.approx((ref.low, ref.high), abs=1e-8)


def test_read_raw_records_groups_and_keeps_order(tmp_path):
    from inspector.runtime import read_raw_records
    path = tmp_path / "s.parquet"
    pq.write_table(pa.table({"x": list(range(10))}), path, row_group_size=3)
    assert [r["x"] for r in read_raw_records(path, [7, 0, 4, 8], ["x"])] == [7, 0, 4, 8]
    with pytest.raises(IndexError):
        read_raw_records(path, [10], ["x"])


# --- Exact ablation (V1-4b), on chunk 0 spectrum 0 ------------------------------------

@pytest.fixture(scope="module")
def explained0(inspector, chunk0):
    spec, rec = _raw_spectrum_record(chunk0, 0)
    exp = inspector.explain(spec)
    assert exp.prediction is not None
    return spec, rec, exp


@pytest.fixture(scope="module")
def ctx0(inspector, explained0):
    spec, _rec, exp = explained0
    return inspector.ablation_context(spec, exp)


@pytest.fixture(scope="module")
def explained_latent(inspector, chunk0):
    spec, _ = _raw_spectrum(chunk0, 0)
    with no_network():
        exp = inspector.explain(spec, latent_ablation=True)
    return spec, exp


def _layer_outputs(inspector, ctx, deltas=None):
    _logp, _top1, acts = inspector._tf_log_probs(ctx.batch, deltas=deltas, capture=True,
                                                 capture_after_delta=deltas is not None)
    return {L: a[0] for L, a in acts.items()}


@needs_data
def test_zero_delta_is_identity(inspector, chunk0, explained0, ctx0):
    """The ablation hook with zero deltas changes nothing: the layer outputs
    match the stored chunk (as in test_activations_match_stored_chunk) and
    log p equals the explanation's."""
    _spec, _rec, exp = explained0
    n_tok = ctx0.features[2].size(0)
    zeros = {L: torch.zeros(1, n_tok, int(inspector.config["dim_model"])) for L in LAYERS}
    logp, _top1, acts = inspector._tf_log_probs(ctx0.batch, deltas=zeros, capture=True,
                                                capture_after_delta=True)
    torch.testing.assert_close(logp[0], ctx0.baseline_log_prob, rtol=0, atol=1e-6)
    per_residue, eos = decoder_to_residue_order(logp[0].tolist(), ctx0.n_residues)
    assert per_residue == pytest.approx([r["log_prob_tf"] for r in exp.prediction["residues"]],
                                        rel=1e-5, abs=1e-5)
    assert eos == pytest.approx(exp.prediction["eos_log_prob_tf"], rel=1e-5, abs=1e-5)
    rows = _stored_rows(chunk0, 0)
    for L in LAYERS:
        stored = torch.load(CHUNKS / f"acts_L{L}_00000.pt", map_location="cpu",
                            weights_only=False)["activations"][rows].float()
        rms = float(stored.pow(2).mean().sqrt())
        torch.testing.assert_close(acts[L][0], stored, rtol=2 ** -7, atol=2 ** -7 * rms)


@needs_data
def test_ablation_preserves_sae_error(inspector, ctx0):
    """x' - x = -a * W_dec[f] at the ablated tokens only, so x' - x_hat' = x - x_hat."""
    L = 4
    feats, sae = ctx0.features[L], inspector.saes[L]
    w = sae.W_dec.detach().cpu()
    f = int(((feats > 0).sum(dim=0) >= 2).nonzero()[0])
    base = _layer_outputs(inspector, ctx0)[L]
    for scope, position in (("global", None), ("local", int((feats[:, f] > 0).nonzero()[0]))):
        pos = ablation_positions(feats, f, scope, position)
        delta = ablation_delta(feats, w, [(f, pos)])
        after = _layer_outputs(inspector, ctx0, {L: delta[None]})[L]
        torch.testing.assert_close(after - base, delta, rtol=0, atol=1e-5)
        others = [t for t in range(feats.size(0)) if t not in pos]
        assert float((after - base)[others].abs().max()) == 0.0
        removed = feats.clone()
        removed[pos, f] = 0
        with torch.no_grad():
            err_before = base - sae.decode(feats)
            err_after = after - sae.decode(removed)
        torch.testing.assert_close(err_after, err_before, rtol=0, atol=1e-4)


@needs_data
def test_batched_ablation_matches_single(inspector, ctx0):
    L = 2
    feats = ctx0.features[L]
    w = inspector.saes[L].W_dec.detach().cpu()
    fs = (feats[0] > 0).nonzero().flatten().tolist()[:3]
    deltas = [ablation_delta(feats, w, [(f, ablation_positions(feats, f, "global"))]) for f in fs]
    together = inspector._run_ablations(ctx0, L, deltas)
    for d, (got, _changed) in zip(deltas, together, strict=True):
        alone, _ = inspector._run_ablations(ctx0, L, [d])[0]
        assert got == pytest.approx(alone, abs=1e-5)
    assert any(abs(sum(g)) > 1e-4 for g, _ in together)        # the ablations do something


@needs_data
def test_local_equals_global_for_single_token_feature(inspector, ctx0):
    for L in LAYERS:
        feats = ctx0.features[L]
        singles = ((feats > 0).sum(dim=0) == 1).nonzero().flatten()
        if singles.numel():
            break
    f = int(singles[0])
    t = int((feats[:, f] > 0).nonzero()[0])
    loc = inspector.ablate(ctx0, L, [f], "local", t, with_summaries=False).results[0]
    glo = inspector.ablate(ctx0, L, [f], "global", with_summaries=False).results[0]
    assert loc["positions"] == glo["positions"] == [t]
    assert loc["delta_log_prob"] == pytest.approx(glo["delta_log_prob"], abs=1e-7)
    assert loc["delta_total"] == pytest.approx(glo["delta_total"], abs=1e-6)


@needs_data
def test_ablation_context_rejects_mismatched_explanation(inspector, chunk0, explained0):
    spec, _rec, exp = explained0
    other, _ = _raw_spectrum(chunk0, 1)
    with pytest.raises(ValueError, match="input"):
        inspector.ablation_context(other, exp)
    # Same precursor and peak count, different peaks: caught by the log p check.
    shuffled = Spectrum(spec.mz, spec.intensity[::-1], spec.precursor_mz, spec.precursor_charge)
    with pytest.raises(ValueError, match="reproduce"):
        inspector.ablation_context(shuffled, exp)


@needs_data
def test_joint_ablation_removes_the_token_reconstruction(inspector, ctx0):
    """Joint ablation at a token leaves x' = (x - x_hat) + b_dec there: the SAE
    error plus the SAE's 'nothing active' bias, so x' - b_dec is the error only."""
    L = 6
    feats, sae = ctx0.features[L], inspector.saes[L]
    w = sae.W_dec.detach().cpu()
    t = int((feats[1:] > 0).sum(dim=1).argmax()) + 1
    base = _layer_outputs(inspector, ctx0)[L]
    after = _layer_outputs(inspector, ctx0, {L: joint_delta(feats, w, t)[None]})[L]
    with torch.no_grad():
        x_hat = sae.decode(feats)
        b_dec = sae.b_dec.detach().cpu()
    torch.testing.assert_close(after[t] - b_dec, base[t] - x_hat[t], rtol=0, atol=1e-4)
    others = [s for s in range(feats.size(0)) if s != t]
    assert float((after - base)[others].abs().max()) == 0.0

    rep = inspector.ablate_token(ctx0, L, t, with_summaries=False)
    n_active = int((feats[t] > 0).sum())
    assert len(rep.results) == rep.joint["n_features"] == n_active
    assert rep.sum_of_singles == pytest.approx(sum(r["delta_total"] for r in rep.results))
    totals = [abs(r["delta_total"]) for r in rep.results]
    assert totals == sorted(totals, reverse=True)
    assert len(rep.joint["delta_log_prob"]) == ctx0.n_residues
    print(f"L{L} token {t}: {n_active} features; joint {rep.joint_delta_total:+.4f}, "
          f"sum of singles {rep.sum_of_singles:+.4f}")


@needs_data
def test_ablation_with_true_peptide_target(inspector, explained0, ctx0):
    spec, rec, exp = explained0
    rs = inspector.residue_set
    truth = [rs.residue_remapping.get(t, t) for t in rs.tokenize(rec["modified_sequence"])]
    ctx_t = inspector.ablation_context(spec, exp, target=rec["modified_sequence"])
    assert ctx_t.target == {"kind": "given", "peptide": "".join(truth), "tokens": truth}
    ce_true = inspector.forward(spec, rec["modified_sequence"]).ce
    torch.testing.assert_close(ctx_t.baseline_log_prob, -ce_true, rtol=0, atol=1e-5)
    assert all(torch.equal(ctx_t.features[L], ctx0.features[L]) for L in LAYERS)

    rep = inspector.ablate_token(ctx_t, 2, 0, with_summaries=False)
    assert rep.target["kind"] == "given" and rep.target["tokens"] == truth
    assert "teacher-forced on the given target peptide" in rep.provenance
    assert all(len(r["delta_log_prob"]) == len(truth) for r in rep.results)
    f = rep.results[0]["feature"]
    pred_rep = inspector.ablate(ctx0, 2, [f], "local", 0, with_summaries=False)
    assert "teacher-forced on the predicted peptide" in pred_rep.provenance
    assert pred_rep.target["kind"] == "prediction"


@needs_data
def test_schema2_round_trip_bounded_with_latent_ablation(inspector, explained_latent):
    spec, exp = explained_latent
    text = exp.to_json()
    back = Explanation.from_json(text)
    assert back == exp and back.to_json() == text
    data = json.loads(text)
    abl = data["ablation"]
    assert abl["label"] == LATENT_LABEL and abl["scope"] == "local" and abl["position"] == 0
    counts = {int(L): len(rep["results"]) for L, rep in abl["layers"].items()}
    bound = inspector.size_bound(counts)
    print(f"json {len(text):,} bytes with latent ablation ({counts}); bound {bound:,}; "
          f"without: bound {inspector.json_size_bound:,}")
    assert len(text) <= bound

    ctx = inspector.ablation_context(spec, exp)
    for L, rep in abl["layers"].items():
        assert counts[int(L)] == int((ctx.features[int(L)][0] > 0).sum())   # every active feature
        assert rep["target"]["kind"] == "prediction" and rep["joint"]["n_features"] == counts[int(L)]
        assert all(r["positions"] == [0] and "summary" not in r for r in rep["results"])
        assert {str(r["feature"]) for r in rep["results"]} <= set(data["features"][L])
        totals = [abs(r["delta_total"]) for r in rep["results"]]
        assert totals == sorted(totals, reverse=True)

    f = int(next(iter(data["features"]["2"])))
    assert feature_row(exp, inspector.dictionaries, 2, f) == inspector.dictionaries[2].lookup(f)
    prov = data["provenance"]
    tampered = {**prov, "dictionary": {**prov["dictionary"],
                                       "2": {**prov["dictionary"]["2"], "sha256_16": "0" * 16}}}
    with pytest.raises(ValueError, match="sha256"):
        feature_row(Explanation.from_dict({**data, "provenance": tampered}), inspector.dictionaries, 2, f)


@needs_data
def test_explain_with_latent_ablation_makes_no_network_calls(explained_latent):
    _spec, exp = explained_latent        # built under no_network()
    assert set(exp.ablation["layers"]) == {str(L) for L in LAYERS}
