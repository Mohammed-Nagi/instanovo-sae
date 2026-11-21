"""Tests for inspector.runtime (explain) and inspector.dictionary.

Run from the repo root: ``PYTHONPATH=src uv run python -m pytest tests``.

Pure tests run anywhere. Data-backed tests need the local InstaNovo checkpoint,
the chunks and SAEs under ``sae_pipeline_outputs/`` and the HF nine-species
cache, and skip without them. The checkpoint is $INSTANOVO_CKPT, else the first
readable of ``checkpoints/instanovo_v1.1.0.ckpt`` and InstaNovo's own cache.

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
import json
import os
import socket
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import annotate
from inspector.dictionary import FeatureDictionary
from inspector.runtime import (
    Explanation,
    Inspector,
    Spectrum,
    decoder_to_residue_order,
    fve,
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


def _readable_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path):
            return True
    except (OSError, zipfile.BadZipFile):
        return False


def _checkpoint() -> Path | None:
    if os.environ.get("INSTANOVO_CKPT"):
        return Path(os.environ["INSTANOVO_CKPT"])
    for p in (REPO / "checkpoints" / "instanovo_v1.1.0.ckpt",
              Path.home() / ".cache" / "instanovo" / "instanovo-v1.1.0.ckpt"):
        if p.is_file() and _readable_zip(p):
            return p
    return None


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
    assert data["schema_version"] == EXPLAIN_SCHEMA_VERSION and data["attribution"] is None
    assert set(data["layers"]) == {str(L) for L in LAYERS}
    assert all(len(t["top"]) <= inspector.top_k for b in data["layers"].values() for t in b["tokens"])
    assert data["provenance"]["preprocessing"]["use_spectrum_utils"] is True
    assert data["provenance"]["decoding"]["max_length"] == int(inspector.config["max_length"])

    for bad in ({**data, "schema_version": EXPLAIN_SCHEMA_VERSION + 1},
                {k: v for k, v in data.items() if k != "features"}):
        with pytest.raises(ValueError):
            Explanation.from_dict(bad)
    with pytest.raises(NotImplementedError):
        inspector.explain(spec, attribution=True)


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
