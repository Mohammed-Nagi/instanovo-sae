"""Tests for evaluate.ChunkStream on the full and the consolidated manifest layouts.

Run from the repo root: ``PYTHONPATH=src uv run python -m pytest tests``.

Full layout: the extract manifest lists every chunk in order, as many as the
annotation manifest; chunks are joined by position. This must behave exactly as
before consolidated manifests were supported. Consolidated layout: the extract
manifest lists only the chunks kept on disk (a subset of the annotation's chunk
ids), some without every layer; chunks are joined by id.

Every chunk's tensors encode its chunk id, so a join with the wrong chunk shows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evaluate import ChunkStream
from schema import ANNOTATION_SCHEMA_VERSION, EXTRACT_SCHEMA_VERSION

D_MODEL, D_DICT, N_TOKENS = 4, 6, 5
REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / "sae_pipeline_outputs"


class _StubSAE:
    d_dict = D_DICT

    def forward_inference(self, x):
        return {"features": x.repeat(1, 2)[:, :D_DICT] * 10}


def _write_chunk_files(root: Path, idx: int, layers: tuple[int, ...]) -> None:
    (root / "extract" / "chunks").mkdir(parents=True, exist_ok=True)
    (root / "annotation" / "labels").mkdir(parents=True, exist_ok=True)
    torch.save({
        "total_tokens": N_TOKENS, "n_spectra": 1,
        "token_to_spectrum": torch.zeros(N_TOKENS, dtype=torch.long),
        "token_to_position": torch.arange(N_TOKENS),
        "spectrum_ids": [f"s{idx}"], "peptides": [f"P{idx}"], "proforma_strings": [f"P{idx}"],
        "modifications": [[]], "precursor_charges": torch.tensor([2]),
        "precursor_mzs": torch.tensor([500.0 + idx]),
    }, root / "extract" / "chunks" / f"meta_{idx:05d}.pt")
    for layer in layers:
        torch.save({"activations": torch.full((N_TOKENS, D_MODEL), float(idx * 100 + layer))},
                   root / "extract" / "chunks" / f"acts_L{layer}_{idx:05d}.pt")
    torch.save({
        "total_tokens": N_TOKENS, "token_labels": torch.zeros(N_TOKENS, 2, dtype=torch.bool),
        "ion_type_ids": torch.full((N_TOKENS,), idx), "peak_mzs": torch.full((N_TOKENS,), float(idx)),
        "peak_intensities": torch.ones(N_TOKENS),
    }, root / "annotation" / "labels" / f"chunk_{idx:05d}.pt")


def _chunk_entry(idx: int, layers: tuple[int, ...]) -> dict:
    return {"idx": idx, "meta": f"chunks/meta_{idx:05d}.pt",
            "activations": {str(l): f"chunks/acts_L{l}_{idx:05d}.pt" for l in layers}}


def _tree(root: Path, n_annotation: int, extract: dict[int, tuple[int, ...]],
          n_chunks: int | None = None) -> Path:
    for idx, layers in extract.items():
        _write_chunk_files(root, idx, layers)
    (root / "extract" / "manifest.json").write_text(json.dumps({
        "schema_version": EXTRACT_SCHEMA_VERSION,
        "n_chunks": len(extract) if n_chunks is None else n_chunks,
        "chunks": [_chunk_entry(i, layers) for i, layers in extract.items()],
    }))
    (root / "annotation" / "annotation_manifest.json").write_text(json.dumps({
        "schema_version": ANNOTATION_SCHEMA_VERSION, "n_chunks": n_annotation,
        "chunks": [{"idx": i, "path": f"labels/chunk_{i:05d}.pt"} for i in range(n_annotation)],
        "registry": {"names": ["a", "b"], "diagnostic": [], "family_of": {}},
        "base_rates": {"a": 0.1, "b": 0.2},
    }))
    return root


def _stream(root: Path, layer: int, **kw) -> ChunkStream:
    return ChunkStream(extract_dir=root / "extract", annotation_dir=root / "annotation",
                       target_layer=layer, sae=_StubSAE(), device="cpu", batch_size=2,
                       dtype=torch.float32, **kw)


def _joined(stream: ChunkStream) -> list[tuple[int, float, int, float]]:
    """(chunk_idx, activation value, annotation ion id, first feature) per chunk."""
    return [(c.chunk_idx, float(c.activations[0, 0]), int(c.ion_type_ids[0]),
             float(c.features[0, 0])) for c in stream]


# --- full layout: unchanged ----------------------------------------------------------


@pytest.fixture
def full(tmp_path) -> Path:
    return _tree(tmp_path, 5, {i: (2, 6) for i in range(5)})


def test_full_manifest_iterates_every_chunk_by_position(full):
    stream = _stream(full, 2)
    assert not stream.subset_manifest
    assert stream.n_chunks == 5
    assert isinstance(stream.meta_paths, list) and isinstance(stream.annotation_paths, list)
    assert _joined(stream) == [(i, i * 100 + 2.0, i, (i * 100 + 2.0) * 10) for i in range(5)]


def test_full_manifest_sample_and_helpers(full):
    stream = _stream(full, 6, chunk_indices=[3, 1])
    assert [c[0] for c in _joined(stream)] == [3, 1]
    assert [float(a[0, 0]) for a in stream.iter_activations()] == [306.0, 106.0]
    assert [m["peptides"][0] for m in stream.iter_metadata()] == ["P3", "P1"]
    assert [int(a["ion_type_ids"][0]) for _m, a in stream.iter_metadata_annotations()] == [3, 1]


def test_full_manifest_feature_subset(full):
    stream = _stream(full, 2, chunk_indices=[0], feature_subset=[5, 0])
    (chunk,) = list(stream)
    assert chunk.features.shape == (N_TOKENS, 2)


def test_full_manifest_errors_are_unchanged(tmp_path):
    root = _tree(tmp_path / "a", 5, {i: (2,) for i in range(5)})
    with pytest.raises(ValueError, match="out of range for a 5-chunk run"):
        _stream(root, 2, chunk_indices=[5])
    with pytest.raises(KeyError, match="Layer 6 not extracted for chunk 0"):
        _stream(root, 6)
    more = _tree(tmp_path / "b", 4, {i: (2,) for i in range(5)})   # 5 extract vs 4 annotation
    with pytest.raises(ValueError, match="Chunk count mismatch: extract=5, annotation=4"):
        _stream(more, 2)
    partial = _tree(tmp_path / "c", 5, {i: (2, 6) for i in range(5)})
    manifest = json.loads((partial / "extract" / "manifest.json").read_text())
    del manifest["chunks"][2]["activations"]["6"]
    (partial / "extract" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(KeyError, match="Layer 6 not extracted for chunk 2"):
        _stream(partial, 6)


# --- consolidated layout ----------------------------------------------------------------


@pytest.fixture
def consolidated(tmp_path) -> Path:
    # Like the local data: chunk 1 has layer 2 only; 52 and 104 have both.
    return _tree(tmp_path, 625, {0: (2, 6), 1: (2,), 52: (2, 6), 104: (2, 6)})


def test_consolidated_manifest_joins_by_chunk_id(consolidated):
    stream = _stream(consolidated, 2)
    assert stream.subset_manifest and stream.n_chunks == 4
    assert _joined(stream) == [(i, i * 100 + 2.0, i, (i * 100 + 2.0) * 10) for i in (0, 1, 52, 104)]


def test_consolidated_manifest_skips_chunks_without_the_layer(consolidated):
    stream = _stream(consolidated, 6)
    assert [c[0] for c in _joined(stream)] == [0, 52, 104]
    assert [c[0] for c in _joined(_stream(consolidated, 6, chunk_indices=[104, 0]))] == [104, 0]
    with pytest.raises(ValueError, match="not in the consolidated manifest for layer 6"):
        _stream(consolidated, 6, chunk_indices=[1])


def test_consolidated_manifest_with_unknown_chunk_id_is_rejected(tmp_path):
    root = _tree(tmp_path, 10, {0: (2,), 52: (2,)})   # 52 is not an annotation chunk
    with pytest.raises(ValueError, match="Chunk count mismatch"):
        _stream(root, 2)


def test_consolidated_manifest_whose_count_disagrees_is_rejected(tmp_path):
    root = _tree(tmp_path, 625, {0: (2,), 52: (2,)}, n_chunks=3)
    with pytest.raises(ValueError, match="Chunk count mismatch"):
        _stream(root, 2)


@pytest.mark.skipif(not (LOCAL / "extract" / "manifest.json").exists(),
                    reason="local extract data not present")
@pytest.mark.parametrize(("layer", "expected"), [
    (2, [0, 1, 52, 104, 156, 208, 260, 312, 364, 416, 468, 520, 572]),
    (6, [0, 52, 104, 156, 208, 260, 312, 364, 416, 468, 520, 572]),
])
def test_local_consolidated_manifest_opens(layer, expected):
    stream = _stream(LOCAL, layer)
    assert stream.subset_manifest
    assert stream._chunk_indices == expected
