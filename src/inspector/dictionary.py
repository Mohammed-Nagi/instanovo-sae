"""dictionary.py -- read the feature dictionary, the contract between pipeline and tool.

One parquet per layer, written by ``build_dictionary.py``: one row per *live*
feature (fired at least once over the corpus). The loader checks the schema
version and layer recorded in the parquet's metadata and fails loudly on a
mismatch rather than serving rows from another layout.

A feature absent from the dictionary never fired over the corpus. When it fires
at runtime, ``lookup`` returns None and the caller reports it as not in the
dictionary; it is not given a description, association or causal value.
"""
from __future__ import annotations

import math
from pathlib import Path

import pyarrow.parquet as pq

from schema import DICTIONARY_SCHEMA_VERSION


def dictionary_path(root: Path, layer: int) -> Path:
    """``<root>/dictionary/layer_{L}.parquet``, root being ``sae_pipeline_outputs``."""
    return Path(root) / "dictionary" / f"layer_{layer}.parquet"


def _clean(value):
    """NaN -> None, so rows serialise to strict JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class FeatureDictionary:
    """The dictionary rows of one layer, keyed by feature index."""

    def __init__(self, layer: int, rows: dict[int, dict], metadata: dict[str, str]):
        self.layer = layer
        self._rows = rows
        self.metadata = metadata

    @classmethod
    def load(cls, root: Path, layer: int) -> FeatureDictionary:
        path = dictionary_path(root, layer)
        if not path.is_file():
            raise FileNotFoundError(
                f"No feature dictionary for layer {layer} at {path}; build it with "
                f"`python -m build_dictionary --layers {layer}`"
            )
        table = pq.read_table(path)
        metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
        version = metadata.get("schema_version")
        if version != str(DICTIONARY_SCHEMA_VERSION):
            raise ValueError(
                f"{path}: dictionary schema {version!r}, this tool reads "
                f"{DICTIONARY_SCHEMA_VERSION}; rebuild the dictionary"
            )
        if metadata.get("layer") != str(layer):
            raise ValueError(f"{path}: metadata says layer {metadata.get('layer')!r}, expected {layer}")

        rows: dict[int, dict] = {}
        for row in table.to_pylist():
            if row["layer"] != layer:
                raise ValueError(f"{path}: row for layer {row['layer']} in the layer-{layer} file")
            rows[int(row["feature_idx"])] = {k: _clean(v) for k, v in row.items() if k != "layer"}
        return cls(layer, rows, metadata)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        """Feature indices, ascending."""
        return iter(sorted(self._rows))

    def __contains__(self, feature_idx: int) -> bool:
        return int(feature_idx) in self._rows

    def lookup(self, feature_idx: int) -> dict | None:
        """The feature's dictionary row (a copy), or None if it is not live in the corpus."""
        row = self._rows.get(int(feature_idx))
        return dict(row) if row is not None else None
