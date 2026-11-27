"""dictionary.py -- read the feature dictionary, the contract between pipeline and tool.

One parquet per layer, written by ``build_dictionary.py``: one row per *live*
feature (fired at least once over the corpus). The loader checks the schema
version and layer recorded in the parquet's metadata and fails loudly on a
mismatch rather than serving rows from another layout.

A feature absent from the dictionary never fired over the corpus. When it fires
at runtime, ``lookup`` returns None and the caller reports it as not in the
dictionary; it is not given a description, association or causal value.

``summary`` is the slim per-feature view the Explanation JSON carries (schema
2); the dashboard reads full rows with ``lookup``, after checking the
explanation's ``sha256_16`` against this file's (``runtime.feature_row``).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from schema import DICTIONARY_SCHEMA_VERSION

# Summary key -> dictionary column. Association with its effect size and the
# concept's chance fraction (V1-3b), the best association passing the display
# rule, causal coverage (null = untested, never "not important"), description
# coverage and score: the evidence indicators a panel shows without the full row.
SUMMARY_FIELDS = {
    "firing_rate": "firing_rate",
    "top_concept": "concept_1",
    "top_concept_f1_dom": "f1_dom_1",
    "top_concept_chance_fraction": "chance_fraction_1",
    "top_concept_meets_display_rule": "meets_display_rule_1",
    "reliable_concept": "reliable_concept_1",
    "reliable_concept_f1_dom": "reliable_f1_dom_1",
    "has_significant_association": "has_significant_association",
    "causal_delta_ce": "phase8_delta_ce",
    "description_blinded": "description_blinded",
    "description_r": "description_r",
    "description_beats_baseline": "description_beats_baseline",
    "latent_token_fraction": "latent_token_fraction",
}
# Keys of every summary: the mapped fields plus two derived ones.
SUMMARY_KEYS = ("in_dictionary", "has_description", *SUMMARY_FIELDS)
UNTESTED = "untested"   # build_dictionary's causal_provenance for unmeasured features


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

    def __init__(self, layer: int, rows: dict[int, dict], metadata: dict[str, str],
                 path: Path | None = None, sha256_16: str | None = None):
        self.layer = layer
        self._rows = rows
        self.metadata = metadata
        self.path = path
        self.sha256_16 = sha256_16    # of the parquet file, so a reader can check it has this one

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
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return cls(layer, rows, metadata, path=path, sha256_16=digest)

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

    def summary(self, feature_idx: int) -> dict:
        """The slim per-feature view, keys SUMMARY_KEYS in that order. A feature
        not in the dictionary gets in_dictionary False and None everywhere else.
        A missing column raises: the file would not be the dictionary this tool reads."""
        row = self._rows.get(int(feature_idx))
        if row is None:
            return {"in_dictionary": False, **{k: None for k in SUMMARY_KEYS[1:]}}
        try:
            mapped = {key: row[col] for key, col in SUMMARY_FIELDS.items()}
        except KeyError as exc:
            raise KeyError(f"layer {self.layer} dictionary has no column {exc}") from None
        return {"in_dictionary": True, "has_description": row.get("description") is not None, **mapped}

    def concept_reliability(self) -> dict[str, dict]:
        """The concept reliability records from the metadata (V1-3b), keyed by
        concept: display name, chance fraction with CI, bug counts, dependence,
        whether the concept meets the display rule, reasons and notes."""
        text = self.metadata.get("concept_reliability")
        if text is None:
            raise KeyError(f"layer {self.layer} dictionary metadata has no concept_reliability")
        return {rec["concept"]: rec for rec in json.loads(text)}

    def display_name(self, concept: str | None) -> str | None:
        """The name the tool shows for a concept (re-documented ones differ from the key)."""
        if concept is None:
            return None
        return self.concept_reliability()[concept]["display_name"]

    def causal_provenance(self) -> str | None:
        """The provenance string shared by every causally measured feature, or
        None if none was measured. Raises if measured features disagree."""
        found = {row.get("causal_provenance") for row in self._rows.values()} - {UNTESTED, None}
        if len(found) > 1:
            raise ValueError(f"layer {self.layer}: {len(found)} distinct causal provenance strings")
        return found.pop() if found else None


# --- V1-5: fit reference and feature examples ---------------------------------------

def fit_reference_path(root: Path) -> Path:
    """``<root>/dictionary/fit_reference.json`` (build_dictionary --examples)."""
    return Path(root) / "dictionary" / "fit_reference.json"


class FitReference:
    """Per-layer quantiles of the per-spectrum centred FVE over the local
    chunks, to place one spectrum's fit (explain()'s fve_centred) among them."""

    def __init__(self, data: dict, path: Path | None = None):
        from schema import FIT_REFERENCE_SCHEMA_VERSION
        if data.get("schema_version") != FIT_REFERENCE_SCHEMA_VERSION:
            raise ValueError(f"{path}: fit reference schema {data.get('schema_version')!r}, this tool "
                             f"reads {FIT_REFERENCE_SCHEMA_VERSION}")
        self.path = path
        self.data = data
        self.probabilities = [float(p) for p in data["probabilities"]]
        self.note = data["note"]
        self.layers = {int(L): v for L, v in data["layers"].items()}
        for L, v in self.layers.items():
            q = v["quantiles"]
            if len(q) != len(self.probabilities) or any(b < a for a, b in zip(q, q[1:])):
                raise ValueError(f"{path}: layer {L} quantiles malformed")

    @classmethod
    def load(cls, root: Path) -> FitReference:
        path = fit_reference_path(root)
        if not path.is_file():
            raise FileNotFoundError(f"No fit reference at {path}; build it with "
                                    f"`python -m build_dictionary --examples --example-features ...`")
        return cls(json.loads(path.read_text(encoding="utf-8")), path)

    def percentile(self, layer: int, fve: float) -> float:
        """Percentage (0-100) of reference spectra at this layer whose centred FVE
        is at or below `fve`, interpolated on the quantile table; 0 below the
        lowest and 100 above the highest reference value."""
        if int(layer) not in self.layers:
            raise KeyError(f"fit reference has no layer {layer}")
        if not math.isfinite(float(fve)):
            raise ValueError(f"fve must be finite, got {fve}")
        q = self.layers[int(layer)]["quantiles"]
        x = float(fve)
        if x < q[0]:
            return 0.0
        if x >= q[-1]:
            return 100.0
        # last index with q[i] <= x, then linear within [q[i], q[i+1]]
        lo, hi = 0, len(q) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if q[mid] <= x:
                lo = mid
            else:
                hi = mid
        p0, p1 = self.probabilities[lo], self.probabilities[hi]
        span = q[hi] - q[lo]
        p = p1 if span <= 0 else p0 + (p1 - p0) * (x - q[lo]) / span
        return 100.0 * p


class FeatureExamples:
    """examples/layer_{L}.parquet and the shared spectrum context
    examples/spectra.parquet (build_dictionary --examples), schema-checked."""

    def __init__(self, layer: int, examples: dict[int, list[dict]], spectra: dict[tuple[int, int], dict],
                 metadata: dict[str, str]):
        self.layer = layer
        self._examples = examples
        self._spectra = spectra
        self.metadata = metadata

    @staticmethod
    def _read(path: Path) -> tuple[list[dict], dict[str, str]]:
        from schema import EXAMPLES_SCHEMA_VERSION
        if not path.is_file():
            raise FileNotFoundError(f"No feature examples at {path}; build them with "
                                    f"`python -m build_dictionary --examples --example-features ...`")
        table = pq.read_table(path)
        metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
        if metadata.get("schema_version") != str(EXAMPLES_SCHEMA_VERSION):
            raise ValueError(f"{path}: examples schema {metadata.get('schema_version')!r}, this tool "
                             f"reads {EXAMPLES_SCHEMA_VERSION}")
        return table.to_pylist(), metadata

    @classmethod
    def load(cls, root: Path, layer: int) -> FeatureExamples:
        base = Path(root) / "dictionary" / "examples"
        rows, metadata = cls._read(base / f"layer_{layer}.parquet")
        if metadata.get("layer") != str(layer):
            raise ValueError(f"examples file says layer {metadata.get('layer')!r}, expected {layer}")
        spectra_rows, _ = cls._read(base / "spectra.parquet")
        spectra = {(r["chunk"], r["spectrum_index"]): r for r in spectra_rows}
        examples: dict[int, list[dict]] = {}
        for r in rows:
            if (r["chunk"], r["spectrum_index"]) not in spectra:
                raise ValueError(f"layer {layer} example of feature {r['feature_idx']} references a "
                                 f"spectrum missing from spectra.parquet")
            examples.setdefault(int(r["feature_idx"]), []).append(r)
        for v in examples.values():
            v.sort(key=lambda r: r["rank"])
        return cls(layer, examples, spectra, metadata)

    def __contains__(self, feature_idx: int) -> bool:
        return int(feature_idx) in self._examples

    def examples(self, feature_idx: int) -> list[dict]:
        """The feature's examples, strongest first, each with its spectrum
        context under "spectrum"; [] for a feature without examples."""
        return [{**r, "spectrum": self._spectra[(r["chunk"], r["spectrum_index"])]}
                for r in self._examples.get(int(feature_idx), [])]
