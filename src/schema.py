"""On-disk schema versions for the pipeline's cached artefacts.

The pipeline caches expensive intermediates (activation chunks, concept labels,
SAE checkpoints) and reuses them across runs, so a stale artefact from an
incompatible run is a realistic failure mode. Each consumer compares the version
recorded in an artefact against the constant below and hard-fails on a
mismatch, rather than silently reading misaligned data.

Bump a constant when an artefact's format OR its content semantics change, so
that reusing a cached copy would give a different answer. Existing artefacts of
that type must then be regenerated.

This module deliberately has no imports, so any consumer can read the constants
(and the run-tag check) without pulling in torch, spectrum_utils, or the
InstaNovo package.
"""

# extract.py: manifest.json, ChunkMeta, and the per-layer activation files.
#
# NOT bumped for the resume-counting fix (extract.py __iter__ now yields skipped
# chunks). The chunk FILES that version guards are unaffected -- only a manifest
# written by a pre-fix resumed run is wrong, and bumping would force a ~415 GB
# re-extraction to repair a file that costs one metadata pass to rebuild.
# run_pipeline.sh's extract-skip gate detects that manifest against the chunks
# on disk instead; see "manifest/chunk disagreement" there.
EXTRACT_SCHEMA_VERSION = 1

# annotate.py: manifest.json and LabelChunkData.
# v2: unmatched peaks no longer carry a fragment charge, so is_fragment_charge_1
# and the fragment_charges metadata differ from v1 for every noise token.
ANNOTATION_SCHEMA_VERSION = 2

# train.py: the SAE checkpoint dict.
SAE_SCHEMA_VERSION = 1

# evaluate.py: report.json and the CSVs beside it. Read back by the Phase 4
# resume path, which reconstructs the Phase 4 tensors from a previous run's
# report plus feature_label_associations.csv and per_feature_stats.csv -- so a
# report whose layout has changed must be rejected rather than half-parsed.
EVAL_SCHEMA_VERSION = 1

# build_dictionary.py: dictionary/layer_{L}.parquet, the contract between the
# pipeline and the inspector tool. Recorded in the parquet's schema metadata.
DICTIONARY_SCHEMA_VERSION = 1

# interpret.py: description runs live in interpret/layer_{L}/<run_tag>/. The
# archive directory beside them keeps superseded runs as provenance; no stage
# reads from it or writes to it, so a run tag must be one plain directory name
# that is not the archive.
INTERPRET_ARCHIVE_DIR = "archive"


def check_run_tag(tag: str) -> str:
    """Return tag if it names one description-run directory; raise ValueError otherwise."""
    if (not tag or tag in (".", "..") or any(c in tag for c in "/\\:")
            or tag.lower() == INTERPRET_ARCHIVE_DIR):
        raise ValueError(
            f"run tag {tag!r} must be a single directory name under interpret/layer_L/, "
            f"not a path and not {INTERPRET_ARCHIVE_DIR!r} (superseded runs kept as "
            f"provenance, which nothing reads)"
        )
    return tag
