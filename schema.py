"""On-disk schema versions for the pipeline's cached artefacts.

The pipeline caches expensive intermediates (activation chunks, concept labels,
SAE checkpoints) and reuses them across runs, so a stale artefact from an
incompatible run is a realistic failure mode. Each consumer compares the version
recorded in an artefact against the constant below and hard-fails on a
mismatch, rather than silently reading misaligned data.

Bump a constant when its format changes incompatibly. Existing artefacts of that
type must then be regenerated.

This module deliberately has no imports, so any consumer can read the constants
without pulling in torch, spectrum_utils, or the InstaNovo package.
"""

# extract.py: manifest.json, ChunkMeta, and the per-layer activation files.
EXTRACT_SCHEMA_VERSION = 1

# annotate.py: manifest.json and LabelChunkData.
ANNOTATION_SCHEMA_VERSION = 1

# train.py: the SAE checkpoint dict.
SAE_SCHEMA_VERSION = 1

# evaluate.py: report.json and the CSVs beside it. Read back by the Phase 4
# resume path, which reconstructs the Phase 4 tensors from a previous run's
# report plus feature_label_associations.csv and per_feature_stats.csv -- so a
# report whose layout has changed must be rejected rather than half-parsed.
EVAL_SCHEMA_VERSION = 1
