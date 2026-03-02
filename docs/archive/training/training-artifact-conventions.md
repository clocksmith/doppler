# Training Artifact Conventions

Canonical naming/versioning conventions for training and UL artifacts.

## UL run directories

Path pattern:

`bench/out/ul/<stage>_<timestamp>/`

Required files:

- `metrics.ndjson`
- `ul_stage1_manifest.json` or `ul_stage2_manifest.json`
- `latents.ndjson` (stage1 only)

## Versioning

1. `ul_stage*_manifest.json` includes:
- `schemaVersion`
- `manifestHash`
- `manifestContentHash`

2. Step metrics entries include:
- `schemaVersion`

3. Training workload contracts include:
- `schemaVersion`

## Report linkage

1. Training suite/bench reports should include UL artifact references under:
- `report.lineage.training.ulArtifacts[]` and/or `report.metrics.ulArtifacts[]`

2. Report references must include:
- `manifestPath`
- `manifestHash` when available
