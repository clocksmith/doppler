# Training Overview

Practical training support in Doppler is contract-driven and currently focused on
verify/calibrate workflows, artifact lineage, and deterministic reporting.

## Scope and claim boundary

- UL: UL-inspired practical two-stage workflow (`stage1_joint`, `stage2_base`).
- Distill: practical two-stage distill workflow (`stage_a`, `stage_b`).
- Scope is operational validation and reproducible artifacts, not paper-equivalent
  SOTA claims.

## Command intents

- Verify path: `verify --config '{"request":{"suite":"training",...}}'`
- Calibrate path: `bench --config '{"request":{"workloadType":"training",...}}'`
- These paths are intentionally non-interchangeable and fail closed on invalid
  field combinations.

## Supported command fields

- Stage and workload: `request.trainingStage`, `request.workloadType="training"`
- Schema/versioning: `request.trainingSchemaVersion` (pinned to `1`)
- Bench steps: `request.trainingBenchSteps`
- Stage linkage: `request.stage1Artifact`, `request.stage1ArtifactHash`,
  `request.stageAArtifact`, `request.stageAArtifactHash`
- Distill inputs: `request.teacherModelId`, `request.studentModelId`,
  `request.distillDatasetPath`, `request.distillLanguagePair`

## Runtime/surface notes

- Node and browser runners both support training command contract fields.
- `--surface auto` does not silently downgrade training flows when Node WebGPU is
  unavailable; it fails with explicit guidance.

## Release expectations

- Verify lane, calibrate lane, and provenance checks must pass before training
  claimable outputs are published.
- Artifacts must be hash-linked and replayable.

## See also

- [Training Artifact Policy](training-artifact-policy.md)
- [Training Migrations](training-migrations.md)
- [Distill Studio Ops](distill-studio-ops.md)
