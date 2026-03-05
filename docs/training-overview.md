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
- Bench/verify max steps: `request.trainingBenchSteps`
- Checkpoint cadence: `request.checkpointEvery` (positive integer)
- Stage linkage: `request.stage1Artifact`, `request.stage1ArtifactHash`,
  `request.stageAArtifact`, `request.stageAArtifactHash`
- Distill inputs: `request.teacherModelId`, `request.studentModelId`,
  `request.distillDatasetPath`, `request.distillLanguagePair`
- Distill scope controls: `request.distillSourceLangs`,
  `request.distillTargetLangs`, `request.distillPairAllowlist`
- Distill row contract gate: `request.strictPairContract=true` (fail closed on
  pair/src/tgt mismatch)
- Resume override controls: `request.forceResume=true` with
  `request.forceResumeReason`, optional `request.forceResumeSource`, and optional
  `request.checkpointOperator` for audited compatibility overrides.

## Operator helper script

- `node tools/run-distill-bench.mjs --mode bench ...` runs deterministic
  contract-gated benchmark lanes.
- `node tools/run-distill-bench.mjs --mode train ...` runs longer stage A/B
  verify flows with resumable checkpoint controls for operational distill runs.

## Training metrics contract (core context fields)

Training metrics entries are objective-aware and include run context needed for
auditable replay:

- `lr` / `effective_lr`
- `seed`
- `model_id`
- `runtime_preset`
- `kernel_path`
- `environment_metadata`
- `memory_stats`
- `build_provenance`

For training/distill objectives, objective-specific config fields are mandatory:

- `kd`: `distill_temperature`, `distill_alpha_kd`, `distill_alpha_ce`,
  `distill_loss_total`
- `triplet`: `distill_triplet_margin`, `distill_triplet_active_count`

## Resume compatibility + audit behavior

- Resume compatibility checks are fail-closed by default when checkpoint
  metadata mismatches expected run metadata.
- `forceResume=true` allows continuation but writes an audit record including
  mismatched fields, source/operator context, timestamp, reason, and prior
  checkpoint metadata hash.
- Resume override evidence is carried into training lineage artifacts/timelines.

## Runtime/surface notes

- Node and browser runners both support training command contract fields.
- `--surface auto` does not silently downgrade training flows when Node WebGPU is
  unavailable; it fails with explicit guidance.

## Release expectations

- Verify lane, calibrate lane, and provenance checks must pass before training
  claimable outputs are published.
- Artifacts must be hash-linked and replayable.
- CI/release gate entrypoint: `npm run ci:training:contract`
- Weekly contract delta artifact: `npm run training:contract:delta`
- Deterministic workload packs must validate against registry:
  `npm run training:workloads:verify`
- Report-id publication artifact must be produced for claim traceability:
  `npm run training:report-ids:publish -- --out <path>`
- Distill quality gates + reproducibility bundle:
  `npm run distill:quality-gate -- --report <report.json> --out-dir <dir>`

## See also

- [Training Artifact Policy](training-artifact-policy.md)
- [Training Migrations](training-migrations.md)
- [Distill Studio Ops](distill-studio-ops.md)
- [Training Contract Governance](training-governance.md)
