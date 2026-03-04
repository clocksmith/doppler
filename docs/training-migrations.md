# Training Migrations

Migration summary for training command schema, UL schema, and metrics schema.

## Current schema baseline

- Training command schema version is pinned to `1`.
- `trainingSchemaVersion` must be `1` for training flows.
- `trainingBenchSteps` is supported for training bench workloads.

## Command contract migration highlights

- Training-only fields are valid only for:
  - `verify --config '{"request":{"suite":"training",...}}'`
  - `bench --config '{"request":{"workloadType":"training",...}}'`
- Invalid field/suite combinations are fail-closed.
- `forceResumeReason` requires `forceResume=true`.

## Metrics schema migration highlights

Objective-aware metrics schema includes explicit unions for:

- `cross_entropy`
- `ul_stage1_joint`
- `ul_stage2_base`
- `kd`
- `triplet`

Each objective requires its own field set and rejects incompatible stage/objective
mixes.

Distill objective config fields are enforced as required payload fields:

- KD (`objective="kd"`): `distill_temperature`, `distill_alpha_kd`,
  `distill_alpha_ce`, `distill_loss_total`
- Triplet (`objective="triplet"`): `distill_triplet_margin`,
  `distill_triplet_active_count`

Core replay context fields are now treated as part of the metrics contract:

- `lr` / `effective_lr`
- `seed`
- `model_id`
- `runtime_preset`
- `kernel_path`
- `environment_metadata`
- `memory_stats`
- `build_provenance`

## UL schema migration highlights

UL config includes explicit stage/version controls, dependency linkage, and
artifact fields suitable for stage-to-stage verification.

## Checkpoint provenance + resume migration highlights

- Checkpoints persist metadata hashes for config/dataset/optimizer lineage.
- Build provenance fields (commit/build/timestamp) are persisted when available.
- Runtime environment metadata includes runtime/surface context and GPU adapter
  metadata when available.
- Resume metadata mismatches fail closed by default.
- Forced resume writes explicit `resumeAudits` records, including source,
  operator, reason, mismatch list, and prior checkpoint metadata hash.

## Migration guidance

- Producers: emit required objective/stage fields explicitly.
- Consumers: treat missing required fields as invalid payloads.
- Preserve deterministic content-hash semantics when introducing new fields.

## Historical notes

Older migration details were removed from the public docs tree during
documentation consolidation. Use git history for prior training migration
iterations and changelog context.
