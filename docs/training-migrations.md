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

## Metrics schema migration highlights

Objective-aware metrics schema includes explicit unions for:

- `cross_entropy`
- `ul_stage1_joint`
- `ul_stage2_base`
- `kd`
- `triplet`

Each objective requires its own field set and rejects incompatible stage/objective
mixes.

## UL schema migration highlights

UL config includes explicit stage/version controls, dependency linkage, and
artifact fields suitable for stage-to-stage verification.

## Migration guidance

- Producers: emit required objective/stage fields explicitly.
- Consumers: treat missing required fields as invalid payloads.
- Preserve deterministic content-hash semantics when introducing new fields.

## Historical notes

Older migration details were removed from the public docs tree during
documentation consolidation. Use git history for prior training migration
iterations and changelog context.
