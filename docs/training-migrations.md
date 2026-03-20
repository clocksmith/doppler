# Training Migrations

Migration summary for the training command surface, workload-pack schema, and artifact/report contracts.

## Current Schema Baseline

- training command schema version is pinned to `1`
- workload-pack schema version is pinned to `1`
- `trainingSchemaVersion` must be `1` for training flows

## Command Contract Migration Highlights

- first-class operator commands are now `lora` and `distill`
- operator commands require `action`
- operator commands require `workloadPath` or `runRoot`
- `watch`, `compare`, and `quality-gate` are run-root-driven and fail closed without `runRoot`
- `eval` and `export` operate from explicit `checkpointPath` or finalized checkpoints already present in the run root
- legacy training-only fields remain valid only for:
  - `verify --config '{"request":{"workload":"training",...}}'`
  - `bench --config '{"request":{"workload":"training","workloadType":"training",...}}'`
- invalid field and surface combinations remain fail closed
- `forceResumeReason`, `forceResumeSource`, and `checkpointOperator` still require `forceResume=true` on legacy harness flows

## Workload-Pack Migration Highlights

- workload packs under `tools/configs/training-workloads/` are now the canonical source of truth for operator runs
- shared workload fields include model IDs, dataset IDs and paths, eval datasets, checkpoint cadence, selection policy, surface support, and training policy
- LoRA workload fields now carry adapter, freeze, export, and activation policy
- distill workload fields now carry stage plan, KD and triplet parameters, pair-policy filters, and subset policy

## Artifact Migration Highlights

- run roots are deterministic under `reports/training/<kind>/<workload-id>/<timestamp>/`
- every run writes `run_contract.json` and `workload.lock.json`
- finalized checkpoints require `checkpoint.complete.json`
- compare and quality-gate artifacts are normal pipeline outputs, not ad hoc summary files
- scoreboards are append-only `scoreboard.ndjson` plus a derived `latest.json`

## Metrics Schema Migration Highlights

Objective-aware metrics schema includes explicit unions for:

- `cross_entropy`
- `ul_stage1_joint`
- `ul_stage2_base`
- `kd`
- `triplet`

Each objective requires its own field set and rejects incompatible stage and objective mixes.

Distill objective config fields are enforced as required payload fields:

- KD (`objective="kd"`): `distill_temperature`, `distill_alpha_kd`, `distill_alpha_ce`, `distill_loss_total`
- triplet (`objective="triplet"`): `distill_triplet_margin`, `distill_triplet_active_count`

Core replay context fields remain part of the metrics contract:

- `lr` / `effective_lr`
- `seed`
- `model_id`
- `runtime_profile`
- `kernel_path`
- `environment_metadata`
- `memory_stats`
- `build_provenance`

## Current Compatibility Notes

- browser surfaces still fail closed for `lora` and `distill`
- `lora run` currently supports the toy training backend only
- distill stage entries still resolve to the internal `stage_a` / `stage_b` contract
- plain `sft` distill workloads are rejected by the current JS distill runner

## Migration Guidance

- producers: emit required workload, checkpoint, eval, and artifact fields explicitly
- consumers: treat missing required fields as invalid payloads
- preserve deterministic content-hash semantics when introducing new fields
- prefer workload-pack updates over new runtime-only flags

## Historical Notes

Older migration details were removed from the public docs tree during documentation consolidation.
Use git history for prior training-migration iterations and changelog context.
