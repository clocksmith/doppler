# Training Handbook

Canonical operational handbook for Doppler training and distillation.

## Scope

- first-class workload-driven `lora` and `distill` operator commands
- legacy `verify` / `bench` training harness lanes for contract gates and calibration
- deterministic run roots, artifact lineage, and claim traceability
- eval, checkpoint watch, compare, and quality-gate workflows

This is operational guidance, not a paper-equivalent SOTA claim surface.

## Primary Commands

### Contract and release gates

```bash
npm run ci:training:contract
npm run training:contract:delta -- --out ./reports/training/contract-delta/latest.json
npm run training:workloads:verify
npm run training:report-ids:publish -- --out ./reports/training/report-ids/latest.json
```

### First-class operator commands

```bash
node tools/doppler-cli.js distill --config '{"request":{"action":"subsets","workloadPath":"tools/configs/training-workloads/distill-translategemma-tiny.json"}}'

node tools/doppler-cli.js distill --config '{"request":{"action":"run","workloadPath":"tools/configs/training-workloads/distill-translategemma-tiny.json"}}'

node tools/doppler-cli.js distill --config '{"request":{"action":"watch","runRoot":"reports/training/distill/distill-translategemma-tiny/2026-03-07T00-00-00.000Z","stopWhenIdle":true}}'

node tools/doppler-cli.js lora --config '{"request":{"action":"run","workloadPath":"tools/configs/training-workloads/lora-toy-tiny.json"}}'

node tools/doppler-cli.js lora --config '{"request":{"action":"export","runRoot":"reports/training/lora/lora-toy-tiny/2026-03-07T00-00-00.000Z"}}'
```

`tools/distillation.js` and `tools/lora.js` are thin wrappers over `tools/doppler-cli.js`.
The canonical request contract lives in `src/tooling/command-api.js`.

### Legacy harness lanes

```bash
node tools/doppler-cli.js verify --config '{"request":{"suite":"training","modelId":"gemma-3-270m-it-wq4k-ef16-hf16"}}' --json

node tools/doppler-cli.js bench --config '{"request":{"suite":"bench","modelId":"gemma-3-270m-it-wq4k-ef16-hf16","workloadType":"training","trainingSchemaVersion":1}}' --json
```

Use these lanes for contract verification, calibration, and legacy training harness coverage.
They are not replacements for the workload-first `lora` / `distill` operator lifecycle.

## Command Surface

### `distill`

Actions:
- `run`
- `stage-a`
- `stage-b`
- `eval`
- `watch`
- `compare`
- `quality-gate`
- `subsets`

Action requirements:
- `run`, `stage-a`, `subsets`: `workloadPath` or `runRoot`
- `stage-b`: `workloadPath` or `runRoot`; provide `stageArtifact` when a validated Stage A artifact is not already discoverable from the run root
- `eval`: `workloadPath` or `runRoot`, plus either `checkpointPath` or finalized checkpoints under the run root
- `watch`, `compare`, `quality-gate`: `runRoot`

### `lora`

Actions:
- `run`
- `eval`
- `watch`
- `export`
- `compare`
- `quality-gate`
- `activate`

Action requirements:
- `run`: `workloadPath` or `runRoot`
- `eval`, `export`: `workloadPath` or `runRoot`, plus either `checkpointPath` or finalized checkpoints under the run root
- `watch`, `compare`, `quality-gate`: `runRoot`
- `activate`: part of the command contract, but the current Node operator runner fails closed and directs activation to the browser provider/runtime surface

## Current Implementation Notes

- `lora` and `distill` are currently Node-only in the command API and fail closed on browser surfaces.
- `lora run` currently supports `baseModelId="training-toy"` with `datasetFormat="toy_linear_classification_jsonl"` only.
- `distill` currently resolves workload stages into the internal `stage_a` / `stage_b` runner contract.
- Distillation workloads that declare `sft` fail closed today; use `objective="kd"` with `trainingStage="stage_a"` or `objective="triplet"` with `trainingStage="stage_b"` until a plain-SFT runner exists.
- Distillation translation eval is currently implemented for `studentGraphMode="transformer_full"` only.

## Workload Packs

Canonical workload packs live under:
- `tools/configs/training-workloads/lora-*.json`
- `tools/configs/training-workloads/distill-*.json`

Each pack is the source of truth for:
- `schemaVersion`, `kind`, `id`, `description`, `claimBoundary`, `seed`
- `baseModelId`, `studentModelId`, `teacherModelId`
- `datasetId`, `datasetPath`, `evalDatasets`
- `trainingSchemaVersion`, `checkpointEvery`, `selectionMetric`, `selectionGoal`, `surfaceSupport`
- `training.optimizer`, `training.batchSize`, `training.accumSteps`, `training.steps`, `training.precision`, `training.gradientClipping`
- pipeline-specific fields in `lora.*` or `distill.*`

LoRA-specific fields include:
- `datasetFormat`
- `taskType`
- `adapter.rank`
- `adapter.alpha`
- `adapter.dropout`
- `adapter.targetModules`
- `freeze`
- `export`
- `activation`

Distill-specific fields include:
- `stagePlan`
- `studentGraphMode`
- `temperature`
- `alphaKd`
- `alphaCe`
- `tripletMargin`
- `sourceLangs`
- `targetLangs`
- `pairAllowlist`
- `strictPairContract`
- `subsetSpec`

Rule: if it changes behavior, it belongs in the workload pack.

## Run Roots and Artifacts

Deterministic run roots:
- `reports/training/lora/<workload-id>/<timestamp>/`
- `reports/training/distill/<workload-id>/<timestamp>/`

Each run root contains:
- `run_contract.json`
- `workload.lock.json`
- `logs/`
- `checkpoints/`
- `eval/`
- `scoreboard/`
- `exports/`
- `compare/`
- `quality-gate/`

Checkpoint readiness is explicit:
- checkpoint metadata is written separately from raw state
- finalized checkpoints are marked with `checkpoint.complete.json`
- watchers evaluate only finalized checkpoints

Canonical artifact classes:
- `training_run_contract`
- `training_checkpoint`
- `training_eval_report`
- `training_scoreboard`
- `training_compare_report`
- `training_quality_gate`
- `lora_adapter_manifest`
- `distill_stage_manifest`
- `subset_manifest`

## Claim Traceability Requirements

A publishable claim must include:
- `reportId`
- `workloadId`
- `workloadPath`
- `workloadSha256`
- `configHash`
- `datasetPath`
- `datasetHash`
- `claimBoundary`

Mapping is deterministic from workload-pack bytes and derived artifact payloads.

## Operating Rhythm

1. Validate contract gates.
2. Validate workload registry integrity.
3. Publish the report-id artifact.
4. Build subsets when required by a distill workload.
5. Execute `distill` or `lora` run / eval / watch lanes.
6. Generate compare and quality-gate artifacts before claim publication.
7. Publish only claimable artifacts with deterministic traceability fields.

## Incident Response

1. Freeze publication for affected workload IDs and report IDs.
2. Rerun contract gates and isolate the failing operator lane or artifact class.
3. Regenerate the run-root artifacts and report-id index.
4. Publish a corrective note with the affected workload IDs, report IDs, and claim boundary.

## Related Policy Docs

- [training-overview.md](training-overview.md)
- [training-governance.md](training-governance.md)
- [training-claim-traceability.md](training-claim-traceability.md)
- [training-operator-playbook.md](training-operator-playbook.md)
- [training-artifact-policy.md](training-artifact-policy.md)
- [training-migrations.md](training-migrations.md)
- [distill-studio-ops.md](distill-studio-ops.md) for legacy compatibility helpers only
