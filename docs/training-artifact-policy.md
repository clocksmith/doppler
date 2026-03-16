# Training Artifact Policy

Canonical policy for training artifact naming, hashing, lineage, and deterministic behavior.

## Run-Root Layout

Deterministic run roots for the first-class operator pipelines:

- `reports/training/lora/<workload-id>/<timestamp>/`
- `reports/training/distill/<workload-id>/<timestamp>/`

Shared contents:

- `run_contract.json`
- `workload.lock.json`
- `logs/`
- `checkpoints/`
- `eval/`
- `scoreboard/`
- `exports/`
- `compare/`
- `quality-gate/`

## Artifact Classes

Shared artifact classes:

- `training_run_contract`
- `training_checkpoint`
- `training_eval_report`
- `training_scoreboard`
- `training_compare_report`
- `training_quality_gate`

Pipeline-specific classes:

- `lora_adapter_manifest`
- `distill_stage_manifest`
- `subset_manifest`

Legacy UL manifests remain part of the broader training lineage story, but UL has not yet been migrated onto the same first-class operator surface as `lora` and `distill`.

## File Conventions

Distill run roots typically include:

- `run_contract.json`
- `workload.lock.json`
- `checkpoints/<stage>/distill_stage_manifest.json`
- `checkpoints/<stage>/<checkpoint-id>/checkpoint.json`
- `checkpoints/<stage>/<checkpoint-id>/checkpoint.complete.json`
- `eval/<stage>/<checkpoint-id>__<eval-dataset-id>.json`
- `scoreboard/<stage>/scoreboard.ndjson`
- `scoreboard/<stage>/latest.json`
- `compare/compare.json`
- `quality-gate/quality-gate.json`
- `exports/subset/subset_manifest.json` when subset building is used

LoRA run roots typically include:

- `run_contract.json`
- `workload.lock.json`
- `checkpoints/<checkpoint-id>/state.json`
- `checkpoints/<checkpoint-id>/checkpoint.json`
- `checkpoints/<checkpoint-id>/checkpoint.complete.json`
- `eval/<checkpoint-id>__<eval-dataset-id>.json`
- `scoreboard/scoreboard.ndjson`
- `scoreboard/latest.json`
- `exports/<checkpoint-id>.adapter.manifest.json`
- `exports/<checkpoint-id>.export.json`
- `compare/compare.json`
- `quality-gate/quality-gate.json`

## Checkpoint Finalization Contract

- checkpoint metadata must be separate from raw tensor state
- finalized checkpoints must write `checkpoint.complete.json`
- watchers evaluate only finalized checkpoints
- checkpoint lineage must preserve config, dataset, and parent-artifact linkage

## Hashing Policy

- Hash function: SHA-256 hex for lineage and provenance linkage
- `workloadSha256`: exact workload pack bytes
- `configHash`: deterministic normalized workload or config hash
- `datasetHash`: deterministic dataset content hash
- `artifactHash`: deterministic normalized artifact payload hash
- file-level hashes and content-level hashes are intentionally distinct and both may be published

## Required Traceability Fields

Each claimable artifact should carry:

- `artifactType`
- `schemaVersion`
- `reportId`
- `workloadId`
- `workloadPath`
- `workloadSha256`
- `configHash`
- `datasetPath`
- `datasetHash`
- `baseModelId`
- `teacherModelId` when applicable
- `studentModelId` when applicable
- `stage`
- `checkpointStep`
- `parentArtifacts`
- `generatedAt`
- `runtime`
- `surface`
- `claimBoundary`

## Lineage Requirements

- Stage2 UL artifacts must reference and validate Stage1 dependency hashes
- Stage B distill artifacts must reference and validate Stage A dependency hashes
- exported LoRA adapter artifacts must preserve the originating checkpoint linkage
- compare and quality-gate reports must point back to the run root and workload lock
- published workload packs include deterministic report-id bindings derived from workload content hashes

## Deterministic Timestamp Behavior

- timestamps are allowed for auditability and run tracking
- deterministic content hashes must exclude volatile timestamp and run-root path fields
- run-root timestamps do not change workload hashes or config hashes

## Validation Tools

Use provenance and workload verification for manifests and reports:

```bash
node tools/verify-training-provenance.js --manifest <manifest.json> [--stage1-manifest <manifest.json>]
node tools/verify-training-provenance.js --report <report.json>
node tools/verify-training-workload-packs.js --registry tools/configs/training-workloads/registry.json
node tools/publish-training-report-ids.js --registry tools/configs/training-workloads/registry.json --out reports/training/report-ids/latest.json
```
