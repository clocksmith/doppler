# Training Rollout Readiness

Rollout checklist for the first-class `lora` and `distill` operator surface.

## Gate Readiness

- `npm run ci:training:contract` passes all lanes
- no lane filtering is used for release checks
- contract-delta artifact is generated for each release cycle

## Operator Surface Readiness

- `lora` and `distill` are present in `src/tooling/command-api.js`
- CLI help and API docs describe the operator commands
- browser surfaces fail closed for unsupported operator actions
- workload packs are validated through the training-workload registry
- run roots write `run_contract.json` and `workload.lock.json`
- finalized checkpoints write `checkpoint.complete.json`
- eval, compare, scoreboard, and quality-gate artifacts are emitted for candidate runs

## Traceability Readiness

- workload registry hashes match workload-pack files
- baseline report IDs are present for all workload packs
- report-id publication artifact is generated and stored
- claimable artifacts carry workload and dataset traceability fields

## Legacy Helper Readiness

- Distill Studio helper scripts are documented as compatibility tooling only
- operator documentation does not depend on Distill Studio naming

## External Readiness Tracks

- operator playbook is published and current
- rollout doc set is published and current
- benchmark publication process is documented and followed
