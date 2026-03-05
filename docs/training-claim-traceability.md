# Training Claim Traceability

Canonical traceability workflow is maintained in [training-handbook.md](training-handbook.md#claim-traceability-requirements).

## Source of truth

- Workload packs: `tools/configs/training-workloads/*.json`
- Registry: `tools/configs/training-workloads/registry.json`
- Published report-id index artifact

## Deterministic mapping

For each workload pack:
1. compute `sha256` from exact bytes
2. derive stable report id
3. publish claim entry with workload id/hash and report id

## Required publication fields

- `reportId`
- `workloadId`
- `workloadPath`
- `workloadSha256`
- `claimBoundary`
