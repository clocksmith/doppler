# Training Claim Traceability

Policy and mechanics for deterministic workload ids and claim report ids.

## Source of truth

- Workload packs: `tools/configs/training-workloads/*.json`
- Workload registry: `tools/configs/training-workloads/registry.json`
- Report-id publication artifact: `training_report_id_index`

## Deterministic mapping

For each workload pack:

1. Compute `sha256` from the exact JSON bytes.
2. Build `baselineReportId` as `trn_<workload-id>_<sha256[0..11]>`.
3. Publish claim entries with workload id, workload sha256, and report id.

This keeps report-id claims stable and auditable across surfaces and CI runs.

## Operator commands

```bash
npm run training:workloads:verify
npm run training:report-ids:publish -- --out reports/training/report-ids/latest.json
```

## Required fields for claim publication

- `reportId`
- `workloadId`
- `workloadPath`
- `workloadSha256`
- `claimBoundary`

Claims missing any required field are out of policy.
