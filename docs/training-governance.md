# Training Contract Governance

Operational governance requirements for training/distillation contract claims.

## Required artifacts per release cycle

- Weekly machine-readable contract delta artifact.
- Training contract release-gate pass in CI.
- Distill quality-gate exports (EN and ES) with reproducibility bundle for any
  distill demo claim.

## Commands

- Emit contract delta:
  - `npm run training:contract:delta -- --out ./reports/training/contract-delta/latest.json`
- Run release gates:
  - `npm run ci:training:contract`
- Verify deterministic training workload packs:
  - `npm run training:workloads:verify`
- Publish report-id traceability index:
  - `npm run training:report-ids:publish -- --out ./reports/training/report-ids/latest.json`
- Run Distill Studio MVP contract modes:
  - `npm run distill:studio:mvp -- replay-teacher --teacher <teacher-report.json> --workload-pack tools/configs/training-workloads/distill-translategemma-tiny.json --out reports/distill-studio/replay.json`
  - `npm run distill:studio:mvp -- branch-compare --teacher <teacher-report.json> --student <student-report.json> --workload-pack tools/configs/training-workloads/distill-translategemma-tiny.json --out reports/distill-studio/compare.json`
  - `npm run distill:studio:mvp -- mini-eval --teacher <teacher-report.json> --student <student-report.json> --holdout <holdout.json> --workload-pack tools/configs/training-workloads/distill-translategemma-tiny.json --out reports/distill-studio/mini-eval.json`
- Emit Distill Studio quality gates:
  - `npm run distill:quality-gate -- --report <report.json> --out-dir <dir>`

## Demo claim policy

- Any benchmark or quality claim shown in demo/UI/docs must include a report id.
- Report id must map to a stored report artifact and reproducibility bundle.
- Report id must map to an entry in the published claim index (`training_report_id_index`).
- Claims without report ids are out of policy.

## Prelaunch red-team review checklist

- Verify unknown-suite and malformed-training requests fail closed.
- Verify command/suite parity between browser and node surfaces.
- Verify training metrics schema validation rejects incomplete objective payloads.
- Verify forced resume writes explicit `resumeAudits`.
- Verify distill quality-gate EN/ES artifacts are present and reproducible.
- Verify deterministic workload registry hashes match pack files.
- Verify report-id index publication references the deterministic workload registry.

## References

- `tools/ci-training-contract-gates.mjs`
- `.github/workflows/training-contract-release-gate.yml`
- `tools/emit-training-contract-delta.mjs`
- `tools/distill-studio-quality-gate.mjs`
- `tools/verify-training-workload-packs.mjs`
- `tools/publish-training-report-ids.mjs`
- `tools/configs/training-workloads/registry.json`
