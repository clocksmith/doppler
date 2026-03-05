# Training Handbook

Canonical operational handbook for Doppler training and distillation.

## Scope

- contract-gated verify/calibrate workflows
- deterministic traceability and artifact lineage
- distill stage operations and incident response

This is operational guidance, not a paper-equivalent SOTA claim surface.

## Primary commands

### Contract and release gates

```bash
npm run ci:training:contract
npm run training:contract:delta -- --out ./reports/training/contract-delta/latest.json
npm run training:workloads:verify
npm run training:report-ids:publish -- --out ./reports/training/report-ids/latest.json
```

### Distill Studio MVP

```bash
npm run distill:studio:mvp -- replay-teacher --teacher <teacher-report.json> --workload-pack tools/configs/training-workloads/distill-translategemma-tiny.json --out reports/distill-studio/replay.json
npm run distill:studio:mvp -- branch-compare --teacher <teacher-report.json> --student <student-report.json> --workload-pack tools/configs/training-workloads/distill-translategemma-tiny.json --out reports/distill-studio/compare.json
npm run distill:studio:mvp -- mini-eval --teacher <teacher-report.json> --student <student-report.json> --holdout <holdout.json> --workload-pack tools/configs/training-workloads/distill-translategemma-tiny.json --out reports/distill-studio/mini-eval.json
npm run distill:quality-gate -- --report <report.json> --out-dir <dir>
```

## Distill runtime fields

Common request fields:
- `trainingStage`, `workloadType`
- `trainingSchemaVersion`
- `checkpointEvery`
- `teacherModelId`, `studentModelId`
- `distillDatasetPath`
- `distillSourceLangs`, `distillTargetLangs`, `distillPairAllowlist`
- `strictPairContract`

Resume override fields:
- `forceResume`
- `forceResumeReason`
- optional `forceResumeSource`
- optional `checkpointOperator`

## Claim traceability requirements

A publishable claim must include:
- `reportId`
- `workloadId`
- `workloadPath`
- `workloadSha256`
- `claimBoundary`

Mapping is deterministic from workload pack bytes.

## Operating rhythm

1. Validate contract gates.
2. Validate workload registry integrity.
3. Publish report-id artifact.
4. Execute distill lanes and quality gates.
5. Publish only claimable artifacts with traceability fields.

## Incident response

1. Freeze publication for affected report IDs.
2. Rerun contract gates and isolate failure lane.
3. Regenerate artifacts and report-id index.
4. Publish corrective note with affected workload/report IDs.

## Related policy docs

- [training-overview.md](training-overview.md)
- [training-governance.md](training-governance.md)
- [training-claim-traceability.md](training-claim-traceability.md)
- [distill-studio-ops.md](distill-studio-ops.md)
- [training-operator-playbook.md](training-operator-playbook.md)
- [training-artifact-policy.md](training-artifact-policy.md)
- [training-migrations.md](training-migrations.md)
