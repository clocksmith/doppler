# Training Operator Playbook

Operational playbook for training/distillation contract release readiness.

## Daily operations

1. Validate training contract gates:
   - `npm run ci:training:contract`
2. Validate workload registry integrity:
   - `npm run training:workloads:verify`
3. Publish report-id index artifact:
   - `npm run training:report-ids:publish -- --out reports/training/report-ids/latest.json`

## Distill Studio MVP operations

1. Replay teacher:
   - `npm run distill:studio:mvp -- replay-teacher --teacher <teacher-report.json> --workload-pack tools/configs/training-workloads/distill-translategemma-tiny.json --out reports/distill-studio/replay.json`
2. Branch compare:
   - `npm run distill:studio:mvp -- branch-compare --teacher <teacher-report.json> --student <student-report.json> --workload-pack tools/configs/training-workloads/distill-translategemma-tiny.json --out reports/distill-studio/compare.json`
3. Mini eval pulses:
   - `npm run distill:studio:mvp -- mini-eval --teacher <teacher-report.json> --student <student-report.json> --holdout <holdout.json> --workload-pack tools/configs/training-workloads/distill-translategemma-tiny.json --out reports/distill-studio/mini-eval.json`
4. Diagnostics + quality gate:
   - `node tools/distill-studio-diagnostics.mjs --report <report.json>`
   - `npm run distill:quality-gate -- --report <report.json> --out-dir reports/distill-studio/gates`

## Incident handling

1. Freeze claim publication for affected report ids.
2. Re-run `npm run ci:training:contract` and isolate failing lane.
3. Regenerate artifacts and report-id index.
4. Publish corrective notes with affected workload ids/report ids.
