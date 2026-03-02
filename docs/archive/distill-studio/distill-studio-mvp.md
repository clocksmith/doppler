# Distill Studio MVP

Distill Studio MVP is deterministic and contract-driven.

## Contract Surface

Inputs:

1. Teacher report JSON (training suite or bench training report).
2. Student report JSON.
3. Optional holdout JSON array.

Outputs:

1. Replay timeline payload.
2. Branch-compare payload.
3. Mini-eval pulse payload.

## Commands

```bash
node tools/distill-studio-mvp.mjs replay-teacher --teacher <teacher-report.json> --out reports/distill-studio/replay.json
node tools/distill-studio-mvp.mjs branch-compare --teacher <teacher-report.json> --student <student-report.json> --out reports/distill-studio/compare.json
node tools/distill-studio-mvp.mjs mini-eval --teacher <teacher-report.json> --student <student-report.json> --holdout <holdout.json> --out reports/distill-studio/mini-eval.json
node tools/distill-studio-diagnostics.mjs --report <report.json>
```

## MVP Safety Rails

1. Only contract-validated reports are accepted.
2. Provenance/report coherence checks are required for diagnostics.
3. Metrics are machine-parseable JSON outputs.

## Scope

- Deterministic replay/compare/eval scaffolding for operator workflows.
- Not a full interactive UI runtime.
