# Distill Studio Ops

Operational guide for Distill Studio MVP workflows, reliability checks, staged
rollout, and incident handling.

## Contract surface

Inputs:

- teacher report JSON
- student report JSON
- optional holdout fixture JSON

Outputs:

- replay timeline payload
- branch-compare payload
- mini-eval pulse payload

## Operator commands

```bash
node tools/distill-studio-mvp.mjs replay-teacher --teacher <teacher-report.json> --out reports/distill-studio/replay.json
node tools/distill-studio-mvp.mjs branch-compare --teacher <teacher-report.json> --student <student-report.json> --out reports/distill-studio/compare.json
node tools/distill-studio-mvp.mjs mini-eval --teacher <teacher-report.json> --student <student-report.json> --holdout <holdout.json> --out reports/distill-studio/mini-eval.json
node tools/distill-studio-diagnostics.mjs --report <report.json>
```

## Rollout stages

1. Internal: deterministic command outputs + provenance checks.
2. Limited: replay/branch-compare only, diagnostics required per candidate.
3. Wider: enable mini-eval and reliability dashboard ingestion.

## Reliability signals

- Contract/schema drift rate
- Provenance check pass rate
- Replay/compare/mini-eval success rates
- Alert and fail-on-alert incident rates

## Incident response (condensed)

1. Capture failing command and report/artifact paths.
2. Run diagnostics and provenance checks.
3. Freeze claimable outputs for affected artifacts.
4. Fix root cause, add regression test, regenerate artifacts, rerun release gates.

## Scope

Distill Studio is deterministic, contract-validated operator tooling.
It is not a full interactive UI runtime.
