# Distill Studio Ops

This page now covers legacy Distill Studio compatibility helpers only.
Operational distillation runs use the first-class `distill` command family documented in [training-handbook.md](training-handbook.md).

## Status

- `distill` is the canonical training/eval/watch/compare/quality-gate operator surface.
- `distill-studio-*` scripts are retained for report analysis, diagnostics, or compatibility workflows.
- New operator behavior must be documented under the `distill` surface, not under Distill Studio naming.

## Legacy Helper Scripts

- `tools/distill-studio-mvp.mjs`
- `tools/distill-studio-diagnostics.mjs`
- `tools/distill-studio-quality-gate.mjs`

These helpers are suitable for post-hoc analysis and historical report workflows.
They are not the canonical run-contract or workload-pack execution path.

## Operator Replacement

Use these commands for active workflows instead:

```bash
node tools/doppler-cli.js distill --config '{"request":{"action":"run","workloadPath":"tools/configs/training-workloads/distill-translategemma-tiny.json"}}'

node tools/doppler-cli.js distill --config '{"request":{"action":"eval","runRoot":"reports/training/distill/distill-translategemma-tiny/2026-03-07T00-00-00.000Z"}}'

node tools/doppler-cli.js distill --config '{"request":{"action":"quality-gate","runRoot":"reports/training/distill/distill-translategemma-tiny/2026-03-07T00-00-00.000Z"}}'
```

## Reliability and Incidents

Use [training-handbook.md](training-handbook.md#incident-response) for incident sequence and release gating.
