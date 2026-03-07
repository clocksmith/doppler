# Training Operator Playbook

This file remains an entrypoint. The canonical operator contract and examples live in [training-handbook.md](training-handbook.md).

## Daily Operations

1. `npm run ci:training:contract`
2. `npm run training:workloads:verify`
3. `npm run training:report-ids:publish -- --out reports/training/report-ids/latest.json`

## Common Operator Sequence

1. For distill workloads with subset policy, run `distill subsets`.
2. Run `distill run` or `lora run` from a workload pack.
3. Use `watch` against the run root when live checkpoint eval is required.
4. Use `compare` and `quality-gate` against the completed run root before publication.

Examples:

```bash
node tools/doppler-cli.js distill --config '{"request":{"action":"run","workloadPath":"tools/configs/training-workloads/distill-translategemma-tiny.json"}}'

node tools/doppler-cli.js distill --config '{"request":{"action":"compare","runRoot":"reports/training/distill/distill-translategemma-tiny/2026-03-07T00-00-00.000Z"}}'

node tools/doppler-cli.js lora --config '{"request":{"action":"quality-gate","runRoot":"reports/training/lora/lora-toy-tiny/2026-03-07T00-00-00.000Z"}}'
```

## Legacy Distill Studio Helpers

Legacy report-analysis and diagnostics helpers are documented in [distill-studio-ops.md](distill-studio-ops.md).
They are not the source of truth for operator execution.

## Incident Handling

Use the canonical incident response in [training-handbook.md](training-handbook.md#incident-response).
