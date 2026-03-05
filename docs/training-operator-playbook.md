# Training Operator Playbook

This file remains as an entrypoint. Daily operations and incident sequence are canonicalized in [training-handbook.md](training-handbook.md).

## Daily operations

1. `npm run ci:training:contract`
2. `npm run training:workloads:verify`
3. `npm run training:report-ids:publish -- --out reports/training/report-ids/latest.json`

## Distill Studio operations

Use the command set in [training-handbook.md](training-handbook.md#distill-studio-mvp).

## Incident handling

Use the canonical incident response in [training-handbook.md](training-handbook.md#incident-response).
