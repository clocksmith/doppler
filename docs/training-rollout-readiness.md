# Training Rollout Readiness

Rollout checklist for training/distillation contract and Distill Studio MVP.

## Gate readiness

- `npm run ci:training:contract` passes all lanes.
- No lane filtering is used for release checks.
- Contract delta artifact is generated for each release cycle.

## Distill Studio MVP readiness

- Replay teacher mode produces traceable payload.
- Branch compare mode produces traceable payload.
- Mini-eval mode produces traceable payload.
- Distill diagnostics and quality-gate outputs are generated for candidate claims.

## Traceability readiness

- Workload registry hashes match workload pack files.
- Baseline report ids are present for all workload packs.
- Report-id publication artifact is generated and stored.

## External readiness tracks (process owned)

- Operator playbook is published and current.
- Rollout doc set is published and current.
- Benchmark publication process is documented and followed.
