# Distill Studio Reliability Dashboard

Operator-facing reliability signals for Distill Studio MVP outputs.

## Required panels

1. Contract health
- training command schema version drift

2. Provenance health
- `verify-training-provenance --report` pass rate
- report-to-manifest linkage error count

3. Workflow health
- replay job success rate
- branch compare success rate
- mini-eval pulse pass ratio

4. Alert health
- telemetry alert frequency by code
- fail-on-alert incident count

## Data sources

- Distill Studio MVP output JSON (`reports/distill-studio/*`)
- training reports (`reports/<model>/*.json`)
- UL manifests (`bench/out/ul/*/ul_stage*.json`)
