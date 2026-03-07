# Training Overview

This file is a concise overview. The canonical operational guide is [training-handbook.md](training-handbook.md).

## Scope and Claim Boundary

Doppler training support is contract-driven and focused on reproducible operator workflows, verification/calibration gates, and deterministic lineage.

## Command Families

### First-class operator commands

- `distill --config '{"request":{"action":"run|stage-a|stage-b|eval|watch|compare|quality-gate|subsets",...}}'`
- `lora --config '{"request":{"action":"run|eval|watch|export|compare|quality-gate|activate",...}}'`

These commands are workload-first and operate from workload packs plus run-root artifacts.

### Legacy harness commands

- `verify --config '{"request":{"suite":"training",...}}'`
- `bench --config '{"request":{"workloadType":"training",...}}'`

These remain valid for contract verification and calibration, but they are not the primary operator surface.

## Contract Highlights

- Canonical workload packs live under `tools/configs/training-workloads/`
- Run roots are deterministic under `reports/training/<kind>/<workload-id>/<timestamp>/`
- Finalized checkpoints use explicit `checkpoint.complete.json` markers
- Eval, scoreboard, compare, and quality-gate artifacts are written as normal pipeline outputs
- Browser surfaces fail closed for `lora` and `distill`

## Current Implementation Notes

- `distill` is the main real-model training/eval operator path in the repo today
- `lora run` is currently a toy training backend and explicitly rejects unsupported real-model workloads
- Distill workloads currently map into the internal `stage_a` / `stage_b` runner contract; plain `sft` is not yet a supported workload stage

## Canonical Docs

- Operational handbook: [training-handbook.md](training-handbook.md)
- Artifact lineage policy: [training-artifact-policy.md](training-artifact-policy.md)
- Operator playbook: [training-operator-playbook.md](training-operator-playbook.md)
- Migration baseline: [training-migrations.md](training-migrations.md)
