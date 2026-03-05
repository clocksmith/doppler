# Training Overview

This file is a concise overview. The canonical operational guide is [training-handbook.md](training-handbook.md).

## Scope and claim boundary

Doppler training support is contract-driven and focused on reproducible verification, calibration, and lineage.

## Command intents

- Verify path: `verify --config '{"request":{"suite":"training",...}}'`
- Calibrate path: `bench --config '{"request":{"workloadType":"training",...}}'`

## Contract highlights

- Stage/workload fields (`trainingStage`, `workloadType`)
- Schema pin (`trainingSchemaVersion`)
- Distill dataset/pair gates (`distillSourceLangs`, `distillTargetLangs`, `distillPairAllowlist`, `strictPairContract`)
- Resume override audit controls (`forceResume*`)

## Canonical docs

- Operational handbook: [training-handbook.md](training-handbook.md)
- Artifact lineage policy: [training-artifact-policy.md](training-artifact-policy.md)
- Migration baseline: [training-migrations.md](training-migrations.md)
