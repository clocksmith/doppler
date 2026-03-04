# Diffusion Expansion Lanes

This document tracks post-closure expansion work for the GPU SD3 diffusion
contract after verify/bench smoke, image regression fixtures, and perf artifact
gates are locked.

## Source of Truth

- Tracker JSON: `tools/configs/diffusion-expansion-lanes.json`
- Current schema: `schemaVersion=1`

## Lanes (Started)

1. Larger targets (`sd3-larger-targets`)
- Scope: extend deterministic coverage to larger image targets.
- Next action: add a 1024x1024 fixture pack and optional CI lane.

2. Quantized diffusion path (`sd3-quantized-path`)
- Scope: define quantized diffusion runtime contract and acceptance checks.
- Next action: pin schema + fail-closed compatibility tests.

3. Broader platform rollout (`sd3-platform-rollout`)
- Scope: publish platform matrix gates for browser + Node WebGPU.
- Next action: add platform-specific artifact checks and rollout thresholds.

## Exit Criteria

- Each lane has contract tests in `tests/config` or `tests/integration`.
- Command-level verify and bench semantics remain identical across surfaces.
- Required diffusion artifacts are emitted in CI for each enabled lane.
