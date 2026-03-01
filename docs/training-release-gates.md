# Training/UL Release Gates

This file defines blocking gates for training and UL-related release claims.

## Blocking Gates

1. Verify lane must pass.
- Workflow: `Training Verify Smoke`
- Coverage:
  - command API routing/parity checks
  - UL config schema checks
  - training metrics schema checks

2. Calibrate lane must pass.
- Workflow: `Training Calibrate Smoke`
- Coverage:
  - bench workload training command contract checks
  - UL artifact contract checks
  - provenance coherence self-test

3. Claim boundary review must hold.
- External language for this lane is limited to:
  - "UL-inspired practical two-stage pipeline"
  - "not paper-equivalent SOTA"
- This gate is currently policy/manual review, not an automated CI check.

## Non-pass Behavior

- Any gate failure blocks training/UL release claims.
- Failures must be resolved or explicitly waived with recorded approver + reason.

## Required Artifacts for Claims

- Stage manifests (`ul_stage1_manifest.json`, `ul_stage2_manifest.json`)
- Step metrics NDJSON
- Hash-linked stage dependency and provenance block
- Training workload contract used (`tools/configs/training-workloads/*.json`)

## Change Discipline

- Runtime-visible metric/schema changes must update:
  - schema validator
  - integration/config tests
  - this gate document when gate scope changes
