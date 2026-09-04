---
name: doppler-perf
description: Localize a measured Doppler prefill, decode, TTFT, or model-load bottleneck through controlled profiling experiments when a baseline receipt is available.
---

# Doppler Performance Diagnosis

## Prerequisites

- Run from the Doppler repository root.
- Provide a saved baseline command/receipt, model and artifact identity, runtime
  profile, surface, cache mode, workload, and hardware identity.
- Read the applicable benchmark, config, JavaScript, and WGSL style guides before
  changing measurement or execution behavior.

## Procedure

1. Reproduce one clean baseline with the original command.
2. Separate model load, TTFT/prefill, and decode measurements.
3. Capture resolved kernel paths, materialized dtypes, dispatch counts, and for decode:
   `decodeRecordMs`, submit wait, readback wait, and orchestration time.
4. Inspect manifest/config routing before proposing a new kernel. This is a diagnostic
   check, not a presumption about the cause.
5. Form one falsifiable hypothesis and vary one declared runtime or code variable.
6. Rerun the identical workload, then revert or retain the candidate according to the
   measured correctness and performance result.
7. Use `doppler-bench` only after tuning when publication-grade evidence is requested.

## Validation

A candidate is supported only by paired baseline/candidate receipts with identical
work and correctness, explicit configuration deltas, and a measured improvement beyond
run variance. The report must identify the localized phase and wall.

## Stop Conditions

Stop when the baseline cannot be reproduced, work or correctness differs, implicit
fallback occurs, or the evidence points to a different owner. Stop before remote or
costly runs without authorization. Do not keep an inconclusive candidate change.

## Outputs

- Baseline and candidate receipt paths.
- Phase/wall diagnosis, tested hypothesis, delta, variance, and disposition.

## Side Effects

Runs profiling and benchmark workloads. Authorized tuning may change runtime profiles,
manifests, orchestration, or kernels; it does not publish claims or promote artifacts.

## References

Read model-specific history only when it matches the named model:
`docs/performance/`, `docs/perf-investigations/`, and
`docs/developer-guides/16-kernel-performance-optimization.md`.
