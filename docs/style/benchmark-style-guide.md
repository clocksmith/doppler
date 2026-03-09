# Doppler Benchmark Style Guide

Implementation conventions for benchmark harnesses and normalized outputs.

Claim policy is canonical in [../benchmark-methodology.md](../benchmark-methodology.md).
Vendor registry/toolchain contract is canonical in
[../../benchmarks/vendors/README.md](../../benchmarks/vendors/README.md).

## Scope

Use this guide for:
- benchmark output schema discipline
- shared-contract vs engine-overlay separation
- implementation hygiene for local and vendor benchmark runners

## Output schema requirements

Benchmark output must validate against:
- `benchmarks/benchmark-schema.json`
- `benchmarks/vendors/schema/*.json`

## Shared vs engine config

Use two explicit objects:
- shared contract: workload, sampling, run counts, cache/load policy
- engine overlay: engine-specific execution knobs only

Do not mix fairness axes and engine-specific knobs in a single object.

## Required implementation rules

- Include canonical timing fields and units.
- Include environment metadata (host, browser, gpu, backend).
- Keep warm/cold behavior explicit.
- Include seed and run-count metadata.
- Fail closed when required normalization fields are missing.

## Profiling and stats

- Label warmup vs timed runs explicitly.
- Emit p50/p95/p99 where supported.
- Keep submit/readback counters explicit when available.

## Maintainer checklist

1. Validate registry/harness definitions.
2. Run compare or vendor bench with explicit workload id.
3. Save normalized artifact.
4. Ensure claim text references exact artifact path and command.

## See also

- [../benchmark-methodology.md](../benchmark-methodology.md)
- [../../benchmarks/vendors/README.md](../../benchmarks/vendors/README.md)
