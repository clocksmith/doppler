# DOPPLER Benchmark Style Guide

This guide defines benchmark implementation conventions.

Policy and claim methodology are canonicalized in [../benchmark-methodology.md](../benchmark-methodology.md).
Vendor registry/toolchain details are canonicalized in [../../benchmarks/vendors/README.md](../../benchmarks/vendors/README.md).

## Scope

Use this guide for:
- schema-friendly benchmark output writing
- harness implementation expectations
- local benchmarking implementation hygiene

## Output schema

Benchmark output must validate against:
- `benchmarks/benchmark-schema.json`
- vendor result schema(s) under `benchmarks/vendors/schema/`

## Shared vs engine config

Keep a strict split:
- shared contract: workload, sampling, runs, cache/load policy
- engine overlay: engine-specific knobs only

Do not mix fairness axes and engine internals in one object.

## Required implementation rules

- include canonical timing fields and units
- include environment metadata (host, browser, gpu, backend)
- keep warm/cold behavior explicit
- include seed and run-count metadata
- fail closed when required normalization fields are missing

## Profiling and stats

- define whether runs are warmup or timed
- report p50/p95/p99 where supported
- keep submit/readback counters explicit when available

## Checklist

1. Validate registry/harness definitions.
2. Run compare or vendor bench with explicit workload id.
3. Save normalized artifact.
4. Ensure claim text references exact artifact path and command.

## Commands

```bash
node tools/vendor-bench.js validate
node tools/vendor-bench.js capabilities
node tools/compare-engines.js --mode all
```
