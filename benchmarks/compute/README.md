# Inspectable compute evidence

This repository-only framework measures declared compute mechanisms, not model
quality, general library superiority, or runtime-kernel promotion eligibility.
It is separate from the public `doppler bench` model-command contract.

```sh
npm run bench:compute
node tools/bench-compute.js --config benchmarks/compute/suite.json --out /absolute/new-run-directory
node tools/bench-compute.js --analyze /absolute/run-directory/receipt.json
```

The runner requires the installed Playwright package and the browser channel
declared in the suite. It launches a disposable profile, serves only a frozen
source snapshot on loopback, and makes no model downloads or deployments.
The output directory must be new; prior attempts are never overwritten.

## Questions and boundaries

- **Fusion:** hold the scalar matmul algorithm, inputs, dtype, and submission
  grouping fixed; fuse bias and SiLU into the producer instead of dispatching
  Doppler's existing `bias_add.wgsl` and `silu.wgsl` shaders.
- **Submission:** hold the complete unfused shader graph fixed; compare one
  queue submission with a submission per dispatch, without an intermediate
  CPU wait. This is not a comparison against an artificially serialized GPU.
- **Geometry:** cover decode-like, prefill-like, non-aligned tails, and wide
  saturated outputs. The matmul fixture is deliberately controlled, not a
  best-available production GEMM or evidence for replacing one.

`suite.json` predeclares all sample counts, numerical budgets, workloads, and
acceptance policy. `linear.js` owns input generation, the independent scalar
reference, and plan construction. `executor.js` understands resource ownership,
bindings, and submission mechanics but has no mathematical operation dispatch.
`analysis.js` owns evidence gates and paired statistics using shared observation
primitives; `report.js` renders only the saved receipt.

## What is retained

Each run contains `receipt.json`, a self-contained `report.html`, the executed
`source/` snapshot with SHA-256 identities, and `data/` float32 binaries. Inputs,
the independent oracle, and one complete output per lane are saved. Every warmup
and timed execution is numerically checked and must match its own lane's saved
output hash. Cross-lane equality uses the numerical budget, not byte identity.
Mismatches retain the actual output and fail the run. The output's guard tail is
part of the saved binary; NaN poisoning checks that every result was written.

The statistical population contains every timed pair, alternating A/B and B/A.
There is no retry-until-pass, outlier removal, or favorable early stopping. The
paired mean-difference bootstrap uses a predeclared family-wise correction across
all shape/experiment rows. Empirical p95/p99 values are not tail-SLA guarantees.
A median improvement with a worse p95 is reported as mixed, not an uncomplicated
win. Statistical intervals depend on the usual paired-sample assumptions; raw
order remains inspectable for thermal drift and autocorrelation.

Primary wall timing covers encoding through queue completion, with observed
dispatch/submission counters. Preparation, input upload, output poisoning,
readback, oracle computation, hashing, and disk writes have separate scopes.
Summed GPU-pass timestamps come from separate diagnostic executions and are not
substituted for primary samples. Zero-duration GPU queries remain visible.
The browser's timestamp resolution and internal driver caches are not controlled.
Logical intermediate traffic is an analytical byte count, not hardware memory
bandwidth or measured cache traffic.

Wrong output, unwritten output, non-finite values, truncation, and guard corruption
exercise the numerical checker. Negative receipt controls reject skipped
dispatches, false correctness, missing pairs, zero time, changed output identity,
profiling contamination, and unbalanced order. Workgroup override probes compile
and execute one shared shader at multiple sizes rather than merely assuming a
language capability.

`--analyze` verifies saved source/artifact hashes, recomputes the CPU oracle,
checks the saved GPU outputs, and regenerates summaries from raw paired samples.
It does not rerun the GPU or establish authenticity against a maliciously edited
receipt. Reproduction binds source and input bytes, not identical timing outcomes.

## Extending the framework

Add a workload adapter with an independent oracle and explicit plans. Reuse the
executor; do not add operation-specific resource or arithmetic branches there.
Add an experiment only with a named treatment, frozen non-treatment axes, and
negative controls for its evidence boundary. Broader workloads must extend the
strict suite validator rather than bypass it.

Promotion into actual model execution requires the existing
[benchmark methodology](../../docs/benchmark-methodology.md) and
[kernel optimization gates](../../docs/developer-guides/16-kernel-performance-optimization.md).
Nothing in this suite updates production kernel selection or public claims.
