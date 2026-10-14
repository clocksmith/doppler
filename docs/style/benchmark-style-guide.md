# Doppler Benchmark Style Guide

Claim policy is canonical in [../benchmark-methodology.md](../benchmark-methodology.md).
Vendor registry/toolchain contract is canonical in
[../../benchmarks/vendors/README.md](../../benchmarks/vendors/README.md).

## Scope

Output schema discipline, shared-contract vs engine-overlay separation, and
hygiene for local and vendor benchmark runners.

## Output schema requirements

Benchmark output must validate against `benchmarks/benchmark-schema.json` and
`benchmarks/vendors/schema/*.json`.

## Shared vs engine config

Use two explicit objects: a shared contract (workload, sampling, run counts,
cache/load policy) and an engine overlay (engine-specific execution knobs only).
Never mix fairness axes and engine-specific knobs in one object.

## Required implementation rules

- Include canonical timing fields and units, environment metadata (host,
  browser, gpu, backend), and seed and run-count metadata.
- Keep warm/cold behavior explicit.
- Fail closed when required normalization fields are missing.

## Profiling and stats

- Label warmup vs timed runs; emit p50/p95/p99 where supported; keep
  submit/readback counters explicit when available.
- Resource utilization telemetry is explicit opt-in evidence. Host process-tree
  RAM/CPU and system RAM summaries are comparable only when sampled by the same
  parent harness. GPU utilization/memory from vendor tools such as `rocm-smi` is
  device-global unless the source proves per-process residency.
- Resource telemetry stays out of required claim gates, and derived efficiency
  metrics stay in optional evidence blocks, until benchmark policy/schema
  promotes them in the same change.

## Maintainer checklist

Before publishing a benchmark claim:

1. Validate registry/harness definitions.
2. Run compare or vendor bench with explicit workload id.
3. Save normalized artifact.
4. Ensure claim text references exact artifact path and command.

For inspectable operator-mechanism experiments, use
[`benchmarks/compute/`](../../benchmarks/compute/README.md): workload and
acceptance policy in its suite contract, operation semantics in the workload
adapter, GPU ownership in the plan executor (no operation-specific branches),
and statistics in the shared observation layer. Its acceptance and claim
boundary is
[`Operator-mechanism evidence`](../benchmark-methodology.md#operator-mechanism-evidence).

- Use `computeSampleStats(samples, { outlierPolicy: 'none' })` for latency
  populations. Reject invalid timing samples before summarizing; retain stalls.
- Attach confidence intervals to their actual estimator, not a median paired
  with a mean's standard error.
- Capture source bytes before execution and execute that snapshot. Store raw
  paired samples, inputs, representative outputs, exact plans, and failures.
- Report observed API counters separately from analytically estimated traffic.
- Keep timestamp-instrumented passes separate from primary timing samples.

## Perf-validated kernel-path promotion

A performance-motivated kernel-path change (fusion, dispatch reduction, dtype
change, etc.) follows two stages. Do not edit the conversion config or
regenerate the artifact manifest until the runtime-override stage shows a real
win on the target hardware.

### Stage 1 — candidate graph

1. Build the candidate as an execution-v1 graph or explicit execution graph
   transform; no string kernel-path IDs or side registry.
2. Test it through a runtime config carrying an inline execution-v1-derived
   `runtime.inference.kernelPath` object, or through the conversion config
   before artifact promotion.
3. Correctness gate: a deterministic greedy probe (temperature=0, topK=1) and a
   per-layer numerical compare against the unmodified path. Fail if generated
   text differs on the model's reference prompt, or max abs logit delta exceeds
   the divergence budget recorded in the promotion note (default `1e-3` for f16
   paths).
4. Perf gate: run `bench` with the candidate on the target hardware. It must
   beat the baseline on the same metric (decode tok/s, TTFT, or prefill tok/s)
   by more than the run-to-run noise band.
5. Record both gates in the report directory under the artifact id.

### Stage 2 — promote to conversion config

Promote only after both gates pass, as a single change:

1. Update `src/config/conversion/<family>/<artifact>.json` so the candidate
   kernel path is the default for the affected step(s).
2. Regenerate the artifact manifest via the converter so
   `manifest.inference.execution` carries the new kernel digest(s) inline.
3. Re-run the correctness probe on the regenerated artifact (no runtime
   override); confirm the same generated text and per-layer values as Stage 1.
4. If the artifact is hosted, refresh the hosted copy in the same change so
   manifest-level and hosted-artifact behavior do not drift.

### Do not

- Promote based on synthetic or fixture-only timings (Class `C` evidence per
  the model-failure action plan rules).
- Edit the conversion config first and chase regressions afterwards.
- Reuse a baseline from a different host or driver revision for a
  hardware-targeted change.
- Skip the per-layer numerical compare for kernel fusions that change op
  ordering, accumulator dtype, or write semantics.

## See also

- [../developer-guides/06-kernel-path-config.md](../developer-guides/06-kernel-path-config.md)
