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

## Perf-validated kernel-path promotion

When proposing a kernel-path change motivated by performance (kernel fusion,
dispatch reduction, dtype change, etc.), follow this two-stage flow. Do not
edit the conversion config or regenerate the artifact manifest until the
runtime-override stage shows a real win on the target hardware.

### Stage 1 — runtime override

1. Add or modify an execution-v1 graph in the relevant conversion config, or
   add a capability transform that rewrites explicit execution-v1 steps.
   Follow [../developer-guides/06-kernel-path-config.md](../developer-guides/06-kernel-path-config.md).
2. For an existing artifact, force only an inline execution-v1-derived
   `runtime.inference.kernelPath` object. Do not use string IDs:

   ```json
   { "runtime": { "inference": { "kernelPath": { "id": "inline-experiment", "...": "..." } } } }
   ```

3. Correctness gate: run a deterministic greedy probe (temperature=0, topK=1)
   and a per-layer numerical compare against the unmodified path. Fail the
   gate if either:
   - generated text differs on the model's reference prompt, or
   - max abs logit delta exceeds the divergence budget recorded in the
     promotion note (default `1e-3` for f16 paths).
4. Perf gate: run `bench` with the override on the target hardware. The
   change must beat the baseline on the same metric (decode tok/s, TTFT, or
   prefill tok/s) by more than the run-to-run noise band.
5. Record both gates in the report directory under the artifact id, so the
   promotion decision is reproducible.

### Stage 2 — promote to conversion config

Promote only after both gates pass. Treat the promotion as a single change:

1. Update the artifact's conversion config under
   `src/config/conversion/<family>/<artifact>.json` so the new kernel-path
   ID is the default for the affected step(s).
2. Regenerate the artifact manifest via the converter so
   `manifest.inference.execution` carries the new kernel digest(s) inline.
3. Re-run the correctness probe against the regenerated artifact (no runtime
   override) and confirm the same generated text and per-layer values as the
   Stage 1 override run.
4. If the artifact is hosted, refresh the hosted copy in the same change so
   `manifest`-level and hosted-artifact behavior do not drift.

### Do not

- Promote based on synthetic or fixture-only timings (Class `C` evidence per
  the model-failure action plan rules).
- Edit the conversion config first and chase regressions afterwards.
- Reuse a baseline measurement from a different host or driver revision when
  promoting a hardware-targeted change.
- Skip the per-layer numerical compare for kernel fusions that change op
  ordering, accumulator dtype, or write semantics.

## See also

- [../benchmark-methodology.md](../benchmark-methodology.md)
- [../../benchmarks/vendors/README.md](../../benchmarks/vendors/README.md)
- [../developer-guides/06-kernel-path-config.md](../developer-guides/06-kernel-path-config.md)
