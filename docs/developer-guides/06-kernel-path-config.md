# Compose Kernel Paths From Execution Graphs

## Goal

Define or change execution identity using existing kernels without adding a
parallel kernel-path registry.

## When To Use This Guide

- You want a different decode or prefill path without writing new WGSL.
- The change is about execution identity, dtype mix, or kernel sequencing.
- You are preparing a candidate graph for benchmark or promotion work.

## Blast Radius

- Conversion config JSON
- Execution graph transforms when capability adaptation is required
- Runtime profiles only for session/dtype/policy knobs

## Required Touch Points

- `src/config/conversion/<family>/<model>.json`
- `src/config/transforms/execution-graph-transforms.js` when the change is a
  capability transform instead of a model-owned graph change
- `src/config/runtime/**` only for session, dtype, or kernel-path policy
- Tests under `tests/config` or `tests/integration`

## Recommended Order

1. Start from the model's conversion config and inspect `inference.execution`.
2. Update the graph steps, entries, dtypes, and `kernelRef` pins there, or add an
   explicit graph transform if the change is capability-driven.
3. Keep runtime profiles free of string `runtime.inference.kernelPath` values.
   Runtime `kernelPath` may be `null` or an inline object generated from
   execution-v1; it may not be a registry ID.
4. Keep every non-cast step explicit: WGSL file, entry point, digest, source,
   destination, and precision policy.
5. Run digest and kernel checks before promoting a converted artifact.
6. Verify the candidate with one deterministic debug or bench run.

## Verification

- `npm run onboarding:check`
- `npm run digests:check-conversion`
- `npm run kernels:check`
- `npm run kernels:reachability:check`
- `npm run config:single-source:check`
- Run one debug pass with the new path selected by the manifest or an inline
  runtime override

## Common Misses

- Reintroducing removed kernel-path registry assets or string kernel-path IDs.
- Updating runtime profiles when the model-owned execution graph is the actual
  source of truth.
- Forgetting to update digest pins after changing WGSL file or entry-point
  identity.
- Referencing a kernel filename or entry point that does not exist.
- Making decode and prefill inconsistent when the model needs both phases.
- Hiding capability adaptation in JS instead of execution graph transforms.

## Related Guides

- [03-model-family-config.md](03-model-family-config.md)
- [04-conversion-config.md](04-conversion-config.md)
- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [13-attention-variant.md](13-attention-variant.md)

## Canonical References

- `src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json`
- `src/config/schema/execution-v1.schema.d.ts`
- [../config.md](../config.md)
- [../conversion-runtime-contract.md](../conversion-runtime-contract.md)
- [../style/config-style-guide.md](../style/config-style-guide.md)
- `src/inference/pipelines/text/execution-v1.js`
