# Add or Change an Execution-V1 Kernel Path

## Goal

Define a new execution plan using existing WGSL kernels.

## When To Use This Guide

- You want a different decode or prefill path without writing new WGSL.
- The change is about execution identity, dtype mix, or kernel sequencing.

## Blast Radius

- JSON execution graph + generated manifest

## Required Touch Points

- Usually one of:
  `src/config/conversion/<family>/<model>.json`
- Tests or fixtures that compile the execution-v1 graph

## Recommended Order

1. Copy the nearest execution-v1 graph from a conversion config.
2. Update `kernels`, `decode`, `prefill`, `preLayer`, `postLayer`, `sampling`,
   session policy, and pinned `kernelRef` entries together.
3. Keep every non-cast step explicit: WGSL file, entry point, digest, source,
   destination, and precision policy.
4. Re-refresh or compile the target artifact so the manifest owns the graph.
5. Run kernel generation checks and one debug run with the graph selected by
   the manifest or an inline execution-v1-derived runtime override.

## Verification

- `npm run onboarding:check`
- `npm run kernels:check`
- `npm run config:single-source:check`
- Run one debug pass with the new path selected by the manifest or an inline
  runtime override

## Common Misses

- Adding a string kernel-path ID or removed kernel-path-registry asset.
- Updating decode but not prefill, or updating the graph without refreshing
  the generated manifest.
- Referencing a kernel filename or entry point that does not exist.
- Making decode and prefill inconsistent when the model needs both paths.
- Letting generated digest mirrors drift from WGSL source.

## Related Guides

- [03-model-family-config.md](03-model-family-config.md)
- [04-conversion-config.md](04-conversion-config.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [13-attention-variant.md](13-attention-variant.md)

## Canonical References

- `src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json`
- `src/config/schema/execution-v1.schema.d.ts`
- [../config.md](../config.md)
- `src/inference/pipelines/text/model-load.js`
