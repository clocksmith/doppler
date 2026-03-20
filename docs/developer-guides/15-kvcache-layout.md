# Add a KV-Cache Layout

## Goal

Add a new KV-cache layout or inference memory-management strategy.

## When To Use This Guide

- Existing contiguous, paged, sliding-window, or tiered paths do not fit the new runtime behavior.
- The change affects allocation, reads, writes, and attention compatibility.

## Blast Radius

- Cache internals + attention integration

## Required Touch Points

- `src/config/schema/kvcache.schema.js` and `.d.ts`
- `src/inference/pipelines/text/init.js`
- `src/inference/kv-cache/*`
- Attention run and record integration when the new layout changes how keys or values are accessed
- Optional kernel-path or runtime-profile updates
- Unit, GPU, and end-to-end decode tests

## Recommended Order

1. Add the layout identifier and any layout-specific fields to the KV-cache schema.
2. Implement the layout class or helper under `src/inference/kv-cache/`.
3. Update `createKVCache()` in `src/inference/pipelines/text/init.js`.
4. Add explicit compatibility or fail-closed checks for attention kernels that cannot use the new layout.
5. Add tests for allocation, read/write behavior, and one end-to-end decode path.

## Verification

- `npm run test:unit`
- `npm run test:gpu`
- `npm run test:gpu:browser`
- Run one end-to-end decode flow with the layout forced through runtime config

## Common Misses

- Forgetting that full-attention models must remain on a contiguous-compatible path.
- Letting the contiguous-to-paged auto-upgrade logic apply when the model or layout should stay explicit.
- Adding the schema value without updating `createKVCache()`.
- Testing only on Node and missing browser-specific divergence.
- Not failing closed when an attention kernel cannot read the new layout correctly.

## Related Guides

- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [13-attention-variant.md](13-attention-variant.md)
- [composite-pipeline-family.md](composite-pipeline-family.md)

## Canonical References

- `src/config/schema/kvcache.schema.js`
- `src/inference/pipelines/text/init.js`
- `src/inference/kv-cache/index.js`
- `src/inference/README.md`
- [../architecture.md](../architecture.md)
