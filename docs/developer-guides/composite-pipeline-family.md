# Add a New Pipeline Family or ModelType Ecosystem

## Goal

Add a fundamentally new pipeline family and expose it cleanly through Doppler's runtime and public surfaces.

## When To Use This Guide

- The model does not fit an existing pipeline family.
- You need a new `modelType`, pipeline implementation, API barrel, and possibly a new subpath export.

## Blast Radius

- Full vertical slice

## Required Touch Points

- Manifest and schema fields that identify the new `modelType`
- `src/inference/pipelines/<family>/`
- `src/inference/pipelines/registry.js`
- Public API barrels such as `src/<family>/index.js` and `.d.ts`
- `package.json` exports
- API docs under `docs/api/`
- Command surface work if the family needs new command semantics
- Tests and API-doc synchronization

## Recommended Order

1. Confirm the work cannot be expressed as [composite-model-family.md](composite-model-family.md).
2. Define the manifest and config contract, usually starting with [07-manifest-runtime-field.md](07-manifest-runtime-field.md).
3. Implement the pipeline and register it through `registerPipeline(...)`.
4. Add the public subpath barrel and any required root or subpath exports.
5. Update `package.json` exports and API docs.
6. Add [11-wgsl-kernel.md](11-wgsl-kernel.md), [12-command-surface.md](12-command-surface.md), or [15-kvcache-layout.md](15-kvcache-layout.md) if the new family requires them.
7. Add tests, sync API docs, and verify end-to-end execution.

## Verification

- `npm run api:docs:check`
- `npm run onboarding:check`
- `npm run test:unit`
- Run one end-to-end pipeline load and execution path
- If new commands are involved, verify browser and Node parity or explicit fail-closed behavior

## Common Misses

- Registering the pipeline factory but forgetting the public API barrel or package export.
- Adding a new public export without updating `.d.ts` and API docs.
- Forcing the new pipeline onto the root surface without deciding whether it should live on a dedicated subpath.
- Leaving `modelType` support half-implemented between manifest, registry, and runtime loading.

## Related Guides

- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [12-command-surface.md](12-command-surface.md)
- [15-kvcache-layout.md](15-kvcache-layout.md)

## Canonical References

- `src/inference/pipelines/registry.js`
- `src/inference/pipelines/text.js`
- `package.json`
- [../architecture.md](../architecture.md)
- [../api/index.md](../api/index.md)
