# Loaders API

> **Deprecated.** `doppler-gpu/loaders` is deprecated in 0.7.0 and will be removed in
> 0.8.0. Load through `doppler-gpu` or `doppler-gpu/compat`; manifest helpers are in
> `doppler-gpu/tooling/manifest` and OPFS storage in `doppler-gpu/tooling/storage`.
> The explicit loader classes have no replacement. See the [changelog](../../CHANGELOG.md).

## Purpose

Explicit loader and manifest/bootstrap helpers for advanced consumers who want more control than `dr.load(...)`.

## Import Path

```js
import { DopplerLoader, createDopplerLoader } from 'doppler-gpu/loaders';
```

## Audience

Advanced consumers building explicit loading flows, custom manifest/bootstrap logic, or multi-model loader orchestration.

## Stability

Public, but advanced. Prefer the root `doppler` facade unless you need explicit loader control.
Current support tier: `tier1` advanced surface. See [Subsystem Support Matrix](../subsystem-support-matrix.md).

## Primary Exports

- `DopplerLoader`
- `getDopplerLoader()`
- `createDopplerLoader()`
- `MultiModelLoader`
- loader-module helpers from `src/loader/index.js` (text-pipeline config and init
  helpers are no longer re-exported here as of 0.7.0)

## Related Surfaces

- [Root API](root.md)
- [Orchestration API](orchestration.md)
- [Generation API](generation.md)
- [Generated export inventory](reference/exports.md)
