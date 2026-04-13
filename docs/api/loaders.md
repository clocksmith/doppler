# Loaders API

## Purpose

Explicit loader and manifest/bootstrap helpers for advanced consumers who want more control than `doppler.load(...)`.

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
- manifest/config bootstrap helpers re-exported from loader-facing modules

## Related Surfaces

- [Root API](root.md)
- [Orchestration API](orchestration.md)
- [Generation API](generation.md)
- [Generated export inventory](reference/exports.md)
