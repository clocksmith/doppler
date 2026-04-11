# Advanced Export Map

## Purpose

Map the older broad root surface to the current dedicated advanced subpaths.

## Current Shape

```js
import { doppler } from 'doppler-gpu';
import { DopplerLoader } from 'doppler-gpu/loaders';
import { KVCache } from 'doppler-gpu/orchestration';
import { createPipeline } from 'doppler-gpu/generation';
```

## Audience

Advanced consumers migrating from the older catch-all root barrel or choosing the right dedicated subpath for new code.

## Stability

Public guidance only. The root facade is intentionally minimal now.

## Subpath Split

### `doppler-gpu/loaders`

- `DopplerLoader`
- `getDopplerLoader()`
- `createDopplerLoader()`
- `MultiModelLoader`
- manifest/config bootstrap helpers used during explicit loader setup

### `doppler-gpu/orchestration`

- `KVCache`
- `Tokenizer`
- `SpeculativeDecoder`
- `ExpertRouter`
- `MoERouter`
- `MultiModelNetwork`
- `MultiPipelinePool`
- `StructuredJsonHeadPipeline`
- `createStructuredJsonHeadPipeline(...)`
- `DreamStructuredPipeline`
- `createDreamStructuredPipeline(...)`
- `EnergyRowHeadPipeline`
- `createEnergyRowHeadPipeline(...)`
- `DreamEnergyHeadPipeline`
- `createDreamEnergyHeadPipeline(...)`
- `ADAPTER_MANIFEST_SCHEMA`
- `validateAdapterManifest(...)`
- `parseAdapterManifest(...)`
- `serializeAdapterManifest(...)`
- `createAdapterManifest(...)`
- `computeLoRAScale(...)`
- `loadLoRAWeights(...)`
- `loadLoRAFromManifest(...)`
- `loadLoRAFromUrl(...)`
- `loadLoRAFromSafetensors(...)`
- `AdapterManager`
- `getAdapterManager()`
- `resetAdapterManager()`
- `AdapterRegistry`
- `getAdapterRegistry()`
- `resetAdapterRegistry()`
- `createMemoryRegistry()`

### GPU helpers

- `LogitMergeKernel`
- `getLogitMergeKernel(...)`
- `mergeLogits(...)`
- `mergeMultipleLogits(...)`

### `doppler-gpu/generation`

- `createPipeline(...)`
- `InferencePipeline`
- `EmbeddingPipeline`
- core text-pipeline types and generation option types

### `doppler-gpu/tooling`

- runtime config/profile helpers
- storage and manifest tooling
- browser/Node command runners
- diagnostics harness helpers

## Migration Rule

- Application code should import `doppler` from `doppler-gpu`.
- Explicit loader work should import from `doppler-gpu/loaders`.
- KV cache, routers, adapters, and logit-merge helpers should import from `doppler-gpu/orchestration`.
- Direct pipeline construction should import from `doppler-gpu/generation`.
- Tooling and command helpers should import from `doppler-gpu/tooling`.

## Code Pointers

- root export surface: [src/index.js](../../src/index.js)
- root type surface: [src/index.d.ts](../../src/index.d.ts)
- loaders subpath: [src/loaders/index.js](../../src/loaders/index.js)
- orchestration subpath: [src/orchestration/index.js](../../src/orchestration/index.js)
- generation subpath: [docs/api/generation.md](generation.md)

## Related Surfaces

- [Root API](root.md)
- [Loaders API](loaders.md)
- [Orchestration API](orchestration.md)
- [Generation API](generation.md)
- [Generated export inventory](reference/exports.md)
