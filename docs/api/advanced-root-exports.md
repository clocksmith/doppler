# Advanced Root Exports

## Purpose

Reference for advanced symbols exported from `doppler-gpu` in addition to the primary `doppler` facade.

## Import Path

```js
import { createPipeline, DopplerLoader, KVCache } from 'doppler-gpu';
```

## Audience

Advanced consumers who intentionally want root-level access to loaders, pipelines, adapter infrastructure, or low-level helpers.

## Stability

Public, but secondary to the root `doppler` facade. Prefer subpath docs when a dedicated subpath exists.

## Export Groups

### Loaders

- `DopplerLoader`
- `getDopplerLoader()`
- `createDopplerLoader()`
- `MultiModelLoader`

Use these when you want explicit loader control rather than `doppler.load()`.

### Text pipeline and orchestration

- `createPipeline(...)`
- `InferencePipeline`
- `EmbeddingPipeline`
- `KVCache`
- `Tokenizer`
- `SpeculativeDecoder`
- `ExpertRouter`
- `MoERouter`
- `MultiModelNetwork`
- `MultiPipelinePool`

These map more naturally to the [Generation API](generation.md), but are still exported from the root package.

### Structured / energy heads

- `StructuredJsonHeadPipeline`
- `createStructuredJsonHeadPipeline(...)`
- `DreamStructuredPipeline`
- `createDreamStructuredPipeline(...)`
- `EnergyRowHeadPipeline`
- `createEnergyRowHeadPipeline(...)`
- `DreamEnergyHeadPipeline`
- `createDreamEnergyHeadPipeline(...)`

### LoRA adapter infrastructure

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

### Tooling compatibility re-exports

The root package still re-exports the shared tooling surface for backward compatibility.
Tooling-only consumers should prefer [`doppler-gpu/tooling`](tooling.md).

## Code Pointers

- root export surface: [src/index.js](../../src/index.js)
- root type surface: [src/index.d.ts](../../src/index.d.ts)
- generation subpath: [docs/api/generation.md](generation.md)
- tooling subpath: [docs/api/tooling.md](tooling.md)

## Related Surfaces

- [Root API](root.md)
- [Generation API](generation.md)
- [Tooling API](tooling.md)
- [Generated export inventory](reference/exports.md)
