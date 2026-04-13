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
Subpath support tiers are defined in [Subsystem Support Matrix](../subsystem-support-matrix.md).

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

- tier1 runtime config/profile helpers
- tier1 storage and manifest tooling
- tier1 browser/Node command runners for canonical verify/debug/bench flows
- diagnostics harness helpers
- experimental browser conversion/file-picker helpers
- experimental P2P/distribution helpers
- experimental Node operator flows (`diagnose`, `lora`, `distill`)

## Migration Rule

- Application code should import `doppler` from `doppler-gpu`.
- Explicit loader work should import from `doppler-gpu/loaders`.
- KV cache, routers, adapters, and logit-merge helpers should import from `doppler-gpu/orchestration`.
- Direct pipeline construction should import from `doppler-gpu/generation`.
- Tooling and command helpers should import from `doppler-gpu/tooling`.
- Treat `doppler-gpu/tooling` as a mixed-tier export: command/storage/registry helpers are promoted first; browser import, P2P, and operator helpers are still experimental.
- Check the subsystem support matrix before treating every exported subpath as equally promoted.

## Code Pointers

- root export surface: [src/index.js](../../src/index.js)
- root type surface: [src/index.d.ts](../../src/index.d.ts)
- loaders subpath: [src/loaders/index.js](../../src/loaders/index.js)
- orchestration subpath: [src/experimental/orchestration/index.js](../../src/experimental/orchestration/index.js)
- generation subpath: [docs/api/generation.md](generation.md)

## Related Surfaces

- [Root API](root.md)
- [Loaders API](loaders.md)
- [Orchestration API](orchestration.md)
- [Generation API](generation.md)
- [Generated export inventory](reference/exports.md)
