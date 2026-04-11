# Public Export Inventory

Auto-generated from `package.json` exports and shipped `.d.ts` entrypoints.
This is a reference inventory, not the behavior guide. Manual API guides live one level up in `docs/api/`.

## `doppler-gpu`

- Audience: app authors
- Stability: preferred public
- Manual guide: [docs/api/root.md](../root.md)
- Types: [src/index.d.ts](../../../src/index.d.ts)
- Implementation: [src/index.js](../../../src/index.js)
- Notes: Primary application-facing surface. Prefer this over lower-level exports.
- Exported symbols:
  - `doppler`
  - `DOPPLER_VERSION`

## `doppler-gpu/tooling`

- Audience: tool builders
- Stability: public advanced
- Manual guide: [docs/api/tooling.md](../tooling.md)
- Types: [src/tooling-exports.d.ts](../../../src/tooling-exports.d.ts)
- Implementation: [src/tooling-exports.js](../../../src/tooling-exports.js)
- Notes: Tooling and command-runner surface, not the main app-facing API.
- Exported symbols:
  - `*`
  - `hasNodeWebGPUSupport`
  - `NodeBrowserCommandRunOptions`
  - `NodeCommandRunOptions`
  - `NodeCommandRunResult`
  - `normalizeNodeBrowserCommand`
  - `normalizeNodeCommand`
  - `runBrowserCommandInNode`
  - `runNodeCommand`

## `doppler-gpu/loaders`

- Audience: advanced loader consumers
- Stability: public advanced
- Manual guide: [docs/api/loaders.md](../loaders.md)
- Types: [src/loaders/index.d.ts](../../../src/loaders/index.d.ts)
- Implementation: [src/loaders/index.js](../../../src/loaders/index.js)
- Notes: Explicit loader and manifest/bootstrap helpers.
- Exported symbols:
  - `*`

## `doppler-gpu/orchestration`

- Audience: advanced runtime consumers
- Stability: public advanced
- Manual guide: [docs/api/orchestration.md](../orchestration.md)
- Types: [src/orchestration/index.d.ts](../../../src/orchestration/index.d.ts)
- Implementation: [src/orchestration/index.js](../../../src/orchestration/index.js)
- Notes: Tokenizer, KV cache, router, adapter, and logit-merge orchestration helpers.
- Exported symbols:
  - `*`
  - `buildConservativeMultimodalGenerationOptions`

## `doppler-gpu/generation`

- Audience: advanced runtime consumers
- Stability: public advanced
- Manual guide: [docs/api/generation.md](../generation.md)
- Types: [src/generation/index.d.ts](../../../src/generation/index.d.ts)
- Implementation: [src/generation/index.js](../../../src/generation/index.js)
- Notes: Lower-level text pipeline construction and pipeline types.
- Exported symbols:
  - `AdvanceEmbeddingResult`
  - `BatchingStats`
  - `createPipeline`
  - `EmbeddingPipeline`
  - `ExpertWeights`
  - `GenerateOptions`
  - `GenerationResult`
  - `InferencePipeline`
  - `KVCacheSnapshot`
  - `LayerWeights`
  - `LogitsStepResult`
  - `LoRAAdapter`
  - `LoRAModuleName`
  - `ParsedModelConfig`
  - `PipelineContexts`
  - `PipelineStats`
  - `PrefillEmbeddingResult`
  - `PrefillResult`
  - `PromptInput`
  - `RouterWeights`
  - `SamplingOptions`

## `doppler-gpu/diffusion`

- Audience: advanced diffusion consumers
- Stability: public advanced
- Manual guide: [docs/api/diffusion.md](../diffusion.md)
- Types: [src/diffusion/index.d.ts](../../../src/diffusion/index.d.ts)
- Implementation: [src/diffusion/index.js](../../../src/diffusion/index.js)
- Notes: Diffusion/image pipeline surface.
- Exported symbols:
  - `assertImageRegressionWithinTolerance`
  - `computeImageFingerprint`
  - `computeImageRegressionMetrics`
  - `createDiffusionPipeline`
  - `createDiffusionWeightLoader`
  - `DiffusionPipeline`
  - `initializeDiffusion`
  - `mergeDiffusionConfig`

## `doppler-gpu/energy`

- Audience: advanced energy consumers
- Stability: public advanced
- Manual guide: [docs/api/energy.md](../energy.md)
- Types: [src/energy/index.d.ts](../../../src/energy/index.d.ts)
- Implementation: [src/energy/index.js](../../../src/energy/index.js)
- Notes: Energy pipeline surface.
- Exported symbols:
  - `computeQuintelEnergy`
  - `createEnergyPipeline`
  - `EnergyPipeline`
  - `mergeQuintelConfig`
  - `runQuintelEnergyLoop`
