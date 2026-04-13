# Public Export Inventory

Auto-generated from `package.json` exports and shipped `.d.ts` entrypoints.
This is a reference inventory, not the behavior guide. Manual API guides live one level up in `docs/api/`.

## `doppler-gpu`

- Audience: app authors
- Stability: preferred public (tier1)
- Manual guide: [docs/api/root.md](../root.md)
- Types: [src/index.d.ts](../../../src/index.d.ts)
- Implementation: [src/index.js](../../../src/index.js)
- Notes: Primary application-facing surface. Prefer this over lower-level exports.
- Support tier source: `src/config/support-tiers/subsystems.json` (api.root-facade)
- Exported symbols:
  - `doppler`
  - `DOPPLER_VERSION`

## `doppler-gpu/tooling`

- Audience: tool builders
- Stability: experimental export
- Manual guide: [docs/api/tooling.md](../tooling.md)
- Types: [src/tooling-exports.d.ts](../../../src/tooling-exports.d.ts)
- Implementation: [src/tooling-exports.js](../../../src/tooling-exports.js)
- Notes: Tooling and command-runner surface, not the main app-facing API.
- Support tier source: `src/config/support-tiers/subsystems.json` (api.tooling-operator-commands)
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

## `doppler-gpu/tooling-experimental`

- Audience: unspecified
- Stability: experimental export
- Types: [src/tooling-experimental-exports.d.ts](../../../src/tooling-experimental-exports.d.ts)
- Implementation: [src/tooling-experimental-exports.js](../../../src/tooling-experimental-exports.js)
- Notes: No manual classification recorded for this export path.
- Support tier source: `src/config/support-tiers/subsystems.json` (api.tooling-p2p-helpers)
- Exported symbols:
  - `*`

## `doppler-gpu/loaders`

- Audience: advanced loader consumers
- Stability: tier1 advanced
- Manual guide: [docs/api/loaders.md](../loaders.md)
- Types: [src/loaders/index.d.ts](../../../src/loaders/index.d.ts)
- Implementation: [src/loaders/index.js](../../../src/loaders/index.js)
- Notes: Explicit loader and manifest/bootstrap helpers.
- Support tier source: `src/config/support-tiers/subsystems.json` (api.loaders-subpath)
- Exported symbols:
  - `*`

## `doppler-gpu/orchestration`

- Audience: advanced runtime consumers
- Stability: experimental export
- Manual guide: [docs/api/orchestration.md](../orchestration.md)
- Types: [src/experimental/orchestration/index.d.ts](../../../src/experimental/orchestration/index.d.ts)
- Implementation: [src/experimental/orchestration/index.js](../../../src/experimental/orchestration/index.js)
- Notes: Tokenizer, KV cache, router, adapter, and logit-merge orchestration helpers.
- Support tier source: `src/config/support-tiers/subsystems.json` (api.orchestration-subpath)
- Exported symbols:
  - `*`
  - `buildConservativeMultimodalGenerationOptions`

## `doppler-gpu/generation`

- Audience: advanced runtime consumers
- Stability: tier1 advanced
- Manual guide: [docs/api/generation.md](../generation.md)
- Types: [src/generation/index.d.ts](../../../src/generation/index.d.ts)
- Implementation: [src/generation/index.js](../../../src/generation/index.js)
- Notes: Lower-level text pipeline construction and pipeline types.
- Support tier source: `src/config/support-tiers/subsystems.json` (api.generation-subpath)
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
- Stability: experimental export
- Manual guide: [docs/api/diffusion.md](../diffusion.md)
- Types: [src/experimental/diffusion/index.d.ts](../../../src/experimental/diffusion/index.d.ts)
- Implementation: [src/experimental/diffusion/index.js](../../../src/experimental/diffusion/index.js)
- Notes: Diffusion/image pipeline surface.
- Support tier source: `src/config/support-tiers/subsystems.json` (api.diffusion-subpath)
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
- Stability: experimental export
- Manual guide: [docs/api/energy.md](../energy.md)
- Types: [src/experimental/energy/index.d.ts](../../../src/experimental/energy/index.d.ts)
- Implementation: [src/experimental/energy/index.js](../../../src/experimental/energy/index.js)
- Notes: Energy pipeline surface.
- Support tier source: `src/config/support-tiers/subsystems.json` (api.energy-subpath)
- Exported symbols:
  - `computeQuintelEnergy`
  - `createEnergyPipeline`
  - `EnergyPipeline`
  - `mergeQuintelConfig`
  - `runQuintelEnergyLoop`
