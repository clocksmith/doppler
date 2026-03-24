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
  - `*`
  - `ADAPTER_MANIFEST_SCHEMA`
  - `AdapterManager`
  - `AdapterManagerEvents`
  - `AdapterManifest`
  - `AdapterMetadata`
  - `AdapterQueryOptions`
  - `AdapterRegistry`
  - `AdapterRegistryEntry`
  - `AdapterSource`
  - `AdapterStackOptions`
  - `AdapterState`
  - `AdapterTensorSpec`
  - `computeLoRAScale`
  - `createAdapterManifest`
  - `createDopplerLoader`
  - `createDreamEnergyHeadPipeline`
  - `createDreamStructuredPipeline`
  - `createEnergyRowHeadPipeline`
  - `createMemoryRegistry`
  - `createPipeline`
  - `createStructuredJsonHeadPipeline`
  - `doppler`
  - `DOPPLER_VERSION`
  - `DopplerLoader`
  - `DreamEnergyHeadPipeline`
  - `DreamStructuredPipeline`
  - `EmbeddingPipeline`
  - `EnableAdapterOptions`
  - `EnergyRowHeadPipeline`
  - `ExpertNode`
  - `ExpertRouter`
  - `ExpertTask`
  - `ExpertWeights`
  - `GenerateOptions`
  - `GenerationResult`
  - `getAdapterManager`
  - `getAdapterRegistry`
  - `getDopplerLoader`
  - `getLogitMergeKernel`
  - `InferencePipeline`
  - `isDreamStructuredModelType`
  - `isStructuredJsonHeadModelType`
  - `KVCache`
  - `KVCacheSnapshot`
  - `LayerWeights`
  - `LoaderStats`
  - `loadLoRAFromManifest`
  - `loadLoRAFromSafetensors`
  - `loadLoRAFromUrl`
  - `loadLoRAWeights`
  - `LoadOptions`
  - `LoadProgress`
  - `LogitMergeKernel`
  - `LoRAAdapter`
  - `LoRALoadOptions`
  - `LoRAModuleName`
  - `LoRAWeightsResult`
  - `mergeLogits`
  - `mergeMultipleLogits`
  - `MoERouter`
  - `MultiModelLoader`
  - `MultiModelNetwork`
  - `MultiPipelinePool`
  - `parseAdapterManifest`
  - `ParsedModelConfig`
  - `RDRRManifest`
  - `resetAdapterManager`
  - `resetAdapterRegistry`
  - `RouterWeights`
  - `SamplingOptions`
  - `serializeAdapterManifest`
  - `ShardInfo`
  - `SpeculativeDecoder`
  - `StructuredJsonHeadPipeline`
  - `TensorLocation`
  - `Tokenizer`
  - `validateAdapterManifest`

## `doppler-gpu/provider`

- Audience: advanced/demo integrations
- Stability: public advanced
- Manual guide: [docs/api/provider.md](../provider.md)
- Types: [src/client/doppler-provider.d.ts](../../../src/client/doppler-provider.d.ts)
- Implementation: [src/client/doppler-provider.js](../../../src/client/doppler-provider.js)
- Notes: Legacy/demo-oriented singleton provider surface.
- Exported symbols:
  - `activateLoRAFromTrainingOutput`
  - `buildChatPrompt`
  - `ChatMessage`
  - `ChatResponse`
  - `default`
  - `destroyDoppler`
  - `DOPPLER_PROVIDER_VERSION`
  - `DopplerCapabilities`
  - `DopplerCapabilitiesType`
  - `dopplerChat`
  - `DopplerProvider`
  - `DopplerProviderInterface`
  - `formatChatMessages`
  - `formatGemmaChat`
  - `formatGptOssChat`
  - `formatLlama3Chat`
  - `generate`
  - `GenerateOptions`
  - `generateWithPrefixKV`
  - `getActiveLoRA`
  - `getAvailableModels`
  - `getCurrentModelId`
  - `getPipeline`
  - `InferredAttentionParams`
  - `initDoppler`
  - `loadLoRAAdapter`
  - `loadModel`
  - `LoadProgressEvent`
  - `ModelEstimate`
  - `prefillKV`
  - `TextModelConfig`
  - `unloadLoRAAdapter`
  - `unloadModel`

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

## `doppler-gpu/internal`

- Audience: internal/advanced
- Stability: exposed internal
- Types: [src/index-internal.d.ts](../../../src/index-internal.d.ts)
- Implementation: [src/index-internal.js](../../../src/index-internal.js)
- Notes: Reachable compatibility surface, not a primary public-doc target.
- Exported symbols:
  - `*`

## `doppler-gpu/generation`

- Audience: advanced runtime consumers
- Stability: public advanced
- Manual guide: [docs/api/generation.md](../generation.md)
- Types: [src/generation/index.d.ts](../../../src/generation/index.d.ts)
- Implementation: [src/generation/index.js](../../../src/generation/index.js)
- Notes: Lower-level text pipeline access.
- Exported symbols:
  - `createDreamStructuredPipeline`
  - `createPipeline`
  - `createStructuredJsonHeadPipeline`
  - `DreamStructuredPipeline`
  - `EmbeddingPipeline`
  - `ExpertWeights`
  - `GenerateOptions`
  - `GenerationResult`
  - `InferencePipeline`
  - `initTokenizer`
  - `initTokenizerFromManifest`
  - `isDreamStructuredModelType`
  - `isStopToken`
  - `isStructuredJsonHeadModelType`
  - `KVCacheSnapshot`
  - `LayerWeights`
  - `loadWeights`
  - `LoRAAdapter`
  - `LoRAModuleName`
  - `ParsedModelConfig`
  - `parseModelConfig`
  - `parseModelConfigFromManifest`
  - `PipelineContexts`
  - `RouterWeights`
  - `SamplingOptions`
  - `StructuredJsonHeadPipeline`

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
