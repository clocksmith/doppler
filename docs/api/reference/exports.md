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
  - `createDopplerProvider`
  - `doppler`
  - `DOPPLER_VERSION`

## `doppler-gpu/provider`

- Audience: unspecified
- Stability: unspecified
- Types: [src/client/provider.d.ts](../../../src/client/provider.d.ts)
- Implementation: [src/client/provider.js](../../../src/client/provider.js)
- Notes: No manual classification recorded for this export path.
- Exported symbols:
  - `DopplerProvider`
  - `FailureClass`
  - `FaultInjectionConfig`
  - `InferenceSource`
  - `ModelHandle`
  - `PolicyMode`
  - `ProviderConfig`
  - `ProviderDiagnosticsConfig`
  - `ProviderFallbackConfig`
  - `ProviderGenerateOptions`
  - `ProviderLocalConfig`
  - `ProviderPolicyConfig`
  - `ProviderReceiptV1`
  - `ProviderResult`
  - `ReceiptDevice`
  - `ReceiptFailure`
  - `ReceiptFallbackDecision`
  - `ReceiptModel`
  - `wrapPipelineAsHandle`

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
  - `checkProgramBundleFile`
  - `checkProgramBundleParity`
  - `exportProgramBundle`
  - `hasNodeWebGPUSupport`
  - `loadProgramBundle`
  - `NodeBrowserCommandRunOptions`
  - `NodeCommandRunOptions`
  - `NodeCommandRunResult`
  - `normalizeNodeBrowserCommand`
  - `normalizeNodeCommand`
  - `PROGRAM_BUNDLE_PARITY_SCHEMA_ID`
  - `ProgramBundleCheckResult`
  - `ProgramBundleExportOptions`
  - `ProgramBundleParityOptions`
  - `ProgramBundleParityResult`
  - `ProgramBundleWriteResult`
  - `runBrowserCommandInNode`
  - `runNodeCommand`
  - `writeProgramBundle`

## `doppler-gpu/tooling/storage`

- Audience: unspecified
- Stability: unspecified
- Types: [src/tooling-exports/storage.d.ts](../../../src/tooling-exports/storage.d.ts)
- Implementation: [src/tooling-exports/storage.js](../../../src/tooling-exports/storage.js)
- Notes: No manual classification recorded for this export path.
- Exported symbols:
  - `computeHash`
  - `deleteModel`
  - `deleteStorageEntry`
  - `ensureModelCached`
  - `exportModelToDirectory`
  - `formatBytes`
  - `getQuotaInfo`
  - `listFilesInStore`
  - `listModels`
  - `listRegisteredModels`
  - `listStorageInventory`
  - `loadFileFromStore`
  - `loadManifestFromStore`
  - `loadShard`
  - `loadTensorsFromStore`
  - `loadTokenizerFromStore`
  - `loadTokenizerModelFromStore`
  - `openModelStore`
  - `registerModel`
  - `removeRegisteredModel`
  - `saveAuxFile`
  - `saveManifest`
  - `saveTensorsToStore`
  - `saveTokenizer`
  - `saveTokenizerModel`
  - `streamFileFromStore`
  - `writeShard`

## `doppler-gpu/tooling/device`

- Audience: unspecified
- Stability: unspecified
- Types: [src/tooling-exports/device.d.ts](../../../src/tooling-exports/device.d.ts)
- Implementation: [src/tooling-exports/device.js](../../../src/tooling-exports/device.js)
- Notes: No manual classification recorded for this export path.
- Exported symbols:
  - `getDevice`
  - `getKernelCapabilities`
  - `getPlatformConfig`
  - `hasPreseededShaderSource`
  - `initDevice`
  - `isWebGPUAvailable`
  - `registerShaderSources`

## `doppler-gpu/tooling/manifest`

- Audience: unspecified
- Stability: unspecified
- Types: [src/tooling-exports/manifest.d.ts](../../../src/tooling-exports/manifest.d.ts)
- Implementation: [src/tooling-exports/manifest.js](../../../src/tooling-exports/manifest.js)
- Notes: No manual classification recorded for this export path.
- Exported symbols:
  - `classifyTensorRole`
  - `clearManifest`
  - `DEFAULT_MANIFEST_INFERENCE`
  - `getManifest`
  - `parseManifest`
  - `setManifest`

## `doppler-gpu/structured`

- Audience: unspecified
- Stability: unspecified
- Types: [src/tooling-exports/structured.d.ts](../../../src/tooling-exports/structured.d.ts)
- Implementation: [src/tooling-exports/structured.js](../../../src/tooling-exports/structured.js)
- Notes: No manual classification recorded for this export path.
- Exported symbols:
  - `*`

## `doppler-gpu/client/model-manager`

- Audience: unspecified
- Stability: unspecified
- Types: [src/client/runtime/model-manager.d.ts](../../../src/client/runtime/model-manager.d.ts)
- Implementation: [src/client/runtime/model-manager.js](../../../src/client/runtime/model-manager.js)
- Notes: No manual classification recorded for this export path.
- Exported symbols:
  - `activateLoRAFromTrainingOutput`
  - `extractTextModelConfig`
  - `fetchArrayBuffer`
  - `getActiveLoRA`
  - `getCurrentModelId`
  - `getPipeline`
  - `loadModel`
  - `readOPFSFile`
  - `shouldAutoTuneKernels`
  - `verifyExplicitModelUrlMatch`
  - `writeOPFSFile`

## `doppler-gpu/models/qwen3`

- Audience: unspecified
- Stability: unspecified
- Types: [src/models/qwen3.d.ts](../../../src/models/qwen3.d.ts)
- Implementation: [src/models/qwen3.js](../../../src/models/qwen3.js)
- Notes: No manual classification recorded for this export path.
- Exported symbols:
  - `FAMILY_ID`
  - `HF_REPO_ID`
  - `KNOWN_MODELS`
  - `resolveHfBaseUrl`
  - `resolveModel`

## `doppler-gpu/models/gemma3`

- Audience: unspecified
- Stability: unspecified
- Types: [src/models/gemma3.d.ts](../../../src/models/gemma3.d.ts)
- Implementation: [src/models/gemma3.js](../../../src/models/gemma3.js)
- Notes: No manual classification recorded for this export path.
- Exported symbols:
  - `FAMILY_ID`
  - `HF_REPO_ID`
  - `KNOWN_MODELS`
  - `resolveHfBaseUrl`
  - `resolveModel`

## `doppler-gpu/models/gemma4`

- Audience: unspecified
- Stability: unspecified
- Types: [src/models/gemma4.d.ts](../../../src/models/gemma4.d.ts)
- Implementation: [src/models/gemma4.js](../../../src/models/gemma4.js)
- Notes: No manual classification recorded for this export path.
- Exported symbols:
  - `FAMILY_ID`
  - `HF_REPO_ID`
  - `KNOWN_MODELS`
  - `resolveHfBaseUrl`
  - `resolveModel`

## `doppler-gpu/models/embeddinggemma`

- Audience: unspecified
- Stability: unspecified
- Types: [src/models/embeddinggemma.d.ts](../../../src/models/embeddinggemma.d.ts)
- Implementation: [src/models/embeddinggemma.js](../../../src/models/embeddinggemma.js)
- Notes: No manual classification recorded for this export path.
- Exported symbols:
  - `FAMILY_ID`
  - `HF_REPO_ID`
  - `KNOWN_MODELS`
  - `resolveHfBaseUrl`
  - `resolveModel`

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
