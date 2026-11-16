export { loadRangeBackedPleProjectionSliceBytes } from "./plan.js";
export { ensurePleGpuHotVocabularyRuntime, ensurePleGpuSplitTablesRuntime, ensurePleScaledProjectionNormWeight, inferPleProjectionNormDtype, resolvePleHotVocabularyCapacity, scalePerLayerProjectionNormWeights } from "./materialize.js";
export { resolveDensePleProjectionWeight } from "./projection-contract.js";
export { destroyPleRuntimeCache, createPleBufferCache, destroyPleBufferCache, getPleHotVocabularyRuntime, prefetchPerLayerRow, hasRangeBackedPerLayerInputEmbeddings, hasGpuSplitPerLayerInputEmbeddings } from "./runtime-cache.js";
export { preparePerLayerInputs, createPerLayerInputTensor } from "./prepare.js";
