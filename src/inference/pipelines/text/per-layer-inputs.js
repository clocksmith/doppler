export { loadRangeBackedPleProjectionSliceBytes } from './per-layer/plan.js';
export { ensurePleGpuHotVocabularyRuntime, ensurePleGpuSplitTablesRuntime, ensurePleScaledProjectionNormWeight, inferPleProjectionNormDtype, resolvePleHotVocabularyCapacity, scalePerLayerProjectionNormWeights } from './per-layer/materialize.js';
export { resolveDensePleProjectionWeight } from './per-layer/projection-contract.js';
export { destroyPleRuntimeCache, createPleBufferCache, destroyPleBufferCache, getPleHotVocabularyRuntime, prefetchPerLayerRow, hasRangeBackedPerLayerInputEmbeddings, hasGpuSplitPerLayerInputEmbeddings } from './per-layer/runtime-cache.js';
export { preparePerLayerInputs, createPerLayerInputTensor } from './per-layer/prepare.js';
