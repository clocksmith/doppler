

// ============================================================================
// Module State
// ============================================================================


const defaultConfig = {
  categories: {},
  layers: [],
  maxDecodeSteps: 5,
  maxAbsThreshold: 10000,
  bufferStats: false,
};


let config = { ...defaultConfig };
let decodeStep = 0;

// ============================================================================
// Configuration API
// ============================================================================


export function setDebugCategories(categories, options) {
  config = {
    ...config,
    ...options,
    categories: { ...config.categories, ...categories },
  };
}


export function resetDebugConfig() {
  config = { ...defaultConfig, categories: {} };
  decodeStep = 0;
}


export function applyPipelineDebugConfig(pipeline) {
  if (!pipeline) return;

  const shouldEnable = pipeline.enabled || (pipeline.categories && pipeline.categories.length > 0);
  if (!shouldEnable) {
    resetDebugConfig();
    return;
  }

  const categories = pipeline.categories && pipeline.categories.length > 0
    ? pipeline.categories
    : ['all'];

  
  const categoryMap = {};
  if (categories.includes('all')) {
    categoryMap.all = true;
  } else {
    for (const cat of categories) {
      categoryMap[cat] = true;
    }
  }

  setDebugCategories(categoryMap, {
    layers: pipeline.layers ?? undefined,
    maxDecodeSteps: pipeline.maxDecodeSteps ?? undefined,
    maxAbsThreshold: pipeline.maxAbsThreshold ?? undefined,
    bufferStats: pipeline.bufferStats ?? undefined,
  });
}


export function getDebugConfig() {
  return { ...config };
}

// ============================================================================
// Decode Step Tracking
// ============================================================================


export function incrementDecodeStep() {
  return ++decodeStep;
}


export function resetDecodeStep() {
  decodeStep = 0;
}


export function getDecodeStep() {
  return decodeStep;
}

// ============================================================================
// Layer Filtering
// ============================================================================


export function shouldDebugLayerOutput(layerIdx, debugLayers) {
  if (debugLayers === null) return false;
  if (debugLayers === undefined || debugLayers.length === 0) {
    // Backward compat: default to layers 0, 2, 17 (where Q4K issues were found)
    return layerIdx === 0 || layerIdx === 2 || layerIdx === 17;
  }
  return debugLayers.includes(layerIdx);
}

// ============================================================================
// Internal Helpers (exported for use by other debug-utils modules)
// ============================================================================


export function isEnabled(category, layerIdx) {
  // Check if category is enabled
  if (!config.categories.all && !config.categories[category]) {
    return false;
  }

  // Check layer filter
  if (layerIdx !== undefined && config.layers?.length) {
    if (!config.layers.includes(layerIdx)) {
      return false;
    }
  }

  // Check decode step limit
  if (config.maxDecodeSteps && decodeStep > config.maxDecodeSteps) {
    // Only apply to non-prefill logs
    if (decodeStep > 0) {
      return false;
    }
  }

  return true;
}


export function formatTag(category, layerIdx, step) {
  let tag = `[${category.toUpperCase()}]`;
  if (layerIdx !== undefined) tag += `[L${layerIdx}]`;
  if (step !== undefined) tag += `[S${step}]`;
  return tag;
}


export function isBufferStatsEnabled() {
  return config.bufferStats ?? false;
}


export function getMaxAbsThreshold() {
  return config.maxAbsThreshold ?? 10000;
}
