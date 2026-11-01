import { getRuntimeConfig } from '../config/runtime.js';
import { getDecodeStep, resetDecodeStep } from './config.js';

// Pipeline observation policy stays distinct from trace categories; both share
// the observation owner and decode-step clock.
/** @returns {import('./pipeline-config.js').PipelineDebugState} */
function readDefaultConfig() {
  const pipeline = getRuntimeConfig().shared.debug.pipeline;
  return {
    categories: {},
    layers: pipeline.layers ?? null,
    maxDecodeSteps: pipeline.maxDecodeSteps,
    maxAbsThreshold: pipeline.maxAbsThreshold,
    bufferStats: pipeline.bufferStats,
  };
}

/** @type {import('./pipeline-config.js').PipelineDebugState | undefined} */
let pipelineDebugConfig;

function getPipelineDebugState() {
  pipelineDebugConfig ??= readDefaultConfig();
  return pipelineDebugConfig;
}

// ============================================================================
// Configuration API
// ============================================================================

/** @type {typeof import('./pipeline-config.js').setDebugCategories} */
export function setDebugCategories(categories, options) {
  const config = getPipelineDebugState();
  pipelineDebugConfig = {
    ...config,
    ...options,
    categories: { ...config.categories, ...categories },
  };
}

/** @type {typeof import('./pipeline-config.js').resetDebugConfig} */
export function resetDebugConfig() {
  pipelineDebugConfig = readDefaultConfig();
  resetDecodeStep();
}

/** @type {typeof import('./pipeline-config.js').applyPipelineDebugConfig} */
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


  /** @type {Record<string, boolean>} */
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

/** @type {typeof import('./pipeline-config.js').getDebugConfig} */
export function getDebugConfig() {
  return { ...getPipelineDebugState() };
}

// ============================================================================
// Layer Filtering
// ============================================================================

/** @type {typeof import('./pipeline-config.js').shouldDebugLayerOutput} */
export function shouldDebugLayerOutput(layerIdx, debugLayers) {
  if (debugLayers === null) return false;
  if (!Array.isArray(debugLayers) || debugLayers.length === 0) return true;
  return debugLayers.includes(layerIdx);
}

// ============================================================================
// Internal Helpers (exported for use by other debug-utils modules)
// ============================================================================

/** @type {typeof import('./pipeline-config.js').isEnabled} */
export function isEnabled(category, layerIdx) {
  const config = getPipelineDebugState();
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
  if (config.maxDecodeSteps && getDecodeStep() > config.maxDecodeSteps) {
    // Only apply to non-prefill logs
    if (getDecodeStep() > 0) {
      return false;
    }
  }

  return true;
}

/** @type {typeof import('./pipeline-config.js').formatTag} */
export function formatTag(category, layerIdx, step) {
  let tag = `[${category.toUpperCase()}]`;
  if (layerIdx !== undefined) tag += `[L${layerIdx}]`;
  if (step !== undefined) tag += `[S${step}]`;
  return tag;
}

/** @type {typeof import('./pipeline-config.js').isBufferStatsEnabled} */
export function isBufferStatsEnabled() {
  return getPipelineDebugState().bufferStats;
}

/** @type {typeof import('./pipeline-config.js').getMaxAbsThreshold} */
export function getMaxAbsThreshold() {
  return getPipelineDebugState().maxAbsThreshold;
}
