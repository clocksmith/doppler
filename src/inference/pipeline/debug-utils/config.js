/**
 * Debug configuration management for pipeline tracing.
 *
 * Manages debug categories, layer filters, and decode step tracking.
 * Enable via: setDebugCategories({ embed: true, layer: true })
 *
 * Categories:
 * - embed: Embedding layer output
 * - layer: Per-layer entry/exit, hidden state stats
 * - attn: Attention computation details
 * - ffn: FFN computation details
 * - kv: KV cache operations
 * - logits: Logits computation and top-k
 * - sample: Sampling decisions
 * - io: GPU buffer read/writes
 * - perf: Timing and benchmarks
 * - kernel: Kernel step debugging - inspect tensor state after each kernel (SLOW!)
 * - all: Enable everything (use sparingly)
 *
 * @module inference/pipeline/debug-utils/config
 */

// ============================================================================
// Module State
// ============================================================================

/** @type {import('./config.js').DebugConfig} */
const defaultConfig = {
  categories: {},
  layers: [],
  maxDecodeSteps: 5,
  maxAbsThreshold: 10000,
  bufferStats: false,
};

/** @type {import('./config.js').DebugConfig} */
let config = { ...defaultConfig };
let decodeStep = 0;

// ============================================================================
// Configuration API
// ============================================================================

/**
 * Set debug categories. Merges with existing config.
 *
 * @example
 * setDebugCategories({ embed: true, layer: true });
 * setDebugCategories({ all: true }); // Enable everything
 * setDebugCategories({ all: true, io: false }); // All except io
 *
 * @param {Partial<Record<import('./config.js').DebugCategory, boolean>>} categories
 * @param {Partial<Omit<import('./config.js').DebugConfig, 'categories'>>} [options]
 * @returns {void}
 */
export function setDebugCategories(categories, options) {
  config = {
    ...config,
    ...options,
    categories: { ...config.categories, ...categories },
  };
}

/**
 * Reset debug config to defaults (all off).
 * @returns {void}
 */
export function resetDebugConfig() {
  config = { ...defaultConfig, categories: {} };
  decodeStep = 0;
}

/**
 * Apply pipeline debug config (runtime.debug.pipeline) to debug-utils.
 * @param {import('../../../config/schema/index.js').PipelineDebugConfigSchema | null | undefined} [pipeline]
 * @returns {void}
 */
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

  /** @type {Partial<Record<import('./config.js').DebugCategory, boolean>>} */
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

/**
 * Get current debug config (for inspection).
 * @returns {import('./config.js').DebugConfig}
 */
export function getDebugConfig() {
  return { ...config };
}

// ============================================================================
// Decode Step Tracking
// ============================================================================

/**
 * Increment decode step counter.
 * @returns {number}
 */
export function incrementDecodeStep() {
  return ++decodeStep;
}

/**
 * Reset decode step counter (call at start of generation).
 * @returns {void}
 */
export function resetDecodeStep() {
  decodeStep = 0;
}

/**
 * Get current decode step.
 * @returns {number}
 */
export function getDecodeStep() {
  return decodeStep;
}

// ============================================================================
// Layer Filtering
// ============================================================================

/**
 * Check if a layer should be debugged based on debugLayers config.
 * @param {number} layerIdx - The layer index to check
 * @param {number[] | null | undefined} debugLayers - Array of layer indices to debug, null means none, undefined/empty means hardcoded defaults
 * @returns {boolean} true if the layer should be debugged
 */
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

/**
 * Check if a debug category is enabled for the given layer.
 * @internal
 * @param {import('./config.js').DebugCategory} category
 * @param {number} [layerIdx]
 * @returns {boolean}
 */
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

/**
 * Format a debug tag with category, layer, and step info.
 * @internal
 * @param {string} category
 * @param {number} [layerIdx]
 * @param {number} [step]
 * @returns {string}
 */
export function formatTag(category, layerIdx, step) {
  let tag = `[${category.toUpperCase()}]`;
  if (layerIdx !== undefined) tag += `[L${layerIdx}]`;
  if (step !== undefined) tag += `[S${step}]`;
  return tag;
}

/**
 * Check if buffer stats collection is enabled.
 * @internal
 * @returns {boolean}
 */
export function isBufferStatsEnabled() {
  return config.bufferStats ?? false;
}

/**
 * Get max abs threshold for explosion warnings.
 * @internal
 * @returns {number}
 */
export function getMaxAbsThreshold() {
  return config.maxAbsThreshold ?? 10000;
}
