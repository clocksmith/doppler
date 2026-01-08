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

import type { PipelineDebugConfigSchema } from '../../../config/schema/index.js';

// ============================================================================
// Debug Configuration Types
// ============================================================================

export type DebugCategory =
  | 'embed'
  | 'layer'
  | 'attn'
  | 'ffn'
  | 'kv'
  | 'logits'
  | 'sample'
  | 'io'
  | 'perf'
  | 'kernel'  // Log after every kernel operation (expensive!)
  | 'all';

export interface DebugConfig {
  /** Which categories are enabled */
  categories: Partial<Record<DebugCategory, boolean>>;
  /** Only log these layer indices (empty = all) */
  layers?: number[];
  /** Only log first N decode steps (0 = all) */
  maxDecodeSteps?: number;
  /** Warn if maxAbs exceeds this */
  maxAbsThreshold?: number;
  /** Log GPU buffer stats (expensive - requires readback) */
  bufferStats?: boolean;
}

// ============================================================================
// Module State
// ============================================================================

const defaultConfig: DebugConfig = {
  categories: {},
  layers: [],
  maxDecodeSteps: 5,
  maxAbsThreshold: 10000,
  bufferStats: false,
};

let config: DebugConfig = { ...defaultConfig };
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
 */
export function setDebugCategories(
  categories: Partial<Record<DebugCategory, boolean>>,
  options?: Partial<Omit<DebugConfig, 'categories'>>
): void {
  config = {
    ...config,
    ...options,
    categories: { ...config.categories, ...categories },
  };
}

/**
 * Reset debug config to defaults (all off).
 */
export function resetDebugConfig(): void {
  config = { ...defaultConfig, categories: {} };
  decodeStep = 0;
}

/**
 * Apply pipeline debug config (runtime.debug.pipeline) to debug-utils.
 */
export function applyPipelineDebugConfig(pipeline?: PipelineDebugConfigSchema | null): void {
  if (!pipeline) return;

  const shouldEnable = pipeline.enabled || (pipeline.categories && pipeline.categories.length > 0);
  if (!shouldEnable) {
    resetDebugConfig();
    return;
  }

  const categories = pipeline.categories && pipeline.categories.length > 0
    ? pipeline.categories
    : ['all'];

  const categoryMap: Partial<Record<DebugCategory, boolean>> = {};
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
 */
export function getDebugConfig(): DebugConfig {
  return { ...config };
}

// ============================================================================
// Decode Step Tracking
// ============================================================================

/**
 * Increment decode step counter.
 */
export function incrementDecodeStep(): number {
  return ++decodeStep;
}

/**
 * Reset decode step counter (call at start of generation).
 */
export function resetDecodeStep(): void {
  decodeStep = 0;
}

/**
 * Get current decode step.
 */
export function getDecodeStep(): number {
  return decodeStep;
}

// ============================================================================
// Layer Filtering
// ============================================================================

/**
 * Check if a layer should be debugged based on debugLayers config.
 * @param layerIdx - The layer index to check
 * @param debugLayers - Array of layer indices to debug, null means none, undefined/empty means hardcoded defaults
 * @returns true if the layer should be debugged
 */
export function shouldDebugLayerOutput(layerIdx: number, debugLayers: number[] | null | undefined): boolean {
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
 */
export function isEnabled(category: DebugCategory, layerIdx?: number): boolean {
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
 */
export function formatTag(category: string, layerIdx?: number, step?: number): string {
  let tag = `[${category.toUpperCase()}]`;
  if (layerIdx !== undefined) tag += `[L${layerIdx}]`;
  if (step !== undefined) tag += `[S${step}]`;
  return tag;
}

/**
 * Check if buffer stats collection is enabled.
 * @internal
 */
export function isBufferStatsEnabled(): boolean {
  return config.bufferStats ?? false;
}

/**
 * Get max abs threshold for explosion warnings.
 * @internal
 */
export function getMaxAbsThreshold(): number {
  return config.maxAbsThreshold ?? 10000;
}
