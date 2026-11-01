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
 * @module debug/pipeline-config
 */

import type { PipelineDebugConfigSchema } from '../config/schema/index.js';

// ============================================================================
// Debug Configuration Types
// ============================================================================

export type PipelineDebugCategory =
  | 'embed'
  | 'layer'
  | 'attn'
  | 'ffn'
  | 'kv'
  | 'logits'
  | 'sample'
  | 'io'
  | 'perf'
  | 'kernel'
  | 'all';

export interface PipelineDebugState {
  /** Which categories are enabled */
  categories: Partial<Record<PipelineDebugCategory, boolean>>;
  /** Only log these layer indices (empty = all) */
  layers?: number[] | null;
  /** Only log first N decode steps (0 = all) */
  maxDecodeSteps?: number;
  /** Warn if maxAbs exceeds this */
  maxAbsThreshold?: number;
  /** Log GPU buffer stats (expensive - requires readback) */
  bufferStats?: boolean;
}

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
  categories: Partial<Record<PipelineDebugCategory, boolean>>,
  options?: Partial<Omit<PipelineDebugState, 'categories'>>
): void;

/**
 * Reset debug config to defaults (all off).
 */
export function resetDebugConfig(): void;

/**
 * Apply pipeline debug config (runtime.shared.debug.pipeline) to debug-utils.
 */
export function applyPipelineDebugConfig(pipeline?: PipelineDebugConfigSchema | null): void;

/**
 * Get current debug config (for inspection).
 */
export function getDebugConfig(): PipelineDebugState;

// ============================================================================
// Layer Filtering
// ============================================================================

/**
 * Check if a layer should be debugged based on debugLayers config.
 * @param layerIdx - The layer index to check
 * @param debugLayers - Array of layer indices to debug, null means none, undefined/empty means hardcoded defaults
 * @returns true if the layer should be debugged
 */
export function shouldDebugLayerOutput(layerIdx: number, debugLayers: number[] | null | undefined): boolean;

// ============================================================================
// Internal Helpers (exported for use by other debug-utils modules)
// ============================================================================

/**
 * Check if a debug category is enabled for the given layer.
 * @internal
 */
export function isEnabled(category: PipelineDebugCategory, layerIdx?: number): boolean;

/**
 * Format a debug tag with category, layer, and step info.
 * @internal
 */
export function formatTag(category: string, layerIdx?: number, step?: number): string;

/**
 * Check if buffer stats collection is enabled.
 * @internal
 */
export function isBufferStatsEnabled(): boolean | undefined;

/**
 * Get max abs threshold for explosion warnings.
 * @internal
 */
export function getMaxAbsThreshold(): number | undefined;
