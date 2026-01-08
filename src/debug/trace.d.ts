/**
 * DOPPLER Debug Module - Trace Logging Interface
 *
 * Category-based tracing for detailed subsystem debugging.
 *
 * @module debug/trace
 */

// ============================================================================
// Trace Interface
// ============================================================================

/**
 * Trace logging interface - only logs if category is enabled.
 */
export declare const trace: {
  /**
   * Trace model loading operations.
   */
  loader(message: string, data?: unknown): void;

  /**
   * Trace kernel execution.
   */
  kernels(message: string, data?: unknown): void;

  /**
   * Trace logit computation.
   */
  logits(message: string, data?: unknown): void;

  /**
   * Trace embedding layer.
   */
  embed(message: string, data?: unknown): void;

  /**
   * Trace attention computation.
   */
  attn(layerIdx: number, message: string, data?: unknown): void;

  /**
   * Trace feed-forward network.
   */
  ffn(layerIdx: number, message: string, data?: unknown): void;

  /**
   * Trace KV cache operations.
   */
  kv(layerIdx: number, message: string, data?: unknown): void;

  /**
   * Trace token sampling.
   */
  sample(message: string, data?: unknown): void;

  /**
   * Trace buffer stats (expensive - requires GPU readback).
   */
  buffers(message: string, data?: unknown): void;

  /**
   * Trace performance timing.
   */
  perf(message: string, data?: unknown): void;
};
