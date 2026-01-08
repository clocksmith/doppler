/**
 * KV Cache Config Schema
 *
 * Configuration for key-value cache: dtype, layout, sizing.
 * Controls memory allocation and access patterns for transformer attention.
 *
 * @module config/schema/kvcache
 */

// =============================================================================
// Default Config
// =============================================================================

/** Default KV cache configuration */
export const DEFAULT_KVCACHE_CONFIG = {
  maxSeqLen: 4096,
  kvDtype: 'f16',
  layout: 'contiguous',
  pageSize: 256,
  windowSize: 1024,
};

/**
 * Sequence length threshold for automatic paged layout selection.
 * Above this threshold, paged layout is preferred for memory efficiency.
 */
export const PAGED_LAYOUT_SEQ_LEN_THRESHOLD = 8192;
