/**
 * KV Cache Config Schema
 *
 * Configuration for key-value cache: dtype, layout, sizing.
 * Controls memory allocation and access patterns for transformer attention.
 *
 * @module config/schema/kvcache
 */

/**
 * Data type for KV cache storage.
 *
 * - 'f16': Half precision (2 bytes per element, lower memory, slight accuracy loss)
 * - 'f32': Full precision (4 bytes per element, higher memory, full accuracy)
 */
export type KVDtype = 'f16' | 'f32';

/**
 * Memory layout for KV cache.
 *
 * - 'contiguous': Single contiguous buffer per layer (simpler, better for short sequences)
 * - 'paged': Page-based allocation (better memory efficiency for variable sequences)
 */
export type KVLayout = 'contiguous' | 'paged';

/**
 * Configuration for the key-value cache.
 *
 * The KV cache stores computed key and value tensors from attention layers
 * to avoid recomputation during autoregressive decoding. These settings
 * control memory allocation, precision, and layout strategies.
 */
export interface KVCacheConfigSchema {
  /** Maximum sequence length the cache can hold */
  maxSeqLen: number;

  /** Data type for cache storage */
  kvDtype: KVDtype;

  /** Memory layout strategy */
  layout: KVLayout;

  /** Page size for paged layout (number of tokens per page) */
  pageSize: number;

  /** Sliding window size for sliding window attention models */
  windowSize: number;
}

/** Default KV cache configuration */
export declare const DEFAULT_KVCACHE_CONFIG: KVCacheConfigSchema;

/**
 * Sequence length threshold for automatic paged layout selection.
 * Above this threshold, paged layout is preferred for memory efficiency.
 */
export declare const PAGED_LAYOUT_SEQ_LEN_THRESHOLD: number;
