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
export type KVLayout = 'contiguous' | 'paged' | 'tiered';

/**
 * Tiered KV cache mode.
 *
 * - 'off': Disable tiering (use layout directly)
 * - 'fp16': Hot ring + cold FP16 pages
 * - 'int8': Cold tier compressed to int8
 * - 'int4': Cold tier compressed to int4 (experimental)
 */
export type KVTieringMode = 'off' | 'fp16' | 'int8' | 'int4';

/**
 * Cold tier compression mode.
 *
 * - 'none': Keep cold tier in FP16/FP32
 * - 'int8': Block-wise int8 compression
 * - 'int4': Block-wise int4 compression
 */
export type KVCompressionMode = 'none' | 'int8' | 'int4';

/**
 * Gating mode for tiered compression.
 *
 * - 'auto': Use device heuristics to enable/disable compression
 * - 'force_on': Always enable compression
 * - 'force_off': Always disable compression
 */
export type KVTieringGatingMode = 'auto' | 'force_on' | 'force_off';

export interface KVTieringCompressionSchema {
  /** Compression mode for cold tier */
  mode: KVCompressionMode;
  /** Compression block size (tokens). Currently only 1 is supported. */
  blockSize: number;
}

export interface KVTieringGatingSchema {
  /** Compression gating strategy */
  mode: KVTieringGatingMode;
  /** Minimum ALU/BW ratio to enable compression (0 disables gating) */
  minAluBwRatio: number;
}

export interface KVTieringConfigSchema {
  /** Tiering mode */
  mode: KVTieringMode;
  /** Number of tokens kept hot in ring buffer */
  hotWindow: number;
  /** Page size for cold tier paging */
  coldPageSize: number;
  /** Cold tier dtype before compression */
  coldDtype: KVDtype;
  /** Cold tier compression settings */
  compression: KVTieringCompressionSchema;
  /** Compression gating settings */
  gating: KVTieringGatingSchema;
}

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

  /** Max sequence length cap when GPU paged layout is unavailable */
  gpuPagedFallbackMaxSeqLen: number;

  /** Data type for cache storage */
  kvDtype: KVDtype;

  /**
   * Force F32 KV cache when attention logit softcapping is enabled.
   * Default false keeps F16 KV cache when supported.
   */
  forceF32Softcap: boolean;

  /** Memory layout strategy */
  layout: KVLayout;

  /** Page size for paged layout (number of tokens per page) */
  pageSize: number;

  /** Sliding window size for sliding window attention models */
  windowSize: number;

  /** Tiered cache configuration (hot ring + cold pages) */
  tiering: KVTieringConfigSchema;
}

/** Default KV cache configuration */
export declare const DEFAULT_KVCACHE_CONFIG: KVCacheConfigSchema;

/**
 * Sequence length threshold for automatic paged layout selection.
 * Above this threshold, paged layout is preferred for memory efficiency.
 */
export declare const PAGED_LAYOUT_SEQ_LEN_THRESHOLD: number;
