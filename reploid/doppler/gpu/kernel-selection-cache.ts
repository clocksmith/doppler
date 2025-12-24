/**
 * Kernel Selection Cache
 *
 * Caches kernel variant selections after first forward pass to avoid
 * repeated capability checks and heuristic evaluation during batch generation.
 *
 * Uses auto-detect approach: profile first forward pass, cache selections
 * for all subsequent tokens in the generation.
 *
 * @module gpu/kernel-selection-cache
 */

interface CacheEntry {
  variant: string;
  timestamp: number;
}

/** Cache storage: "op:key" -> variant */
const cache = new Map<string, CacheEntry>();

/** Whether cache has been warmed (first forward pass complete) */
let isWarmed = false;

/** Statistics for debugging */
let stats = {
  hits: 0,
  misses: 0,
  sets: 0,
};

/**
 * Get cached kernel selection for an operation
 * @param op - Operation name (e.g., 'matmul', 'attention')
 * @param key - Cache key (e.g., 'M_N_K_dtype' for matmul)
 * @returns Cached variant string or null if not cached
 */
export function getCachedSelection(op: string, key: string): string | null {
  const entry = cache.get(`${op}:${key}`);
  if (entry) {
    stats.hits++;
    return entry.variant;
  }
  stats.misses++;
  return null;
}

/**
 * Cache a kernel selection
 * @param op - Operation name
 * @param key - Cache key
 * @param variant - Selected variant to cache
 */
export function setCachedSelection(op: string, key: string, variant: string): void {
  cache.set(`${op}:${key}`, {
    variant,
    timestamp: performance.now(),
  });
  stats.sets++;
}

/**
 * Get or compute kernel selection with caching
 * @param op - Operation name
 * @param key - Cache key
 * @param compute - Function to compute variant if not cached
 * @returns Cached or computed variant
 */
export function getOrComputeSelection(
  op: string,
  key: string,
  compute: () => string
): string {
  const cached = getCachedSelection(op, key);
  if (cached !== null) {
    return cached;
  }

  const variant = compute();
  setCachedSelection(op, key, variant);
  return variant;
}

/**
 * Mark the cache as warmed (first forward pass complete)
 * After warming, the cache will be used for all subsequent operations.
 */
export function markWarmed(): void {
  if (!isWarmed) {
    isWarmed = true;
    console.log(`[KernelCache] Warmed with ${cache.size} cached selections`);
  }
}

/**
 * Check if cache has been warmed
 */
export function isKernelCacheWarmed(): boolean {
  return isWarmed;
}

/**
 * Clear the cache (useful for testing or model switching)
 */
export function clearKernelCache(): void {
  cache.clear();
  isWarmed = false;
  stats = { hits: 0, misses: 0, sets: 0 };
}

/**
 * Get cache statistics
 */
export function getKernelCacheStats(): {
  size: number;
  warmed: boolean;
  hits: number;
  misses: number;
  sets: number;
  hitRate: number;
} {
  const total = stats.hits + stats.misses;
  return {
    size: cache.size,
    warmed: isWarmed,
    hits: stats.hits,
    misses: stats.misses,
    sets: stats.sets,
    hitRate: total > 0 ? stats.hits / total : 0,
  };
}

/**
 * Generate a cache key for matmul operations
 */
export function matmulCacheKey(
  M: number,
  N: number,
  K: number,
  aDtype: string | null,
  bDtype: string | null,
  transposeB: boolean
): string {
  return `${M}_${N}_${K}_${aDtype ?? 'f32'}_${bDtype ?? 'f32'}_${transposeB ? 't' : 'n'}`;
}

/**
 * Generate a cache key for attention operations
 */
export function attentionCacheKey(
  mode: 'prefill' | 'decode',
  seqLen: number,
  numHeads: number,
  headDim: number,
  useF16KV: boolean
): string {
  return `${mode}_${seqLen}_${numHeads}_${headDim}_${useF16KV ? 'f16' : 'f32'}`;
}

/**
 * Generate a cache key for dequantization operations
 */
export function dequantCacheKey(
  quantType: string,
  outputDtype: string
): string {
  return `${quantType}_${outputDtype}`;
}
