/**
 * Uniform Buffer Cache
 *
 * Caches small uniform buffers by content hash to avoid repeated allocations.
 * WebLLM-inspired optimization: uniform buffers with identical contents are reused
 * across kernel dispatches instead of being created fresh and destroyed each time.
 */

interface UniformCacheEntry {
  buffer: GPUBuffer;
  lastUsed: number;
  refCount: number;
}

interface UniformCacheStats {
  hits: number;
  misses: number;
  evictions: number;
  currentSize: number;
}

/**
 * Uniform Buffer Cache
 *
 * Provides content-addressed caching for uniform buffers. Buffers with
 * identical contents share the same GPU buffer, reducing allocation overhead.
 *
 * IMPORTANT: Evicted buffers are NOT destroyed immediately. They are queued
 * for deferred destruction to avoid use-after-destroy bugs when command
 * buffers reference cached uniforms that get evicted before submit.
 * Call flushPendingDestruction() after GPU work completes.
 */
export declare class UniformBufferCache {
  private cache: Map<string, UniformCacheEntry>;
  private stats: UniformCacheStats;
  private pendingDestruction: GPUBuffer[];
  private readonly maxEntries: number;
  private readonly maxAgeMs: number;

  constructor(maxEntries?: number, maxAgeMs?: number);

  /**
   * Get or create a uniform buffer with the given contents.
   * Returns a cached buffer if one exists with identical data.
   */
  getOrCreate(data: ArrayBuffer | SharedArrayBuffer, label: string): GPUBuffer;

  /**
   * Release a reference to a cached buffer.
   * Buffer is NOT destroyed - it stays in cache for reuse.
   * Call this instead of buffer.destroy() for cached uniforms.
   */
  release(buffer: GPUBuffer): void;

  /**
   * Evict stale entries (older than maxAgeMs).
   * Buffers are queued for deferred destruction.
   */
  evictStale(): number;

  /**
   * Clear all cached buffers.
   * Also flushes any pending destruction queue.
   */
  clear(): void;

  /**
   * Destroy all buffers in the pending destruction queue.
   * Call this after GPU work completes (e.g., after onSubmittedWorkDone).
   *
   * This is critical for avoiding use-after-destroy bugs: when the uniform
   * cache evicts a buffer that's still referenced by a pending command buffer,
   * the buffer is queued here instead of being destroyed immediately.
   */
  flushPendingDestruction(): number;

  /**
   * Get the number of buffers pending destruction.
   */
  getPendingDestructionCount(): number;

  /**
   * Check if a buffer is managed by this cache
   */
  isCached(buffer: GPUBuffer): boolean;

  /**
   * Get cache statistics
   */
  getStats(): UniformCacheStats & { hitRate: string; pendingDestruction: number };
}

/**
 * Release or destroy a uniform buffer appropriately.
 * If the buffer is cached, releases it back to the cache.
 * If not cached, destroys it directly.
 */
export function releaseUniformBuffer(buffer: GPUBuffer): void;

/**
 * Get the global uniform buffer cache instance
 */
export function getUniformCache(): UniformBufferCache;

/**
 * Reset the global uniform cache (useful for testing or device loss)
 */
export function resetUniformCache(): void;
