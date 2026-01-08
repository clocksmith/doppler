/**
 * Memory Monitor - Memory statistics and logging during model loading.
 *
 * Provides utilities for tracking memory usage across JS heap, GPU buffers,
 * and shard cache during the model loading process.
 *
 * @module loader/memory-monitor
 */

import { formatBytes } from '../storage/quota.js';
import { getBufferPool } from '../gpu/buffer-pool.js';
import { log } from '../debug/index.js';

// ============================================================================
// Memory Snapshot
// ============================================================================

/**
 * Capture current memory statistics.
 *
 * @returns {import('./memory-monitor.js').MemorySnapshot} Memory snapshot with available stats
 */
export function captureMemorySnapshot() {
  /** @type {import('./memory-monitor.js').MemorySnapshot} */
  const snapshot = {};

  // JS Heap (Chrome only)
  const perfMemory = /** @type {Performance & { memory?: { usedJSHeapSize?: number; totalJSHeapSize?: number; jsHeapSizeLimit?: number } }} */ (performance).memory;

  if (perfMemory) {
    snapshot.jsHeapUsed = perfMemory.usedJSHeapSize ?? 0;
    snapshot.jsHeapTotal = perfMemory.totalJSHeapSize ?? 0;
    snapshot.jsHeapLimit = perfMemory.jsHeapSizeLimit ?? 0;
  }

  // GPU buffer pool stats
  try {
    const pool = getBufferPool();
    const poolStats = pool.getStats();
    snapshot.gpu = {
      currentBytes: poolStats.currentBytesAllocated,
      activeBuffers: poolStats.activeBuffers,
      pooledBuffers: poolStats.pooledBuffers,
      peakBytes: poolStats.peakBytesAllocated,
    };
  } catch {
    // Buffer pool not initialized yet
  }

  return snapshot;
}

/**
 * Format memory snapshot for logging.
 *
 * @param {string} phase - Loading phase label
 * @param {number} elapsed - Elapsed seconds since start
 * @param {import('./memory-monitor.js').MemorySnapshot} snapshot - Memory snapshot
 * @param {number} shardCacheBytes - Shard cache total bytes
 * @param {number} shardCount - Number of cached shards
 * @param {number} layerCount - Number of loaded layers
 * @param {number} gpuBufferCount - Number of GPU buffers
 * @returns {string} Formatted log string
 */
export function formatMemoryStats(
  phase,
  elapsed,
  snapshot,
  shardCacheBytes,
  shardCount,
  layerCount,
  gpuBufferCount
) {
  /** @type {string[]} */
  const stats = [`[${elapsed.toFixed(1)}s] Memory (${phase}):`];

  if (snapshot.jsHeapUsed !== undefined) {
    stats.push(
      `Heap=${formatBytes(snapshot.jsHeapUsed)}/${formatBytes(snapshot.jsHeapTotal ?? 0)} ` +
      `(limit=${formatBytes(snapshot.jsHeapLimit ?? 0)})`
    );
  }

  if (snapshot.gpu) {
    stats.push(
      `GPU=${formatBytes(snapshot.gpu.currentBytes)} ` +
      `(${snapshot.gpu.activeBuffers} active, ${snapshot.gpu.pooledBuffers} pooled, ` +
      `peak=${formatBytes(snapshot.gpu.peakBytes)})`
    );
  }

  stats.push(`ShardCache=${formatBytes(shardCacheBytes)} (${shardCount} shards)`);
  stats.push(`Layers=${layerCount}, GPUBuffers=${gpuBufferCount}`);

  return stats.join(' | ');
}

// ============================================================================
// Memory Monitor Class
// ============================================================================

/**
 * Memory monitor for tracking loading progress.
 *
 * Manages periodic memory logging during model loading.
 */
export class MemoryMonitor {
  /** @type {number} */
  #startTime = 0;

  /** @type {ReturnType<typeof setInterval> | null} */
  #interval = null;

  /** @type {number} */
  #logIntervalMs;

  /**
   * @param {number} [logIntervalMs=30000]
   */
  constructor(logIntervalMs = 30000) {
    this.#logIntervalMs = logIntervalMs;
  }

  /**
   * Start memory monitoring.
   *
   * @param {() => { shardCacheBytes: number; shardCount: number; layerCount: number; gpuBufferCount: number }} getState - Function to get current loader state for logging
   */
  start(getState) {
    this.#startTime = performance.now();
    this.#log('start', getState());

    this.#interval = setInterval(() => {
      this.#log('loading', getState());
    }, this.#logIntervalMs);
  }

  /**
   * Stop memory monitoring.
   *
   * @param {'complete' | 'failed'} phase - Final phase label ('complete' or 'failed')
   * @param {() => { shardCacheBytes: number; shardCount: number; layerCount: number; gpuBufferCount: number }} getState - Function to get current loader state
   */
  stop(phase, getState) {
    if (this.#interval) {
      clearInterval(this.#interval);
      this.#interval = null;
    }
    this.#log(phase, getState());
  }

  /**
   * Log memory stats for a phase.
   *
   * @param {string} phase
   * @param {{ shardCacheBytes: number; shardCount: number; layerCount: number; gpuBufferCount: number }} state
   */
  #log(phase, state) {
    const elapsed = (performance.now() - this.#startTime) / 1000;
    const snapshot = captureMemorySnapshot();
    const message = formatMemoryStats(
      phase,
      elapsed,
      snapshot,
      state.shardCacheBytes,
      state.shardCount,
      state.layerCount,
      state.gpuBufferCount
    );
    log.info('Loader', message);
  }

  /**
   * Get elapsed time since monitoring started.
   *
   * @returns {number}
   */
  getElapsed() {
    return (performance.now() - this.#startTime) / 1000;
  }
}
