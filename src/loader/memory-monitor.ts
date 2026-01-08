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
// Types
// ============================================================================

export interface MemorySnapshot {
  /** JS heap used (Chrome only) */
  jsHeapUsed?: number;
  /** JS heap total (Chrome only) */
  jsHeapTotal?: number;
  /** JS heap limit (Chrome only) */
  jsHeapLimit?: number;
  /** GPU buffer pool stats */
  gpu?: {
    currentBytes: number;
    activeBuffers: number;
    pooledBuffers: number;
    peakBytes: number;
  };
  /** Shard cache stats */
  shardCache?: {
    totalBytes: number;
    shardCount: number;
  };
  /** Loaded model state */
  modelState?: {
    layerCount: number;
    gpuBufferCount: number;
  };
}

export interface MemoryMonitorState {
  startTime: number;
  interval: ReturnType<typeof setInterval> | null;
}

// ============================================================================
// Memory Snapshot
// ============================================================================

/**
 * Capture current memory statistics.
 *
 * @returns Memory snapshot with available stats
 */
export function captureMemorySnapshot(): MemorySnapshot {
  const snapshot: MemorySnapshot = {};

  // JS Heap (Chrome only)
  const perfMemory = (performance as Performance & {
    memory?: {
      usedJSHeapSize?: number;
      totalJSHeapSize?: number;
      jsHeapSizeLimit?: number;
    };
  }).memory;

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
 * @param phase - Loading phase label
 * @param elapsed - Elapsed seconds since start
 * @param snapshot - Memory snapshot
 * @param shardCacheBytes - Shard cache total bytes
 * @param shardCount - Number of cached shards
 * @param layerCount - Number of loaded layers
 * @param gpuBufferCount - Number of GPU buffers
 * @returns Formatted log string
 */
export function formatMemoryStats(
  phase: string,
  elapsed: number,
  snapshot: MemorySnapshot,
  shardCacheBytes: number,
  shardCount: number,
  layerCount: number,
  gpuBufferCount: number
): string {
  const stats: string[] = [`[${elapsed.toFixed(1)}s] Memory (${phase}):`];

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
  private startTime = 0;
  private interval: ReturnType<typeof setInterval> | null = null;

  constructor(private logIntervalMs: number = 30000) {}

  /**
   * Start memory monitoring.
   *
   * @param getState - Function to get current loader state for logging
   */
  start(getState: () => { shardCacheBytes: number; shardCount: number; layerCount: number; gpuBufferCount: number }): void {
    this.startTime = performance.now();
    this.log('start', getState());

    this.interval = setInterval(() => {
      this.log('loading', getState());
    }, this.logIntervalMs);
  }

  /**
   * Stop memory monitoring.
   *
   * @param phase - Final phase label ('complete' or 'failed')
   * @param getState - Function to get current loader state
   */
  stop(
    phase: 'complete' | 'failed',
    getState: () => { shardCacheBytes: number; shardCount: number; layerCount: number; gpuBufferCount: number }
  ): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    this.log(phase, getState());
  }

  /**
   * Log memory stats for a phase.
   */
  private log(
    phase: string,
    state: { shardCacheBytes: number; shardCount: number; layerCount: number; gpuBufferCount: number }
  ): void {
    const elapsed = (performance.now() - this.startTime) / 1000;
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
   */
  getElapsed(): number {
    return (performance.now() - this.startTime) / 1000;
  }
}
