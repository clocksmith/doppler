/**
 * Memory Monitor - Memory statistics and logging during model loading.
 *
 * Provides utilities for tracking memory usage across JS heap, GPU buffers,
 * and shard cache during the model loading process.
 *
 * @module loader/memory-monitor
 */

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
export declare function captureMemorySnapshot(): MemorySnapshot;

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
export declare function formatMemoryStats(
  phase: string,
  elapsed: number,
  snapshot: MemorySnapshot,
  shardCacheBytes: number,
  shardCount: number,
  layerCount: number,
  gpuBufferCount: number
): string;

// ============================================================================
// Memory Monitor Class
// ============================================================================

/**
 * Memory monitor for tracking loading progress.
 *
 * Manages periodic memory logging during model loading.
 */
export declare class MemoryMonitor {
  constructor(logIntervalMs?: number);

  /**
   * Start memory monitoring.
   *
   * @param getState - Function to get current loader state for logging
   */
  start(getState: () => { shardCacheBytes: number; shardCount: number; layerCount: number; gpuBufferCount: number }): void;

  /**
   * Stop memory monitoring.
   *
   * @param phase - Final phase label ('complete' or 'failed')
   * @param getState - Function to get current loader state
   */
  stop(
    phase: 'complete' | 'failed',
    getState: () => { shardCacheBytes: number; shardCount: number; layerCount: number; gpuBufferCount: number }
  ): void;

  /**
   * Get elapsed time since monitoring started.
   */
  getElapsed(): number;
}
