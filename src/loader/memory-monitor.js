

import { formatBytes } from '../storage/quota.js';
import { getBufferPool } from '../gpu/buffer-pool.js';
import { log } from '../debug/index.js';

// ============================================================================
// Memory Snapshot
// ============================================================================


export function captureMemorySnapshot() {
  
  const snapshot = {};

  // JS Heap (Chrome only)
  const perfMemory =  (performance).memory;

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


export function formatMemoryStats(
  phase,
  elapsed,
  snapshot,
  shardCacheBytes,
  shardCount,
  layerCount,
  gpuBufferCount
) {
  
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


export class MemoryMonitor {
  
  #startTime = 0;

  
  #interval = null;

  
  #logIntervalMs;

  
  constructor(logIntervalMs = 30000) {
    this.#logIntervalMs = logIntervalMs;
  }

  
  start(getState) {
    this.#startTime = performance.now();
    this.#log('start', getState());

    this.#interval = setInterval(() => {
      this.#log('loading', getState());
    }, this.#logIntervalMs);
  }

  
  stop(phase, getState) {
    if (this.#interval) {
      clearInterval(this.#interval);
      this.#interval = null;
    }
    this.#log(phase, getState());
  }

  
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

  
  getElapsed() {
    return (performance.now() - this.#startTime) / 1000;
  }
}
