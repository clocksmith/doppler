

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
      currentBytesRequested: poolStats.currentBytesRequested,
      activeBuffers: poolStats.activeBuffers,
      pooledBuffers: poolStats.pooledBuffers,
      peakBytes: poolStats.peakBytesAllocated,
      peakBytesRequested: poolStats.peakBytesRequested,
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


  #snapshots = [];


  #captureSnapshots;


  constructor(logIntervalMs = 30000, captureSnapshots = false) {
    this.#logIntervalMs = logIntervalMs;
    this.#captureSnapshots = captureSnapshots;
    this.#snapshots = [];
  }


  start(getState) {
    this.#startTime = performance.now();
    this.#snapshots = [];
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


  captureSnapshot(phase) {
    const elapsed = (performance.now() - this.#startTime) / 1000;
    const snapshot = captureMemorySnapshot();
    if (this.#captureSnapshots) {
      this.#snapshots.push({
        timestamp: elapsed,
        phase,
        ...snapshot,
      });
    }
    return snapshot;
  }


  getSnapshots() {
    return [...this.#snapshots];
  }


  clearSnapshots() {
    this.#snapshots = [];
  }


  #log(phase, state) {
    const elapsed = (performance.now() - this.#startTime) / 1000;
    const snapshot = captureMemorySnapshot();

    // Store snapshot if capture is enabled
    if (this.#captureSnapshots) {
      this.#snapshots.push({
        timestamp: elapsed,
        phase,
        ...snapshot,
        shardCacheBytes: state.shardCacheBytes,
        shardCount: state.shardCount,
        layerCount: state.layerCount,
        gpuBufferCount: state.gpuBufferCount,
      });
    }

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

// ============================================================================
// Time Series Collector for Benchmarks
// ============================================================================


export class MemoryTimeSeries {

  #samples = [];


  #interval = null;


  #startTime = 0;


  #sampleIntervalMs;


  constructor(sampleIntervalMs = 100) {
    this.#sampleIntervalMs = sampleIntervalMs;
    this.#samples = [];
  }


  start() {
    this.#startTime = performance.now();
    this.#samples = [];
    this.#capture('start');

    this.#interval = setInterval(() => {
      this.#capture('sample');
    }, this.#sampleIntervalMs);
  }


  stop() {
    if (this.#interval) {
      clearInterval(this.#interval);
      this.#interval = null;
    }
    this.#capture('end');
  }


  mark(phase) {
    this.#capture(phase);
  }


  #capture(phase) {
    const elapsed = performance.now() - this.#startTime;
    const snapshot = captureMemorySnapshot();

    this.#samples.push({
      t: Math.round(elapsed),
      phase,
      gpu: snapshot.gpu?.currentBytes ?? 0,
      gpuPeak: snapshot.gpu?.peakBytes ?? 0,
      gpuRequested: snapshot.gpu?.currentBytesRequested ?? 0,
      jsHeap: snapshot.jsHeapUsed ?? 0,
    });
  }


  getSamples() {
    return [...this.#samples];
  }


  getSummary() {
    if (this.#samples.length === 0) {
      return { samples: 0, duration: 0, gpuPeak: 0, gpuStart: 0, gpuEnd: 0 };
    }

    const first = this.#samples[0];
    const last = this.#samples[this.#samples.length - 1];
    let gpuPeak = 0;
    for (const s of this.#samples) {
      if (s.gpu > gpuPeak) gpuPeak = s.gpu;
    }

    return {
      samples: this.#samples.length,
      duration: last.t - first.t,
      gpuPeak,
      gpuStart: first.gpu,
      gpuEnd: last.gpu,
    };
  }
}
