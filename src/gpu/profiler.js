

import { getDevice, hasFeature, FEATURES } from './device.js';
import { allowReadback } from './perf-guards.js';
import { log } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { computeBasicStats } from '../debug/stats.js';










export class GPUProfiler {
  
  #device;
  
  #hasTimestampQuery;

  // Query set for timestamp queries (if supported)
  
  #querySet = null;
  
  #queryBuffer = null;
  
  #readbackBuffer = null;
  
  #queryCapacity = 0;
  
  #maxSamples = 0;
  
  #maxDurationMs = 0;

  // Tracking state
  
  #activeLabels = new Map();
  
  #nextQueryIndex = 0;
  
  #pendingResolves = [];

  // Results storage
  
  #results = new Map();

  // CPU fallback timing
  
  #cpuTimings = new Map();

  
  constructor(device = null) {
    this.#device = device || getDevice();
    this.#hasTimestampQuery = this.#device?.features?.has(FEATURES.TIMESTAMP_QUERY) ?? false;
    const runtimeProfiler = getRuntimeConfig().shared?.debug?.profiler;
    if (!runtimeProfiler) {
      throw new Error('runtime.shared.debug.profiler is required.');
    }
    this.#queryCapacity = runtimeProfiler.queryCapacity;
    this.#maxSamples = runtimeProfiler.maxSamples;
    this.#maxDurationMs = runtimeProfiler.maxDurationMs;

    // Initialize query resources if timestamp queries available
    if (this.#hasTimestampQuery && this.#device) {
      this.#initQueryResources();
    }
  }

  
  #initQueryResources() {
    if (!this.#device) return;

    try {
      this.#querySet = this.#device.createQuerySet({
        type: 'timestamp',
        count: this.#queryCapacity * 2, // Start and end for each measurement
      });

      // Buffer to hold query results (8 bytes per timestamp)
      this.#queryBuffer = this.#device.createBuffer({
        size: this.#queryCapacity * 2 * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });

      // Readback buffer
      this.#readbackBuffer = this.#device.createBuffer({
        size: this.#queryCapacity * 2 * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    } catch (e) {
      log.warn('GPUProfiler', `Failed to create timestamp query resources: ${e}`);
      this.#hasTimestampQuery = false;
    }
  }

  
  begin(label) {
    if (this.#activeLabels.has(label)) {
      log.warn('GPUProfiler', `Label "${label}" already active`);
      return;
    }

    const startTime = performance.now();

    // CPU timing for begin/end; GPU timestamps require writeTimestamp() in a pass.
    this.#activeLabels.set(label, {
      cpuStartTime: startTime,
    });
  }

  
  end(label) {
    const active = this.#activeLabels.get(label);
    if (!active) {
      log.warn('GPUProfiler', `No active measurement for label "${label}"`);
      return;
    }

    const endTime = performance.now();
    this.#activeLabels.delete(label);

    if (this.#hasTimestampQuery && 'startQueryIndex' in active) {
      // GPU timing will be resolved later
      this.#pendingResolves.push({
        label,
        startIndex: active.startQueryIndex,
        endIndex: active.startQueryIndex + 1,
        cpuStartTime: active.cpuStartTime,
        cpuEndTime: endTime,
      });
    } else {
      // CPU fallback - record immediately
      this.#recordResult(label, endTime - active.cpuStartTime);
    }
  }

  
  writeTimestamp(pass, label, isEnd = false) {
    if (!this.#hasTimestampQuery || !this.#querySet) return;

    
    let queryIndex;
    if (!isEnd) {
      // Start timestamp
      queryIndex = this.#nextQueryIndex;
      this.#nextQueryIndex += 2;
      this.#activeLabels.set(label, {
        startQueryIndex: queryIndex,
        cpuStartTime: performance.now(),
      });
    } else {
      // End timestamp
      const active = this.#activeLabels.get(label);
      if (!active || !('startQueryIndex' in active)) return;
      queryIndex = active.startQueryIndex + 1;
      this.#activeLabels.delete(label);
      this.#pendingResolves.push({
        label,
        startIndex: active.startQueryIndex,
        endIndex: queryIndex,
        cpuStartTime: active.cpuStartTime,
        cpuEndTime: performance.now(),
      });
    }

    // Note: writeTimestamp is deprecated in modern WebGPU spec but still works in Chrome
    // Future: migrate to timestampWrites in GPUComputePassDescriptor
     (pass).writeTimestamp(this.#querySet, queryIndex);
  }

  
  async resolve() {
    if (!this.#hasTimestampQuery || this.#pendingResolves.length === 0) {
      return;
    }

    if (!this.#device || !this.#querySet || !this.#queryBuffer || !this.#readbackBuffer) {
      log.warn('GPUProfiler', 'Missing required resources for resolve');
      return;
    }

    const encoder = this.#device.createCommandEncoder();

    // Resolve all timestamps to buffer
    const maxIndex = Math.max(...this.#pendingResolves.map(p => p.endIndex)) + 1;
    encoder.resolveQuerySet(this.#querySet, 0, maxIndex, this.#queryBuffer, 0);

    // Copy to readback buffer
    encoder.copyBufferToBuffer(
      this.#queryBuffer,
      0,
      this.#readbackBuffer,
      0,
      maxIndex * 8
    );

    this.#device.queue.submit([encoder.finish()]);

    if (!allowReadback('GPUProfiler.resolve')) {
      return;
    }

    // Read back timestamps
    await this.#readbackBuffer.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(this.#readbackBuffer.getMappedRange());

    // Process pending resolves
    for (const pending of this.#pendingResolves) {
      const startNs = timestamps[pending.startIndex];
      const endNs = timestamps[pending.endIndex];

      // Convert nanoseconds to milliseconds
      const durationMs = Number(endNs - startNs) / 1_000_000;

      // Sanity check - use CPU timing if GPU timing seems wrong
      if (durationMs < 0 || durationMs > this.#maxDurationMs) {
        // Fallback to CPU timing
        this.#recordResult(pending.label, pending.cpuEndTime - pending.cpuStartTime);
      } else {
        this.#recordResult(pending.label, durationMs);
      }
    }

    this.#readbackBuffer.unmap();
    this.#pendingResolves = [];
    this.#nextQueryIndex = 0;
  }

  
  #recordResult(label, timeMs) {
    if (!this.#results.has(label)) {
      this.#results.set(label, {
        times: [],
        min: Infinity,
        max: -Infinity,
        sum: 0,
        count: 0,
      });
    }

    const result = this.#results.get(label);
    result.times.push(timeMs);
    result.min = Math.min(result.min, timeMs);
    result.max = Math.max(result.max, timeMs);
    result.sum += timeMs;
    result.count++;

    // Keep only last N samples for running average
    if (result.times.length > this.#maxSamples) {
      const removed = result.times.shift();
      result.sum -= removed;
      result.count--;
      // Recalculate min/max if needed (expensive, so only do occasionally)
      if (result.times.length % 20 === 0) {
        result.min = Math.min(...result.times);
        result.max = Math.max(...result.times);
      }
    }
  }

  
  getResults() {
    
    const output = {};

    for (const [label, data] of this.#results) {
      const stats = computeBasicStats(data.times);
      output[label] = {
        avg: stats.mean,
        min: stats.min,
        max: stats.max,
        count: stats.count,
        total: stats.total,
      };
    }

    return output;
  }

  
  getResult(label) {
    const data = this.#results.get(label);
    if (!data) return null;

    const stats = computeBasicStats(data.times);
    return {
      avg: stats.mean,
      min: stats.min,
      max: stats.max,
      count: stats.count,
      total: stats.total,
    };
  }

  
  reset() {
    this.#results.clear();
    this.#activeLabels.clear();
    this.#pendingResolves = [];
    this.#nextQueryIndex = 0;
  }

  
  getReport() {
    const results = this.getResults();
    const labels = Object.keys(results).sort();

    if (labels.length === 0) {
      return 'No profiling data collected';
    }

    let report = 'GPU Profiler Results\n';
    report += '\u2500'.repeat(60) + '\n';
    report += 'Label'.padEnd(30) + 'Avg (ms)'.padStart(10) + 'Min'.padStart(10) + 'Max'.padStart(10) + '\n';
    report += '\u2500'.repeat(60) + '\n';

    for (const label of labels) {
      const r = results[label];
      report += label.padEnd(30);
      report += r.avg.toFixed(3).padStart(10);
      report += r.min.toFixed(3).padStart(10);
      report += r.max.toFixed(3).padStart(10);
      report += '\n';
    }

    return report;
  }

  
  isGPUTimingAvailable() {
    return this.#hasTimestampQuery;
  }

  
  destroy() {
    if (this.#querySet) {
      this.#querySet.destroy();
      this.#querySet = null;
    }
    if (this.#queryBuffer) {
      this.#queryBuffer.destroy();
      this.#queryBuffer = null;
    }
    if (this.#readbackBuffer) {
      this.#readbackBuffer.destroy();
      this.#readbackBuffer = null;
    }
    this.#results.clear();
    this.#activeLabels.clear();
  }
}

// Global profiler instance

let globalProfiler = null;


export function getProfiler() {
  if (!globalProfiler) {
    globalProfiler = new GPUProfiler();
  }
  return globalProfiler;
}


export function createProfiler(device) {
  return new GPUProfiler(device);
}


export async function timeOperation(
  label,
  fn
) {
  const profiler = getProfiler();
  profiler.begin(label);
  const result = await fn();
  profiler.end(label);
  await profiler.resolve();

  const timing = profiler.getResult(label);
  return {
    result,
    timeMs: timing?.avg ?? 0,
  };
}

export default GPUProfiler;
