/**
 * GPU Profiler - Timestamp-based Performance Profiling
 *
 * Provides GPU-side timing using WebGPU timestamp queries.
 * Falls back to CPU timing when timestamp queries unavailable.
 *
 * Usage:
 *   const profiler = new GPUProfiler(device);
 *   profiler.begin('matmul');
 *   // ... dispatch compute pass ...
 *   profiler.end('matmul');
 *   await profiler.resolve();
 *   log.info('GPUProfiler', 'Results', profiler.getResults());
 */

import { getDevice, hasFeature, FEATURES } from './device.js';
import { allowReadback } from './perf-guards.js';
import { log } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { DEFAULT_PROFILER_CONFIG } from '../config/schema/debug.schema.js';
import { computeBasicStats } from '../debug/stats.js';

/**
 * @typedef {Object} ActiveMeasurement
 * @property {number} startQueryIndex
 * @property {number} cpuStartTime
 */

/**
 * @typedef {Object} CpuMeasurement
 * @property {number} cpuStartTime
 */

/**
 * @typedef {Object} PendingResolve
 * @property {string} label
 * @property {number} startIndex
 * @property {number} endIndex
 * @property {number} cpuStartTime
 * @property {number} cpuEndTime
 */

/**
 * @typedef {Object} ResultData
 * @property {number[]} times
 * @property {number} min
 * @property {number} max
 * @property {number} sum
 * @property {number} count
 */

/**
 * GPU Profiler using timestamp queries
 */
export class GPUProfiler {
  /** @type {GPUDevice | null} */
  #device;
  /** @type {boolean} */
  #hasTimestampQuery;

  // Query set for timestamp queries (if supported)
  /** @type {GPUQuerySet | null} */
  #querySet = null;
  /** @type {GPUBuffer | null} */
  #queryBuffer = null;
  /** @type {GPUBuffer | null} */
  #readbackBuffer = null;
  /** @type {number} */
  #queryCapacity = DEFAULT_PROFILER_CONFIG.queryCapacity;
  /** @type {number} */
  #maxSamples = DEFAULT_PROFILER_CONFIG.maxSamples;
  /** @type {number} */
  #maxDurationMs = DEFAULT_PROFILER_CONFIG.maxDurationMs;

  // Tracking state
  /** @type {Map<string, ActiveMeasurement | CpuMeasurement>} */
  #activeLabels = new Map();
  /** @type {number} */
  #nextQueryIndex = 0;
  /** @type {PendingResolve[]} */
  #pendingResolves = [];

  // Results storage
  /** @type {Map<string, ResultData>} */
  #results = new Map();

  // CPU fallback timing
  /** @type {Map<string, number>} */
  #cpuTimings = new Map();

  /**
   * @param {GPUDevice | null} [device] - WebGPU device (uses global if not provided)
   */
  constructor(device = null) {
    this.#device = device || getDevice();
    this.#hasTimestampQuery = this.#device?.features?.has(FEATURES.TIMESTAMP_QUERY) ?? false;
    const runtimeProfiler = getRuntimeConfig().shared?.debug?.profiler ?? DEFAULT_PROFILER_CONFIG;
    this.#queryCapacity = runtimeProfiler.queryCapacity ?? DEFAULT_PROFILER_CONFIG.queryCapacity;
    this.#maxSamples = runtimeProfiler.maxSamples ?? DEFAULT_PROFILER_CONFIG.maxSamples;
    this.#maxDurationMs = runtimeProfiler.maxDurationMs ?? DEFAULT_PROFILER_CONFIG.maxDurationMs;

    // Initialize query resources if timestamp queries available
    if (this.#hasTimestampQuery && this.#device) {
      this.#initQueryResources();
    }
  }

  /**
   * Initialize GPU query resources
   */
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

  /**
   * Begin timing a labeled region.
   * Uses CPU timing; use writeTimestamp() inside passes for GPU timestamps.
   * @param {string} label - Unique label for this measurement
   * @returns {void}
   */
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

  /**
   * End timing a labeled region
   * @param {string} label - Label started with begin()
   * @returns {void}
   */
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

  /**
   * Write timestamp to query set within a compute pass
   * Call this instead of begin/end when inside a pass
   * @param {GPUComputePassEncoder} pass - Compute pass encoder
   * @param {string} label - Label for this measurement
   * @param {boolean} [isEnd] - true for end timestamp
   * @returns {void}
   */
  writeTimestamp(pass, label, isEnd = false) {
    if (!this.#hasTimestampQuery || !this.#querySet) return;

    /** @type {number} */
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
    /** @type {any} */ (pass).writeTimestamp(this.#querySet, queryIndex);
  }

  /**
   * Resolve pending timestamp queries and update results
   * Call this after command buffer submission
   * @returns {Promise<void>}
   */
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

  /**
   * Record a timing result
   * @param {string} label
   * @param {number} timeMs
   */
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

  /**
   * Get profiling results
   * @returns {Record<string, import('./profiler.js').ProfileResult>}
   */
  getResults() {
    /** @type {Record<string, import('./profiler.js').ProfileResult>} */
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

  /**
   * Get result for a specific label
   * @param {string} label - Label to get result for
   * @returns {import('./profiler.js').ProfileResult | null}
   */
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

  /**
   * Reset all profiling data
   * @returns {void}
   */
  reset() {
    this.#results.clear();
    this.#activeLabels.clear();
    this.#pendingResolves = [];
    this.#nextQueryIndex = 0;
  }

  /**
   * Get formatted report string
   * @returns {string}
   */
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

  /**
   * Check if timestamp queries are available
   * @returns {boolean}
   */
  isGPUTimingAvailable() {
    return this.#hasTimestampQuery;
  }

  /**
   * Destroy profiler resources
   * @returns {void}
   */
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
/** @type {GPUProfiler | null} */
let globalProfiler = null;

/**
 * Get the global profiler instance
 * @returns {GPUProfiler}
 */
export function getProfiler() {
  if (!globalProfiler) {
    globalProfiler = new GPUProfiler();
  }
  return globalProfiler;
}

/**
 * Create a new profiler instance
 * @param {GPUDevice | null} [device] - Optional GPU device
 * @returns {GPUProfiler}
 */
export function createProfiler(device) {
  return new GPUProfiler(device);
}

/**
 * Convenience function to time a single operation
 * @template T
 * @param {string} label - Label for the operation
 * @param {() => Promise<T>} fn - Async function to time
 * @returns {Promise<{ result: T; timeMs: number }>}
 */
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
