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
 *   console.log(profiler.getResults());
 */

import { getDevice, hasFeature, FEATURES } from './device.js';
import { allowReadback } from './perf-guards.js';
import type { ProfileEvent, ProfileSession } from '../types/gpu.js';

/**
 * Profiling result for a single label
 */
export interface ProfileResult {
  /** Average time in milliseconds */
  avg: number;
  /** Minimum time in milliseconds */
  min: number;
  /** Maximum time in milliseconds */
  max: number;
  /** Number of samples */
  count: number;
  /** Total time in milliseconds */
  total: number;
}

/**
 * Internal tracking state for active measurements
 */
interface ActiveMeasurement {
  startQueryIndex: number;
  cpuStartTime: number;
}

/**
 * Internal tracking state for CPU-only measurements
 */
interface CpuMeasurement {
  cpuStartTime: number;
}

/**
 * Pending timestamp query resolution
 */
interface PendingResolve {
  label: string;
  startIndex: number;
  endIndex: number;
  cpuStartTime: number;
  cpuEndTime: number;
}

/**
 * Internal result storage
 */
interface ResultData {
  times: number[];
  min: number;
  max: number;
  sum: number;
  count: number;
}

/**
 * GPU Profiler using timestamp queries
 */
export class GPUProfiler {
  private device: GPUDevice | null;
  private hasTimestampQuery: boolean;

  // Query set for timestamp queries (if supported)
  private querySet: GPUQuerySet | null = null;
  private queryBuffer: GPUBuffer | null = null;
  private readbackBuffer: GPUBuffer | null = null;
  private queryCapacity = 256; // Max number of timestamp pairs

  // Tracking state
  private activeLabels: Map<string, ActiveMeasurement | CpuMeasurement> = new Map();
  private nextQueryIndex = 0;
  private pendingResolves: PendingResolve[] = [];

  // Results storage
  private results: Map<string, ResultData> = new Map();

  // CPU fallback timing
  private cpuTimings: Map<string, number> = new Map();

  /**
   * @param device - WebGPU device (uses global if not provided)
   */
  constructor(device: GPUDevice | null = null) {
    this.device = device || getDevice();
    this.hasTimestampQuery = this.device?.features?.has(FEATURES.TIMESTAMP_QUERY) ?? false;

    // Initialize query resources if timestamp queries available
    if (this.hasTimestampQuery && this.device) {
      this._initQueryResources();
    }
  }

  /**
   * Initialize GPU query resources
   * @private
   */
  private _initQueryResources(): void {
    if (!this.device) return;

    try {
      this.querySet = this.device.createQuerySet({
        type: 'timestamp',
        count: this.queryCapacity * 2, // Start and end for each measurement
      });

      // Buffer to hold query results (8 bytes per timestamp)
      this.queryBuffer = this.device.createBuffer({
        size: this.queryCapacity * 2 * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });

      // Readback buffer
      this.readbackBuffer = this.device.createBuffer({
        size: this.queryCapacity * 2 * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    } catch (e) {
      console.warn('[GPUProfiler] Failed to create timestamp query resources:', e);
      this.hasTimestampQuery = false;
    }
  }

  /**
   * Begin timing a labeled region
   * @param label - Unique label for this measurement
   */
  begin(label: string): void {
    if (this.activeLabels.has(label)) {
      console.warn(`[GPUProfiler] Label "${label}" already active`);
      return;
    }

    const startTime = performance.now();

    if (this.hasTimestampQuery) {
      const queryIndex = this.nextQueryIndex;
      this.nextQueryIndex += 2; // Reserve start and end slots

      if (queryIndex >= this.queryCapacity * 2) {
        console.warn('[GPUProfiler] Query capacity exceeded, resetting');
        this.nextQueryIndex = 0;
      }

      this.activeLabels.set(label, {
        startQueryIndex: queryIndex,
        cpuStartTime: startTime,
      });
    } else {
      // CPU fallback
      this.activeLabels.set(label, {
        cpuStartTime: startTime,
      });
    }
  }

  /**
   * End timing a labeled region
   * @param label - Label started with begin()
   */
  end(label: string): void {
    const active = this.activeLabels.get(label);
    if (!active) {
      console.warn(`[GPUProfiler] No active measurement for label "${label}"`);
      return;
    }

    const endTime = performance.now();
    this.activeLabels.delete(label);

    if (this.hasTimestampQuery && 'startQueryIndex' in active) {
      // GPU timing will be resolved later
      this.pendingResolves.push({
        label,
        startIndex: active.startQueryIndex,
        endIndex: active.startQueryIndex + 1,
        cpuStartTime: active.cpuStartTime,
        cpuEndTime: endTime,
      });
    } else {
      // CPU fallback - record immediately
      this._recordResult(label, endTime - active.cpuStartTime);
    }
  }

  /**
   * Write timestamp to query set within a compute pass
   * Call this instead of begin/end when inside a pass
   * @param pass - Compute pass encoder
   * @param label - Label for this measurement
   * @param isEnd - true for end timestamp
   */
  writeTimestamp(pass: GPUComputePassEncoder, label: string, isEnd = false): void {
    if (!this.hasTimestampQuery || !this.querySet) return;

    let queryIndex: number;
    if (!isEnd) {
      // Start timestamp
      queryIndex = this.nextQueryIndex;
      this.nextQueryIndex += 2;
      this.activeLabels.set(label, {
        startQueryIndex: queryIndex,
        cpuStartTime: performance.now(),
      });
    } else {
      // End timestamp
      const active = this.activeLabels.get(label);
      if (!active || !('startQueryIndex' in active)) return;
      queryIndex = active.startQueryIndex + 1;
      this.activeLabels.delete(label);
      this.pendingResolves.push({
        label,
        startIndex: active.startQueryIndex,
        endIndex: queryIndex,
        cpuStartTime: active.cpuStartTime,
        cpuEndTime: performance.now(),
      });
    }

    // Note: writeTimestamp is deprecated in modern WebGPU spec but still works in Chrome
    // Future: migrate to timestampWrites in GPUComputePassDescriptor
    (pass as any).writeTimestamp(this.querySet, queryIndex);
  }

  /**
   * Resolve pending timestamp queries and update results
   * Call this after command buffer submission
   */
  async resolve(): Promise<void> {
    if (!this.hasTimestampQuery || this.pendingResolves.length === 0) {
      return;
    }

    if (!this.device || !this.querySet || !this.queryBuffer || !this.readbackBuffer) {
      console.warn('[GPUProfiler] Missing required resources for resolve');
      return;
    }

    const encoder = this.device.createCommandEncoder();

    // Resolve all timestamps to buffer
    const maxIndex = Math.max(...this.pendingResolves.map(p => p.endIndex)) + 1;
    encoder.resolveQuerySet(this.querySet, 0, maxIndex, this.queryBuffer, 0);

    // Copy to readback buffer
    encoder.copyBufferToBuffer(
      this.queryBuffer,
      0,
      this.readbackBuffer,
      0,
      maxIndex * 8
    );

    this.device.queue.submit([encoder.finish()]);

    if (!allowReadback('GPUProfiler.resolve')) {
      return;
    }

    // Read back timestamps
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(this.readbackBuffer.getMappedRange());

    // Process pending resolves
    for (const pending of this.pendingResolves) {
      const startNs = timestamps[pending.startIndex];
      const endNs = timestamps[pending.endIndex];

      // Convert nanoseconds to milliseconds
      const durationMs = Number(endNs - startNs) / 1_000_000;

      // Sanity check - use CPU timing if GPU timing seems wrong
      if (durationMs < 0 || durationMs > 60000) {
        // Fallback to CPU timing
        this._recordResult(pending.label, pending.cpuEndTime - pending.cpuStartTime);
      } else {
        this._recordResult(pending.label, durationMs);
      }
    }

    this.readbackBuffer.unmap();
    this.pendingResolves = [];
    this.nextQueryIndex = 0;
  }

  /**
   * Record a timing result
   * @private
   */
  private _recordResult(label: string, timeMs: number): void {
    if (!this.results.has(label)) {
      this.results.set(label, {
        times: [],
        min: Infinity,
        max: -Infinity,
        sum: 0,
        count: 0,
      });
    }

    const result = this.results.get(label)!;
    result.times.push(timeMs);
    result.min = Math.min(result.min, timeMs);
    result.max = Math.max(result.max, timeMs);
    result.sum += timeMs;
    result.count++;

    // Keep only last 100 samples for running average
    if (result.times.length > 100) {
      const removed = result.times.shift()!;
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
   */
  getResults(): Record<string, ProfileResult> {
    const output: Record<string, ProfileResult> = {};

    for (const [label, data] of this.results) {
      output[label] = {
        avg: data.sum / data.count,
        min: data.min,
        max: data.max,
        count: data.count,
        total: data.sum,
      };
    }

    return output;
  }

  /**
   * Get result for a specific label
   * @param label - Label to get result for
   */
  getResult(label: string): ProfileResult | null {
    const data = this.results.get(label);
    if (!data) return null;

    return {
      avg: data.sum / data.count,
      min: data.min,
      max: data.max,
      count: data.count,
      total: data.sum,
    };
  }

  /**
   * Reset all profiling data
   */
  reset(): void {
    this.results.clear();
    this.activeLabels.clear();
    this.pendingResolves = [];
    this.nextQueryIndex = 0;
  }

  /**
   * Get formatted report string
   */
  getReport(): string {
    const results = this.getResults();
    const labels = Object.keys(results).sort();

    if (labels.length === 0) {
      return 'No profiling data collected';
    }

    let report = 'GPU Profiler Results\n';
    report += '─'.repeat(60) + '\n';
    report += 'Label'.padEnd(30) + 'Avg (ms)'.padStart(10) + 'Min'.padStart(10) + 'Max'.padStart(10) + '\n';
    report += '─'.repeat(60) + '\n';

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
   */
  isGPUTimingAvailable(): boolean {
    return this.hasTimestampQuery;
  }

  /**
   * Destroy profiler resources
   */
  destroy(): void {
    if (this.querySet) {
      this.querySet.destroy();
      this.querySet = null;
    }
    if (this.queryBuffer) {
      this.queryBuffer.destroy();
      this.queryBuffer = null;
    }
    if (this.readbackBuffer) {
      this.readbackBuffer.destroy();
      this.readbackBuffer = null;
    }
    this.results.clear();
    this.activeLabels.clear();
  }
}

// Global profiler instance
let globalProfiler: GPUProfiler | null = null;

/**
 * Get the global profiler instance
 */
export function getProfiler(): GPUProfiler {
  if (!globalProfiler) {
    globalProfiler = new GPUProfiler();
  }
  return globalProfiler;
}

/**
 * Create a new profiler instance
 * @param device - Optional GPU device
 */
export function createProfiler(device?: GPUDevice | null): GPUProfiler {
  return new GPUProfiler(device);
}

/**
 * Convenience function to time a single operation
 * @param label - Label for the operation
 * @param fn - Async function to time
 */
export async function timeOperation<T>(
  label: string,
  fn: () => Promise<T>
): Promise<{ result: T; timeMs: number }> {
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
