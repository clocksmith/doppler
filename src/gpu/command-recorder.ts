/**
 * CommandRecorder - Batched GPU Command Recording
 *
 * Enables recording multiple GPU operations into a single command buffer,
 * avoiding per-kernel submit overhead. Manages temporary buffers automatically.
 *
 * Usage:
 *   const recorder = new CommandRecorder(device);
 *   recordMatmul(recorder, A, B, M, N, K);
 *   recordRMSNorm(recorder, input, weight, eps);
 *   // ... more operations
 *   await recorder.submit();  // Single GPU submission + cleanup
 *
 * Performance impact:
 *   Without batching: 260+ submits per forward pass (~50-100ms overhead)
 *   With batching: 1 submit per forward pass (~0.5ms overhead)
 *
 * Profiling mode:
 *   const recorder = new CommandRecorder(device, 'decode', { profile: true });
 *   // ... record operations ...
 *   recorder.submit();
 *   const timings = await recorder.resolveProfileTimings();
 *   log.info('CommandRecorder', 'Kernel timings', timings);
 */

import { getDevice, hasFeature, FEATURES } from './device.js';
import { allowReadback, trackAllocation } from './perf-guards.js';
import { getUniformCache } from './uniform-cache.js';
import { log } from '../debug/index.js';

/** Statistics about recorded operations */
export interface RecorderStats {
  opCount: number;
  tempBufferCount: number;
  submitted: boolean;
}

/** Options for CommandRecorder */
export interface RecorderOptions {
  /** Enable GPU timestamp profiling (requires 'timestamp-query' feature) */
  profile?: boolean;
}

/** Profiling timing entry */
interface ProfileEntry {
  label: string;
  startQueryIndex: number;
  endQueryIndex: number;
}

/** Profiling result - maps kernel label to time in milliseconds */
export type ProfileTimings = Record<string, number>;

/**
 * CommandRecorder wraps a GPUCommandEncoder and manages temporary resources.
 */
export class CommandRecorder {
  readonly device: GPUDevice;
  readonly label: string;
  private encoder: GPUCommandEncoder;

  /** Temporary buffers to destroy after submit */
  private tempBuffers: GPUBuffer[];

  /** Track if already submitted */
  private submitted: boolean;

  /** Operation count for debugging */
  private opCount: number;

  // Profiling state
  private profilingEnabled: boolean;
  private querySet: GPUQuerySet | null = null;
  private queryBuffer: GPUBuffer | null = null;
  private readbackBuffer: GPUBuffer | null = null;
  private profileEntries: ProfileEntry[] = [];
  private nextQueryIndex = 0;
  private static readonly MAX_QUERIES = 512; // 256 kernel pairs

  /**
   * @param device - GPU device (auto-detected if not provided)
   * @param label - Label for debugging
   * @param options - Recorder options (profiling, etc.)
   */
  constructor(device: GPUDevice | null = null, label: string = 'command_recorder', options: RecorderOptions = {}) {
    this.device = device || getDevice();
    if (!this.device) {
      throw new Error('[CommandRecorder] No GPU device available');
    }

    this.label = label;
    this.encoder = this.device.createCommandEncoder({ label });

    // Temporary buffers to destroy after submit
    this.tempBuffers = [];

    // Track if already submitted
    this.submitted = false;

    // Operation count for debugging
    this.opCount = 0;

    // Initialize profiling if requested and available
    this.profilingEnabled = options.profile === true && hasFeature(FEATURES.TIMESTAMP_QUERY);
    if (this.profilingEnabled) {
      this._initProfiling();
    }
  }

  /**
   * Initialize GPU timestamp query resources for profiling.
   * @private
   */
  private _initProfiling(): void {
    try {
      this.querySet = this.device.createQuerySet({
        type: 'timestamp',
        count: CommandRecorder.MAX_QUERIES,
      });

      // Buffer to hold query results (8 bytes per timestamp = BigUint64)
      this.queryBuffer = this.device.createBuffer({
        label: `${this.label}_query_buffer`,
        size: CommandRecorder.MAX_QUERIES * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });

      // Readback buffer
      this.readbackBuffer = this.device.createBuffer({
        label: `${this.label}_readback_buffer`,
        size: CommandRecorder.MAX_QUERIES * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    } catch (e) {
      log.warn('CommandRecorder', `Failed to initialize profiling: ${e}`);
      this.profilingEnabled = false;
    }
  }

  /**
   * Check if profiling is enabled and available.
   */
  isProfilingEnabled(): boolean {
    return this.profilingEnabled;
  }

  /**
   * Create a temporary buffer that will be destroyed after submit.
   * Use for uniform buffers and other per-operation temporaries.
   *
   * @param size - Buffer size in bytes
   * @param usage - Buffer usage flags
   * @param label - Buffer label for debugging
   * @returns GPUBuffer
   */
  createTempBuffer(size: number, usage: GPUBufferUsageFlags, label: string = 'temp_buffer'): GPUBuffer {
    if (this.submitted) {
      throw new Error('[CommandRecorder] Cannot create buffers after submit');
    }

    const buffer = this.device.createBuffer({
      label: `${this.label}_${label}_${this.tempBuffers.length}`,
      size,
      usage,
    });
    trackAllocation(size, label);

    this.tempBuffers.push(buffer);
    return buffer;
  }

  /**
   * Create an indirect dispatch buffer initialized with workgroup counts.
   * Buffer usage includes STORAGE so GPU kernels can update counts.
   */
  createIndirectDispatchBuffer(
    workgroups: [number, number, number] | Uint32Array = [0, 0, 0],
    label: string = 'indirect_dispatch'
  ): GPUBuffer {
    const data = workgroups instanceof Uint32Array
      ? workgroups
      : new Uint32Array(workgroups);
    const size = Math.max(12, data.byteLength);
    const buffer = this.createTempBuffer(
      size,
      GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label
    );
    const source = data.buffer as ArrayBuffer;
    this.device.queue.writeBuffer(buffer, 0, source, data.byteOffset, data.byteLength);
    return buffer;
  }

  /**
   * Update an indirect dispatch buffer with new workgroup counts.
   */
  writeIndirectDispatchBuffer(
    buffer: GPUBuffer,
    workgroups: [number, number, number] | Uint32Array,
    offset: number = 0
  ): void {
    if (this.submitted) {
      throw new Error('[CommandRecorder] Cannot write buffers after submit');
    }
    const data = workgroups instanceof Uint32Array
      ? workgroups
      : new Uint32Array(workgroups);
    const source = data.buffer as ArrayBuffer;
    this.device.queue.writeBuffer(buffer, offset, source, data.byteOffset, data.byteLength);
  }

  /**
   * Create a uniform buffer, write data, and track for cleanup.
   * Uses content-addressed caching for identical uniform data.
   *
   * @param data - Data to write
   * @param label - Buffer label
   * @returns GPUBuffer
   */
  createUniformBuffer(data: ArrayBuffer | ArrayBufferView, label: string = 'uniforms'): GPUBuffer {
    // Convert ArrayBufferView to ArrayBuffer for caching
    const arrayBuffer = data instanceof ArrayBuffer
      ? data
      : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);

    // Use content-addressed cache for uniform buffers
    // Cache handles creation, writeBuffer, and lifecycle - no cleanup needed
    return getUniformCache().getOrCreate(arrayBuffer, label);
  }

  /**
   * Begin a compute pass on the encoder.
   * When profiling is enabled, injects timestampWrites to measure GPU execution time.
   *
   * @param label - Pass label for debugging (used as key in profile results)
   * @returns GPUComputePassEncoder
   */
  beginComputePass(label: string = 'compute_pass'): GPUComputePassEncoder {
    if (this.submitted) {
      throw new Error('[CommandRecorder] Cannot begin pass after submit');
    }
    this.opCount++;

    const passLabel = `${this.label}_${label}_${this.opCount}`;

    // If profiling enabled, add timestamp writes
    if (this.profilingEnabled && this.querySet && this.nextQueryIndex + 2 <= CommandRecorder.MAX_QUERIES) {
      const startIndex = this.nextQueryIndex;
      const endIndex = startIndex + 1;
      this.nextQueryIndex += 2;

      // Track this entry for later resolution
      this.profileEntries.push({
        label,
        startQueryIndex: startIndex,
        endQueryIndex: endIndex,
      });

      return this.encoder.beginComputePass({
        label: passLabel,
        timestampWrites: {
          querySet: this.querySet,
          beginningOfPassWriteIndex: startIndex,
          endOfPassWriteIndex: endIndex,
        },
      });
    }

    // Non-profiling path
    return this.encoder.beginComputePass({
      label: passLabel,
    });
  }

  /**
   * Get the raw encoder for advanced use cases.
   * @returns GPUCommandEncoder
   */
  getEncoder(): GPUCommandEncoder {
    if (this.submitted) {
      throw new Error('[CommandRecorder] Cannot access encoder after submit');
    }
    return this.encoder;
  }

  /**
   * Track an externally created buffer for cleanup after submit.
   * Use for buffers created outside the recorder that need cleanup.
   *
   * @param buffer - Buffer to track for destruction
   */
  trackTemporaryBuffer(buffer: GPUBuffer): void {
    if (this.submitted) {
      throw new Error('[CommandRecorder] Cannot track buffers after submit');
    }
    this.tempBuffers.push(buffer);
  }

  /**
   * Submit all recorded commands and clean up temporary buffers.
   * After calling this, the recorder cannot be reused.
   */
  submit(): void {
    if (this.submitted) {
      throw new Error('[CommandRecorder] Already submitted');
    }

    // Submit commands
    this.device.queue.submit([this.encoder.finish()]);
    this.submitted = true;

    // Destroy temporary buffers
    for (const buffer of this.tempBuffers) {
      buffer.destroy();
    }
    this.tempBuffers = [];
  }

  /**
   * Submit and wait for GPU to complete (useful for debugging/profiling).
   * Also flushes the uniform cache's pending destruction queue to clean up
   * any evicted buffers that were referenced by this command buffer.
   * @returns Promise that resolves when GPU work is done
   */
  async submitAndWait(): Promise<void> {
    this.submit();
    await this.device.queue.onSubmittedWorkDone();
    // Safe to destroy evicted uniform buffers now that GPU work is complete
    getUniformCache().flushPendingDestruction();
  }

  /**
   * Get statistics about recorded operations.
   * @returns Statistics object
   */
  getStats(): RecorderStats {
    return {
      opCount: this.opCount,
      tempBufferCount: this.tempBuffers.length,
      submitted: this.submitted,
    };
  }

  /**
   * Abort recording without submitting (cleanup only).
   * Use if an error occurs during recording.
   */
  abort(): void {
    if (this.submitted) return;

    // Destroy temp buffers without submitting
    for (const buffer of this.tempBuffers) {
      buffer.destroy();
    }
    this.tempBuffers = [];
    this._destroyProfilingResources();
    this.submitted = true; // Prevent further use
  }

  /**
   * Resolve profiling timestamps and return per-kernel timings.
   * Must be called after submit() and GPU work is done.
   *
   * Returns a map of kernel label to execution time in milliseconds.
   * Labels with multiple invocations are aggregated (e.g., 'matmul' across all layers).
   *
   * @returns Promise resolving to timing map, or null if profiling not enabled
   */
  async resolveProfileTimings(): Promise<ProfileTimings | null> {
    if (!this.profilingEnabled || !this.querySet || !this.queryBuffer || !this.readbackBuffer) {
      return null;
    }

    if (!this.submitted) {
      throw new Error('[CommandRecorder] Must submit before resolving timings');
    }

    if (this.profileEntries.length === 0) {
      return {};
    }

    // Wait for GPU work to complete
    await this.device.queue.onSubmittedWorkDone();

    // Resolve queries to buffer
    const maxIndex = Math.max(...this.profileEntries.map(e => e.endQueryIndex)) + 1;
    const resolveEncoder = this.device.createCommandEncoder({ label: 'profile_resolve' });
    resolveEncoder.resolveQuerySet(this.querySet, 0, maxIndex, this.queryBuffer, 0);
    resolveEncoder.copyBufferToBuffer(this.queryBuffer, 0, this.readbackBuffer, 0, maxIndex * 8);
    this.device.queue.submit([resolveEncoder.finish()]);

    if (!allowReadback('CommandRecorder.resolveProfileTimings')) {
      return null;
    }

    // Read back timestamps
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(this.readbackBuffer.getMappedRange());

    // Aggregate timings by label
    const timings: ProfileTimings = {};

    for (const entry of this.profileEntries) {
      const startNs = timestamps[entry.startQueryIndex];
      const endNs = timestamps[entry.endQueryIndex];
      const durationMs = Number(endNs - startNs) / 1_000_000;

      // Skip invalid timings
      if (durationMs < 0 || durationMs > 60000) {
        continue;
      }

      // Aggregate by label
      if (timings[entry.label] !== undefined) {
        timings[entry.label] += durationMs;
      } else {
        timings[entry.label] = durationMs;
      }
    }

    this.readbackBuffer.unmap();

    // Clean up profiling resources after use
    this._destroyProfilingResources();

    return timings;
  }

  /**
   * Get a formatted profiling report.
   * Must be called after resolveProfileTimings().
   *
   * @param timings - Timings from resolveProfileTimings()
   * @returns Formatted string report
   */
  static formatProfileReport(timings: ProfileTimings): string {
    const entries = Object.entries(timings).sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((sum, [, t]) => sum + t, 0);

    let report = 'GPU Profile Report\n';
    report += '─'.repeat(50) + '\n';
    report += 'Kernel'.padEnd(25) + 'Time (ms)'.padStart(12) + '%'.padStart(8) + '\n';
    report += '─'.repeat(50) + '\n';

    for (const [label, time] of entries) {
      const pct = (time / total * 100).toFixed(1);
      report += label.padEnd(25) + time.toFixed(2).padStart(12) + pct.padStart(8) + '\n';
    }

    report += '─'.repeat(50) + '\n';
    report += 'TOTAL'.padEnd(25) + total.toFixed(2).padStart(12) + '100.0'.padStart(8) + '\n';

    return report;
  }

  /**
   * Clean up profiling resources.
   * @private
   */
  private _destroyProfilingResources(): void {
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
    this.profileEntries = [];
  }
}

/**
 * Create a new CommandRecorder.
 * @param label - Label for debugging
 * @param options - Recorder options
 * @returns CommandRecorder instance
 */
export function createCommandRecorder(label: string = 'command_recorder', options?: RecorderOptions): CommandRecorder {
  return new CommandRecorder(null, label, options);
}

/**
 * Create a profiling-enabled CommandRecorder.
 * Falls back to non-profiling if timestamp-query not available.
 *
 * @param label - Label for debugging
 * @returns CommandRecorder with profiling enabled
 */
export function createProfilingRecorder(label: string = 'profiled_recorder'): CommandRecorder {
  return new CommandRecorder(null, label, { profile: true });
}

export default CommandRecorder;
