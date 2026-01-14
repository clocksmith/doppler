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

let didLogQueryClamp = false;
let didLogQueryFallback = false;

/**
 * @typedef {Object} ProfileEntry
 * @property {string} label
 * @property {number} startQueryIndex
 * @property {number} endQueryIndex
 */

/**
 * CommandRecorder wraps a GPUCommandEncoder and manages temporary resources.
 */
export class CommandRecorder {
  /** @type {GPUDevice} */
  device;
  /** @type {string} */
  label;
  /** @type {GPUCommandEncoder} */
  #encoder;

  /** @type {GPUBuffer[]} Temporary buffers to destroy after submit */
  #tempBuffers;
  /** @type {Promise<void> | null} */
  #cleanupPromise = null;

  /** @type {boolean} Track if already submitted */
  #submitted;

  /** @type {number} Operation count for debugging */
  #opCount;

  // Profiling state
  /** @type {boolean} */
  #profilingEnabled;
  /** @type {GPUQuerySet | null} */
  #querySet = null;
  /** @type {GPUBuffer | null} */
  #queryBuffer = null;
  /** @type {GPUBuffer | null} */
  #readbackBuffer = null;
  /** @type {ProfileEntry[]} */
  #profileEntries = [];
  /** @type {number} */
  #nextQueryIndex = 0;
  /** @type {number} */
  #queryCapacity = 0;
  /** @type {number} */
  static MAX_QUERIES = 16384; // Upper bound; device limits may be lower.
  static DEFAULT_QUERY_LIMIT = 4096; // Safe fallback when maxQuerySetSize is unavailable.

  /**
   * @param {GPUDevice | null} [device] - GPU device (auto-detected if not provided)
   * @param {string} [label] - Label for debugging
   * @param {import('./command-recorder.js').RecorderOptions} [options] - Recorder options (profiling, etc.)
   */
  constructor(device = null, label = 'command_recorder', options = {}) {
    this.device = device || getDevice();
    if (!this.device) {
      throw new Error('[CommandRecorder] No GPU device available');
    }

    this.label = label;
    this.#encoder = this.device.createCommandEncoder({ label });

    // Temporary buffers to destroy after submit
    this.#tempBuffers = [];
    this.#cleanupPromise = null;

    // Track if already submitted
    this.#submitted = false;

    // Operation count for debugging
    this.#opCount = 0;

    // Initialize profiling if requested and available
    this.#profilingEnabled = options.profile === true && hasFeature(FEATURES.TIMESTAMP_QUERY);
    if (this.#profilingEnabled) {
      this.#initProfiling();
    }
  }

  /**
   * Initialize GPU timestamp query resources for profiling.
   */
  #initProfiling() {
    try {
      const deviceLimit = this.device.limits?.maxQuerySetSize;
      const hasDeviceLimit = Number.isFinite(deviceLimit) && deviceLimit > 0;
      const limit = hasDeviceLimit
        ? deviceLimit
        : CommandRecorder.DEFAULT_QUERY_LIMIT;
      this.#queryCapacity = Math.min(CommandRecorder.MAX_QUERIES, limit);
      if (hasDeviceLimit && this.#queryCapacity < CommandRecorder.MAX_QUERIES && !didLogQueryClamp) {
        log.warn(
          'CommandRecorder',
          `Clamping MAX_QUERIES to device limit: ${this.#queryCapacity}/${CommandRecorder.MAX_QUERIES}`
        );
        didLogQueryClamp = true;
      } else if (!hasDeviceLimit && !didLogQueryFallback) {
        log.warn(
          'CommandRecorder',
          `maxQuerySetSize unavailable; using fallback ${CommandRecorder.DEFAULT_QUERY_LIMIT}`
        );
        didLogQueryFallback = true;
      }

      this.#querySet = this.device.createQuerySet({
        type: 'timestamp',
        count: this.#queryCapacity,
      });

      // Buffer to hold query results (8 bytes per timestamp = BigUint64)
      this.#queryBuffer = this.device.createBuffer({
        label: `${this.label}_query_buffer`,
        size: this.#queryCapacity * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });

      // Readback buffer
      this.#readbackBuffer = this.device.createBuffer({
        label: `${this.label}_readback_buffer`,
        size: this.#queryCapacity * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    } catch (e) {
      log.warn('CommandRecorder', `Failed to initialize profiling: ${e}`);
      this.#profilingEnabled = false;
    }
  }

  /**
   * Check if profiling is enabled and available.
   * @returns {boolean}
   */
  isProfilingEnabled() {
    return this.#profilingEnabled;
  }

  /**
   * Create a temporary buffer that will be destroyed after submit.
   * Use for uniform buffers and other per-operation temporaries.
   *
   * @param {number} size - Buffer size in bytes
   * @param {GPUBufferUsageFlags} usage - Buffer usage flags
   * @param {string} [label] - Buffer label for debugging
   * @returns {GPUBuffer}
   */
  createTempBuffer(size, usage, label = 'temp_buffer') {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot create buffers after submit');
    }

    const buffer = this.device.createBuffer({
      label: `${this.label}_${label}_${this.#tempBuffers.length}`,
      size,
      usage,
    });
    trackAllocation(size, label);

    this.#tempBuffers.push(buffer);
    return buffer;
  }

  /**
   * Create an indirect dispatch buffer initialized with workgroup counts.
   * Buffer usage includes STORAGE so GPU kernels can update counts.
   * @param {[number, number, number] | Uint32Array} [workgroups]
   * @param {string} [label]
   * @returns {GPUBuffer}
   */
  createIndirectDispatchBuffer(
    workgroups = [0, 0, 0],
    label = 'indirect_dispatch'
  ) {
    const data = workgroups instanceof Uint32Array
      ? workgroups
      : new Uint32Array(workgroups);
    const size = Math.max(12, data.byteLength);
    const buffer = this.createTempBuffer(
      size,
      GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label
    );
    const source = /** @type {ArrayBuffer} */ (data.buffer);
    this.device.queue.writeBuffer(buffer, 0, source, data.byteOffset, data.byteLength);
    return buffer;
  }

  /**
   * Update an indirect dispatch buffer with new workgroup counts.
   * @param {GPUBuffer} buffer
   * @param {[number, number, number] | Uint32Array} workgroups
   * @param {number} [offset]
   * @returns {void}
   */
  writeIndirectDispatchBuffer(
    buffer,
    workgroups,
    offset = 0
  ) {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot write buffers after submit');
    }
    const data = workgroups instanceof Uint32Array
      ? workgroups
      : new Uint32Array(workgroups);
    const source = /** @type {ArrayBuffer} */ (data.buffer);
    this.device.queue.writeBuffer(buffer, offset, source, data.byteOffset, data.byteLength);
  }

  /**
   * Create a uniform buffer, write data, and track for cleanup.
   * Uses content-addressed caching for identical uniform data.
   *
   * @param {ArrayBuffer | ArrayBufferView} data - Data to write
   * @param {string} [label] - Buffer label
   * @returns {GPUBuffer}
   */
  createUniformBuffer(data, label = 'uniforms') {
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
   * @param {string} [label] - Pass label for debugging (used as key in profile results)
   * @returns {GPUComputePassEncoder}
   */
  beginComputePass(label = 'compute_pass') {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot begin pass after submit');
    }
    this.#opCount++;

    const passLabel = `${this.label}_${label}_${this.#opCount}`;

    // If profiling enabled, add timestamp writes
    if (this.#profilingEnabled && this.#querySet && this.#nextQueryIndex + 2 <= this.#queryCapacity) {
      const startIndex = this.#nextQueryIndex;
      const endIndex = startIndex + 1;
      this.#nextQueryIndex += 2;

      // Track this entry for later resolution
      this.#profileEntries.push({
        label,
        startQueryIndex: startIndex,
        endQueryIndex: endIndex,
      });

      return this.#encoder.beginComputePass({
        label: passLabel,
        timestampWrites: {
          querySet: this.#querySet,
          beginningOfPassWriteIndex: startIndex,
          endOfPassWriteIndex: endIndex,
        },
      });
    }

    // Non-profiling path
    return this.#encoder.beginComputePass({
      label: passLabel,
    });
  }

  /**
   * Get the raw encoder for advanced use cases.
   * @returns {GPUCommandEncoder}
   */
  getEncoder() {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot access encoder after submit');
    }
    return this.#encoder;
  }

  /**
   * Track an externally created buffer for cleanup after submit.
   * Use for buffers created outside the recorder that need cleanup.
   *
   * @param {GPUBuffer} buffer - Buffer to track for destruction
   * @returns {void}
   */
  trackTemporaryBuffer(buffer) {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot track buffers after submit');
    }
    this.#tempBuffers.push(buffer);
  }

  /**
   * Submit all recorded commands and clean up temporary buffers.
   * After calling this, the recorder cannot be reused.
   * @returns {void}
   */
  submit() {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Already submitted');
    }

    // Submit commands
    this.device.queue.submit([this.#encoder.finish()]);
    this.#submitted = true;

    const buffersToDestroy = this.#tempBuffers;
    this.#tempBuffers = [];

    this.#cleanupPromise = this.device.queue.onSubmittedWorkDone().then(() => {
      for (const buffer of buffersToDestroy) {
        buffer.destroy();
      }
      // Safe to destroy evicted uniform buffers now that GPU work is complete
      getUniformCache().flushPendingDestruction();
    }).catch((err) => {
      log.warn('CommandRecorder', `Deferred cleanup failed: ${/** @type {Error} */ (err).message}`);
    });
  }

  /**
   * Submit and wait for GPU to complete (useful for debugging/profiling).
   * Also flushes the uniform cache's pending destruction queue to clean up
   * any evicted buffers that were referenced by this command buffer.
   * @returns {Promise<void>}
   */
  async submitAndWait() {
    this.submit();
    if (this.#cleanupPromise) {
      await this.#cleanupPromise;
    } else {
      await this.device.queue.onSubmittedWorkDone();
      // Safe to destroy evicted uniform buffers now that GPU work is complete
      getUniformCache().flushPendingDestruction();
    }
  }

  /**
   * Get statistics about recorded operations.
   * @returns {import('./command-recorder.js').RecorderStats}
   */
  getStats() {
    return {
      opCount: this.#opCount,
      tempBufferCount: this.#tempBuffers.length,
      submitted: this.#submitted,
    };
  }

  /**
   * Abort recording without submitting (cleanup only).
   * Use if an error occurs during recording.
   * @returns {void}
   */
  abort() {
    if (this.#submitted) return;

    // Destroy temp buffers without submitting
    for (const buffer of this.#tempBuffers) {
      buffer.destroy();
    }
    this.#tempBuffers = [];
    this.#destroyProfilingResources();
    this.#submitted = true; // Prevent further use
  }

  /**
   * Resolve profiling timestamps and return per-kernel timings.
   * Must be called after submit() and GPU work is done.
   *
   * Returns a map of kernel label to execution time in milliseconds.
   * Labels with multiple invocations are aggregated (e.g., 'matmul' across all layers).
   *
   * @returns {Promise<import('./command-recorder.js').ProfileTimings | null>}
   */
  async resolveProfileTimings() {
    if (!this.#profilingEnabled || !this.#querySet || !this.#queryBuffer || !this.#readbackBuffer) {
      return null;
    }

    if (!this.#submitted) {
      throw new Error('[CommandRecorder] Must submit before resolving timings');
    }

    if (this.#profileEntries.length === 0) {
      return {};
    }

    // Wait for GPU work to complete
    await this.device.queue.onSubmittedWorkDone();

    // Resolve queries to buffer
    const maxIndex = Math.max(...this.#profileEntries.map(e => e.endQueryIndex)) + 1;
    const resolveEncoder = this.device.createCommandEncoder({ label: 'profile_resolve' });
    resolveEncoder.resolveQuerySet(this.#querySet, 0, maxIndex, this.#queryBuffer, 0);
    resolveEncoder.copyBufferToBuffer(this.#queryBuffer, 0, this.#readbackBuffer, 0, maxIndex * 8);
    this.device.queue.submit([resolveEncoder.finish()]);

    if (!allowReadback('CommandRecorder.resolveProfileTimings')) {
      return null;
    }

    // Read back timestamps
    await this.#readbackBuffer.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(this.#readbackBuffer.getMappedRange());

    // Aggregate timings by label
    /** @type {import('./command-recorder.js').ProfileTimings} */
    const timings = {};

    for (const entry of this.#profileEntries) {
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

    this.#readbackBuffer.unmap();

    // Clean up profiling resources after use
    this.#destroyProfilingResources();

    return timings;
  }

  /**
   * Get a formatted profiling report.
   * Must be called after resolveProfileTimings().
   *
   * @param {import('./command-recorder.js').ProfileTimings} timings - Timings from resolveProfileTimings()
   * @returns {string}
   */
  static formatProfileReport(timings) {
    const entries = Object.entries(timings).sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((sum, [, t]) => sum + t, 0);

    let report = 'GPU Profile Report\n';
    report += '\u2500'.repeat(50) + '\n';
    report += 'Kernel'.padEnd(25) + 'Time (ms)'.padStart(12) + '%'.padStart(8) + '\n';
    report += '\u2500'.repeat(50) + '\n';

    for (const [label, time] of entries) {
      const pct = (time / total * 100).toFixed(1);
      report += label.padEnd(25) + time.toFixed(2).padStart(12) + pct.padStart(8) + '\n';
    }

    report += '\u2500'.repeat(50) + '\n';
    report += 'TOTAL'.padEnd(25) + total.toFixed(2).padStart(12) + '100.0'.padStart(8) + '\n';

    return report;
  }

  /**
   * Clean up profiling resources.
   */
  #destroyProfilingResources() {
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
    this.#profileEntries = [];
  }
}

/**
 * Create a new CommandRecorder.
 * @param {string} [label] - Label for debugging
 * @param {import('./command-recorder.js').RecorderOptions} [options] - Recorder options
 * @returns {CommandRecorder}
 */
export function createCommandRecorder(label = 'command_recorder', options) {
  return new CommandRecorder(null, label, options);
}

/**
 * Create a profiling-enabled CommandRecorder.
 * Falls back to non-profiling if timestamp-query not available.
 *
 * @param {string} [label] - Label for debugging
 * @returns {CommandRecorder}
 */
export function createProfilingRecorder(label = 'profiled_recorder') {
  return new CommandRecorder(null, label, { profile: true });
}

export default CommandRecorder;
