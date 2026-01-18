

import { getDevice, hasFeature, FEATURES } from './device.js';
import { allowReadback, trackAllocation } from './perf-guards.js';
import { getUniformCache } from './uniform-cache.js';
import { releaseBuffer } from '../memory/buffer-pool.js';
import { log } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

let didLogQueryClamp = false;
let didLogQueryFallback = false;




export class CommandRecorder {
  
  device;
  
  label;
  
  #encoder;


  #tempBuffers;

  // Pooled buffers to release (not destroy) after submit

  #pooledBuffers;

  #cleanupPromise = null;

  
  #submitted;

  
  #opCount;

  // Profiling state
  
  #profilingEnabled;
  
  #querySet = null;
  
  #queryBuffer = null;
  
  #readbackBuffer = null;
  
  #profileEntries = [];
  
  #nextQueryIndex = 0;
  
  #queryCapacity = 0;
  
  
  constructor(device = null, label = 'command_recorder', options = {}) {
    this.device = device || getDevice();
    if (!this.device) {
      throw new Error('[CommandRecorder] No GPU device available');
    }

    this.label = label;
    this.#encoder = this.device.createCommandEncoder({ label });

    // Temporary buffers to destroy after submit (created directly by recorder)
    this.#tempBuffers = [];
    // Pooled buffers to release after submit (came from buffer pool)
    this.#pooledBuffers = [];
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

  
  #initProfiling() {
    try {
      const runtimeProfiler = getRuntimeConfig().shared?.debug?.profiler;
      if (!runtimeProfiler) {
        throw new Error('runtime.shared.debug.profiler is required.');
      }
      const { maxQueries, defaultQueryLimit } = runtimeProfiler;
      const deviceLimit = this.device.limits?.maxQuerySetSize;
      const hasDeviceLimit = Number.isFinite(deviceLimit) && deviceLimit > 0;
      const limit = hasDeviceLimit
        ? deviceLimit
        : defaultQueryLimit;
      this.#queryCapacity = Math.min(maxQueries, limit);
      if (hasDeviceLimit && this.#queryCapacity < maxQueries && !didLogQueryClamp) {
        log.warn(
          'CommandRecorder',
          `Clamping MAX_QUERIES to device limit: ${this.#queryCapacity}/${maxQueries}`
        );
        didLogQueryClamp = true;
      } else if (!hasDeviceLimit && !didLogQueryFallback) {
        log.warn(
          'CommandRecorder',
          `maxQuerySetSize unavailable; using fallback ${defaultQueryLimit}`
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

  
  isProfilingEnabled() {
    return this.#profilingEnabled;
  }

  
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
    const source =  (data.buffer);
    this.device.queue.writeBuffer(buffer, 0, source, data.byteOffset, data.byteLength);
    return buffer;
  }

  
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
    const source =  (data.buffer);
    this.device.queue.writeBuffer(buffer, offset, source, data.byteOffset, data.byteLength);
  }

  
  createUniformBuffer(data, label = 'uniforms') {
    // Convert ArrayBufferView to ArrayBuffer for caching
    const arrayBuffer = data instanceof ArrayBuffer
      ? data
      : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);

    // Use content-addressed cache for uniform buffers
    // Cache handles creation, writeBuffer, and lifecycle - no cleanup needed
    return getUniformCache().getOrCreate(arrayBuffer, label);
  }

  
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

  
  getEncoder() {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot access encoder after submit');
    }
    return this.#encoder;
  }


  trackTemporaryBuffer(buffer) {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Cannot track buffers after submit');
    }
    // Track as pooled buffer - will be released back to pool on cleanup
    this.#pooledBuffers.push(buffer);
  }

  
  submit() {
    if (this.#submitted) {
      throw new Error('[CommandRecorder] Already submitted');
    }

    // Submit commands
    this.device.queue.submit([this.#encoder.finish()]);
    this.#submitted = true;

    const buffersToDestroy = this.#tempBuffers;
    const buffersToRelease = this.#pooledBuffers;
    this.#tempBuffers = [];
    this.#pooledBuffers = [];

    this.#cleanupPromise = this.device.queue.onSubmittedWorkDone().then(() => {
      // Destroy buffers created directly by the recorder
      for (const buffer of buffersToDestroy) {
        buffer.destroy();
      }
      // Release pooled buffers back to the pool
      for (const buffer of buffersToRelease) {
        releaseBuffer(buffer);
      }
      // Safe to destroy evicted uniform buffers now that GPU work is complete
      getUniformCache().flushPendingDestruction();
    }).catch((err) => {
      log.warn('CommandRecorder', `Deferred cleanup failed: ${ (err).message}`);
    });
  }

  
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


  getStats() {
    return {
      opCount: this.#opCount,
      tempBufferCount: this.#tempBuffers.length,
      pooledBufferCount: this.#pooledBuffers.length,
      submitted: this.#submitted,
    };
  }


  abort() {
    if (this.#submitted) return;

    // Destroy temp buffers without submitting
    for (const buffer of this.#tempBuffers) {
      buffer.destroy();
    }
    // Release pooled buffers back to pool
    for (const buffer of this.#pooledBuffers) {
      releaseBuffer(buffer);
    }
    this.#tempBuffers = [];
    this.#pooledBuffers = [];
    this.#destroyProfilingResources();
    this.#submitted = true; // Prevent further use
  }

  
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


export function createCommandRecorder(label = 'command_recorder', options) {
  return new CommandRecorder(null, label, options);
}


export function createProfilingRecorder(label = 'profiled_recorder') {
  return new CommandRecorder(null, label, { profile: true });
}

export default CommandRecorder;
