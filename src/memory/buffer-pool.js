

import { getDevice, getDeviceLimits } from '../gpu/device.js';
import { allowReadback, trackAllocation } from '../gpu/perf-guards.js';
import { log, trace, isTraceEnabled } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';


export const BufferUsage =  ({
  STORAGE: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  STORAGE_READ: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  UNIFORM: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  STAGING_READ: GPUMapMode.READ | GPUBufferUsage.COPY_DST,
  STAGING_WRITE: GPUMapMode.WRITE | GPUBufferUsage.COPY_SRC,
});


function alignTo(size, alignment) {
  return Math.ceil(size / alignment) * alignment;
}


function getSizeBucket(
  size,
  maxAllowedSize = Infinity,
  bucketConfig = getRuntimeConfig().shared.bufferPool.bucket
) {
  // Minimum bucket from config
  const minBucket = bucketConfig.minBucketSizeBytes;
  if (size <= minBucket) return minBucket;

  // Avoid power-of-two rounding for very large buffers.
  // For weights and large activations, rounding 600MB -> 1GB can cause OOM even when the
  // exact-sized buffer would fit. Use coarse-grained bucketing to retain most pooling
  // benefits without 2x blowups.
  const largeThreshold = bucketConfig.largeBufferThresholdBytes;
  if (size >= largeThreshold) {
    const largeStep = bucketConfig.largeBufferStepBytes;
    const bucket = Math.ceil(size / largeStep) * largeStep;
    if (bucket > maxAllowedSize) {
      return alignTo(size, minBucket);
    }
    return bucket;
  }

  // Round up to next power of 2
  // Use Math.pow instead of bit shift to avoid 32-bit signed integer overflow
  // (1 << 31 = -2147483648 in JavaScript due to signed 32-bit arithmetic)
  const bits = 32 - Math.clz32(size - 1);
  const bucket = Math.pow(2, bits);

  // If bucket exceeds device limit, fall back to aligned size
  if (bucket > maxAllowedSize) {
    return alignTo(size, minBucket);
  }
  return bucket;
}


export class BufferPool {
  // Pools organized by usage and size bucket
  
  #pools;

  // Active buffers (currently in use)
  
  #activeBuffers;

  // Buffer metadata for leak detection (debug mode)
  
  #bufferMetadata;

  // Requested sizes per buffer (unbucketed intent)

  #requestedSizes;

  // Buffer ID tracking (for trace)

  #bufferIds;

  #bufferLabels;

  #nextBufferId;

  // Deferred destruction queue (buffers destroyed after GPU work completes)
  
  #pendingDestruction;
  
  #destructionScheduled;

  // Statistics
  
  #stats;

  // Configuration
  
  #config;

  // Schema-based configuration
  
  #schemaConfig;

  // Debug mode flag
  
  #debugMode;

  
  constructor(debugMode = false, schemaConfig) {
    this.#pools = new Map();
    this.#activeBuffers = new Set();
    this.#bufferMetadata = new Map();
    this.#requestedSizes = new Map();
    this.#bufferIds = new WeakMap();
    this.#bufferLabels = new WeakMap();
    this.#nextBufferId = 1;
    this.#debugMode = debugMode;
    this.#schemaConfig = schemaConfig ?? getRuntimeConfig().shared.bufferPool;
    this.#pendingDestruction = new Set();
    this.#destructionScheduled = false;

    this.#stats = {
      allocations: 0,
      reuses: 0,
      totalBytesAllocated: 0,
      peakBytesAllocated: 0,
      currentBytesAllocated: 0,
      totalBytesRequested: 0,
      peakBytesRequested: 0,
      currentBytesRequested: 0,
    };

    // Initialize from schema config
    this.#config = {
      maxPoolSizePerBucket: this.#schemaConfig.limits.maxBuffersPerBucket,
      maxTotalPooledBuffers: this.#schemaConfig.limits.maxTotalPooledBuffers,
      enablePooling: true,
      alignmentBytes: this.#schemaConfig.alignment.alignmentBytes,
    };
  }

  
  acquire(size, usage = BufferUsage.STORAGE, label = 'pooled_buffer') {
    const device = getDevice();
    if (!device) {
      throw new Error('Device not initialized');
    }

    // Check device limits before allocation
    const limits = getDeviceLimits();
    const maxSize = limits?.maxBufferSize || Infinity;
    const maxStorageSize = limits?.maxStorageBufferBindingSize || Infinity;
    const isStorageBuffer = (usage & GPUBufferUsage.STORAGE) !== 0;

    // Align size and compute bucket, respecting device limits
    const alignedSize = alignTo(size, this.#config.alignmentBytes);
    const maxAllowedBucket = isStorageBuffer ? Math.min(maxSize, maxStorageSize) : maxSize;
    const bucket = getSizeBucket(alignedSize, maxAllowedBucket, this.#schemaConfig.bucket);

    if (bucket > maxSize) {
      throw new Error(
        `Buffer size ${bucket} exceeds device maxBufferSize (${maxSize}). ` +
        `Requested: ${size} bytes, bucketed to: ${bucket} bytes.`
      );
    }

    if (isStorageBuffer && bucket > maxStorageSize) {
      throw new Error(
        `Storage buffer size ${bucket} exceeds device maxStorageBufferBindingSize (${maxStorageSize}). ` +
        `Consider splitting into smaller buffers or using a different strategy.`
      );
    }

    // Try to get from pool
    if (this.#config.enablePooling) {
      const pooled = this.#getFromPool(bucket, usage);
      if (pooled) {
        this.#activeBuffers.add(pooled);
        this.#stats.reuses++;
        this.#requestedSizes.set(pooled, alignedSize);
        this.#bufferLabels.set(pooled, label);
        this.#stats.currentBytesRequested += alignedSize;
        this.#stats.peakBytesRequested = Math.max(
          this.#stats.peakBytesRequested,
          this.#stats.currentBytesRequested
        );

        // Track metadata in debug mode
        if (this.#debugMode) {
          this.#trackBuffer(pooled, bucket, usage, label);
        }

        this.#traceAcquire(pooled, bucket, alignedSize, usage, label, true);

        return pooled;
      }
    }

    // Allocate new buffer
    const buffer = device.createBuffer({
      label: `${label}_${bucket}`,
      size: bucket,
      usage,
    });

    this.#activeBuffers.add(buffer);
    this.#stats.allocations++;
    this.#stats.totalBytesAllocated += bucket;
    this.#stats.currentBytesAllocated += bucket;
    this.#stats.peakBytesAllocated = Math.max(
      this.#stats.peakBytesAllocated,
      this.#stats.currentBytesAllocated
    );
    this.#stats.totalBytesRequested += alignedSize;
    this.#stats.currentBytesRequested += alignedSize;
    this.#stats.peakBytesRequested = Math.max(
      this.#stats.peakBytesRequested,
      this.#stats.currentBytesRequested
    );
    trackAllocation(bucket, label);
    this.#requestedSizes.set(buffer, alignedSize);
    this.#bufferLabels.set(buffer, label);

    // Track metadata in debug mode
    if (this.#debugMode) {
      this.#trackBuffer(buffer, bucket, usage, label);
    }

    this.#traceAcquire(buffer, bucket, alignedSize, usage, label, false);

    return buffer;
  }

  
  release(buffer) {
    if (!this.#activeBuffers.has(buffer)) {
      log.warn('BufferPool', 'Releasing buffer not tracked as active');
      return;
    }

    this.#activeBuffers.delete(buffer);
    const requestedSize = this.#requestedSizes.get(buffer) ?? 0;
    this.#stats.currentBytesRequested -= requestedSize;
    this.#requestedSizes.delete(buffer);

    // Remove metadata in debug mode
    if (this.#debugMode) {
      this.#bufferMetadata.delete(buffer);
    }

    if (!this.#config.enablePooling) {
      this.#deferDestroy(buffer);
      this.#stats.currentBytesAllocated -= buffer.size;
      this.#traceRelease(buffer, requestedSize, false);
      return;
    }

    // Return to pool if there's room
    const bucket = buffer.size;
    const usage = buffer.usage;

    if (!this.#pools.has(usage)) {
      this.#pools.set(usage, new Map());
    }
    const usagePool = this.#pools.get(usage);

    if (!usagePool.has(bucket)) {
      usagePool.set(bucket, []);
    }
    const bucketPool = usagePool.get(bucket);

    let pooled = false;
    if (bucketPool.length < this.#config.maxPoolSizePerBucket &&
        this.#getTotalPooledCount() < this.#config.maxTotalPooledBuffers) {
      bucketPool.push(buffer);
      pooled = true;
    } else {
      // Pool is full; defer destruction until GPU work completes.
      this.#deferDestroy(buffer);
      this.#stats.currentBytesAllocated -= buffer.size;
    }

    this.#traceRelease(buffer, requestedSize, pooled);
  }

  
  isActiveBuffer(buffer) {
    return this.#activeBuffers.has(buffer);
  }

  
  getRequestedSize(buffer) {
    return this.#requestedSizes.get(buffer) ?? buffer.size;
  }

  
  #deferDestroy(buffer) {
    this.#pendingDestruction.add(buffer);
    if (this.#destructionScheduled) {
      return;
    }
    const device = getDevice();
    if (!device) {
      // No device context; destroy immediately as a fallback.
      for (const pending of this.#pendingDestruction) {
        pending.destroy();
      }
      this.#pendingDestruction.clear();
      this.#destructionScheduled = false;
      return;
    }

    this.#destructionScheduled = true;
    device.queue.onSubmittedWorkDone()
      .then(() => {
        for (const pending of this.#pendingDestruction) {
          pending.destroy();
        }
        this.#pendingDestruction.clear();
        this.#destructionScheduled = false;
      })
      .catch((err) => {
        log.warn('BufferPool', `Deferred destruction failed: ${ (err).message}`);
        this.#pendingDestruction.clear();
        this.#destructionScheduled = false;
      });
  }

  
  #getFromPool(bucket, usage) {
    const usagePool = this.#pools.get(usage);
    if (!usagePool) return null;

    const bucketPool = usagePool.get(bucket);
    if (!bucketPool || bucketPool.length === 0) return null;

    return bucketPool.pop();
  }

  
  #getTotalPooledCount() {
    let count = 0;
    for (const usagePool of this.#pools.values()) {
      for (const bucketPool of usagePool.values()) {
        count += bucketPool.length;
      }
    }
    return count;
  }

  
  #trackBuffer(buffer, size, usage, label) {
    
    const metadata = {
      size,
      usage,
      label,
      acquiredAt: Date.now(),
    };

    // Capture stack trace for leak detection
    if (Error.captureStackTrace) {
      const obj = {};
      Error.captureStackTrace(obj);
      metadata.stackTrace =  (obj).stack;
    }

    this.#bufferMetadata.set(buffer, metadata);
  }

  #getBufferId(buffer) {
    let id = this.#bufferIds.get(buffer);
    if (!id) {
      id = this.#nextBufferId++;
      this.#bufferIds.set(buffer, id);
    }
    return id;
  }

  #traceAcquire(buffer, bucket, requestedSize, usage, label, reused) {
    if (!isTraceEnabled('buffers')) {
      return;
    }
    const id = this.#getBufferId(buffer);
    const mode = reused ? 'reuse' : 'new';
    trace.buffers(
      `Acquire ${mode} id=${id} size=${bucket} req=${requestedSize} usage=${usage} label=${label}`
    );
  }

  #traceRelease(buffer, requestedSize, pooled) {
    if (!isTraceEnabled('buffers')) {
      return;
    }
    const id = this.#getBufferId(buffer);
    const label = this.#bufferLabels.get(buffer) ?? 'unknown';
    const action = pooled ? 'pool' : 'destroy';
    trace.buffers(
      `Release id=${id} size=${buffer.size} req=${requestedSize} usage=${buffer.usage} label=${label} action=${action}`
    );
  }

  
  detectLeaks(thresholdMs = 60000) {
    if (!this.#debugMode) {
      log.warn('BufferPool', 'Leak detection requires debug mode');
      return [];
    }

    const now = Date.now();
    
    const leaks = [];

    for (const [buffer, metadata] of this.#bufferMetadata.entries()) {
      if (this.#activeBuffers.has(buffer)) {
        const age = now - metadata.acquiredAt;
        if (age > thresholdMs) {
          leaks.push(metadata);
        }
      }
    }

    return leaks;
  }

  
  createStagingBuffer(size) {
    return this.acquire(size, BufferUsage.STAGING_READ, 'staging_read');
  }

  
  createUploadBuffer(size) {
    return this.acquire(size, BufferUsage.STAGING_WRITE, 'staging_write');
  }

  
  createUniformBuffer(size) {
    // Uniform buffers have stricter alignment (256 bytes typically)
    const alignedSize = alignTo(size, 256);
    return this.acquire(alignedSize, BufferUsage.UNIFORM, 'uniform');
  }

  
  uploadData(buffer, data, offset = 0) {
    const device = getDevice();
    if (!device) {
      throw new Error('Device not initialized');
    }
    device.queue.writeBuffer(buffer, offset,  (data));
  }

  
  async readBuffer(buffer, size = buffer.size) {
    if (!allowReadback('BufferPool.readBuffer')) {
      return new ArrayBuffer(0);
    }

    const device = getDevice();
    if (!device) {
      throw new Error('Device not initialized');
    }

    // Create staging buffer
    const staging = this.createStagingBuffer(size);

    // Copy to staging
    const encoder = device.createCommandEncoder({ label: 'readback_encoder' });
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);

    // Map and read
    await staging.mapAsync(GPUMapMode.READ);
    const data = staging.getMappedRange(0, size).slice(0);
    staging.unmap();

    // Release staging buffer
    this.release(staging);

    return data;
  }

  
  clearPool() {
    for (const usagePool of this.#pools.values()) {
      for (const bucketPool of usagePool.values()) {
        for (const buffer of bucketPool) {
          buffer.destroy();
          this.#stats.currentBytesAllocated -= buffer.size;
        }
        bucketPool.length = 0;
      }
    }
    this.#pools.clear();
    for (const buffer of this.#pendingDestruction) {
      buffer.destroy();
    }
    this.#pendingDestruction.clear();
    this.#destructionScheduled = false;
  }

  
  destroy() {
    // Destroy active buffers
    for (const buffer of this.#activeBuffers) {
      buffer.destroy();
    }
    this.#activeBuffers.clear();
    this.#bufferMetadata.clear();

    // Clear pools
    this.clearPool();

    this.#stats.currentBytesAllocated = 0;
    this.#stats.currentBytesRequested = 0;
    this.#requestedSizes.clear();
  }

  
  getStats() {
    return {
      ...this.#stats,
      activeBuffers: this.#activeBuffers.size,
      pooledBuffers: this.#getTotalPooledCount(),
      hitRate: this.#stats.allocations > 0
        ? (this.#stats.reuses / (this.#stats.allocations + this.#stats.reuses) * 100).toFixed(1) + '%'
        : '0%',
    };
  }

  
  configure(config) {
    Object.assign(this.#config, config);
  }
}

// Global buffer pool instance

let globalPool = null;


export function getBufferPool() {
  if (!globalPool) {
    globalPool = new BufferPool();
  }
  return globalPool;
}


export function createBufferPool(debugMode, schemaConfig) {
  return new BufferPool(debugMode, schemaConfig);
}


export function destroyBufferPool() {
  if (globalPool) {
    globalPool.destroy();
    globalPool = null;
  }
}

// Convenience exports for common operations

export const createStagingBuffer = (size) => getBufferPool().createStagingBuffer(size);

export const createUploadBuffer = (size) => getBufferPool().createUploadBuffer(size);

export const createUniformBuffer = (size) => getBufferPool().createUniformBuffer(size);

export const acquireBuffer = (size, usage, label) =>
  getBufferPool().acquire(size, usage, label);

export const releaseBuffer = (buffer) => getBufferPool().release(buffer);

export const isBufferActive = (buffer) =>
  getBufferPool().isActiveBuffer(buffer);

export const getBufferRequestedSize = (buffer) =>
  getBufferPool().getRequestedSize(buffer);

export const uploadData = (buffer, data, offset) =>
  getBufferPool().uploadData(buffer, data, offset);

export const readBuffer = (buffer, size) =>
  getBufferPool().readBuffer(buffer, size);


export async function withBuffer(
  size,
  usage,
  fn
) {
  const pool = getBufferPool();
  const buffer = pool.acquire(size, usage);
  try {
    return await fn(buffer);
  } finally {
    pool.release(buffer);
  }
}
