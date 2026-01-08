/**
 * GPU Buffer Pool - Efficient Buffer Allocation and Reuse
 *
 * Manages GPU buffer allocation with pooling for reuse,
 * reducing allocation overhead during inference.
 */

import { getDevice, getDeviceLimits } from './device.js';
import { allowReadback, trackAllocation } from './perf-guards.js';
import { log, trace } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

/**
 * Buffer usage flags for different operations
 */
export const BufferUsage = /** @type {const} */ ({
  STORAGE: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  STORAGE_READ: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  UNIFORM: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  STAGING_READ: GPUMapMode.READ | GPUBufferUsage.COPY_DST,
  STAGING_WRITE: GPUMapMode.WRITE | GPUBufferUsage.COPY_SRC,
});

/**
 * Round size up to alignment boundary
 * @param {number} size
 * @param {number} alignment
 * @returns {number}
 */
function alignTo(size, alignment) {
  return Math.ceil(size / alignment) * alignment;
}

/**
 * Get size bucket for pooling (power of 2 rounding)
 * @param {number} size
 * @param {number} [maxAllowedSize]
 * @param {import('../config/schema/index.js').BufferPoolConfigSchema['bucket']} [bucketConfig]
 * @returns {number}
 */
function getSizeBucket(
  size,
  maxAllowedSize = Infinity,
  bucketConfig = getRuntimeConfig().bufferPool.bucket
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

/**
 * Buffer Pool for efficient GPU memory reuse
 */
export class BufferPool {
  // Pools organized by usage and size bucket
  /** @type {Map<GPUBufferUsageFlags, Map<number, GPUBuffer[]>>} */
  #pools;

  // Active buffers (currently in use)
  /** @type {Set<GPUBuffer>} */
  #activeBuffers;

  // Buffer metadata for leak detection (debug mode)
  /** @type {Map<GPUBuffer, {size: number, usage: GPUBufferUsageFlags, label?: string, acquiredAt: number, stackTrace?: string}>} */
  #bufferMetadata;

  // Deferred destruction queue (buffers destroyed after GPU work completes)
  /** @type {Set<GPUBuffer>} */
  #pendingDestruction;
  /** @type {boolean} */
  #destructionScheduled;

  // Statistics
  /** @type {{allocations: number, reuses: number, totalBytesAllocated: number, peakBytesAllocated: number, currentBytesAllocated: number}} */
  #stats;

  // Configuration
  /** @type {import('./buffer-pool.js').PoolConfig} */
  #config;

  // Schema-based configuration
  /** @type {import('../config/schema/index.js').BufferPoolConfigSchema} */
  #schemaConfig;

  // Debug mode flag
  /** @type {boolean} */
  #debugMode;

  /**
   * @param {boolean} [debugMode]
   * @param {import('../config/schema/index.js').BufferPoolConfigSchema} [schemaConfig]
   */
  constructor(debugMode = false, schemaConfig) {
    this.#pools = new Map();
    this.#activeBuffers = new Set();
    this.#bufferMetadata = new Map();
    this.#debugMode = debugMode;
    this.#schemaConfig = schemaConfig ?? getRuntimeConfig().bufferPool;
    this.#pendingDestruction = new Set();
    this.#destructionScheduled = false;

    this.#stats = {
      allocations: 0,
      reuses: 0,
      totalBytesAllocated: 0,
      peakBytesAllocated: 0,
      currentBytesAllocated: 0,
    };

    // Initialize from schema config
    this.#config = {
      maxPoolSizePerBucket: this.#schemaConfig.limits.maxBuffersPerBucket,
      maxTotalPooledBuffers: this.#schemaConfig.limits.maxTotalPooledBuffers,
      enablePooling: true,
      alignmentBytes: this.#schemaConfig.alignment.alignmentBytes,
    };
  }

  /**
   * Get or create a buffer of the specified size
   * @param {number} size
   * @param {GPUBufferUsageFlags} [usage]
   * @param {string} [label]
   * @returns {GPUBuffer}
   */
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

        // Track metadata in debug mode
        if (this.#debugMode) {
          this.#trackBuffer(pooled, bucket, usage, label);
        }

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
    trackAllocation(bucket, label);

    // Track metadata in debug mode
    if (this.#debugMode) {
      this.#trackBuffer(buffer, bucket, usage, label);
    }

    return buffer;
  }

  /**
   * Release a buffer back to the pool
   * @param {GPUBuffer} buffer
   * @returns {void}
   */
  release(buffer) {
    if (!this.#activeBuffers.has(buffer)) {
      log.warn('BufferPool', 'Releasing buffer not tracked as active');
      return;
    }

    this.#activeBuffers.delete(buffer);

    // Remove metadata in debug mode
    if (this.#debugMode) {
      this.#bufferMetadata.delete(buffer);
    }

    if (!this.#config.enablePooling) {
      this.#deferDestroy(buffer);
      this.#stats.currentBytesAllocated -= buffer.size;
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

    if (bucketPool.length < this.#config.maxPoolSizePerBucket &&
        this.#getTotalPooledCount() < this.#config.maxTotalPooledBuffers) {
      bucketPool.push(buffer);
    } else {
      // Pool is full; defer destruction until GPU work completes.
      this.#deferDestroy(buffer);
      this.#stats.currentBytesAllocated -= buffer.size;
    }
  }

  /**
   * Defer buffer destruction until all submitted GPU work completes.
   * This avoids destroying buffers still referenced by in-flight command buffers.
   * @param {GPUBuffer} buffer
   */
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
        log.warn('BufferPool', `Deferred destruction failed: ${/** @type {Error} */ (err).message}`);
        this.#pendingDestruction.clear();
        this.#destructionScheduled = false;
      });
  }

  /**
   * Get a buffer from the pool if available
   * @param {number} bucket
   * @param {GPUBufferUsageFlags} usage
   * @returns {GPUBuffer | null}
   */
  #getFromPool(bucket, usage) {
    const usagePool = this.#pools.get(usage);
    if (!usagePool) return null;

    const bucketPool = usagePool.get(bucket);
    if (!bucketPool || bucketPool.length === 0) return null;

    return bucketPool.pop();
  }

  /**
   * Get total count of pooled buffers
   * @returns {number}
   */
  #getTotalPooledCount() {
    let count = 0;
    for (const usagePool of this.#pools.values()) {
      for (const bucketPool of usagePool.values()) {
        count += bucketPool.length;
      }
    }
    return count;
  }

  /**
   * Track buffer metadata for leak detection (debug mode)
   * @param {GPUBuffer} buffer
   * @param {number} size
   * @param {GPUBufferUsageFlags} usage
   * @param {string} [label]
   */
  #trackBuffer(buffer, size, usage, label) {
    /** @type {{size: number, usage: GPUBufferUsageFlags, label?: string, acquiredAt: number, stackTrace?: string}} */
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
      metadata.stackTrace = /** @type {any} */ (obj).stack;
    }

    this.#bufferMetadata.set(buffer, metadata);
  }

  /**
   * Detect leaked buffers (debug mode)
   * @param {number} [thresholdMs]
   * @returns {{size: number, usage: GPUBufferUsageFlags, label?: string, acquiredAt: number, stackTrace?: string}[]}
   */
  detectLeaks(thresholdMs = 60000) {
    if (!this.#debugMode) {
      log.warn('BufferPool', 'Leak detection requires debug mode');
      return [];
    }

    const now = Date.now();
    /** @type {{size: number, usage: GPUBufferUsageFlags, label?: string, acquiredAt: number, stackTrace?: string}[]} */
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

  /**
   * Create a staging buffer for CPU readback
   * @param {number} size
   * @returns {GPUBuffer}
   */
  createStagingBuffer(size) {
    return this.acquire(size, BufferUsage.STAGING_READ, 'staging_read');
  }

  /**
   * Create a staging buffer for CPU upload
   * @param {number} size
   * @returns {GPUBuffer}
   */
  createUploadBuffer(size) {
    return this.acquire(size, BufferUsage.STAGING_WRITE, 'staging_write');
  }

  /**
   * Create a uniform buffer
   * @param {number} size
   * @returns {GPUBuffer}
   */
  createUniformBuffer(size) {
    // Uniform buffers have stricter alignment (256 bytes typically)
    const alignedSize = alignTo(size, 256);
    return this.acquire(alignedSize, BufferUsage.UNIFORM, 'uniform');
  }

  /**
   * Upload data to GPU buffer
   * @param {GPUBuffer} buffer
   * @param {ArrayBuffer | ArrayBufferView} data
   * @param {number} [offset]
   * @returns {void}
   */
  uploadData(buffer, data, offset = 0) {
    const device = getDevice();
    if (!device) {
      throw new Error('Device not initialized');
    }
    device.queue.writeBuffer(buffer, offset, /** @type {GPUAllowSharedBufferSource} */ (data));
  }

  /**
   * Read data from GPU buffer
   * NOTE: GPU readbacks are expensive (0.5-2ms overhead per call). Use sparingly.
   * @param {GPUBuffer} buffer
   * @param {number} [size]
   * @returns {Promise<ArrayBuffer>}
   */
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

  /**
   * Clear all pooled buffers
   * @returns {void}
   */
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

  /**
   * Destroy all buffers (active and pooled)
   * @returns {void}
   */
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
  }

  /**
   * Get pool statistics
   * @returns {import('./buffer-pool.js').PoolStats}
   */
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

  /**
   * Configure pool settings
   * @param {Partial<import('./buffer-pool.js').PoolConfig>} config
   * @returns {void}
   */
  configure(config) {
    Object.assign(this.#config, config);
  }
}

// Global buffer pool instance
/** @type {BufferPool | null} */
let globalPool = null;

/**
 * Get the global buffer pool
 * @returns {BufferPool}
 */
export function getBufferPool() {
  if (!globalPool) {
    globalPool = new BufferPool();
  }
  return globalPool;
}

/**
 * Create a standalone buffer pool
 * @param {boolean} [debugMode]
 * @param {import('../config/schema/index.js').BufferPoolConfigSchema} [schemaConfig]
 * @returns {BufferPool}
 */
export function createBufferPool(debugMode, schemaConfig) {
  return new BufferPool(debugMode, schemaConfig);
}

/**
 * Destroy the global buffer pool
 * @returns {void}
 */
export function destroyBufferPool() {
  if (globalPool) {
    globalPool.destroy();
    globalPool = null;
  }
}

// Convenience exports for common operations
/** @type {(size: number) => GPUBuffer} */
export const createStagingBuffer = (size) => getBufferPool().createStagingBuffer(size);
/** @type {(size: number) => GPUBuffer} */
export const createUploadBuffer = (size) => getBufferPool().createUploadBuffer(size);
/** @type {(size: number) => GPUBuffer} */
export const createUniformBuffer = (size) => getBufferPool().createUniformBuffer(size);
/** @type {(size: number, usage?: GPUBufferUsageFlags, label?: string) => GPUBuffer} */
export const acquireBuffer = (size, usage, label) =>
  getBufferPool().acquire(size, usage, label);
/** @type {(buffer: GPUBuffer) => void} */
export const releaseBuffer = (buffer) => getBufferPool().release(buffer);
/** @type {(buffer: GPUBuffer, data: ArrayBuffer | ArrayBufferView, offset?: number) => void} */
export const uploadData = (buffer, data, offset) =>
  getBufferPool().uploadData(buffer, data, offset);
/** @type {(buffer: GPUBuffer, size?: number) => Promise<ArrayBuffer>} */
export const readBuffer = (buffer, size) =>
  getBufferPool().readBuffer(buffer, size);

/**
 * Scoped buffer helper - automatically releases buffer when done
 * @template T
 * @param {number} size
 * @param {GPUBufferUsageFlags} usage
 * @param {(buffer: GPUBuffer) => Promise<T>} fn
 * @returns {Promise<T>}
 */
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
