/**
 * GPU Buffer Pool - Efficient Buffer Allocation and Reuse
 *
 * Manages GPU buffer allocation with pooling for reuse,
 * reducing allocation overhead during inference.
 */

import { getDevice, getDeviceLimits } from './device.js';
import { allowReadback, trackAllocation } from './perf-guards.js';
import type { GpuBufferHandle, BufferRequest } from '../types/gpu.js';
import { log, trace } from '../debug/index.js';
import {
  type BufferPoolConfigSchema,
  DEFAULT_BUFFER_POOL_CONFIG,
} from '../config/schema/index.js';

/**
 * Pool statistics
 */
export interface PoolStats {
  allocations: number;
  reuses: number;
  totalBytesAllocated: number;
  peakBytesAllocated: number;
  currentBytesAllocated: number;
  activeBuffers: number;
  pooledBuffers: number;
  hitRate: string;
}

/**
 * Pool configuration
 */
export interface PoolConfig {
  maxPoolSizePerBucket: number;
  maxTotalPooledBuffers: number;
  enablePooling: boolean;
  alignmentBytes: number;
}

/**
 * Internal buffer stats for tracking
 */
interface InternalStats {
  allocations: number;
  reuses: number;
  totalBytesAllocated: number;
  peakBytesAllocated: number;
  currentBytesAllocated: number;
}

/**
 * Tracked buffer metadata for leak detection
 */
interface BufferMetadata {
  size: number;
  usage: GPUBufferUsageFlags;
  label?: string;
  acquiredAt: number;
  stackTrace?: string;
}

/**
 * Buffer usage flags for different operations
 */
export const BufferUsage = {
  STORAGE: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  STORAGE_READ: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  UNIFORM: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  STAGING_READ: GPUMapMode.READ | GPUBufferUsage.COPY_DST,
  STAGING_WRITE: GPUMapMode.WRITE | GPUBufferUsage.COPY_SRC,
} as const;

/**
 * Round size up to alignment boundary
 */
function alignTo(size: number, alignment: number): number {
  return Math.ceil(size / alignment) * alignment;
}

/**
 * Get size bucket for pooling (power of 2 rounding)
 */
function getSizeBucket(
  size: number,
  maxAllowedSize: number = Infinity,
  bucketConfig = DEFAULT_BUFFER_POOL_CONFIG.bucket
): number {
  // Minimum bucket from config
  const minBucket = bucketConfig.minBucketSizeBytes;
  if (size <= minBucket) return minBucket;

  // Avoid power-of-two rounding for very large buffers.
  // For weights and large activations, rounding 600MB → 1GB can cause OOM even when the
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
  // Map<usage, Map<sizeBucket, GPUBuffer[]>>
  private pools: Map<GPUBufferUsageFlags, Map<number, GPUBuffer[]>>;

  // Active buffers (currently in use)
  private activeBuffers: Set<GPUBuffer>;

  // Buffer metadata for leak detection (debug mode)
  private bufferMetadata: Map<GPUBuffer, BufferMetadata>;

  // Statistics
  private stats: InternalStats;

  // Configuration
  private config: PoolConfig;

  // Schema-based configuration
  private schemaConfig: BufferPoolConfigSchema;

  // Debug mode flag
  private debugMode: boolean;

  constructor(debugMode: boolean = false, schemaConfig?: BufferPoolConfigSchema) {
    this.pools = new Map();
    this.activeBuffers = new Set();
    this.bufferMetadata = new Map();
    this.debugMode = debugMode;
    this.schemaConfig = schemaConfig ?? DEFAULT_BUFFER_POOL_CONFIG;

    this.stats = {
      allocations: 0,
      reuses: 0,
      totalBytesAllocated: 0,
      peakBytesAllocated: 0,
      currentBytesAllocated: 0,
    };

    // Initialize from schema config
    this.config = {
      maxPoolSizePerBucket: this.schemaConfig.limits.maxBuffersPerBucket,
      maxTotalPooledBuffers: this.schemaConfig.limits.maxTotalPooledBuffers,
      enablePooling: true,
      alignmentBytes: this.schemaConfig.alignment.alignmentBytes,
    };
  }

  /**
   * Get or create a buffer of the specified size
   */
  acquire(size: number, usage: GPUBufferUsageFlags = BufferUsage.STORAGE, label: string = 'pooled_buffer'): GPUBuffer {
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
    const alignedSize = alignTo(size, this.config.alignmentBytes);
    const maxAllowedBucket = isStorageBuffer ? Math.min(maxSize, maxStorageSize) : maxSize;
    const bucket = getSizeBucket(alignedSize, maxAllowedBucket, this.schemaConfig.bucket);

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
    if (this.config.enablePooling) {
      const pooled = this._getFromPool(bucket, usage);
      if (pooled) {
        this.activeBuffers.add(pooled);
        this.stats.reuses++;

        // Track metadata in debug mode
        if (this.debugMode) {
          this._trackBuffer(pooled, bucket, usage, label);
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

    this.activeBuffers.add(buffer);
    this.stats.allocations++;
    this.stats.totalBytesAllocated += bucket;
    this.stats.currentBytesAllocated += bucket;
    this.stats.peakBytesAllocated = Math.max(
      this.stats.peakBytesAllocated,
      this.stats.currentBytesAllocated
    );
    trackAllocation(bucket, label);

    // Track metadata in debug mode
    if (this.debugMode) {
      this._trackBuffer(buffer, bucket, usage, label);
    }

    return buffer;
  }

  /**
   * Release a buffer back to the pool
   */
  release(buffer: GPUBuffer): void {
    if (!this.activeBuffers.has(buffer)) {
      log.warn('BufferPool', 'Releasing buffer not tracked as active');
      return;
    }

    this.activeBuffers.delete(buffer);

    // Remove metadata in debug mode
    if (this.debugMode) {
      this.bufferMetadata.delete(buffer);
    }

    if (!this.config.enablePooling) {
      buffer.destroy();
      this.stats.currentBytesAllocated -= buffer.size;
      return;
    }

    // Return to pool if there's room
    const bucket = buffer.size;
    const usage = buffer.usage;

    if (!this.pools.has(usage)) {
      this.pools.set(usage, new Map());
    }
    const usagePool = this.pools.get(usage)!;

    if (!usagePool.has(bucket)) {
      usagePool.set(bucket, []);
    }
    const bucketPool = usagePool.get(bucket)!;

    if (bucketPool.length < this.config.maxPoolSizePerBucket &&
        this._getTotalPooledCount() < this.config.maxTotalPooledBuffers) {
      bucketPool.push(buffer);
    } else {
      // Pool is full, destroy buffer
      buffer.destroy();
      this.stats.currentBytesAllocated -= buffer.size;
    }
  }

  /**
   * Get a buffer from the pool if available
   */
  private _getFromPool(bucket: number, usage: GPUBufferUsageFlags): GPUBuffer | null {
    const usagePool = this.pools.get(usage);
    if (!usagePool) return null;

    const bucketPool = usagePool.get(bucket);
    if (!bucketPool || bucketPool.length === 0) return null;

    return bucketPool.pop()!;
  }

  /**
   * Get total count of pooled buffers
   */
  private _getTotalPooledCount(): number {
    let count = 0;
    for (const usagePool of this.pools.values()) {
      for (const bucketPool of usagePool.values()) {
        count += bucketPool.length;
      }
    }
    return count;
  }

  /**
   * Track buffer metadata for leak detection (debug mode)
   */
  private _trackBuffer(buffer: GPUBuffer, size: number, usage: GPUBufferUsageFlags, label?: string): void {
    const metadata: BufferMetadata = {
      size,
      usage,
      label,
      acquiredAt: Date.now(),
    };

    // Capture stack trace for leak detection
    if (Error.captureStackTrace) {
      const obj = {};
      Error.captureStackTrace(obj);
      metadata.stackTrace = (obj as any).stack;
    }

    this.bufferMetadata.set(buffer, metadata);
  }

  /**
   * Detect leaked buffers (debug mode)
   */
  detectLeaks(thresholdMs: number = 60000): BufferMetadata[] {
    if (!this.debugMode) {
      log.warn('BufferPool', 'Leak detection requires debug mode');
      return [];
    }

    const now = Date.now();
    const leaks: BufferMetadata[] = [];

    for (const [buffer, metadata] of this.bufferMetadata.entries()) {
      if (this.activeBuffers.has(buffer)) {
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
   */
  createStagingBuffer(size: number): GPUBuffer {
    return this.acquire(size, BufferUsage.STAGING_READ, 'staging_read');
  }

  /**
   * Create a staging buffer for CPU upload
   */
  createUploadBuffer(size: number): GPUBuffer {
    return this.acquire(size, BufferUsage.STAGING_WRITE, 'staging_write');
  }

  /**
   * Create a uniform buffer
   */
  createUniformBuffer(size: number): GPUBuffer {
    // Uniform buffers have stricter alignment (256 bytes typically)
    const alignedSize = alignTo(size, 256);
    return this.acquire(alignedSize, BufferUsage.UNIFORM, 'uniform');
  }

  /**
   * Upload data to GPU buffer
   */
  uploadData(buffer: GPUBuffer, data: ArrayBuffer | ArrayBufferView, offset: number = 0): void {
    const device = getDevice();
    if (!device) {
      throw new Error('Device not initialized');
    }
    device.queue.writeBuffer(buffer, offset, data as GPUAllowSharedBufferSource);
  }

  /**
   * Read data from GPU buffer
   * NOTE: GPU readbacks are expensive (0.5-2ms overhead per call). Use sparingly.
   */
  async readBuffer(buffer: GPUBuffer, size: number = buffer.size): Promise<ArrayBuffer> {
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
   */
  clearPool(): void {
    for (const usagePool of this.pools.values()) {
      for (const bucketPool of usagePool.values()) {
        for (const buffer of bucketPool) {
          buffer.destroy();
          this.stats.currentBytesAllocated -= buffer.size;
        }
        bucketPool.length = 0;
      }
    }
    this.pools.clear();
  }

  /**
   * Destroy all buffers (active and pooled)
   */
  destroy(): void {
    // Destroy active buffers
    for (const buffer of this.activeBuffers) {
      buffer.destroy();
    }
    this.activeBuffers.clear();
    this.bufferMetadata.clear();

    // Clear pools
    this.clearPool();

    this.stats.currentBytesAllocated = 0;
  }

  /**
   * Get pool statistics
   */
  getStats(): PoolStats {
    return {
      ...this.stats,
      activeBuffers: this.activeBuffers.size,
      pooledBuffers: this._getTotalPooledCount(),
      hitRate: this.stats.allocations > 0
        ? (this.stats.reuses / (this.stats.allocations + this.stats.reuses) * 100).toFixed(1) + '%'
        : '0%',
    };
  }

  /**
   * Configure pool settings
   */
  configure(config: Partial<PoolConfig>): void {
    Object.assign(this.config, config);
  }
}

// Global buffer pool instance
let globalPool: BufferPool | null = null;

/**
 * Get the global buffer pool
 */
export function getBufferPool(): BufferPool {
  if (!globalPool) {
    globalPool = new BufferPool();
  }
  return globalPool;
}

/**
 * Create a standalone buffer pool
 */
export function createBufferPool(debugMode?: boolean, schemaConfig?: BufferPoolConfigSchema): BufferPool {
  return new BufferPool(debugMode, schemaConfig);
}

/**
 * Destroy the global buffer pool
 */
export function destroyBufferPool(): void {
  if (globalPool) {
    globalPool.destroy();
    globalPool = null;
  }
}

// Convenience exports for common operations
export const createStagingBuffer = (size: number): GPUBuffer => getBufferPool().createStagingBuffer(size);
export const createUploadBuffer = (size: number): GPUBuffer => getBufferPool().createUploadBuffer(size);
export const createUniformBuffer = (size: number): GPUBuffer => getBufferPool().createUniformBuffer(size);
export const acquireBuffer = (size: number, usage?: GPUBufferUsageFlags, label?: string): GPUBuffer =>
  getBufferPool().acquire(size, usage, label);
export const releaseBuffer = (buffer: GPUBuffer): void => getBufferPool().release(buffer);
export const uploadData = (buffer: GPUBuffer, data: ArrayBuffer | ArrayBufferView, offset?: number): void =>
  getBufferPool().uploadData(buffer, data, offset);
export const readBuffer = (buffer: GPUBuffer, size?: number): Promise<ArrayBuffer> =>
  getBufferPool().readBuffer(buffer, size);

/**
 * Scoped buffer helper - automatically releases buffer when done
 */
export async function withBuffer<T>(
  size: number,
  usage: GPUBufferUsageFlags,
  fn: (buffer: GPUBuffer) => Promise<T>
): Promise<T> {
  const pool = getBufferPool();
  const buffer = pool.acquire(size, usage);
  try {
    return await fn(buffer);
  } finally {
    pool.release(buffer);
  }
}
