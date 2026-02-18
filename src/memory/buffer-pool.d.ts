/**
 * GPU Buffer Pool - Efficient Buffer Allocation and Reuse
 *
 * Manages GPU buffer allocation with pooling for reuse,
 * reducing allocation overhead during inference.
 */

import type { BufferPoolConfigSchema } from '../config/schema/index.js';

/**
 * Pool statistics
 */
export interface PoolStats {
  allocations: number;
  reuses: number;
  totalBytesAllocated: number;
  peakBytesAllocated: number;
  currentBytesAllocated: number;
  totalBytesRequested: number;
  peakBytesRequested: number;
  currentBytesRequested: number;
  activeBuffers: number;
  pooledBuffers: number;
  budgetMaxBytes: number;
  budgetUtilization: number;
  hitRate: string;
}

export interface LabelStats {
  label: string;
  bytes: number;
  count: number;
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
 * Buffer usage flags for different operations
 */
export declare const BufferUsage: {
  readonly STORAGE: number;
  readonly STORAGE_READ: number;
  readonly UNIFORM: number;
  readonly STAGING_READ: number;
  readonly STAGING_WRITE: number;
};

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
 * Buffer Pool for efficient GPU memory reuse
 */
export declare class BufferPool {
  constructor(debugMode?: boolean, schemaConfig?: BufferPoolConfigSchema);

  /**
   * Get or create a buffer of the specified size
   */
  acquire(size: number, usage?: GPUBufferUsageFlags, label?: string): GPUBuffer;

  /**
   * Release a buffer back to the pool
   */
  release(buffer: GPUBuffer): void;

  /**
   * Check if a buffer is currently tracked as active by the pool
   */
  isActiveBuffer(buffer: GPUBuffer): boolean;

  /**
   * Get the requested size for a pooled buffer
   */
  getRequestedSize(buffer: GPUBuffer): number;

  /**
   * Detect leaked buffers (debug mode)
   */
  detectLeaks(thresholdMs?: number): BufferMetadata[];

  /**
   * Create a staging buffer for CPU readback
   */
  createStagingBuffer(size: number): GPUBuffer;

  /**
   * Create a staging buffer for CPU upload
   */
  createUploadBuffer(size: number): GPUBuffer;

  /**
   * Create a uniform buffer
   */
  createUniformBuffer(size: number): GPUBuffer;

  /**
   * Upload data to GPU buffer
   */
  uploadData(buffer: GPUBuffer, data: ArrayBuffer | ArrayBufferView, offset?: number): void;

  /**
   * Read data from GPU buffer
   * NOTE: GPU readbacks are expensive (0.5-2ms overhead per call). Use sparingly.
   */
  readBuffer(buffer: GPUBuffer, size?: number): Promise<ArrayBuffer>;

  /**
   * Clear all pooled buffers
   */
  clearPool(): void;

  /**
   * Destroy all buffers (active and pooled)
   */
  destroy(): void;

  /**
   * Get pool statistics
   */
  getStats(): PoolStats;

  /**
   * Get aggregated stats by buffer label (active buffers only)
   */
  getLabelStats(): LabelStats[];

  /**
   * Configure pool settings
   */
  configure(config: Partial<PoolConfig>): void;

  /**
   * Force budget reclaim by trimming pooled buffers.
   */
  forceReclaim(targetRatio?: number | null): void;
}

/**
 * Get the global buffer pool
 */
export function getBufferPool(): BufferPool;

/**
 * Create a standalone buffer pool
 */
export function createBufferPool(debugMode?: boolean, schemaConfig?: BufferPoolConfigSchema): BufferPool;

/**
 * Destroy the global buffer pool
 */
export function destroyBufferPool(): void;

// Convenience exports for common operations
export declare const createStagingBuffer: (size: number) => GPUBuffer;
export declare const createUploadBuffer: (size: number) => GPUBuffer;
export declare const createUniformBuffer: (size: number) => GPUBuffer;
export declare const acquireBuffer: (size: number, usage?: GPUBufferUsageFlags, label?: string) => GPUBuffer;
export declare const releaseBuffer: (buffer: GPUBuffer) => void;
export declare const isBufferActive: (buffer: GPUBuffer) => boolean;
export declare const getBufferRequestedSize: (buffer: GPUBuffer) => number;
export declare const uploadData: (buffer: GPUBuffer, data: ArrayBuffer | ArrayBufferView, offset?: number) => void;
export declare const readBuffer: (buffer: GPUBuffer, size?: number) => Promise<ArrayBuffer>;
export declare const forceBufferPoolReclaim: (targetRatio?: number | null) => void;

/**
 * Scoped buffer helper - automatically releases buffer when done
 */
export function withBuffer<T>(
  size: number,
  usage: GPUBufferUsageFlags,
  fn: (buffer: GPUBuffer) => Promise<T>
): Promise<T>;
