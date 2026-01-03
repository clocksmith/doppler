/**
 * Heap Manager
 * Agent-A | Domain: memory/
 *
 * Manages memory allocation for model weights:
 * - Memory64 mode: Single large WASM heap
 * - Segmented mode: Multiple 4GB ArrayBuffers with virtual addressing
 *
 * @module memory/heap-manager
 */

import { MemoryStrategy, SegmentedLimits } from './capability.js';

/**
 * Memory segment for segmented heap mode
 */
export interface MemorySegment {
  index: number;
  buffer: ArrayBuffer;
  used: number;
}

/**
 * Allocation result
 */
export interface AllocationResult {
  /** Virtual address for this allocation */
  virtualAddress: number;
  /** Allocated size in bytes */
  size: number;
  /** Direct view into the buffer */
  view: Uint8Array;
  /** Which strategy was used */
  strategy: MemoryStrategy;
  /** Segment index (segmented only) */
  segmentIndex?: number;
  /** Offset within segment (segmented only) */
  segmentOffset?: number;
}

/**
 * Heap manager statistics
 */
export interface HeapStats {
  strategy: MemoryStrategy | null;
  totalAllocated: number;
  segmentCount: number;
  memory64HeapSize: number;
  allocated?: number;
  limit?: number;
  maxSize?: number;
}

/**
 * HeapManager - Unified interface for both memory strategies
 */
export declare class HeapManager {
  /**
   * Initialize heap manager based on detected capabilities
   */
  init(): Promise<void>;

  /**
   * Allocate buffer for model data
   * @param size - Size in bytes
   */
  allocate(size: number): AllocationResult;

  /**
   * Read data from virtual address
   * @param virtualAddress - Virtual address to read from
   * @param length - Number of bytes to read
   */
  read(virtualAddress: number, length: number): Uint8Array;

  /**
   * Write data to virtual address
   * @param virtualAddress - Virtual address to write to
   * @param data - Data to write
   */
  write(virtualAddress: number, data: Uint8Array): void;

  /**
   * Get raw buffer for GPU upload
   * @param virtualAddress - Virtual address
   * @param length - Number of bytes
   */
  getBufferSlice(virtualAddress: number, length: number): ArrayBuffer;

  /**
   * Get memory stats
   */
  getStats(): HeapStats;

  /**
   * Free all memory (for model unload)
   */
  reset(): void;
}

/**
 * Get the global heap manager instance
 */
export function getHeapManager(): HeapManager;
