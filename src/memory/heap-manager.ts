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

import { getMemoryCapabilities, type MemoryStrategy, type SegmentedLimits } from './capability.js';
import { AddressTable } from './address-table.js';

// ============================================================================
// Constants
// ============================================================================

const GB = 1024 * 1024 * 1024;
const MB = 1024 * 1024;
const PAGE_SIZE = 65536; // WASM page = 64KB

// ============================================================================
// Types and Interfaces
// ============================================================================

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

// ============================================================================
// Heap Manager Class
// ============================================================================

/**
 * HeapManager - Unified interface for both memory strategies
 */
export class HeapManager {
  private strategy: MemoryStrategy | null = null;
  private memory64Heap: WebAssembly.Memory | null = null;
  private segments: MemorySegment[] = [];
  private addressTable: AddressTable | null = null;
  private initialized = false;
  private totalAllocated = 0;

  /**
   * Initialize heap manager based on detected capabilities
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    const caps = await getMemoryCapabilities();
    this.strategy = caps.strategy;

    if (this.strategy === 'MEMORY64') {
      await this._initMemory64(caps.maxHeapSize!);
    } else {
      await this._initSegmented(caps.segmentedLimits!);
    }

    this.initialized = true;
    console.log(`[HeapManager] Initialized with strategy: ${this.strategy}`);
  }

  /**
   * Initialize Memory64 heap (single large WASM memory)
   */
  private async _initMemory64(maxSize: number): Promise<void> {
    // Start with 1GB, grow as needed
    const initialPages = Math.ceil(GB / PAGE_SIZE);
    const maxPages = Math.ceil(maxSize / PAGE_SIZE);

    try {
      this.memory64Heap = new WebAssembly.Memory({
        initial: initialPages,
        maximum: maxPages,
        // memory64: true would go here when syntax is finalized
      });
      console.log(
        `[HeapManager] Memory64 heap: ${initialPages} initial pages, ${maxPages} max`
      );
    } catch (err) {
      console.error(
        '[HeapManager] Memory64 init failed, falling back to segmented:',
        err
      );
      this.strategy = 'SEGMENTED';
      await this._initSegmented({ maxSegmentSize: 4 * GB, recommendedSegments: 8 });
    }
  }

  /**
   * Initialize segmented heap (multiple ArrayBuffers)
   */
  private async _initSegmented(limits: SegmentedLimits): Promise<void> {
    this.addressTable = new AddressTable(limits.maxSegmentSize);
    this.segments = [];

    // Pre-allocate first segment
    this._allocateSegment();

    console.log(
      `[HeapManager] Segmented heap: ${limits.maxSegmentSize / GB}GB per segment`
    );
  }

  /**
   * Allocate a new segment
   */
  private _allocateSegment(): MemorySegment {
    const segmentSize = this.addressTable!.segmentSize;

    try {
      const segment: MemorySegment = {
        index: this.segments.length,
        buffer: new ArrayBuffer(segmentSize),
        used: 0,
      };
      this.segments.push(segment);
      console.log(
        `[HeapManager] Allocated segment ${segment.index}: ${(segmentSize / MB).toFixed(0)}MB`
      );
      return segment;
    } catch (e) {
      // If allocation fails, try smaller sizes
      const fallbackSizes = [512 * MB, 256 * MB, 128 * MB];

      for (const size of fallbackSizes) {
        if (size >= segmentSize) continue; // Already tried this size
        try {
          const segment: MemorySegment = {
            index: this.segments.length,
            buffer: new ArrayBuffer(size),
            used: 0,
          };
          this.segments.push(segment);
          // Update address table's segment size for consistency
          this.addressTable!.segmentSize = size;
          console.warn(`[HeapManager] Allocation fallback to ${size / MB}MB segment`);
          return segment;
        } catch {
          continue;
        }
      }

      throw new Error(
        `Failed to allocate segment: ${(e as Error).message}. Try closing other tabs.`
      );
    }
  }

  /**
   * Allocate buffer for model data
   * @param size - Size in bytes
   */
  allocate(size: number): AllocationResult {
    if (!this.initialized) {
      throw new Error('HeapManager not initialized. Call init() first.');
    }

    if (this.strategy === 'MEMORY64') {
      return this._allocateMemory64(size);
    } else {
      return this._allocateSegmented(size);
    }
  }

  /**
   * Allocate from Memory64 heap
   */
  private _allocateMemory64(size: number): AllocationResult {
    const buffer = this.memory64Heap!.buffer;
    const offset = this.totalAllocated;

    // Grow if needed
    if (offset + size > buffer.byteLength) {
      const neededPages = Math.ceil((offset + size - buffer.byteLength) / PAGE_SIZE);
      this.memory64Heap!.grow(neededPages);
    }

    this.totalAllocated += size;

    return {
      virtualAddress: offset,
      size,
      view: new Uint8Array(this.memory64Heap!.buffer, offset, size),
      strategy: 'MEMORY64',
    };
  }

  /**
   * Allocate from segmented heap
   */
  private _allocateSegmented(size: number): AllocationResult {
    // Find segment with enough space, or allocate new one
    let segment = this.segments.find((s) => s.buffer.byteLength - s.used >= size);

    if (!segment) {
      segment = this._allocateSegment();
    }

    const offset = segment.used;
    segment.used += size;
    this.totalAllocated += size;

    const virtualAddress = this.addressTable!.encode(segment.index, offset);

    return {
      virtualAddress,
      size,
      view: new Uint8Array(segment.buffer, offset, size),
      segmentIndex: segment.index,
      segmentOffset: offset,
      strategy: 'SEGMENTED',
    };
  }

  /**
   * Read data from virtual address
   * @param virtualAddress - Virtual address to read from
   * @param length - Number of bytes to read
   */
  read(virtualAddress: number, length: number): Uint8Array {
    if (this.strategy === 'MEMORY64') {
      return new Uint8Array(this.memory64Heap!.buffer, virtualAddress, length);
    } else {
      const { segmentIndex, offset } = this.addressTable!.decode(virtualAddress);
      const segment = this.segments[segmentIndex];
      return new Uint8Array(segment.buffer, offset, length);
    }
  }

  /**
   * Write data to virtual address
   * @param virtualAddress - Virtual address to write to
   * @param data - Data to write
   */
  write(virtualAddress: number, data: Uint8Array): void {
    const view = this.read(virtualAddress, data.length);
    view.set(data);
  }

  /**
   * Get raw buffer for GPU upload
   * @param virtualAddress - Virtual address
   * @param length - Number of bytes
   */
  getBufferSlice(virtualAddress: number, length: number): ArrayBuffer {
    if (this.strategy === 'MEMORY64') {
      // Return a copy for GPU upload (can't share WASM memory directly)
      const slice = new ArrayBuffer(length);
      new Uint8Array(slice).set(
        new Uint8Array(this.memory64Heap!.buffer, virtualAddress, length)
      );
      return slice;
    } else {
      const { segmentIndex, offset } = this.addressTable!.decode(virtualAddress);
      const segment = this.segments[segmentIndex];
      return segment.buffer.slice(offset, offset + length);
    }
  }

  /**
   * Get memory stats
   */
  getStats(): HeapStats {
    return {
      strategy: this.strategy,
      totalAllocated: this.totalAllocated,
      segmentCount: this.segments.length,
      memory64HeapSize: this.memory64Heap?.buffer.byteLength || 0,
    };
  }

  /**
   * Free all memory (for model unload)
   */
  reset(): void {
    if (this.strategy === 'SEGMENTED') {
      this.segments = [];
      this._allocateSegment();
    }
    // Memory64 heap can't be shrunk, but we can reset allocation pointer
    this.totalAllocated = 0;
  }
}

// ============================================================================
// Singleton
// ============================================================================

/** Singleton instance */
let heapManagerInstance: HeapManager | null = null;

/**
 * Get the global heap manager instance
 */
export function getHeapManager(): HeapManager {
  if (!heapManagerInstance) {
    heapManagerInstance = new HeapManager();
  }
  return heapManagerInstance;
}
