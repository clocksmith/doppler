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

import { getMemoryCapabilities } from './capability.js';
import { AddressTable } from './address-table.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { log } from '../debug/index.js';

// ============================================================================
// Constants
// ============================================================================

const GB = 1024 * 1024 * 1024;
const MB = 1024 * 1024;
const PAGE_SIZE = 65536; // WASM page = 64KB

// ============================================================================
// Heap Manager Class
// ============================================================================

/**
 * HeapManager - Unified interface for both memory strategies
 */
export class HeapManager {
  /** @type {import('./capability.js').MemoryStrategy | null} */
  #strategy = null;
  /** @type {WebAssembly.Memory | null} */
  #memory64Heap = null;
  /** @type {import('./heap-manager.js').MemorySegment[]} */
  #segments = [];
  /** @type {AddressTable | null} */
  #addressTable = null;
  /** @type {boolean} */
  #initialized = false;
  /** @type {number} */
  #totalAllocated = 0;

  /**
   * Initialize heap manager based on detected capabilities
   * @returns {Promise<void>}
   */
  async init() {
    if (this.#initialized) return;

    const caps = await getMemoryCapabilities();
    this.#strategy = caps.strategy;

    if (this.#strategy === 'MEMORY64') {
      await this.#initMemory64(/** @type {number} */ (caps.maxHeapSize));
    } else {
      await this.#initSegmented(/** @type {import('./capability.js').SegmentedLimits} */ (caps.segmentedLimits));
    }

    this.#initialized = true;
    log.info('HeapManager', `Initialized with strategy: ${this.#strategy}`);
  }

  /**
   * Initialize Memory64 heap (single large WASM memory)
   * @param {number} maxSize
   * @returns {Promise<void>}
   */
  async #initMemory64(maxSize) {
    // Start with 1GB, grow as needed
    const initialPages = Math.ceil(GB / PAGE_SIZE);
    const maxPages = Math.ceil(maxSize / PAGE_SIZE);

    try {
      this.#memory64Heap = new WebAssembly.Memory({
        initial: initialPages,
        maximum: maxPages,
        // memory64: true would go here when syntax is finalized
      });
      log.info(
        'HeapManager',
        `Memory64 heap: ${initialPages} initial pages, ${maxPages} max`
      );
    } catch (err) {
      log.error(
        'HeapManager',
        `Memory64 init failed, falling back to segmented: ${/** @type {Error} */ (err).message}`
      );
      this.#strategy = 'SEGMENTED';
      const { fallbackSegmentSizeBytes } = getRuntimeConfig().memory.segmentAllocation;
      await this.#initSegmented({ maxSegmentSize: fallbackSegmentSizeBytes, recommendedSegments: 8 });
    }
  }

  /**
   * Initialize segmented heap (multiple ArrayBuffers)
   * @param {import('./capability.js').SegmentedLimits} limits
   * @returns {Promise<void>}
   */
  async #initSegmented(limits) {
    this.#addressTable = new AddressTable(limits.maxSegmentSize);
    this.#segments = [];

    // Pre-allocate first segment
    this.#allocateSegment();

    log.info(
      'HeapManager',
      `Segmented heap: ${limits.maxSegmentSize / GB}GB per segment`
    );
  }

  /**
   * Allocate a new segment
   * @returns {import('./heap-manager.js').MemorySegment}
   */
  #allocateSegment() {
    const segmentSize = /** @type {AddressTable} */ (this.#addressTable).segmentSize;

    try {
      /** @type {import('./heap-manager.js').MemorySegment} */
      const segment = {
        index: this.#segments.length,
        buffer: new ArrayBuffer(segmentSize),
        used: 0,
      };
      this.#segments.push(segment);
      log.info(
        'HeapManager',
        `Allocated segment ${segment.index}: ${(segmentSize / MB).toFixed(0)}MB`
      );
      return segment;
    } catch (e) {
      // If allocation fails, try smaller sizes
      const { segmentFallbackSizes } = getRuntimeConfig().memory.segmentAllocation;

      for (const size of segmentFallbackSizes) {
        if (size >= segmentSize) continue; // Already tried this size
        try {
          /** @type {import('./heap-manager.js').MemorySegment} */
          const segment = {
            index: this.#segments.length,
            buffer: new ArrayBuffer(size),
            used: 0,
          };
          this.#segments.push(segment);
          // Update address table's segment size for consistency
          /** @type {AddressTable} */ (this.#addressTable).segmentSize = size;
          log.warn('HeapManager', `Allocation fallback to ${size / MB}MB segment`);
          return segment;
        } catch {
          continue;
        }
      }

      throw new Error(
        `Failed to allocate segment: ${/** @type {Error} */ (e).message}. Try closing other tabs.`
      );
    }
  }

  /**
   * Allocate buffer for model data
   * @param {number} size - Size in bytes
   * @returns {import('./heap-manager.js').AllocationResult}
   */
  allocate(size) {
    if (!this.#initialized) {
      throw new Error('HeapManager not initialized. Call init() first.');
    }

    if (this.#strategy === 'MEMORY64') {
      return this.#allocateMemory64(size);
    } else {
      return this.#allocateSegmented(size);
    }
  }

  /**
   * Allocate from Memory64 heap
   * @param {number} size
   * @returns {import('./heap-manager.js').AllocationResult}
   */
  #allocateMemory64(size) {
    const buffer = /** @type {WebAssembly.Memory} */ (this.#memory64Heap).buffer;
    const offset = this.#totalAllocated;

    // Grow if needed
    if (offset + size > buffer.byteLength) {
      const neededPages = Math.ceil((offset + size - buffer.byteLength) / PAGE_SIZE);
      /** @type {WebAssembly.Memory} */ (this.#memory64Heap).grow(neededPages);
    }

    this.#totalAllocated += size;

    return {
      virtualAddress: offset,
      size,
      view: new Uint8Array(/** @type {WebAssembly.Memory} */ (this.#memory64Heap).buffer, offset, size),
      strategy: 'MEMORY64',
    };
  }

  /**
   * Allocate from segmented heap
   * @param {number} size
   * @returns {import('./heap-manager.js').AllocationResult}
   */
  #allocateSegmented(size) {
    // Find segment with enough space, or allocate new one
    let segment = this.#segments.find((s) => s.buffer.byteLength - s.used >= size);

    if (!segment) {
      segment = this.#allocateSegment();
    }

    const offset = segment.used;
    segment.used += size;
    this.#totalAllocated += size;

    const virtualAddress = /** @type {AddressTable} */ (this.#addressTable).encode(segment.index, offset);

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
   * @param {number} virtualAddress - Virtual address to read from
   * @param {number} length - Number of bytes to read
   * @returns {Uint8Array}
   */
  read(virtualAddress, length) {
    if (this.#strategy === 'MEMORY64') {
      return new Uint8Array(/** @type {WebAssembly.Memory} */ (this.#memory64Heap).buffer, virtualAddress, length);
    } else {
      const { segmentIndex, offset } = /** @type {AddressTable} */ (this.#addressTable).decode(virtualAddress);
      const segment = this.#segments[segmentIndex];
      return new Uint8Array(segment.buffer, offset, length);
    }
  }

  /**
   * Write data to virtual address
   * @param {number} virtualAddress - Virtual address to write to
   * @param {Uint8Array} data - Data to write
   * @returns {void}
   */
  write(virtualAddress, data) {
    const view = this.read(virtualAddress, data.length);
    view.set(data);
  }

  /**
   * Get raw buffer for GPU upload
   * @param {number} virtualAddress - Virtual address
   * @param {number} length - Number of bytes
   * @returns {ArrayBuffer}
   */
  getBufferSlice(virtualAddress, length) {
    if (this.#strategy === 'MEMORY64') {
      // Return a copy for GPU upload (can't share WASM memory directly)
      const slice = new ArrayBuffer(length);
      new Uint8Array(slice).set(
        new Uint8Array(/** @type {WebAssembly.Memory} */ (this.#memory64Heap).buffer, virtualAddress, length)
      );
      return slice;
    } else {
      const { segmentIndex, offset } = /** @type {AddressTable} */ (this.#addressTable).decode(virtualAddress);
      const segment = this.#segments[segmentIndex];
      return segment.buffer.slice(offset, offset + length);
    }
  }

  /**
   * Get memory stats
   * @returns {import('./heap-manager.js').HeapStats}
   */
  getStats() {
    return {
      strategy: this.#strategy,
      totalAllocated: this.#totalAllocated,
      segmentCount: this.#segments.length,
      memory64HeapSize: this.#memory64Heap?.buffer.byteLength || 0,
    };
  }

  /**
   * Free all memory (for model unload)
   * @returns {void}
   */
  reset() {
    if (this.#strategy === 'SEGMENTED') {
      this.#segments = [];
      this.#allocateSegment();
    }
    // Memory64 heap can't be shrunk, but we can reset allocation pointer
    this.#totalAllocated = 0;
  }
}

// ============================================================================
// Singleton
// ============================================================================

/** @type {HeapManager | null} */
let heapManagerInstance = null;

/**
 * Get the global heap manager instance
 * @returns {HeapManager}
 */
export function getHeapManager() {
  if (!heapManagerInstance) {
    heapManagerInstance = new HeapManager();
  }
  return heapManagerInstance;
}
