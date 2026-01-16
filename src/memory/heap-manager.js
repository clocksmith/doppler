

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


export class HeapManager {
  
  #strategy = null;
  
  #memory64Heap = null;
  
  #segments = [];
  
  #addressTable = null;
  
  #initialized = false;
  
  #totalAllocated = 0;

  
  async init() {
    if (this.#initialized) return;

    const caps = await getMemoryCapabilities();
    this.#strategy = caps.strategy;

    if (this.#strategy === 'MEMORY64') {
      await this.#initMemory64( (caps.maxHeapSize));
    } else {
      await this.#initSegmented( (caps.segmentedLimits));
    }

    this.#initialized = true;
    log.info('HeapManager', `Initialized with strategy: ${this.#strategy}`);
  }

  
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
        `Memory64 init failed, falling back to segmented: ${ (err).message}`
      );
      this.#strategy = 'SEGMENTED';
      const { fallbackSegmentSizeBytes } = getRuntimeConfig().shared.memory.segmentAllocation;
      await this.#initSegmented({ maxSegmentSize: fallbackSegmentSizeBytes, recommendedSegments: 8 });
    }
  }

  
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

  
  #allocateSegment() {
    const segmentSize =  (this.#addressTable).segmentSize;

    try {
      
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
      const { segmentFallbackSizes } = getRuntimeConfig().shared.memory.segmentAllocation;

      for (const size of segmentFallbackSizes) {
        if (size >= segmentSize) continue; // Already tried this size
        try {
          
          const segment = {
            index: this.#segments.length,
            buffer: new ArrayBuffer(size),
            used: 0,
          };
          this.#segments.push(segment);
          // Update address table's segment size for consistency
           (this.#addressTable).segmentSize = size;
          log.warn('HeapManager', `Allocation fallback to ${size / MB}MB segment`);
          return segment;
        } catch {
          continue;
        }
      }

      throw new Error(
        `Failed to allocate segment: ${ (e).message}. Try closing other tabs.`
      );
    }
  }

  
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

  
  #allocateMemory64(size) {
    const buffer =  (this.#memory64Heap).buffer;
    const offset = this.#totalAllocated;

    // Grow if needed
    if (offset + size > buffer.byteLength) {
      const neededPages = Math.ceil((offset + size - buffer.byteLength) / PAGE_SIZE);
       (this.#memory64Heap).grow(neededPages);
    }

    this.#totalAllocated += size;

    return {
      virtualAddress: offset,
      size,
      view: new Uint8Array( (this.#memory64Heap).buffer, offset, size),
      strategy: 'MEMORY64',
    };
  }

  
  #allocateSegmented(size) {
    // Find segment with enough space, or allocate new one
    let segment = this.#segments.find((s) => s.buffer.byteLength - s.used >= size);

    if (!segment) {
      segment = this.#allocateSegment();
    }

    const offset = segment.used;
    segment.used += size;
    this.#totalAllocated += size;

    const virtualAddress =  (this.#addressTable).encode(segment.index, offset);

    return {
      virtualAddress,
      size,
      view: new Uint8Array(segment.buffer, offset, size),
      segmentIndex: segment.index,
      segmentOffset: offset,
      strategy: 'SEGMENTED',
    };
  }

  
  read(virtualAddress, length) {
    if (this.#strategy === 'MEMORY64') {
      return new Uint8Array( (this.#memory64Heap).buffer, virtualAddress, length);
    } else {
      const { segmentIndex, offset } =  (this.#addressTable).decode(virtualAddress);
      const segment = this.#segments[segmentIndex];
      return new Uint8Array(segment.buffer, offset, length);
    }
  }

  
  write(virtualAddress, data) {
    const view = this.read(virtualAddress, data.length);
    view.set(data);
  }

  
  getBufferSlice(virtualAddress, length) {
    if (this.#strategy === 'MEMORY64') {
      // Return a copy for GPU upload (can't share WASM memory directly)
      const slice = new ArrayBuffer(length);
      new Uint8Array(slice).set(
        new Uint8Array( (this.#memory64Heap).buffer, virtualAddress, length)
      );
      return slice;
    } else {
      const { segmentIndex, offset } =  (this.#addressTable).decode(virtualAddress);
      const segment = this.#segments[segmentIndex];
      return segment.buffer.slice(offset, offset + length);
    }
  }

  
  getStats() {
    return {
      strategy: this.#strategy,
      totalAllocated: this.#totalAllocated,
      segmentCount: this.#segments.length,
      memory64HeapSize: this.#memory64Heap?.buffer.byteLength || 0,
    };
  }

  
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


let heapManagerInstance = null;


export function getHeapManager() {
  if (!heapManagerInstance) {
    heapManagerInstance = new HeapManager();
  }
  return heapManagerInstance;
}
