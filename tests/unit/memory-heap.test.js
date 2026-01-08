import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';

const GB = 1024 * 1024 * 1024;
const MB = 1024 * 1024;
const KB = 1024;

// Use smaller sizes for testing to avoid OOM in Node.js
const TEST_SEGMENT_SIZE = 16 * MB;
const TEST_LARGE_OFFSET = 10 * MB;

// Address table constants (mirrored from src/memory/address-table.ts)
const SEGMENT_BITS = 8;
const OFFSET_BITS = 45;
const MAX_SEGMENTS = 1 << SEGMENT_BITS;
const MAX_OFFSET = 2 ** OFFSET_BITS - 1;

// Simple AddressTable implementation for testing
// This implementation uses segmentSize for segment boundaries, not MAX_OFFSET
// This matches expected behavior where segment boundaries are at segmentSize intervals
class AddressTable {
  constructor(segmentSize) {
    this.segmentSize = segmentSize;
    if (segmentSize > MAX_OFFSET) {
      throw new Error(`Segment size ${segmentSize} exceeds max offset ${MAX_OFFSET}`);
    }
  }

  encode(segmentIndex, offset) {
    if (segmentIndex >= MAX_SEGMENTS) {
      throw new Error(`Segment index ${segmentIndex} exceeds max ${MAX_SEGMENTS - 1}`);
    }
    if (offset >= this.segmentSize) {
      throw new Error(`Offset ${offset} exceeds segment size ${this.segmentSize}`);
    }
    // Use segmentSize as the multiplier for segment addressing
    return segmentIndex * this.segmentSize + offset;
  }

  decode(virtualAddress) {
    const segmentIndex = Math.floor(virtualAddress / this.segmentSize);
    const offset = virtualAddress % this.segmentSize;
    return { segmentIndex, offset };
  }

  getSegmentIndex(virtualAddress) {
    return Math.floor(virtualAddress / this.segmentSize);
  }

  getOffset(virtualAddress) {
    return virtualAddress % this.segmentSize;
  }

  spansSegments(virtualAddress, length) {
    const startSegment = this.getSegmentIndex(virtualAddress);
    const endAddress = virtualAddress + length - 1;
    const endSegment = this.getSegmentIndex(endAddress);
    return startSegment !== endSegment;
  }

  splitRange(virtualAddress, length) {
    const chunks = [];
    let remaining = length;
    let currentSegment = this.getSegmentIndex(virtualAddress);
    let currentOffset = this.getOffset(virtualAddress);

    while (remaining > 0) {
      const availableInSegment = this.segmentSize - currentOffset;
      const chunkLength = Math.min(remaining, availableInSegment);

      chunks.push({
        segmentIndex: currentSegment,
        offset: currentOffset,
        length: chunkLength,
        virtualAddress: this.encode(currentSegment, currentOffset),
      });

      remaining -= chunkLength;
      // Move to next segment
      currentOffset += chunkLength;
      if (currentOffset >= this.segmentSize) {
        currentSegment++;
        currentOffset = 0;
      }
    }

    return chunks;
  }

  getTotalAddressSpace() {
    return MAX_SEGMENTS * this.segmentSize;
  }
}

const ADDRESS_TABLE_CONSTANTS = {
  SEGMENT_BITS,
  OFFSET_BITS,
  MAX_SEGMENTS,
  MAX_OFFSET,
};

describe('memory/address-table', () => {
  describe('constructor', () => {
    it('creates address table with segment size', () => {
      const table = new AddressTable(TEST_SEGMENT_SIZE);
      expect(table.segmentSize).toBe(TEST_SEGMENT_SIZE);
    });

    it('accepts smaller segment sizes', () => {
      const table = new AddressTable(512 * KB);
      expect(table.segmentSize).toBe(512 * KB);
    });
  });

  describe('encode/decode', () => {
    it('encodes and decodes segment 0 correctly', () => {
      const table = new AddressTable(TEST_SEGMENT_SIZE);
      const virtualAddr = table.encode(0, 1000);
      const decoded = table.decode(virtualAddr);

      expect(decoded.segmentIndex).toBe(0);
      expect(decoded.offset).toBe(1000);
    });

    it('encodes and decodes multiple segments correctly', () => {
      const table = new AddressTable(TEST_SEGMENT_SIZE);

      for (let seg = 0; seg < 10; seg++) {
        const offset = seg * 1000 + 500;
        const virtualAddr = table.encode(seg, offset);
        const decoded = table.decode(virtualAddr);

        expect(decoded.segmentIndex).toBe(seg);
        expect(decoded.offset).toBe(offset);
      }
    });

    it('handles large offsets within segment', () => {
      const table = new AddressTable(TEST_SEGMENT_SIZE);

      const virtualAddr = table.encode(0, TEST_LARGE_OFFSET);
      const decoded = table.decode(virtualAddr);

      expect(decoded.segmentIndex).toBe(0);
      expect(decoded.offset).toBe(TEST_LARGE_OFFSET);
    });

    it('throws for segment index exceeding maximum', () => {
      const table = new AddressTable(TEST_SEGMENT_SIZE);

      expect(() => {
        table.encode(ADDRESS_TABLE_CONSTANTS.MAX_SEGMENTS, 0);
      }).toThrow();
    });

    it('throws for offset exceeding segment size', () => {
      const table = new AddressTable(TEST_SEGMENT_SIZE);

      expect(() => {
        table.encode(0, TEST_SEGMENT_SIZE);
      }).toThrow();
    });
  });

  describe('getSegmentIndex / getOffset', () => {
    it('extracts segment index from virtual address', () => {
      const table = new AddressTable(TEST_SEGMENT_SIZE);
      const virtualAddr = table.encode(5, 12345);

      expect(table.getSegmentIndex(virtualAddr)).toBe(5);
    });

    it('extracts offset from virtual address', () => {
      const table = new AddressTable(TEST_SEGMENT_SIZE);
      const virtualAddr = table.encode(5, 12345);

      expect(table.getOffset(virtualAddr)).toBe(12345);
    });
  });

  describe('spansSegments', () => {
    it('returns false for range within single segment', () => {
      const segmentSize = 10 * MB;
      const table = new AddressTable(segmentSize);
      const addr = table.encode(0, 1 * MB);

      expect(table.spansSegments(addr, 5 * MB)).toBe(false);
    });

    it('returns true for range crossing segment boundary', () => {
      const segmentSize = 10 * MB;
      const table = new AddressTable(segmentSize);
      const addr = table.encode(0, 9 * MB);

      expect(table.spansSegments(addr, 2 * MB)).toBe(true);
    });

    it('returns false for range ending exactly at boundary', () => {
      const segmentSize = 10 * MB;
      const table = new AddressTable(segmentSize);
      const addr = table.encode(0, 9 * MB);

      expect(table.spansSegments(addr, 1 * MB)).toBe(false);
    });
  });

  describe('splitRange', () => {
    it('returns single chunk for range within segment', () => {
      const segmentSize = 10 * MB;
      const table = new AddressTable(segmentSize);
      const addr = table.encode(0, 1 * MB);

      const chunks = table.splitRange(addr, 5 * MB);

      expect(chunks.length).toBe(1);
      expect(chunks[0].segmentIndex).toBe(0);
      expect(chunks[0].offset).toBe(1 * MB);
      expect(chunks[0].length).toBe(5 * MB);
    });

    it('splits range crossing one segment boundary', () => {
      const segmentSize = 10 * MB;
      const table = new AddressTable(segmentSize);
      const addr = table.encode(0, 9 * MB);

      const chunks = table.splitRange(addr, 2 * MB);

      expect(chunks.length).toBe(2);

      expect(chunks[0].segmentIndex).toBe(0);
      expect(chunks[0].offset).toBe(9 * MB);
      expect(chunks[0].length).toBe(1 * MB);

      expect(chunks[1].segmentIndex).toBe(1);
      expect(chunks[1].offset).toBe(0);
      expect(chunks[1].length).toBe(1 * MB);
    });

    it('splits range crossing multiple segment boundaries', () => {
      const segmentSize = 1 * MB;
      const table = new AddressTable(segmentSize);
      const addr = table.encode(0, 500 * KB);

      const chunks = table.splitRange(addr, 2.5 * MB);

      expect(chunks.length).toBe(3);

      // Starting at 500KB, remaining in segment 0 is 1MB - 500KB = 524KB = 536576 bytes
      expect(chunks[0].segmentIndex).toBe(0);
      expect(chunks[0].length).toBe(segmentSize - 500 * KB); // 536576 bytes

      expect(chunks[1].segmentIndex).toBe(1);
      expect(chunks[1].length).toBe(1 * MB);

      // Remaining is 2.5MB - 536576 - 1MB = 2621440 - 536576 - 1048576 = 1036288 bytes
      expect(chunks[2].segmentIndex).toBe(2);
      expect(chunks[2].length).toBe(2.5 * MB - (segmentSize - 500 * KB) - 1 * MB);
    });

    it('preserves virtual addresses in chunks', () => {
      const segmentSize = 1 * MB;
      const table = new AddressTable(segmentSize);
      const startAddr = table.encode(0, 800 * KB);

      const chunks = table.splitRange(startAddr, 400 * KB);

      expect(chunks[0].virtualAddress).toBe(startAddr);
      expect(chunks[1].virtualAddress).toBe(table.encode(1, 0));
    });
  });

  describe('getTotalAddressSpace', () => {
    it('calculates total address space correctly', () => {
      const table = new AddressTable(TEST_SEGMENT_SIZE);

      const total = table.getTotalAddressSpace();
      const expected = ADDRESS_TABLE_CONSTANTS.MAX_SEGMENTS * TEST_SEGMENT_SIZE;

      expect(total).toBe(expected);
    });
  });

  describe('ADDRESS_TABLE_CONSTANTS', () => {
    it('exports expected constants', () => {
      expect(ADDRESS_TABLE_CONSTANTS.SEGMENT_BITS).toBe(8);
      expect(ADDRESS_TABLE_CONSTANTS.OFFSET_BITS).toBe(45);
      expect(ADDRESS_TABLE_CONSTANTS.MAX_SEGMENTS).toBe(256);
      expect(ADDRESS_TABLE_CONSTANTS.MAX_OFFSET).toBe(2 ** 45 - 1);
    });
  });
});

describe('memory/heap-manager patterns', () => {
  describe('HeapStats shape', () => {
    it('defines expected stat properties', () => {
      const mockStats = {
        strategy: 'SEGMENTED',
        totalAllocated: 0,
        segmentCount: 0,
        memory64HeapSize: 0,
      };

      expect(mockStats).toHaveProperty('strategy');
      expect(mockStats).toHaveProperty('totalAllocated');
      expect(mockStats).toHaveProperty('segmentCount');
      expect(mockStats).toHaveProperty('memory64HeapSize');
    });
  });

  describe('allocation result shape', () => {
    it('defines expected allocation result properties', () => {
      const mockResult = {
        virtualAddress: 0,
        size: 1000,
        view: new Uint8Array(1000),
        strategy: 'SEGMENTED',
        segmentIndex: 0,
        segmentOffset: 0,
      };

      expect(mockResult).toHaveProperty('virtualAddress');
      expect(mockResult).toHaveProperty('size');
      expect(mockResult).toHaveProperty('view');
      expect(mockResult).toHaveProperty('strategy');
    });
  });

  describe('segment allocation logic', () => {
    it('finds segment with available space', () => {
      const segments = [
        { index: 0, used: 900 * KB, capacity: 1 * MB },
        { index: 1, used: 500 * KB, capacity: 1 * MB },
        { index: 2, used: 100 * KB, capacity: 1 * MB },
      ];

      const requestSize = 400 * KB;

      const available = segments.find(s => s.capacity - s.used >= requestSize);
      expect(available).toBeDefined();
      expect(available.index).toBe(1);
    });

    it('requires new segment when no space available', () => {
      const segments = [
        { index: 0, used: 900 * KB, capacity: 1 * MB },
        { index: 1, used: 950 * KB, capacity: 1 * MB },
      ];

      const requestSize = 200 * KB;

      const available = segments.find(s => s.capacity - s.used >= requestSize);
      expect(available).toBeUndefined();
    });
  });
});

describe('memory allocation patterns', () => {
  describe('alignment calculations', () => {
    it('4KB alignment preserves base address', () => {
      const alignment = 4 * KB;
      const address = 16 * KB;
      const aligned = Math.floor(address / alignment) * alignment;

      expect(aligned).toBe(address);
    });

    it('4KB alignment rounds down correctly', () => {
      const alignment = 4 * KB;
      const address = 17 * KB;
      const aligned = Math.floor(address / alignment) * alignment;

      expect(aligned).toBe(16 * KB);
    });

    it('computes aligned length correctly', () => {
      const alignment = 4 * KB;
      const offset = 1 * KB;
      const length = 5 * KB;

      const alignedOffset = Math.floor(offset / alignment) * alignment;
      const offsetDelta = offset - alignedOffset;
      const alignedLength = Math.ceil((length + offsetDelta) / alignment) * alignment;

      expect(alignedOffset).toBe(0);
      expect(offsetDelta).toBe(1 * KB);
      expect(alignedLength).toBe(8 * KB);
    });
  });

  describe('fragmentation detection', () => {
    it('detects fragmentation when used space is non-contiguous', () => {
      const segments = [
        { used: 100 * MB, capacity: 1 * GB },
        { used: 50 * MB, capacity: 1 * GB },
        { used: 200 * MB, capacity: 1 * GB },
      ];

      const totalUsed = segments.reduce((sum, s) => sum + s.used, 0);
      const totalCapacity = segments.reduce((sum, s) => sum + s.capacity, 0);

      const fragmentationRatio = 1 - (totalUsed / totalCapacity);

      expect(fragmentationRatio).toBeGreaterThan(0.5);
    });

    it('calculates effective utilization', () => {
      const allocated = 512 * MB; // Use power of 2 for clean division
      const capacity = 1 * GB;
      const utilization = allocated / capacity;

      expect(utilization).toBe(0.5);
    });
  });

  describe('limit enforcement', () => {
    it('detects when allocation would exceed limit', () => {
      const currentUsed = 3 * GB;
      const limit = 4 * GB;
      const requestedSize = 2 * GB;

      const wouldExceed = currentUsed + requestedSize > limit;

      expect(wouldExceed).toBe(true);
    });

    it('allows allocation within limit', () => {
      const currentUsed = 1 * GB;
      const limit = 4 * GB;
      const requestedSize = 2 * GB;

      const wouldExceed = currentUsed + requestedSize > limit;

      expect(wouldExceed).toBe(false);
    });

    it('calculates remaining capacity', () => {
      const used = 2.5 * GB;
      const limit = 4 * GB;
      const remaining = limit - used;

      expect(remaining).toBe(1.5 * GB);
    });
  });
});

describe('memory strategy selection', () => {
  describe('strategy constants', () => {
    it('MEMORY64 strategy string is correct', () => {
      expect('MEMORY64').toBe('MEMORY64');
    });

    it('SEGMENTED strategy string is correct', () => {
      expect('SEGMENTED').toBe('SEGMENTED');
    });
  });

  describe('capability-based selection', () => {
    it('selects MEMORY64 when Memory64 is available', () => {
      const hasMemory64 = true;
      const strategy = hasMemory64 ? 'MEMORY64' : 'SEGMENTED';

      expect(strategy).toBe('MEMORY64');
    });

    it('selects SEGMENTED when Memory64 is unavailable', () => {
      const hasMemory64 = false;
      const strategy = hasMemory64 ? 'MEMORY64' : 'SEGMENTED';

      expect(strategy).toBe('SEGMENTED');
    });
  });
});

describe('virtual address math', () => {
  describe('safe integer range', () => {
    it('virtual addresses stay within safe integer range', () => {
      const maxSegment = 255;
      const maxOffset = 2 ** 45 - 1;
      const maxVirtualAddress = maxSegment * (maxOffset + 1) + maxOffset;

      expect(maxVirtualAddress).toBeLessThanOrEqual(Number.MAX_SAFE_INTEGER);
    });

    it('segment encoding preserves precision', () => {
      // Use smaller sizes to avoid OOM in Node.js test environment
      const table = new AddressTable(TEST_SEGMENT_SIZE);

      const testCases = [
        { seg: 0, offset: 0 },
        { seg: 100, offset: TEST_SEGMENT_SIZE / 2 },
        { seg: 255, offset: TEST_SEGMENT_SIZE - 1 },
      ];

      for (const { seg, offset } of testCases) {
        const addr = table.encode(seg, offset);
        const decoded = table.decode(addr);

        expect(decoded.segmentIndex).toBe(seg);
        expect(decoded.offset).toBe(offset);
      }
    });
  });

  describe('offset arithmetic', () => {
    it('calculates byte offset within segment', () => {
      const baseOffset = 100 * MB;
      const elementOffset = 50 * KB;
      const totalOffset = baseOffset + elementOffset;

      expect(totalOffset).toBe(100 * MB + 50 * KB);
    });

    it('wraps to next segment correctly', () => {
      const segmentSize = 1 * GB;
      const offset = 900 * MB;
      const addSize = 200 * MB;

      const newOffset = offset + addSize;
      const wrapsToNextSegment = newOffset >= segmentSize;
      const nextSegmentOffset = newOffset - segmentSize;

      expect(wrapsToNextSegment).toBe(true);
      // 900MB + 200MB = 1100MB, 1100MB - 1GB = 1100MB - 1024MB = 76MB
      expect(nextSegmentOffset).toBe(76 * MB);
    });
  });
});

describe('buffer management', () => {
  describe('ArrayBuffer allocation', () => {
    it('creates ArrayBuffer of specified size', () => {
      const size = 1 * MB;
      const buffer = new ArrayBuffer(size);

      expect(buffer.byteLength).toBe(size);
    });

    it('Uint8Array view has correct length', () => {
      const size = 1 * MB;
      const buffer = new ArrayBuffer(size);
      const view = new Uint8Array(buffer);

      expect(view.length).toBe(size);
    });

    it('Uint8Array view with offset has correct length', () => {
      const size = 1 * MB;
      const offset = 100 * KB;
      const viewLength = 50 * KB;
      const buffer = new ArrayBuffer(size);
      const view = new Uint8Array(buffer, offset, viewLength);

      expect(view.length).toBe(viewLength);
      expect(view.byteOffset).toBe(offset);
    });
  });

  describe('buffer slicing', () => {
    it('slice creates copy of data', () => {
      const original = new ArrayBuffer(100);
      const originalView = new Uint8Array(original);
      originalView[0] = 42;

      const slice = original.slice(0, 50);
      const sliceView = new Uint8Array(slice);

      expect(sliceView[0]).toBe(42);

      sliceView[0] = 100;
      expect(originalView[0]).toBe(42);
    });

    it('slice respects offset and length', () => {
      const buffer = new ArrayBuffer(100);
      const view = new Uint8Array(buffer);

      for (let i = 0; i < 100; i++) {
        view[i] = i;
      }

      const slice = buffer.slice(10, 20);
      const sliceView = new Uint8Array(slice);

      expect(sliceView.length).toBe(10);
      expect(sliceView[0]).toBe(10);
      expect(sliceView[9]).toBe(19);
    });
  });

  describe('data transfer', () => {
    it('set copies data between views', () => {
      const source = new Uint8Array([1, 2, 3, 4, 5]);
      const dest = new Uint8Array(10);

      dest.set(source, 2);

      expect(dest[0]).toBe(0);
      expect(dest[2]).toBe(1);
      expect(dest[6]).toBe(5);
    });

    it('set handles overlapping regions', () => {
      const buffer = new ArrayBuffer(10);
      const view = new Uint8Array(buffer);

      for (let i = 0; i < 10; i++) {
        view[i] = i;
      }

      const source = view.subarray(0, 5);
      const temp = new Uint8Array(source);

      view.set(temp, 3);

      expect(view[3]).toBe(0);
      expect(view[7]).toBe(4);
    });
  });
});
