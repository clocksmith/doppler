import assert from 'node:assert/strict';

import { AddressTable, ADDRESS_TABLE_CONSTANTS } from '../../src/memory/address-table.js';
import { HeapManager } from '../../src/memory/heap-manager.js';

// ============================================================================
// AddressTable constants
// ============================================================================

{
  assert.equal(ADDRESS_TABLE_CONSTANTS.SEGMENT_BITS, 8);
  assert.equal(ADDRESS_TABLE_CONSTANTS.OFFSET_BITS, 45);
  assert.equal(ADDRESS_TABLE_CONSTANTS.MAX_SEGMENTS, 256);
  assert.equal(ADDRESS_TABLE_CONSTANTS.MAX_OFFSET, 2 ** 45 - 1);
}

// ============================================================================
// AddressTable: encode/decode round-trip
// ============================================================================

{
  const segmentSize = 1024 * 1024 * 1024; // 1GB
  const at = new AddressTable(segmentSize);

  // Segment 0, offset 0
  const addr0 = at.encode(0, 0);
  const dec0 = at.decode(addr0);
  assert.equal(dec0.segmentIndex, 0);
  assert.equal(dec0.offset, 0);

  // Segment 0, offset 100
  const addr1 = at.encode(0, 100);
  const dec1 = at.decode(addr1);
  assert.equal(dec1.segmentIndex, 0);
  assert.equal(dec1.offset, 100);

  // Segment 1, offset 50
  const addr2 = at.encode(1, 50);
  const dec2 = at.decode(addr2);
  assert.equal(dec2.segmentIndex, 1);
  assert.equal(dec2.offset, 50);

  // High segment index
  const addr3 = at.encode(255, 0);
  const dec3 = at.decode(addr3);
  assert.equal(dec3.segmentIndex, 255);
  assert.equal(dec3.offset, 0);

  // High offset
  const maxOffset = segmentSize - 1;
  const addr4 = at.encode(0, maxOffset);
  const dec4 = at.decode(addr4);
  assert.equal(dec4.segmentIndex, 0);
  assert.equal(dec4.offset, maxOffset);

  // Segment 255 with large offset
  const addr5 = at.encode(255, maxOffset);
  const dec5 = at.decode(addr5);
  assert.equal(dec5.segmentIndex, 255);
  assert.equal(dec5.offset, maxOffset);
}

// ============================================================================
// AddressTable: getSegmentIndex / getOffset helpers
// ============================================================================

{
  const at = new AddressTable(4096);

  const addr = at.encode(3, 500);
  assert.equal(at.getSegmentIndex(addr), 3);
  assert.equal(at.getOffset(addr), 500);
}

// ============================================================================
// AddressTable: spansSegments
// ============================================================================

{
  const segmentSize = 1024;
  const at = new AddressTable(segmentSize);

  // Fits within one segment
  const addr1 = at.encode(0, 0);
  assert.equal(at.spansSegments(addr1, 100), false);
  assert.equal(at.spansSegments(addr1, segmentSize), false);

  // Spans to next segment
  const addr2 = at.encode(0, segmentSize - 10);
  assert.equal(at.spansSegments(addr2, 20), true);

  // Single byte at end of segment does not span
  assert.equal(at.spansSegments(at.encode(0, segmentSize - 1), 1), false);
}

// ============================================================================
// AddressTable: splitRange within one segment
// ============================================================================

{
  const segmentSize = 1024;
  const at = new AddressTable(segmentSize);

  const addr = at.encode(0, 100);
  const chunks = at.splitRange(addr, 200);
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].segmentIndex, 0);
  assert.equal(chunks[0].offset, 100);
  assert.equal(chunks[0].length, 200);
}

// ============================================================================
// AddressTable: splitRange across two segments
// ============================================================================

{
  const segmentSize = 1024;
  const at = new AddressTable(segmentSize);

  const addr = at.encode(0, 900);
  const chunks = at.splitRange(addr, 300);

  assert.equal(chunks.length, 2);

  // First chunk: remainder of segment 0
  assert.equal(chunks[0].segmentIndex, 0);
  assert.equal(chunks[0].offset, 900);
  assert.equal(chunks[0].length, 124); // 1024 - 900

  // Second chunk: beginning of segment 1
  assert.equal(chunks[1].segmentIndex, 1);
  assert.equal(chunks[1].offset, 0);
  assert.equal(chunks[1].length, 176); // 300 - 124
}

// ============================================================================
// AddressTable: splitRange across three segments
// ============================================================================

{
  const segmentSize = 100;
  const at = new AddressTable(segmentSize);

  const addr = at.encode(0, 80);
  const chunks = at.splitRange(addr, 250);

  assert.equal(chunks.length, 3);
  assert.equal(chunks[0].segmentIndex, 0);
  assert.equal(chunks[0].length, 20); // 100 - 80
  assert.equal(chunks[1].segmentIndex, 1);
  assert.equal(chunks[1].length, 100); // full segment
  assert.equal(chunks[2].segmentIndex, 2);
  assert.equal(chunks[2].length, 130); // 250 - 20 - 100

  // Sum of chunk lengths matches requested length
  const totalLen = chunks.reduce((s, c) => s + c.length, 0);
  assert.equal(totalLen, 250);
}

// ============================================================================
// AddressTable: getTotalAddressSpace
// ============================================================================

{
  const segmentSize = 4096;
  const at = new AddressTable(segmentSize);
  assert.equal(at.getTotalAddressSpace(), 256 * segmentSize);
}

// ============================================================================
// AddressTable: encode validation - segment index out of range
// ============================================================================

{
  const at = new AddressTable(1024);
  assert.throws(
    () => at.encode(256, 0),
    /exceeds max/
  );
}

// ============================================================================
// AddressTable: encode validation - offset exceeds segment size
// ============================================================================

{
  const at = new AddressTable(1024);
  assert.throws(
    () => at.encode(0, 1024),
    /exceeds segment size/
  );
  assert.throws(
    () => at.encode(0, 2000),
    /exceeds segment size/
  );
}

// ============================================================================
// AddressTable: constructor validation - segment too large
// ============================================================================

{
  assert.throws(
    () => new AddressTable(2 ** 45),
    /exceeds max offset/
  );
}

// ============================================================================
// HeapManager: allocate before init throws
// ============================================================================

{
  const hm = new HeapManager();
  assert.throws(
    () => hm.allocate(100),
    /not initialized/
  );
}

// ============================================================================
// HeapManager: init sets up SEGMENTED strategy in Node
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const stats = hm.getStats();
  assert.equal(stats.strategy, 'SEGMENTED');
  assert.equal(stats.totalAllocated, 0);
  assert.ok(stats.segmentCount >= 1);
}

// ============================================================================
// HeapManager: double init is idempotent
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();
  await hm.init();

  const stats = hm.getStats();
  assert.equal(stats.strategy, 'SEGMENTED');
}

// ============================================================================
// HeapManager: basic allocation returns valid result
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const alloc = hm.allocate(256);
  assert.equal(alloc.size, 256);
  assert.equal(alloc.strategy, 'SEGMENTED');
  assert.equal(typeof alloc.virtualAddress, 'number');
  assert.ok(alloc.view instanceof Uint8Array);
  assert.equal(alloc.view.length, 256);
  assert.equal(typeof alloc.segmentIndex, 'number');
  assert.equal(typeof alloc.segmentOffset, 'number');
}

// ============================================================================
// HeapManager: sequential allocations produce non-overlapping regions
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const a1 = hm.allocate(100);
  const a2 = hm.allocate(200);
  const a3 = hm.allocate(300);

  assert.equal(a2.virtualAddress, a1.virtualAddress + a1.size);
  assert.equal(a3.virtualAddress, a2.virtualAddress + a2.size);

  const stats = hm.getStats();
  assert.equal(stats.totalAllocated, 600);
}

// ============================================================================
// HeapManager: write and read data
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const alloc = hm.allocate(8);
  const data = new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]);
  hm.write(alloc.virtualAddress, data);

  const readBack = hm.read(alloc.virtualAddress, 8);
  assert.deepEqual([...readBack], [10, 20, 30, 40, 50, 60, 70, 80]);
}

// ============================================================================
// HeapManager: write isolation between allocations
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const a1 = hm.allocate(4);
  const a2 = hm.allocate(4);

  hm.write(a1.virtualAddress, new Uint8Array([1, 2, 3, 4]));
  hm.write(a2.virtualAddress, new Uint8Array([5, 6, 7, 8]));

  assert.deepEqual([...hm.read(a1.virtualAddress, 4)], [1, 2, 3, 4]);
  assert.deepEqual([...hm.read(a2.virtualAddress, 4)], [5, 6, 7, 8]);
}

// ============================================================================
// HeapManager: getBufferSlice returns a copy
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const alloc = hm.allocate(16);
  const data = new Uint8Array(16);
  for (let i = 0; i < 16; i++) data[i] = i + 1;
  hm.write(alloc.virtualAddress, data);

  const slice = hm.getBufferSlice(alloc.virtualAddress, 8);
  assert.equal(slice.byteLength, 8);
  assert.deepEqual([...new Uint8Array(slice)], [1, 2, 3, 4, 5, 6, 7, 8]);

  // Modifying slice does not affect heap
  new Uint8Array(slice)[0] = 99;
  const heapByte = hm.read(alloc.virtualAddress, 1);
  assert.equal(heapByte[0], 1);
}

// ============================================================================
// HeapManager: zero-size allocation returns valid empty result
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const alloc = hm.allocate(0);
  assert.equal(alloc.size, 0);
  assert.equal(alloc.view.length, 0);
}

// ============================================================================
// HeapManager: allocate via view produces readable data
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const alloc = hm.allocate(4);
  alloc.view[0] = 0xAA;
  alloc.view[1] = 0xBB;
  alloc.view[2] = 0xCC;
  alloc.view[3] = 0xDD;

  const readBack = hm.read(alloc.virtualAddress, 4);
  assert.equal(readBack[0], 0xAA);
  assert.equal(readBack[1], 0xBB);
  assert.equal(readBack[2], 0xCC);
  assert.equal(readBack[3], 0xDD);
}

// ============================================================================
// HeapManager: reset clears totalAllocated and re-creates first segment
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  hm.allocate(1024);
  hm.allocate(2048);
  assert.equal(hm.getStats().totalAllocated, 3072);

  hm.reset();

  const stats = hm.getStats();
  assert.equal(stats.totalAllocated, 0);
  assert.ok(stats.segmentCount >= 1);

  // Can allocate again after reset
  const alloc = hm.allocate(512);
  assert.equal(alloc.size, 512);
  assert.equal(hm.getStats().totalAllocated, 512);
}

// ============================================================================
// HeapManager: many small allocations maintain capacity tracking
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const count = 100;
  const size = 64;
  for (let i = 0; i < count; i++) {
    hm.allocate(size);
  }

  assert.equal(hm.getStats().totalAllocated, count * size);
}

// ============================================================================
// HeapManager: large allocation that fits within segment
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const size = 4 * 1024 * 1024; // 4MB
  const alloc = hm.allocate(size);
  assert.equal(alloc.size, size);
  assert.equal(alloc.view.length, size);

  // Write at beginning and end
  alloc.view[0] = 0x01;
  alloc.view[size - 1] = 0xFF;

  const first = hm.read(alloc.virtualAddress, 1);
  assert.equal(first[0], 0x01);

  const last = hm.read(alloc.virtualAddress + size - 1, 1);
  assert.equal(last[0], 0xFF);
}

// ============================================================================
// HeapManager: getStats shape
// ============================================================================

{
  const hm = new HeapManager();
  await hm.init();

  const stats = hm.getStats();
  assert.equal(typeof stats.strategy, 'string');
  assert.equal(typeof stats.totalAllocated, 'number');
  assert.equal(typeof stats.segmentCount, 'number');
  assert.equal(typeof stats.memory64HeapSize, 'number');
  assert.equal(stats.memory64HeapSize, 0);
}

// ============================================================================
// AddressTable: encode/decode consistency across many values
// ============================================================================

{
  const at = new AddressTable(1024 * 1024); // 1MB segments
  const pairs = [
    [0, 0],
    [0, 1],
    [0, 1023],
    [1, 0],
    [127, 500000],
    [255, 1024 * 1024 - 1],
  ];

  for (const [seg, off] of pairs) {
    const addr = at.encode(seg, off);
    const decoded = at.decode(addr);
    assert.equal(decoded.segmentIndex, seg, `segment mismatch for [${seg}, ${off}]`);
    assert.equal(decoded.offset, off, `offset mismatch for [${seg}, ${off}]`);
  }
}

// ============================================================================
// AddressTable: splitRange zero-length returns empty array
// ============================================================================

{
  const at = new AddressTable(1024);
  const chunks = at.splitRange(at.encode(0, 0), 0);
  assert.equal(chunks.length, 0);
}

// ============================================================================
// AddressTable: splitRange exactly one segment
// ============================================================================

{
  const segmentSize = 512;
  const at = new AddressTable(segmentSize);

  const chunks = at.splitRange(at.encode(2, 0), segmentSize);
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0].segmentIndex, 2);
  assert.equal(chunks[0].offset, 0);
  assert.equal(chunks[0].length, segmentSize);
}

console.log('heap-manager.test: ok');
