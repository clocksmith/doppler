import assert from 'node:assert/strict';

import { createPagedByteReader } from '../../src/formats/paged-source-reader.js';

function countingSource(size, { name = 'counting source', delay = false } = {}) {
  const bytes = Uint8Array.from({ length: size }, (_, index) => index % 251);
  const calls = [];
  return {
    name,
    size,
    calls,
    bytes,
    readRange(offset, length) {
      calls.push([offset, length]);
      const slice = bytes.slice(offset, offset + length);
      return delay ? new Promise((resolve) => setTimeout(() => resolve(slice), 1)) : slice;
    },
  };
}

const options = { pageBytes: 16, parser: 'Test parser', defaultLabel: 'test source' };

{
  // Three simultaneous reads of one page share a single readRange.
  const source = countingSource(64, { delay: true });
  const reader = createPagedByteReader(source, options);
  const [a, b, c] = await Promise.all([reader.readSlice(0, 4), reader.readSlice(4, 4), reader.readUint32(8)]);
  assert.deepEqual(Array.from(a), [0, 1, 2, 3]);
  assert.deepEqual(Array.from(b), [4, 5, 6, 7]);
  assert.equal(c, new DataView(source.bytes.buffer, 8, 4).getUint32(0, true));
  assert.deepEqual(source.calls, [[0, 16]], 'one page fetch for three concurrent reads');
  await reader.readSlice(2, 2);
  assert.equal(source.calls.length, 1, 'a settled page is served from cache');
}

{
  // Cross-page reads assemble bytes in order and fetch each page once.
  const source = countingSource(64);
  const reader = createPagedByteReader(source, options);
  const slice = await reader.readSlice(12, 24);
  assert.deepEqual(Array.from(slice), Array.from(source.bytes.subarray(12, 36)));
  assert.deepEqual(source.calls, [[0, 16], [16, 16], [32, 16]]);
  assert.equal(reader.label, 'counting source');
  assert.equal(reader.size, 64);
  assert.deepEqual(Array.from(await reader.readSlice(60, 0)), []);
  assert.equal(await reader.readUint8(63), 63);
  assert.equal(await reader.readInt16(0), new DataView(source.bytes.buffer, 0, 2).getInt16(0, true));
  assert.equal(await reader.readFloat64(16), new DataView(source.bytes.buffer, 16, 8).getFloat64(0, true));
  assert.equal(await reader.readBigUint64(8), new DataView(source.bytes.buffer, 8, 8).getBigUint64(0, true));
}

{
  // Bounds, short reads and bad sources fail with the owning parser's label.
  const source = countingSource(40);
  const reader = createPagedByteReader(source, options);
  await assert.rejects(() => reader.readSlice(36, 8), /Test parser: read exceeds counting source bounds/);
  await assert.rejects(() => reader.readSlice(Number.NaN, 4), /Test parser: invalid read request/);
  const short = { name: 'short', size: 40, readRange: () => new Uint8Array(3) };
  await assert.rejects(() => createPagedByteReader(short, options).readSlice(0, 4), /Test parser: short read for short page 0/);
  const retry = createPagedByteReader({ size: 40, readRange: (() => { let first = true; return () => { if (first) { first = false; throw new Error('transient'); } return new Uint8Array(16); }; })() }, options);
  await assert.rejects(() => retry.readSlice(0, 4), /transient/);
  assert.deepEqual(Array.from(await retry.readSlice(0, 4)), [0, 0, 0, 0], 'a failed page fetch is not cached');
  assert.equal(retry.label, 'test source');
  assert.throws(() => createPagedByteReader({ size: 4, readRange() {} }, options), /source\.size must be a finite byte length >= 8/);
  assert.throws(() => createPagedByteReader({ size: 40 }, options), /source\.readRange\(offset, length\) is required/);
  assert.throws(() => createPagedByteReader(source, { ...options, pageBytes: 0 }), /pageBytes must be a positive integer/);
}

console.log('paged-source-reader: shared paging, in-flight reuse and bounds verified');
