import assert from 'node:assert/strict';

import { resolveProjectionSliceOffsetBytes } from '../../src/inference/pipelines/text/attention/projections.js';

function makeWeight(dtype, layout = 'row', shape = [2048, 1024]) {
  return {
    buffer: {},
    dtype,
    layout,
    shape,
  };
}

{
  const q4kWeight = makeWeight('q4k', 'row', [4096, 1024]);
  const offset = resolveProjectionSliceOffsetBytes(q4kWeight, 2048, 1024);
  // Q4K packs 256 logical elements into 144 bytes.
  const expected = 2048 * Math.ceil(1024 / 256) * 144;
  assert.equal(offset, expected);
}

{
  const f16Weight = makeWeight('f16', 'row', [4096, 1024]);
  const offset = resolveProjectionSliceOffsetBytes(f16Weight, 2048, 1024);
  assert.equal(offset, 2048 * 1024 * 2);
}

{
  const f32Weight = makeWeight('f32', 'row', [4096, 1024]);
  const offset = resolveProjectionSliceOffsetBytes(f32Weight, 2048, 1024);
  assert.equal(offset, 2048 * 1024 * 4);
}

{
  const q4kColWeight = makeWeight('q4k', 'column', [4096, 1024]);
  assert.throws(
    () => resolveProjectionSliceOffsetBytes(q4kColWeight, 2048, 1024),
    /unsupported q4k layout/i
  );
}

{
  const offset = resolveProjectionSliceOffsetBytes(makeWeight('q4k'), 0, 1024);
  assert.equal(offset, 0);
}

console.log('attention-projection-offsets.test: ok');
