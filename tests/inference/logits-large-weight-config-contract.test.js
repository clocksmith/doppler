import assert from 'node:assert/strict';

import { createDopplerConfig } from '../../src/config/schema/index.js';
import { float32ToFloat16 } from '../../src/converter/quantizer.js';
import { extractLmHeadChunk, resolveLmHeadChunkRows } from '../../src/inference/pipelines/text/logits/gpu.js';

const device = {
  limits: {
    maxStorageBufferBindingSize: 1024,
    maxBufferSize: 2048,
  },
};

const largeWeights = createDopplerConfig().runtime.inference.largeWeights;

assert.equal(resolveLmHeadChunkRows(device, 1, 16, largeWeights), 14);
assert.equal(
  resolveLmHeadChunkRows(device, 1, 16, {
    ...largeWeights,
    lmHeadChunkRows: 9,
  }),
  9
);

{
  const values = new Float32Array([
    1, 2, 3, 4,
    5, 6, 7, 8,
  ]);
  const packed = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    packed[index] = float32ToFloat16(values[index]);
  }
  const bytes = new Uint8Array(packed.buffer.slice(0));
  const rangeCalls = [];
  const chunk = await extractLmHeadChunk(
    {
      kind: 'tensor_range_source',
      sourceDtype: 'f16',
      async loadRange(byteOffset, byteLength) {
        rangeCalls.push([byteOffset, byteLength]);
        return bytes.slice(byteOffset, byteOffset + byteLength);
      },
    },
    'row',
    4,
    2,
    1,
    1,
    'f16'
  );
  assert.deepEqual(rangeCalls, [[8, 8]]);
  assert.deepEqual(Array.from(chunk), [5, 6, 7, 8]);
}

{
  const values = new Float32Array([
    1, 2, 3,
    4, 5, 6,
  ]);
  const packed = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    packed[index] = float32ToFloat16(values[index]);
  }
  const bytes = new Uint8Array(packed.buffer.slice(0));
  const rangeCalls = [];
  const chunk = await extractLmHeadChunk(
    {
      kind: 'tensor_range_source',
      sourceDtype: 'f16',
      async loadRange(byteOffset, byteLength) {
        rangeCalls.push([byteOffset, byteLength]);
        return bytes.slice(byteOffset, byteOffset + byteLength);
      },
    },
    'column',
    2,
    3,
    1,
    2,
    'f16'
  );
  assert.deepEqual(rangeCalls, [
    [2, 4],
    [8, 4],
  ]);
  assert.deepEqual(Array.from(chunk), [2, 3, 5, 6]);
}

console.log('logits-large-weight-config-contract.test: ok');
