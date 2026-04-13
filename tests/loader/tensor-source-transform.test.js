import assert from 'node:assert/strict';

const { assembleShardData } = await import('../../src/loader/tensors/tensor-reader.js');
const { loadTensorRange } = await import('../../src/loader/tensors/tensor-reader.js');
const { loadTensorToCPU } = await import('../../src/loader/tensors/tensor-loader.js');

function assertApproxArray(actual, expected, epsilon = 1e-3) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index++) {
    const delta = Math.abs(actual[index] - expected[index]);
    assert.ok(
      delta <= epsilon,
      `value mismatch at ${index}: expected ${expected[index]}, got ${actual[index]}`
    );
  }
}

const int8Location = {
  shardIndex: 0,
  offset: 0,
  size: 4,
  shape: [2, 2],
  dtype: 'F16',
  role: 'matmul',
  sourceTransform: {
    kind: 'affine_dequant',
    scheme: 'per_tensor_affine',
    sourceDtype: 'INT8',
    targetDtype: 'F16',
    scale: 0.25,
    zeroPoint: 1,
  },
};
const int8Bytes = Uint8Array.from([1, 2, 3, 4]);
const int8Dequantized = await assembleShardData(
  int8Location,
  'model.layers.0.self_attn.q_proj.weight',
  async () => int8Bytes.buffer.slice(0),
  async (_index, offset, length) => int8Bytes.slice(offset, offset + length).buffer
);
assert.equal(int8Dequantized.byteLength, 8);
const int8Cpu = loadTensorToCPU(int8Dequantized, int8Location);
assert.ok(int8Cpu instanceof Float32Array);
assertApproxArray(Array.from(int8Cpu), [0, 0.25, 0.5, 0.75]);

const int4Location = {
  shardIndex: 0,
  offset: 0,
  size: 2,
  shape: [4],
  dtype: 'F16',
  role: 'matmul',
  sourceTransform: {
    kind: 'affine_dequant',
    scheme: 'per_tensor_affine',
    sourceDtype: 'INT4',
    targetDtype: 'F16',
    scale: 0.5,
    zeroPoint: 0,
  },
};
const int4Bytes = Uint8Array.from([0x21, 0xfe]);
const int4Dequantized = await assembleShardData(
  int4Location,
  'model.layers.0.mlp.gate_proj.weight',
  async () => int4Bytes.buffer.slice(0),
  async (_index, offset, length) => int4Bytes.slice(offset, offset + length).buffer
);
const int4Cpu = loadTensorToCPU(int4Dequantized, int4Location);
assert.ok(int4Cpu instanceof Float32Array);
assertApproxArray(Array.from(int4Cpu), [0.5, 1.0, -1.0, -0.5]);

const litertRowwiseLocation = {
  shardIndex: 0,
  offset: 0,
  size: 2,
  shape: [2, 2],
  dtype: 'F16',
  role: 'matmul',
  sourceTransform: {
    kind: 'litert_rowwise_dequant',
    scheme: 'per_row_affine',
    sourceDtype: 'INT4',
    targetDtype: 'F16',
    storageEncoding: 'offset_binary',
    scaleSource: {
      shard: 1,
      offset: 0,
      size: 8,
    },
    rowSumSource: {
      shard: 2,
      offset: 0,
      size: 8,
    },
  },
};
const litertRawBytes = Uint8Array.from([0x98, 0xba]);
const litertScaleBytes = new Uint8Array(Float32Array.from([0.5, 1.0]).buffer);
const litertRowSumBytes = new Uint8Array(Int32Array.from([1, 1]).buffer);
const litertShards = new Map([
  [0, litertRawBytes],
  [1, litertScaleBytes],
  [2, litertRowSumBytes],
]);
const litertDequantized = await assembleShardData(
  litertRowwiseLocation,
  'model.language_model.layers.0.self_attn.q_proj.weight',
  async (index) => litertShards.get(index).buffer.slice(0),
  async (index, offset, length) => litertShards.get(index).slice(offset, offset + length).buffer
);
const litertCpu = loadTensorToCPU(litertDequantized, litertRowwiseLocation);
assert.ok(litertCpu instanceof Float32Array);
assertApproxArray(Array.from(litertCpu), [0, 0.5, 0, 1]);

const litertRowRange = await loadTensorRange(
  litertRowwiseLocation,
  'model.language_model.layers.0.self_attn.q_proj.weight',
  4,
  4,
  async (index, offset, length) => litertShards.get(index).slice(offset, offset + length).buffer
);
const litertRowRangeCpu = loadTensorToCPU(litertRowRange, {
  ...litertRowwiseLocation,
  shape: [2],
  size: 4,
  sourceTransform: undefined,
  dtype: 'F16',
});
assert.ok(litertRowRangeCpu instanceof Float32Array);
assertApproxArray(Array.from(litertRowRangeCpu), [0, 1]);

const litertInt2Location = {
  shardIndex: 0,
  offset: 0,
  size: 2,
  shape: [2, 4],
  dtype: 'F16',
  role: 'matmul',
  sourceTransform: {
    kind: 'litert_rowwise_dequant',
    scheme: 'per_row_affine',
    sourceDtype: 'INT2',
    targetDtype: 'F16',
    storageEncoding: 'offset_binary',
    scaleSource: {
      shard: 1,
      offset: 0,
      size: 8,
    },
    rowSumSource: {
      shard: 2,
      offset: 0,
      size: 8,
    },
  },
};
const litertInt2RawBytes = Uint8Array.from([0xe4, 0x1b]);
const litertInt2ScaleBytes = new Uint8Array(Float32Array.from([0.5, 1.0]).buffer);
const litertInt2RowSumBytes = new Uint8Array(Int32Array.from([0, 0]).buffer);
const litertInt2Shards = new Map([
  [0, litertInt2RawBytes],
  [1, litertInt2ScaleBytes],
  [2, litertInt2RowSumBytes],
]);
const litertInt2Dequantized = await assembleShardData(
  litertInt2Location,
  'model.language_model.layers.15.mlp.gate_proj.weight',
  async (index) => litertInt2Shards.get(index).buffer.slice(0),
  async (index, offset, length) => litertInt2Shards.get(index).slice(offset, offset + length).buffer
);
const litertInt2Cpu = loadTensorToCPU(litertInt2Dequantized, litertInt2Location);
assert.ok(litertInt2Cpu instanceof Float32Array);
assertApproxArray(Array.from(litertInt2Cpu), [-0.75, -0.25, 0.25, 0.75, 1.5, 0.5, -0.5, -1.5]);

console.log('tensor-source-transform.test: ok');
