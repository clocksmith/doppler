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

function cpuLocation(dtype, size, shape) {
  return {
    shardIndex: 0,
    offset: 0,
    size,
    shape,
    dtype,
    role: 'matmul',
  };
}

{
  const f16Bytes = Uint8Array.from([0x00, 0x3c, 0x00, 0xc0, 0x00, 0x38, 0x00, 0x34]);
  const aligned = loadTensorToCPU(f16Bytes, cpuLocation('F16', f16Bytes.byteLength, [4]));
  assert.ok(aligned instanceof Float32Array);
  assertApproxArray(Array.from(aligned), [1, -2, 0.5, 0.25]);

  const backing = Uint8Array.from([0xff, ...f16Bytes]);
  const unalignedBytes = backing.subarray(1);
  const unaligned = loadTensorToCPU(unalignedBytes, cpuLocation('F16', unalignedBytes.byteLength, [4]));
  assert.ok(unaligned instanceof Float32Array);
  assertApproxArray(Array.from(unaligned), [1, -2, 0.5, 0.25]);
}

{
  const bf16Bytes = Uint8Array.from([0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3f, 0x80, 0x3e]);
  const aligned = loadTensorToCPU(bf16Bytes, cpuLocation('BF16', bf16Bytes.byteLength, [4]));
  assert.ok(aligned instanceof Float32Array);
  assertApproxArray(Array.from(aligned), [1, -2, 0.5, 0.25]);

  const backing = Uint8Array.from([0xff, ...bf16Bytes]);
  const unalignedBytes = backing.subarray(1);
  const unaligned = loadTensorToCPU(unalignedBytes, cpuLocation('BF16', unalignedBytes.byteLength, [4]));
  assert.ok(unaligned instanceof Float32Array);
  assertApproxArray(Array.from(unaligned), [1, -2, 0.5, 0.25]);
}

{
  const f32Source = new Float32Array([1, -2, 0.5, 0.25]);
  const exact = loadTensorToCPU(f32Source, cpuLocation('F32', f32Source.byteLength, [4]));
  assert.ok(exact instanceof Float32Array);
  assert.equal(exact.buffer, f32Source.buffer);
  assertApproxArray(Array.from(exact), [1, -2, 0.5, 0.25]);

  const backing = new Uint8Array(f32Source.byteLength + 4);
  backing.set(new Uint8Array(f32Source.buffer), 4);
  const rangedBytes = backing.subarray(4);
  const ranged = loadTensorToCPU(rangedBytes, cpuLocation('F32', rangedBytes.byteLength, [4]));
  assert.ok(ranged instanceof Float32Array);
  assert.notEqual(ranged.buffer, backing.buffer);
  assertApproxArray(Array.from(ranged), [1, -2, 0.5, 0.25]);
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
    scaleSemantics: 'step',
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

const litertAxisSignedNoSumLocation = {
  shardIndex: 0,
  offset: 0,
  size: 4,
  shape: [2, 4],
  dtype: 'F16',
  role: 'embedding',
  sourceTransform: {
    kind: 'litert_axis_dequant',
    scheme: 'per_axis_affine',
    sourceDtype: 'INT4',
    targetDtype: 'F16',
    storageEncoding: 'signed',
    scaleSemantics: 'step',
    storageShape: [2, 4],
    quantAxis: 1,
    scaleSource: {
      shard: 1,
      offset: 0,
      size: 8,
    },
  },
};
const litertAxisSignedNoSumRawBytes = Uint8Array.from([0x10, 0x2f, 0xe3, 0x10]);
const litertAxisSignedNoSumScaleBytes = new Uint8Array(Float32Array.from([0.5, 0.25]).buffer);
const litertAxisSignedNoSumShards = new Map([
  [0, litertAxisSignedNoSumRawBytes],
  [1, litertAxisSignedNoSumScaleBytes],
]);
const litertAxisSignedNoSumDequantized = await assembleShardData(
  litertAxisSignedNoSumLocation,
  'model.language_model.layers.0.embed_tokens_per_layer.weight',
  async (index) => litertAxisSignedNoSumShards.get(index).buffer.slice(0),
  async (index, offset, length) => litertAxisSignedNoSumShards.get(index).slice(offset, offset + length).buffer
);
const litertAxisSignedNoSumCpu = loadTensorToCPU(
  litertAxisSignedNoSumDequantized,
  litertAxisSignedNoSumLocation
);
assert.ok(litertAxisSignedNoSumCpu instanceof Float32Array);
assertApproxArray(
  Array.from(litertAxisSignedNoSumCpu),
  [0, 0.5, -0.5, 1, 0.75, -0.5, 0, 0.25]
);

const litertAxisUint8CompanionLocation = {
  shardIndex: 0,
  offset: 0,
  size: 8,
  shape: [2, 4],
  dtype: 'F16',
  role: 'embedding',
  sourceTransform: {
    kind: 'litert_axis_dequant',
    scheme: 'per_axis_affine',
    sourceDtype: 'UINT8',
    targetDtype: 'F16',
    storageEncoding: 'signed',
    scaleSemantics: 'step',
    storageShape: [2, 4],
    quantAxis: 1,
    scaleCompanionDtype: 'UINT8',
    scaleCompanionDequant: {
      scale: 0.01,
      zeroPoint: 0,
    },
    scaleSource: {
      shard: 1,
      offset: 0,
      size: 8,
    },
  },
};
const litertAxisUint8CompanionRawBytes = Uint8Array.from([0, 1, 2, 3, 4, 3, 2, 1]);
const litertAxisUint8CompanionScaleBytes = Uint8Array.from([1, 2, 3, 4, 4, 3, 2, 1]);
const litertAxisUint8CompanionShards = new Map([
  [0, litertAxisUint8CompanionRawBytes],
  [1, litertAxisUint8CompanionScaleBytes],
]);
const litertAxisUint8CompanionDequantized = await assembleShardData(
  litertAxisUint8CompanionLocation,
  'model.language_model.layers.0.embed_tokens_per_layer.weight',
  async (index) => litertAxisUint8CompanionShards.get(index).buffer.slice(0),
  async (index, offset, length) => litertAxisUint8CompanionShards.get(index).slice(offset, offset + length).buffer
);
const litertAxisUint8CompanionCpu = loadTensorToCPU(
  litertAxisUint8CompanionDequantized,
  litertAxisUint8CompanionLocation
);
assert.ok(litertAxisUint8CompanionCpu instanceof Float32Array);
assertApproxArray(
  Array.from(litertAxisUint8CompanionCpu),
  [
    0,
    0.019989013671875,
    0.05999755859375,
    0.1199951171875,
    0.159912109375,
    0.0899658203125,
    0.03997802734375,
    0.0099945068359375,
  ]
);

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
    scaleSemantics: 'step',
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

const litertQmaxAbsLocation = {
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
    scaleSemantics: 'qmax_abs',
    scaleDivisor: 8,
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
const litertQmaxAbsRawBytes = Uint8Array.from([0x98, 0xba]);
const litertQmaxAbsScaleBytes = new Uint8Array(Float32Array.from([0.5, 1.0]).buffer);
const litertQmaxAbsRowSumBytes = new Uint8Array(Int32Array.from([1, 1]).buffer);
const litertQmaxAbsShards = new Map([
  [0, litertQmaxAbsRawBytes],
  [1, litertQmaxAbsScaleBytes],
  [2, litertQmaxAbsRowSumBytes],
]);
const litertQmaxAbsDequantized = await assembleShardData(
  litertQmaxAbsLocation,
  'model.language_model.layers.0.self_attn.k_proj.weight',
  async (index) => litertQmaxAbsShards.get(index).buffer.slice(0),
  async (index, offset, length) => litertQmaxAbsShards.get(index).slice(offset, offset + length).buffer
);
const litertQmaxAbsCpu = loadTensorToCPU(litertQmaxAbsDequantized, litertQmaxAbsLocation);
assert.ok(litertQmaxAbsCpu instanceof Float32Array);
assertApproxArray(Array.from(litertQmaxAbsCpu), [0, 0.0625, 0, 0.125]);

const litertBlockedAxisLocation = {
  shardIndex: 0,
  offset: 0,
  size: 4,
  shape: [2, 8],
  dtype: 'F16',
  role: 'embedding',
  sourceTransform: {
    kind: 'litert_axis_blocked_dequant',
    scheme: 'per_axis_affine',
    sourceDtype: 'INT2',
    targetDtype: 'F16',
    storageEncoding: 'offset_binary',
    scaleSemantics: 'step',
    storageShape: [2, 2],
    quantAxis: 0,
    storageBlockSize: 4,
    storageLaneOrder: [0, 1, 2, 3],
    scaleSource: {
      shard: 1,
      offset: 0,
      size: 8,
    },
    sumSource: {
      shard: 2,
      offset: 0,
      size: 8,
    },
  },
};
const litertBlockedAxisRawBytes = Uint8Array.from([0xe4, 0x1b, 0x1b, 0xe4]);
const litertBlockedAxisScaleBytes = new Uint8Array(Float32Array.from([0.5, 1.0]).buffer);
const litertBlockedAxisSumBytes = new Uint8Array(Int32Array.from([0, 0]).buffer);
const litertBlockedAxisShards = new Map([
  [0, litertBlockedAxisRawBytes],
  [1, litertBlockedAxisScaleBytes],
  [2, litertBlockedAxisSumBytes],
]);
const litertBlockedAxisDequantized = await assembleShardData(
  litertBlockedAxisLocation,
  'model.language_model.embed_tokens.weight',
  async (index) => litertBlockedAxisShards.get(index).buffer.slice(0),
  async (index, offset, length) => litertBlockedAxisShards.get(index).slice(offset, offset + length).buffer
);
const litertBlockedAxisCpu = loadTensorToCPU(litertBlockedAxisDequantized, litertBlockedAxisLocation);
assert.ok(litertBlockedAxisCpu instanceof Float32Array);
assertApproxArray(
  Array.from(litertBlockedAxisCpu),
  [
    -0.75, -0.25, 0.25, 0.75, 0.75, 0.25, -0.25, -0.75,
    1.5, 0.5, -0.5, -1.5, -1.5, -0.5, 0.5, 1.5,
  ]
);

const litertBlockedAxisRowRange = await loadTensorRange(
  litertBlockedAxisLocation,
  'model.language_model.embed_tokens.weight',
  16,
  16,
  async (index, offset, length) => litertBlockedAxisShards.get(index).slice(offset, offset + length).buffer
);
const litertBlockedAxisRowRangeCpu = loadTensorToCPU(litertBlockedAxisRowRange, {
  ...litertBlockedAxisLocation,
  shape: [8],
  size: 16,
  sourceTransform: undefined,
  dtype: 'F16',
});
assert.ok(litertBlockedAxisRowRangeCpu instanceof Float32Array);
assertApproxArray(Array.from(litertBlockedAxisRowRangeCpu), [1.5, 0.5, -0.5, -1.5, -1.5, -0.5, 0.5, 1.5]);

const litertBlockedAxisQmaxAbsLocation = {
  ...litertBlockedAxisLocation,
  sourceTransform: {
    ...litertBlockedAxisLocation.sourceTransform,
    scaleSemantics: 'qmax_abs',
    scaleDivisor: 3,
  },
};
const litertBlockedAxisQmaxAbsDequantized = await assembleShardData(
  litertBlockedAxisQmaxAbsLocation,
  'model.language_model.embed_tokens.weight',
  async (index) => litertBlockedAxisShards.get(index).buffer.slice(0),
  async (index, offset, length) => litertBlockedAxisShards.get(index).slice(offset, offset + length).buffer
);
const litertBlockedAxisQmaxAbsCpu = loadTensorToCPU(
  litertBlockedAxisQmaxAbsDequantized,
  litertBlockedAxisQmaxAbsLocation
);
assert.ok(litertBlockedAxisQmaxAbsCpu instanceof Float32Array);
assertApproxArray(
  Array.from(litertBlockedAxisQmaxAbsCpu),
  [
    -0.25, -0.08331298828125, 0.08331298828125, 0.25,
    0.25, 0.08331298828125, -0.08331298828125, -0.25,
    0.5, 0.1666259765625, -0.1666259765625, -0.5,
    -0.5, -0.1666259765625, 0.1666259765625, 0.5,
  ]
);

console.log('tensor-source-transform.test: ok');
