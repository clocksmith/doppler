import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

class FakeBuffer {
  constructor({ size, usage = 0 }) {
    this.size = size;
    this.usage = usage;
    this.bytes = new Uint8Array(size);
  }

  destroy() {}
}

globalThis.GPUBuffer = FakeBuffer;

const { setDevice } = await import('../../src/gpu/device.js');
const { createCpuWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const {
  decodeRangeChunkIntoOutput,
  embed,
  resolveEmbeddingScale,
  selectPreloadedCpuEmbeddingValues,
} = await import('../../src/inference/pipelines/text/embed.js');
const { releaseBuffer } = await import('../../src/memory/buffer-pool.js');

const fakeDevice = {
  lost: new Promise(() => {}),
  features: new Set(),
  limits: {
    maxStorageBufferBindingSize: 1 << 30,
    maxBufferSize: 1 << 30,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupStorageSize: 16384,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 1,
    maxComputeWorkgroupSizeZ: 1,
  },
  queue: {
    submit() {},
    writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size = undefined) {
      const source = ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        : new Uint8Array(data);
      const start = Number(dataOffset) || 0;
      const length = size == null ? source.byteLength - start : Number(size);
      buffer.bytes.set(source.subarray(start, start + length), bufferOffset);
    },
    onSubmittedWorkDone() {
      return Promise.resolve();
    },
  },
  createBuffer(descriptor) {
    return new FakeBuffer(descriptor);
  },
  createBindGroup() {
    return {};
  },
};

setDevice(fakeDevice, { platformConfig: null });

const hiddenSize = 4;
const f16Bytes = new Uint8Array(new Uint16Array([
  0x3C00,
  0x3800,
  0x4000,
  0xBC00,
]).buffer);
const decoded = new Float32Array(hiddenSize);
decodeRangeChunkIntoOutput(f16Bytes, 'f16', decoded, 0, hiddenSize);
assert.ok(Math.abs(decoded[0] - 1) < 1e-6);
assert.ok(Math.abs(decoded[1] - 0.5) < 1e-6);
assert.ok(Math.abs(decoded[2] - 2) < 1e-6);
assert.ok(Math.abs(decoded[3] + 1) < 1e-6);

assert.equal(resolveEmbeddingScale({ scaleEmbeddings: false, embeddingScale: null }, hiddenSize), 1);
assert.equal(resolveEmbeddingScale({ scaleEmbeddings: true, embeddingScale: null }, hiddenSize), 2);
assert.equal(resolveEmbeddingScale({ scaleEmbeddings: false, embeddingScale: 3 }, hiddenSize), 3);

const cpuWeight = createCpuWeightBuffer(
  {
    kind: 'tensor_range_source',
    sourceDtype: 'f16',
    async loadRange() {
      return f16Bytes;
    },
  },
  'f16',
  'row',
  [1, hiddenSize],
  'cpu_embedding_reference'
);
await assert.rejects(
  () => embed([0], cpuWeight, {
    hiddenSize,
    vocabSize: 1,
    scaleEmbeddings: false,
    embeddingScale: null,
    activationDtype: 'f32',
    embeddingDtype: 'f16',
  }),
  /CPU-resident embedding gather requires a verified preloaded row/
);

assert.deepEqual(
  Array.from(selectPreloadedCpuEmbeddingValues({
    preloadedCpuRow: new Float32Array([1, 2, 3, 4]),
    numTokens: 1,
    inputHiddenSize: 4,
    hiddenSize: 2,
    hiddenOffset: 1,
  })),
  [2, 3]
);
assert.deepEqual(
  Array.from(selectPreloadedCpuEmbeddingValues({
    preloadedCpuBatchedRows: new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
    ]),
    numTokens: 2,
    inputHiddenSize: 4,
    hiddenSize: 2,
    hiddenOffset: 1,
  })),
  [2, 3, 6, 7]
);

const preloadedTensor = await embed([0], cpuWeight, {
  hiddenSize: 2,
  vocabSize: 1,
  scaleEmbeddings: false,
  embeddingScale: null,
  activationDtype: 'f32',
  embeddingDtype: 'f16',
  inputHiddenSize: 4,
  hiddenOffset: 1,
  preloadedCpuRow: new Float32Array([1, 2, 3, 4]),
});
assert.equal(preloadedTensor.dtype, 'f32');
assert.deepEqual(
  Array.from(new Float32Array(preloadedTensor.buffer.bytes.buffer, 0, 2)),
  [2, 3]
);
releaseBuffer(preloadedTensor.buffer);

await assert.rejects(
  () => embed([0], cpuWeight, {
    hiddenSize,
    vocabSize: 1,
    scaleEmbeddings: false,
    embeddingScale: null,
    activationDtype: 'f16',
    embeddingDtype: 'f16',
  }),
  /requires shader-f16 support/
);

setDevice(null, { platformConfig: null });

console.log('embed-cpu-f16-gather.test: ok');
