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
} = await import('../../src/inference/pipelines/text/embed.js');

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
  new Uint16Array(f16Bytes.buffer),
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
  /CPU-resident embedding gather is not a production path/
);

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
