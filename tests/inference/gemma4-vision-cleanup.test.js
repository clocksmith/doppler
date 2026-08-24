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
  constructor({ size, usage, label }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice() {
  return {
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize: 1 << 20,
      maxBufferSize: 1 << 20,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 8,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    queue: {
      submit() {},
      writeBuffer() {},
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
}

const { setDevice } = await import('../../src/gpu/device.js');
const {
  destroyBufferPool,
  getBufferPool,
} = await import('../../src/memory/buffer-pool.js');
const { encodeGemma4Image } = await import('../../src/inference/pipelines/vision/gemma4.js');

setDevice(createFakeDevice(), { platformConfig: null });

try {
  await assert.rejects(
    () => encodeGemma4Image({
      pixels: new Uint8Array(2 * 2 * 3),
      width: 2,
      height: 2,
      softTokenBudget: 4,
      visionConfig: {
        hiddenActivation: 'gelu_pytorch_tanh',
        standardize: false,
        useClippedLinears: true,
        hiddenSize: 4,
        patchSize: 1,
        poolingKernelSize: 1,
        ropeTheta: 10000,
        positionEmbeddingSize: 2,
        defaultOutputLength: 4,
        eps: 1e-6,
      },
      weights: {
        patchPositionEmbeddingTable: null,
      },
    }),
    /typed GPU table tensor/
  );

  assert.equal(getBufferPool().getStats().activeBuffers, 0);
} finally {
  destroyBufferPool();
  setDevice(null);
}

console.log('gemma4-vision-cleanup.test: ok');
