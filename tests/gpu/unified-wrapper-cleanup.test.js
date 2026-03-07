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

const { setDevice } = await import('../../src/gpu/device.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { runModulate } = await import('../../src/gpu/kernels/modulate.js');
const { runReLU } = await import('../../src/gpu/kernels/relu.js');
const { runRepeatChannels } = await import('../../src/gpu/kernels/repeat_channels.js');
const { runPixelShuffle } = await import('../../src/gpu/kernels/pixel_shuffle.js');
const { runConv2D } = await import('../../src/gpu/kernels/conv2d.js');
const { runDepthwiseConv2D } = await import('../../src/gpu/kernels/depthwise_conv2d.js');
const { runGroupedPointwiseConv2D } = await import('../../src/gpu/kernels/grouped_pointwise_conv2d.js');

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice({ createBindGroupThrowAt = null } = {}) {
  let createBindGroupCount = 0;

  return {
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
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
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
    },
    createBindGroup() {
      createBindGroupCount += 1;
      if (createBindGroupCount === createBindGroupThrowAt) {
        throw new Error(`createBindGroup failed at ${createBindGroupCount}`);
      }
      return {};
    },
    createBindGroupLayout() {
      return {};
    },
    createPipelineLayout() {
      return {};
    },
    createShaderModule() {
      return {
        async getCompilationInfo() {
          return { messages: [] };
        },
      };
    },
    async createComputePipelineAsync() {
      return {
        getBindGroupLayout() {
          return {};
        },
      };
    },
  };
}

function resetRuntimeState(device = null) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

function assertPoolIsClean() {
  const stats = getBufferPool().getStats();
  assert.equal(stats.activeBuffers, 0);
  assert.equal(stats.currentBytesAllocated, 0);
}

function createExternalTensor(size, shape, label) {
  const buffer = new FakeBuffer({
    size: size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  return createTensor(buffer, 'f32', shape, label);
}

async function assertCleanupOnThrow(run) {
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  await assert.rejects(run, /createBindGroup failed at 1/);
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const input = createExternalTensor(8, [2, 4], 'mod_input');
  const mod = createExternalTensor(12, [3, 4], 'mod_state');
  await assertCleanupOnThrow(() => runModulate(input, mod, { numTokens: 2, hiddenSize: 4 }));
}

{
  const input = createExternalTensor(4, [4], 'relu_input');
  await assertCleanupOnThrow(() => runReLU(input, { count: 4 }));
}

{
  const input = createExternalTensor(8, [2, 2, 2], 'repeat_input');
  await assertCleanupOnThrow(() => runRepeatChannels(input, {
    inChannels: 2,
    height: 2,
    width: 2,
    repeats: 2,
  }));
}

{
  const input = createExternalTensor(16, [4, 2, 2], 'pixel_input');
  await assertCleanupOnThrow(() => runPixelShuffle(input, {
    outChannels: 1,
    outHeight: 4,
    outWidth: 4,
    gridWidth: 2,
    gridHeight: 2,
    patchSize: 2,
    patchChannels: 4,
  }));
}

{
  const input = createExternalTensor(9, [1, 3, 3], 'conv_input');
  const weight = createExternalTensor(9, [1, 1, 3, 3], 'conv_weight');
  await assertCleanupOnThrow(() => runConv2D(input, weight, null, {
    inChannels: 1,
    outChannels: 1,
    height: 3,
    width: 3,
    kernelH: 3,
    kernelW: 3,
  }));
}

{
  const input = createExternalTensor(9, [1, 3, 3], 'depthwise_input');
  const weight = createExternalTensor(9, [1, 3, 3], 'depthwise_weight');
  await assertCleanupOnThrow(() => runDepthwiseConv2D(input, weight, null, {
    channels: 1,
    height: 3,
    width: 3,
    kernelH: 3,
    kernelW: 3,
  }));
}

{
  const input = createExternalTensor(18, [2, 3, 3], 'grouped_input');
  const weight = createExternalTensor(4, [2, 2, 1, 1], 'grouped_weight');
  await assertCleanupOnThrow(() => runGroupedPointwiseConv2D(input, weight, null, {
    inChannels: 2,
    outChannels: 2,
    height: 3,
    width: 3,
    groups: 1,
  }));
}

console.log('unified-wrapper-cleanup.test: ok');
