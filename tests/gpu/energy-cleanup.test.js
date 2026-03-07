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
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const {
  runEnergyEval,
  runEnergyQuintelReduce,
} = await import('../../src/gpu/kernels/energy.js');

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
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
      writeBuffer() {},
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
    createBindGroup() {
      createBindGroupCount += 1;
      if (createBindGroupCount === createBindGroupThrowAt) {
        throw new Error(`createBindGroup failed at ${createBindGroupCount}`);
      }
      return {};
    },
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
    },
  };
}

function createExternalTensor(size, label) {
  const buffer = new FakeBuffer({
    size: size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  return createTensor(buffer, 'f32', [size], label);
}

function resetRuntimeState(device = null) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
}

function assertPoolIsClean() {
  const stats = getBufferPool().getStats();
  assert.equal(stats.activeBuffers, 0);
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  const state = createExternalTensor(4, 'state');
  const target = createExternalTensor(4, 'target');

  await assert.rejects(
    () => runEnergyEval(state, target, { count: 4 }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice());
  const state = createExternalTensor(4, 'state');

  await assert.rejects(
    () => runEnergyQuintelReduce(state, { size: 2, rules: { mirrorX: true } }),
    /flags must be resolved before dispatch/
  );
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  const state = createExternalTensor(4, 'state');

  await assert.rejects(
    () => runEnergyQuintelReduce(state, { size: 2, flags: 1 }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

console.log('energy-cleanup.test: ok');
