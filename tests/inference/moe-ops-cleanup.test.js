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

globalThis.GPUShaderStage = {
  COMPUTE: 0x2,
};

const { setDevice } = await import('../../src/gpu/device.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const {
  destroyBufferPool,
  getBufferPool,
} = await import('../../src/memory/buffer-pool.js');
const { moeFeedForwardCPU } = await import('../../src/inference/pipelines/text/moe-cpu.js');
const { doConv } = await import('../../src/inference/pipelines/text/ops.js');

class FakeBuffer {
  constructor({ size, usage, initialBytes = null }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
    this.bytes = initialBytes ? new Uint8Array(initialBytes) : null;
  }

  destroy() {
    this.destroyed = true;
  }
}

const ORIGINAL_GPU_BUFFER = globalThis.GPUBuffer;
globalThis.GPUBuffer = FakeBuffer;

function createFakeDevice({ createBindGroupThrowAt = null } = {}) {
  let createBindGroupCount = 0;
  return {
    lost: new Promise(() => {}),
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
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        finish() {
          return {};
        },
      };
    },
  };
}

function resetRuntime(device = null) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

function assertPoolIsClean() {
  const stats = getBufferPool().getStats();
  assert.equal(stats.activeBuffers, 0);
}

function createExternalTensor(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  return createTensor(
    new FakeBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      initialBytes: new Uint8Array(data.buffer.slice(0)),
    }),
    'f32',
    shape,
    label
  );
}

try {
  {
    const device = createFakeDevice({ createBindGroupThrowAt: 1 });
    resetRuntime(device);

    await assert.rejects(
      () => moeFeedForwardCPU(
        new Float32Array([1]),
        1,
        {
          expertFormat: 'mixtral',
          numExperts: 1,
          hiddenSize: 1,
          intermediateSize: 1,
          hiddenActivation: 'silu',
          swigluLimit: null,
          kernelPath: null,
        },
        {
          route() {
            return [{ indices: [0], weights: [1] }];
          },
        },
        new Map([
          ['layer_0_expert_0', {
            gate: new FakeBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }),
            up: new FakeBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }),
            down: new FakeBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }),
          }],
        ]),
        null,
        0
      ),
      /createBindGroup failed at 1/
    );

    assertPoolIsClean();
    resetRuntime();
  }

  {
    const device = createFakeDevice({ createBindGroupThrowAt: 2 });
    resetRuntime(device);

    const inputTensor = createExternalTensor([1], [1, 1], 'conv_input');
    await assert.rejects(
      () => doConv(
        inputTensor,
        new FakeBuffer({ size: 8, usage: GPUBufferUsage.STORAGE }),
        null,
        new FakeBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }),
        {
          numTokens: 1,
          hiddenSize: 1,
          layerIdx: 0,
          label: 'conv_cleanup',
          kernelPath: null,
          swigluLimit: null,
        }
      ),
      /createBindGroup failed at 2/
    );

    assertPoolIsClean();
    resetRuntime();
  }
} finally {
  resetRuntime();
}

console.log('moe-ops-cleanup.test: ok');
if (ORIGINAL_GPU_BUFFER === undefined) {
  delete globalThis.GPUBuffer;
} else {
  globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
}
