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
const { CommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const {
  destroyBufferPool,
  getBufferPool,
} = await import('../../src/memory/buffer-pool.js');
const { runLayerAttentionGPU } = await import('../../src/inference/pipelines/text/attention/run.js');
const { recordLayerAttentionGPU } = await import('../../src/inference/pipelines/text/attention/record.js');

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
      minStorageBufferOffsetAlignment: 16,
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
        clearBuffer() {},
        copyBufferToBuffer() {},
        finish() {
          return {};
        },
      };
    },
    destroy() {},
  };
}

function resetRuntime(device = null) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

function createInputTensor() {
  return createTensor(
    new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
    'f32',
    [1, 4],
    'attn_input'
  );
}

function createLayerWeights() {
  return {
    qProj: createWeightBuffer(new FakeBuffer({ size: 64, usage: GPUBufferUsage.STORAGE }), 'f32', 'row', [4, 4], 'q_proj'),
    kProj: createWeightBuffer(new FakeBuffer({ size: 64, usage: GPUBufferUsage.STORAGE }), 'f32', 'row', [4, 4], 'k_proj'),
    vProj: createWeightBuffer(new FakeBuffer({ size: 64, usage: GPUBufferUsage.STORAGE }), 'f32', 'row', [4, 4], 'v_proj'),
    oProj: createWeightBuffer(new FakeBuffer({ size: 64, usage: GPUBufferUsage.STORAGE }), 'f32', 'row', [4, 4], 'o_proj'),
  };
}

function createConfig() {
  return {
    layerIdx: 0,
    numTokens: 1,
    isPrefill: true,
    numHeads: 1,
    numKVHeads: 1,
    headDim: 4,
    hiddenSize: 4,
    rmsNormEps: 1e-5,
    currentSeqLen: 0,
    slidingWindow: null,
    layerType: 'full_attention',
    residualTensor: null,
    attnSoftcap: null,
    queryPreAttnScalar: 4,
    activationDtype: 'f32',
    causalAttention: true,
    ropeRotaryDim: 4,
    ropeInterleaved: false,
  };
}

function assertPoolIsClean() {
  assert.equal(getBufferPool().getStats().activeBuffers, 0);
}

try {
  {
    resetRuntime(createFakeDevice({ createBindGroupThrowAt: 4 }));
    await assert.rejects(() => runLayerAttentionGPU(
      createInputTensor(),
      createLayerWeights(),
      createConfig(),
      {},
      false,
      {},
      (weight) => weight,
      null,
      null,
      null
    ));
    assertPoolIsClean();
    resetRuntime();
  }

  {
    const device = createFakeDevice({ createBindGroupThrowAt: 4 });
    resetRuntime(device);
    const recorder = new CommandRecorder(device, 'attention_record_cleanup');
    await assert.rejects(() => recordLayerAttentionGPU(
      recorder,
      createInputTensor(),
      createLayerWeights(),
      createConfig(),
      {},
      false,
      {},
      (weight) => weight,
      null,
      null,
      null
    ));
    await recorder.submitAndWait();
    assertPoolIsClean();
    resetRuntime();
  }
} finally {
  resetRuntime();
}

console.log('attention-run-cleanup.test: ok');
