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
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { projectAttentionQKV } = await import('../../src/inference/pipelines/text/attention/projections.js');
const { clearDequantCache, setCachedDequant } = await import('../../src/inference/pipelines/text/moe-cache.js');
const { applyLoRA } = await import('../../src/inference/pipelines/text/lora-apply.js');
const {
  acquireBuffer,
  destroyBufferPool,
  getBufferPool,
  releaseBuffer,
} = await import('../../src/memory/buffer-pool.js');

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

function createExternalTensor(size, dtype, label) {
  const bytesPerElement = dtype === 'f16' ? 2 : 4;
  return createTensor(
    new FakeBuffer({
      size: size * bytesPerElement,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }),
    dtype,
    [1, size],
    label
  );
}

function assertPoolIsClean() {
  const stats = getBufferPool().getStats();
  assert.equal(stats.activeBuffers, 0);
}

try {
  {
    await assert.rejects(
      () => projectAttentionQKV({
        normed: createExternalTensor(4, 'f32', 'normed_missing_q'),
        layerWeights: {
          kProj: createWeightBuffer(new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE }), 'f32', 'row', [2, 4], 'k_proj'),
          vProj: createWeightBuffer(new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE }), 'f32', 'row', [2, 4], 'v_proj'),
        },
        numTokens: 1,
        numHeads: 1,
        numKVHeads: 1,
        headDim: 2,
        hiddenSize: 4,
        layerIdx: 0,
        kernelPath: null,
        matmulOutputDtype: 'f32',
        getWeightBuffer: (weight) => weight,
        lora: null,
        releaseTemporary: () => {},
      }),
      /Attention projection requires qProj/
    );
  }

  {
    const device = createFakeDevice({ createBindGroupThrowAt: 2 });
    resetRuntime(device);
    await assert.rejects(
      () => projectAttentionQKV({
        normed: createExternalTensor(16, 'f32', 'normed_gate_failure'),
        layerWeights: {
          qProj: createWeightBuffer(new FakeBuffer({ size: 16 * 8 * 4, usage: GPUBufferUsage.STORAGE }), 'f32', 'row', [8, 16], 'q_proj'),
        },
        numTokens: 1,
        numHeads: 1,
        numKVHeads: 1,
        headDim: 4,
        hiddenSize: 16,
        layerIdx: 0,
        kernelPath: null,
        matmulOutputDtype: 'f32',
        getWeightBuffer: (weight) => weight,
        lora: null,
        attentionOutputGate: true,
        releaseTemporary: (buffer) => releaseBuffer(buffer),
      }),
      /createBindGroup failed at 2/
    );
    assertPoolIsClean();
    resetRuntime();
  }

  {
    const device = createFakeDevice();
    resetRuntime(device);
    const gateUp = acquireBuffer(64, GPUBufferUsage.STORAGE, 'moe_gate_up');
    const down = acquireBuffer(64, GPUBufferUsage.STORAGE, 'moe_down');
    setCachedDequant(0, 0, 'f32', gateUp, down);
    clearDequantCache();
    assertPoolIsClean();
    resetRuntime();
  }

  {
    const device = createFakeDevice({ createBindGroupThrowAt: 2 });
    resetRuntime(device);
    const input = createExternalTensor(4, 'f32', 'lora_input');
    const baseOutput = createExternalTensor(4, 'f32', 'lora_base');
    const loraA = createWeightBuffer(new FakeBuffer({ size: 4 * 1 * 4, usage: GPUBufferUsage.STORAGE }), 'f32', 'row', [1, 4], 'lora_a');
    const loraB = createWeightBuffer(new FakeBuffer({ size: 1 * 4 * 4, usage: GPUBufferUsage.STORAGE }), 'f32', 'row', [4, 1], 'lora_b');

    await assert.rejects(
      () => applyLoRA(
        input,
        baseOutput,
        { rank: 1, scale: 1, a: loraA, b: loraB },
        { M: 1, N: 4, K: 4 },
        (weight) => weight
      ),
      /createBindGroup failed at 2/
    );
    assertPoolIsClean();
    resetRuntime();
  }
} finally {
  clearDequantCache();
  resetRuntime();
}

console.log('attention-projections-cleanup.test: ok');
if (ORIGINAL_GPU_BUFFER === undefined) {
  delete globalThis.GPUBuffer;
} else {
  globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
}
