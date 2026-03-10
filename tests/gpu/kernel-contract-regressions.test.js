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

const { createTensor } = await import('../../src/gpu/tensor.js');
const { setDevice } = await import('../../src/gpu/device.js');
const { hasRequiredFeatures } = await import('../../src/gpu/kernels/feature-check.js');
const { getKernelConfig } = await import('../../src/gpu/kernels/kernel-configs.js');
const { dequantize } = await import('../../src/gpu/kernels/dequant.js');
const { runSoftmax, runSoftmaxTopK } = await import('../../src/gpu/kernels/softmax.js');
const { runRoPE } = await import('../../src/gpu/kernels/rope.js');
const { runSiLU } = await import('../../src/gpu/kernels/silu.js');
const { runMatmul } = await import('../../src/gpu/kernels/matmul.js');
const { destroyBufferPool, getBufferPool, releaseBuffer } = await import('../../src/memory/buffer-pool.js');

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
  const createdBuffers = [];
  let createBindGroupCount = 0;

  return {
    createdBuffers,
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
      const buffer = new FakeBuffer({ size, usage });
      createdBuffers.push(buffer);
      return buffer;
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

function createExternalTensor(elementCount, dtype, label) {
  const bytesPerElement = dtype === 'f16' ? 2 : 4;
  const buffer = new FakeBuffer({
    size: elementCount * bytesPerElement,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  return createTensor(buffer, dtype, [elementCount], label);
}

try {
  {
    const biasAddConfig = getKernelConfig('bias_add', 'default');
    assert.equal(biasAddConfig.uniforms.size, 32);
  }

  {
    const fusedResidualConfig = getKernelConfig('fused_matmul_residual', 'default');
    assert.equal(fusedResidualConfig.uniforms.size, 32);
    assert.deepEqual(fusedResidualConfig.requires, ['shader-f16']);
  }

  {
    const ropeConfig = getKernelConfig('rope', 'default');
    assert.deepEqual(
      ropeConfig.uniforms.fields.map((field) => field.name),
      ['seq_len', 'num_heads', 'head_dim', 'start_pos', 'rope_base', 'rope_scale', 'rotary_dim', 'interleaved']
    );
  }

  assert.equal(hasRequiredFeatures(['subgroups-f16'], {
    hasF16: true,
    hasSubgroups: true,
    hasSubgroupsF16: false,
  }), false);

  {
    const input = createExternalTensor(4, 'f16', 'softmax_f16');
    await assert.rejects(
      () => runSoftmax(input, -1, { batchSize: 1, size: 4 }),
      /Softmax requires f32 input/
    );
  }

  {
    const input = createExternalTensor(8, 'f16', 'rope_input');
    const freqsCos = createExternalTensor(8, 'f16', 'rope_cos');
    const freqsSin = createExternalTensor(8, 'f16', 'rope_sin');
    await assert.rejects(
      () => runRoPE(input, freqsCos, freqsSin, 1, {
        numHeads: 1,
        headDim: 8,
        rotaryDim: 4,
      }),
      /RoPE f16 kernel requires rotaryDim === headDim and interleaved === false/
    );
  }

  {
    const device = createFakeDevice();
    resetRuntime(device);
    const quantized = new FakeBuffer({
      size: 144,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    await assert.rejects(
      () => dequantize(quantized, 1, { outputDtype: 'f16' }),
      /f16 output requires shader-f16 support/
    );
    resetRuntime();
  }

  {
    const device = createFakeDevice();
    resetRuntime(device);
    const input = createExternalTensor(8, 'f32', 'matmul_input');
    const weights = new FakeBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    await assert.rejects(
      () => runMatmul(input, weights, 1, 2, 8, {
        bDtype: 'f16',
        preferF16: true,
        transposeB: true,
      }),
      /f16 weights require shader-f16 support/
    );
    resetRuntime();
  }

  {
    const device = createFakeDevice();
    resetRuntime(device);
    const input = createExternalTensor(4, 'f32', 'silu_input');
    const result = await runSiLU(input, { size: 4, swigluLimit: null });
    const uniformBuffer = device.createdBuffers.find((buffer) => (buffer.usage & GPUBufferUsage.UNIFORM) !== 0);
    assert.ok(uniformBuffer);
    assert.equal(uniformBuffer.destroyed, true);
    releaseBuffer(result.buffer);
    resetRuntime();
  }

  {
    const device = createFakeDevice({ createBindGroupThrowAt: 1 });
    resetRuntime(device);
    const logits = new FakeBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    await assert.rejects(
      () => runSoftmaxTopK(logits, 1, 4, 2),
      /createBindGroup failed at 1/
    );
    for (const buffer of device.createdBuffers) {
      assert.equal(buffer.destroyed, true);
    }
    resetRuntime();
  }
} finally {
  resetRuntime();
}

console.log('kernel-contract-regressions.test: ok');
if (ORIGINAL_GPU_BUFFER === undefined) {
  delete globalThis.GPUBuffer;
} else {
  globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
}
