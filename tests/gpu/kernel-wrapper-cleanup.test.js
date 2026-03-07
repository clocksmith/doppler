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
const { CommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const {
  acquireBuffer,
  destroyBufferPool,
  getBufferPool,
  releaseBuffer,
} = await import('../../src/memory/buffer-pool.js');
const {
  dequantize,
  recordDequantize,
} = await import('../../src/gpu/kernels/dequant.js');
const {
  runFusedFFN,
  recordFusedFFN,
} = await import('../../src/gpu/kernels/fused_ffn.js');
const {
  runMatmulResidualFused,
  recordMatmulResidualFused,
} = await import('../../src/gpu/kernels/fused_matmul_residual.js');
const {
  runMatmulRMSNormFused,
  recordMatmulRMSNormFused,
} = await import('../../src/gpu/kernels/fused_matmul_rmsnorm.js');
const { runGroupNorm } = await import('../../src/gpu/kernels/groupnorm.js');
const { runGeLU } = await import('../../src/gpu/kernels/gelu.js');
const { runLayerNorm } = await import('../../src/gpu/kernels/layernorm.js');
const { runRMSNorm } = await import('../../src/gpu/kernels/rmsnorm.js');
const { runSiLU } = await import('../../src/gpu/kernels/silu.js');
const { runUpsample2D } = await import('../../src/gpu/kernels/upsample2d.js');

class FakeBuffer {
  constructor({ size, usage, initialBytes = null }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
    this.bytes = initialBytes ? new Uint8Array(initialBytes) : null;
  }

  ensureBytes(minSize = this.size) {
    if (!this.bytes || this.bytes.length < minSize) {
      const next = new Uint8Array(minSize);
      if (this.bytes) {
        next.set(this.bytes.subarray(0, Math.min(this.bytes.length, next.length)));
      }
      this.bytes = next;
    }
    return this.bytes;
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
    features: new Set(['subgroups']),
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
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
    },
    createCommandEncoder() {
      return {
        finish() {
          return { ops: [] };
        },
      };
    },
  };
}

function createExternalTensor(values, shape, label, dtype = 'f32') {
  if (dtype === 'f16') {
    const data = values instanceof Uint16Array ? values : new Uint16Array(values);
    const buffer = new FakeBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      initialBytes: new Uint8Array(data.buffer.slice(0)),
    });
    return createTensor(buffer, 'f16', shape, label);
  }
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  const buffer = new FakeBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    initialBytes: new Uint8Array(data.buffer.slice(0)),
  });
  return createTensor(buffer, 'f32', shape, label);
}

function createWeightLike(size, dtype = 'f32') {
  return {
    buffer: new FakeBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
    dtype,
    shape: [1, size / (dtype === 'f16' ? 2 : 4)],
  };
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
  const input = createExternalTensor([1], [1, 1], 'ffn_input');
  const gate = createWeightLike(4);
  const up = createWeightLike(4);
  await assert.rejects(
    () => runFusedFFN(input, gate, up, 1, 1, { swigluLimit: null }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  const recorder = new CommandRecorder(createFakeDevice({ createBindGroupThrowAt: 1 }), 'ffn_record');
  setDevice(recorder.device, { platformConfig: null });
  const input = createExternalTensor([1], [1, 1], 'ffn_input');
  const gate = createWeightLike(4);
  const up = createWeightLike(4);
  await assert.rejects(
    () => recordFusedFFN(recorder, input, gate, up, 1, 1, { swigluLimit: null }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  const input = createExternalTensor([1], [1, 1], 'residual_input');
  const residual = createExternalTensor([1], [1, 1], 'residual');
  await assert.rejects(
    () => runMatmulResidualFused(input, new FakeBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }), residual, { N: 1, K: 1 }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ createBindGroupThrowAt: 1 });
  resetRuntimeState(device);
  const recorder = new CommandRecorder(device, 'residual_record');
  const input = createExternalTensor([1], [1, 1], 'residual_input');
  const residual = createExternalTensor([1], [1, 1], 'residual');
  await assert.rejects(
    () => recordMatmulResidualFused(recorder, input, new FakeBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }), residual, { N: 1, K: 1 }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ createBindGroupThrowAt: 1 });
  resetRuntimeState(device);
  const input = createExternalTensor([1], [1, 1], 'rms_input');
  const weight = new FakeBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  const normWeightBuffer = acquireBuffer(4, undefined, 'test_norm_weight');
  await assert.rejects(
    () => runMatmulRMSNormFused(input, weight, normWeightBuffer, { N: 1, K: 1, eps: 1e-5 }),
    /createBindGroup failed at 1/
  );
  releaseBuffer(normWeightBuffer);
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ createBindGroupThrowAt: 1 });
  resetRuntimeState(device);
  const recorder = new CommandRecorder(device, 'rms_record');
  const input = createExternalTensor([1], [1, 1], 'rms_input');
  const weight = new FakeBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  const normWeightBuffer = acquireBuffer(4, undefined, 'test_norm_weight');
  await assert.rejects(
    () => recordMatmulRMSNormFused(recorder, input, weight, normWeightBuffer, { N: 1, K: 1, eps: 1e-5 }),
    /createBindGroup failed at 1/
  );
  releaseBuffer(normWeightBuffer);
  assertPoolIsClean();
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  await assert.rejects(
    () => dequantize(new FakeBuffer({ size: 144, usage: GPUBufferUsage.STORAGE }), 1, {}),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ createBindGroupThrowAt: 1 });
  resetRuntimeState(device);
  const recorder = new CommandRecorder(device, 'dequant_record');
  await assert.rejects(
    () => recordDequantize(recorder, new FakeBuffer({ size: 144, usage: GPUBufferUsage.STORAGE }), 1, {}),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  const input = createExternalTensor([1, 2, 3, 4], [4, 1, 1], 'groupnorm_input');
  const weight = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  const bias = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  await assert.rejects(
    () => runGroupNorm(input, weight, bias, {
      channels: 4,
      height: 1,
      width: 1,
      numGroups: 2,
      eps: 1e-5,
    }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice());
  const input = createExternalTensor([1, 2, 3], [3, 1, 1], 'groupnorm_input');
  const weight = new FakeBuffer({ size: 12, usage: GPUBufferUsage.STORAGE });
  const bias = new FakeBuffer({ size: 12, usage: GPUBufferUsage.STORAGE });
  await assert.rejects(
    () => runGroupNorm(input, weight, bias, {
      channels: 3,
      height: 1,
      width: 1,
      numGroups: 2,
      eps: 1e-5,
    }),
    /channels to be divisible by numGroups/
  );
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  const input = createExternalTensor([1, 2, 3, 4], [4], 'gelu_input');
  await assert.rejects(
    () => runGeLU(input, { size: 4 }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  const input = createExternalTensor([1, 2, 3, 4], [1, 4], 'layernorm_input');
  const weight = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  const bias = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  await assert.rejects(
    () => runLayerNorm(input, weight, bias, 1e-5, { batchSize: 1, hiddenSize: 4 }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  const input = createExternalTensor([1, 2, 3, 4], [1, 4], 'rmsnorm_input');
  const weight = new FakeBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  await assert.rejects(
    () => runRMSNorm(input, weight, 1e-5, { batchSize: 1, hiddenSize: 4 }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  const input = createExternalTensor([1, 2, 3, 4], [4], 'silu_input');
  await assert.rejects(
    () => runSiLU(input, { size: 4, swigluLimit: null }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  resetRuntimeState(createFakeDevice({ createBindGroupThrowAt: 1 }));
  const input = createExternalTensor([1, 2, 3, 4], [1, 2, 2], 'upsample_input');
  await assert.rejects(
    () => runUpsample2D(input, { channels: 1, height: 2, width: 2, scale: 2 }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

console.log('kernel-wrapper-cleanup.test: ok');
