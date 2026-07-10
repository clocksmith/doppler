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

const { CommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { FEATURES, setDevice } = await import('../../src/gpu/device.js');
const { resetUniformCache } = await import('../../src/gpu/uniform-cache.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { recordLmHeadArgmax, recordLmHeadArgmaxF16 } = await import('../../src/gpu/kernels/lm-head-argmax.js');
const { configurePerfGuards } = await import('../../src/gpu/perf-guards.js');
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { clearPipelineCaches } = await import('../../src/gpu/kernels/utils.js');
const { Q4K_BLOCK_BYTES, q4kBlockCount } = await import('../../src/config/schema/index.js');
const { resetRuntimeConfig, setRuntimeConfig } = await import('../../src/config/runtime.js');

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
    this.__dopplerFakeGPUBuffer = true;
  }

  async mapAsync() {}

  getMappedRange(_offset = 0, size = this.size) {
    return new ArrayBuffer(size);
  }

  unmap() {}

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice() {
  const computePasses = [];
  const features = new Set([FEATURES.SHADER_F16]);

  return {
    computePasses,
    features,
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
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
      maxQuerySetSize: 64,
    },
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
    },
    createShaderModule(descriptor) {
      return {
        ...descriptor,
        async getCompilationInfo() {
          return { messages: [] };
        },
      };
    },
    createBindGroupLayout(descriptor) {
      return { ...descriptor };
    },
    createPipelineLayout(descriptor) {
      return { ...descriptor };
    },
    createBindGroup(descriptor) {
      return { ...descriptor };
    },
    async createComputePipelineAsync(descriptor) {
      return { ...descriptor };
    },
    createCommandEncoder() {
      return {
        beginComputePass(descriptor = {}) {
          const passRecord = {
            label: descriptor.label ?? null,
            ended: false,
            dispatches: [],
            pipelines: [],
          };
          computePasses.push(passRecord);
          return {
            setPipeline(pipeline) {
              passRecord.pipeline = pipeline;
              passRecord.pipelines.push(pipeline);
            },
            setBindGroup(index, bindGroup) {
              passRecord.bindGroup = { index, bindGroup };
            },
            dispatchWorkgroups(x, y, z) {
              passRecord.dispatches.push({ x, y, z });
            },
            end() {
              passRecord.ended = true;
            },
          };
        },
        finish() {
          return {};
        },
      };
    },
  };
}

function resetRuntimeState(device) {
  resetRuntimeConfig();
  destroyBufferPool();
  resetUniformCache();
  clearPipelineCaches();
  setDevice(device, { platformConfig: null });
  getBufferPool().configure({ enablePooling: false });
  configurePerfGuards({
    allowGPUReadback: true,
    trackSubmitCount: false,
    trackAllocations: false,
    logExpensiveOps: false,
    strictMode: false,
  });
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  const recorder = new CommandRecorder(device, 'lm_head_argmax_coalesced');
  const hiddenSize = 16;
  const vocabSize = 128;
  const input = {
    buffer: device.createBuffer({
      size: hiddenSize * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    }),
    dtype: 'f32',
  };
  const lmHead = createWeightBuffer(
    device.createBuffer({
      size: vocabSize * hiddenSize * Uint16Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    }),
    'f16',
    'row',
    [vocabSize, hiddenSize],
    'lm_head'
  );
  const output = device.createBuffer({
    size: Uint32Array.BYTES_PER_ELEMENT * 2,
    usage: GPUBufferUsage.STORAGE,
  });

  await recordLmHeadArgmaxF16(recorder, input, lmHead, {
    vocabSize,
    hiddenSize,
    padTokenId: null,
    logitSoftcap: 0,
    outputBuffer: output,
    outputIndex: 1,
  });

  assert.equal(device.computePasses.length, 1);
  assert.equal(device.computePasses[0].ended, false);
  assert.deepEqual(device.computePasses[0].dispatches, [
    { x: 2, y: 1, z: 1 },
    { x: 1, y: 1, z: 1 },
  ]);
  assert.equal(
    Object.hasOwn(device.computePasses[0].pipelines[0].compute.constants, 'USE_FULL_BLOCK_FAST_PATH'),
    false,
    'f16 LM-head argmax must not receive Q4K-only full-block specialization'
  );
  assert.deepEqual(recorder.getStats().opLabelCounts, {
    lm_head_argmax_phase1: 1,
    lm_head_argmax_phase2: 1,
  });
  assert.equal(recorder.getStats().opCount, 2);
  assert.equal(recorder.getStats().computePassCount, 1);

  recorder.abort();
  destroyBufferPool();
  resetUniformCache();
  clearPipelineCaches();
  setDevice(null);
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  const recorder = new CommandRecorder(device, 'lm_head_argmax_q4k_defaults_to_f16');
  const hiddenSize = 16;
  const vocabSize = 128;
  const input = {
    buffer: device.createBuffer({
      size: hiddenSize * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    }),
    dtype: 'f32',
  };
  const f16Materialization = device.createBuffer({
    size: vocabSize * hiddenSize * Uint16Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });
  const lmHead = createWeightBuffer(
    device.createBuffer({
      size: vocabSize * q4kBlockCount(hiddenSize) * Q4K_BLOCK_BYTES,
      usage: GPUBufferUsage.STORAGE,
    }),
    'q4k',
    'row',
    [vocabSize, hiddenSize],
    'lm_head_q4k_with_f16_materialization',
    {
      f16: {
        buffer: f16Materialization,
        layout: 'row',
      },
    }
  );
  const output = device.createBuffer({
    size: Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });

  await recordLmHeadArgmax(recorder, input, lmHead, {
    vocabSize,
    hiddenSize,
    padTokenId: null,
    logitSoftcap: 0,
    outputBuffer: output,
    outputIndex: 0,
  });

  assert.equal(device.computePasses.length, 1);
  assert.deepEqual(recorder.getStats().opLabelCounts, {
    lm_head_argmax_phase1: 1,
    lm_head_argmax_phase2: 1,
  });
  assert.equal(
    Object.hasOwn(device.computePasses[0].pipelines[0].compute.constants, 'USE_FULL_BLOCK_FAST_PATH'),
    false,
    'Q4K LM-head argmax must stay disabled unless lmHeadArgmaxQ4K is explicitly configured'
  );

  recorder.abort();
  destroyBufferPool();
  resetUniformCache();
  clearPipelineCaches();
  setDevice(null);
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);
  setRuntimeConfig({
    inference: {
      session: {
        lmHeadArgmaxQ4K: {
          useFullBlockFastPath: true,
          colsPerWorkgroup: 128,
          threadsPerCol: 2,
        },
      },
    },
  });

  const recorder = new CommandRecorder(device, 'lm_head_argmax_q4k_profile_shape');
  const hiddenSize = 16;
  const vocabSize = 512;
  const input = {
    buffer: device.createBuffer({
      size: hiddenSize * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    }),
    dtype: 'f32',
  };
  const lmHead = createWeightBuffer(
    device.createBuffer({
      size: vocabSize * q4kBlockCount(hiddenSize) * Q4K_BLOCK_BYTES,
      usage: GPUBufferUsage.STORAGE,
    }),
    'q4k',
    'row',
    [vocabSize, hiddenSize],
    'lm_head_q4k_profile_shape'
  );
  const output = device.createBuffer({
    size: Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });

  await recordLmHeadArgmax(recorder, input, lmHead, {
    vocabSize,
    hiddenSize,
    padTokenId: null,
    logitSoftcap: 0,
    outputBuffer: output,
    outputIndex: 0,
  });

  assert.deepEqual(device.computePasses[0].dispatches, [
    { x: 4, y: 1, z: 1 },
    { x: 1, y: 1, z: 1 },
  ]);
  assert.equal(device.computePasses[0].pipelines[0].compute.constants.COLS_PER_WG, 128);
  assert.equal(device.computePasses[0].pipelines[0].compute.constants.THREADS_PER_COL, 2);
  assert.equal(device.computePasses[0].pipelines[0].compute.constants.USE_FULL_BLOCK_FAST_PATH, 1);

  recorder.abort();
  destroyBufferPool();
  resetUniformCache();
  clearPipelineCaches();
  setDevice(null);
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);
  setRuntimeConfig({
    inference: {
      session: {
        lmHeadArgmaxQ4K: {
          useFullBlockFastPath: true,
        },
      },
    },
  });

  const recorder = new CommandRecorder(device, 'lm_head_argmax_q4k_coalesced');
  const hiddenSize = 16;
  const vocabSize = 128;
  const input = {
    buffer: device.createBuffer({
      size: hiddenSize * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    }),
    dtype: 'f32',
  };
  const lmHead = createWeightBuffer(
    device.createBuffer({
      size: vocabSize * q4kBlockCount(hiddenSize) * Q4K_BLOCK_BYTES,
      usage: GPUBufferUsage.STORAGE,
    }),
    'q4k',
    'row',
    [vocabSize, hiddenSize],
    'lm_head_q4k'
  );
  const output = device.createBuffer({
    size: Uint32Array.BYTES_PER_ELEMENT * 2,
    usage: GPUBufferUsage.STORAGE,
  });

  await recordLmHeadArgmax(recorder, input, lmHead, {
    vocabSize,
    hiddenSize,
    padTokenId: null,
    logitSoftcap: 0,
    outputBuffer: output,
    outputIndex: 1,
  });

  assert.equal(device.computePasses.length, 1);
  assert.equal(device.computePasses[0].ended, false);
  assert.deepEqual(device.computePasses[0].dispatches, [
    { x: 2, y: 1, z: 1 },
    { x: 1, y: 1, z: 1 },
  ]);
  assert.equal(
    device.computePasses[0].pipelines[0].compute.constants.USE_FULL_BLOCK_FAST_PATH,
    1,
    'Q4K LM-head argmax must receive the profile-gated full-block specialization'
  );
  assert.equal(
    Object.hasOwn(device.computePasses[0].pipelines[0].compute.constants, 'USE_FULL_BLOCK_FAST_PATH'),
    true,
    'Q4K LM-head argmax must expose the profile-gated full-block specialization'
  );
  assert.deepEqual(recorder.getStats().opLabelCounts, {
    lm_head_argmax_q4k_phase1: 1,
    lm_head_argmax_q4k_phase2: 1,
  });
  assert.equal(recorder.getStats().opCount, 2);
  assert.equal(recorder.getStats().computePassCount, 1);

  recorder.abort();
  destroyBufferPool();
  resetUniformCache();
  clearPipelineCaches();
  setDevice(null);
}
