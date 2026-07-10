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
const { recordAttention } = await import('../../src/gpu/kernels/attention.js');
const { configurePerfGuards } = await import('../../src/gpu/perf-guards.js');
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const { clearPipelineCaches } = await import('../../src/gpu/kernels/utils.js');
const { resetRuntimeConfig, setRuntimeConfig } = await import('../../src/config/runtime.js');

class FakeBuffer {
  constructor({ size, usage }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
    this.__dopplerFakeGPUBuffer = true;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createFakeDevice() {
  const computePasses = [];
  const pipelineDescriptors = [];
  const features = new Set([FEATURES.SHADER_F16, FEATURES.SUBGROUPS]);

  return {
    computePasses,
    pipelineDescriptors,
    features,
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    limits: {
      maxStorageBufferBindingSize: 1 << 24,
      maxBufferSize: 1 << 24,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 32768,
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
      pipelineDescriptors.push(descriptor);
      return {
        ...descriptor,
        getBindGroupLayout(index) {
          return { label: `attention_bind_group_layout_${index}` };
        },
      };
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

async function recordOnlineDecodeAttention(device, options = {}) {
  const recorder = new CommandRecorder(device, 'attention_online_constants');
  const numHeads = 1;
  const numKVHeads = 1;
  const headDim = options.headDim ?? 128;
  const kvLen = 512;
  const q = {
    buffer: device.createBuffer({
      size: numHeads * headDim * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    }),
    dtype: 'f32',
  };
  const k = {
    buffer: device.createBuffer({
      size: kvLen * numKVHeads * headDim * Uint16Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    }),
    dtype: 'f16',
  };
  const v = {
    buffer: device.createBuffer({
      size: kvLen * numKVHeads * headDim * Uint16Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    }),
    dtype: 'f16',
  };
  const outputGate = options.outputGate === true ? {
    buffer: device.createBuffer({
      size: numHeads * headDim * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    }),
    dtype: 'f32',
    shape: [1, numHeads * headDim],
  } : null;
  await recordAttention(recorder, q, k, v, null, numHeads, headDim, {
    seqLen: 1,
    kvLen,
    numKVHeads,
    causal: true,
    startPos: kvLen - 1,
    layerIdx: 0,
    kvLayout: options.kvLayout,
    kernelPath: options.kernelPath,
    outputGate,
  });
  return recorder;
}

function head256KernelPath() {
  return {
    id: 'test-head256-online-decode',
    name: 'test-head256-online-decode',
    decode: {
      steps: [
        { op: 'attention', kernel: 'attention_decode_online_head256_f16kv_output_gate.wgsl', entry: 'main' },
      ],
    },
  };
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  const recorder = await recordOnlineDecodeAttention(device);

  assert.equal(device.pipelineDescriptors.length, 1);
  assert.equal(device.pipelineDescriptors[0].label, 'attention_decode_online_f16kv_pipeline');
  assert.equal(device.pipelineDescriptors[0].compute.constants, undefined);
  assert.deepEqual(device.computePasses[0].dispatches, [{ x: 1, y: 1, z: 1 }]);

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
        attentionDecodeOnline: {
          useOutputGateFusion: true,
        },
      },
    },
  });

  const recorder = await recordOnlineDecodeAttention(device, {
    headDim: 256,
    kernelPath: head256KernelPath(),
    outputGate: true,
  });

  assert.equal(device.pipelineDescriptors.length, 1);
  assert.equal(device.pipelineDescriptors[0].label, 'attention_decode_online_head256_f16kv_output_gate_pipeline');
  assert.equal(device.pipelineDescriptors[0].compute.constants, undefined);
  assert.equal(device.computePasses[0].bindGroup.bindGroup.entries.some((entry) => entry.binding === 7), true);
  assert.deepEqual(device.computePasses[0].dispatches, [{ x: 1, y: 1, z: 1 }]);

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
        attentionDecodeOnline: {
          workgroupSize: 128,
        },
      },
    },
  });

  const recorder = await recordOnlineDecodeAttention(device);

  assert.equal(device.pipelineDescriptors.length, 1);
  assert.equal(device.pipelineDescriptors[0].label, 'attention_decode_online_f16kv_pipeline_WORKGROUP_SIZE=128');
  assert.equal(device.pipelineDescriptors[0].compute.constants.WORKGROUP_SIZE, 128);
  assert.deepEqual(device.computePasses[0].dispatches, [{ x: 1, y: 1, z: 1 }]);

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
        attentionDecodeOnline: {
          useDirectContiguousKVLayout: true,
        },
      },
    },
  });

  const recorder = await recordOnlineDecodeAttention(device, {
    headDim: 256,
    kernelPath: head256KernelPath(),
    outputGate: true,
  });

  assert.equal(device.pipelineDescriptors.length, 1);
  assert.equal(device.pipelineDescriptors[0].label, 'attention_decode_online_head256_f16kv_output_gate_pipeline_USE_DIRECT_KV_LAYOUT=1');
  assert.equal(device.pipelineDescriptors[0].compute.constants.USE_DIRECT_KV_LAYOUT, 1);
  assert.deepEqual(device.computePasses[0].dispatches, [{ x: 1, y: 1, z: 1 }]);

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
        attentionDecodeOnline: {
          useDirectContiguousKVLayout: true,
        },
      },
    },
  });

  const recorder = await recordOnlineDecodeAttention(device, {
    headDim: 256,
    kernelPath: head256KernelPath(),
    kvLayout: 'paged',
    outputGate: true,
  });

  assert.equal(device.pipelineDescriptors.length, 1);
  assert.equal(device.pipelineDescriptors[0].label, 'attention_decode_online_head256_f16kv_output_gate_pipeline');
  assert.equal(device.pipelineDescriptors[0].compute.constants, undefined);
  assert.deepEqual(device.computePasses[0].dispatches, [{ x: 1, y: 1, z: 1 }]);

  recorder.abort();
  destroyBufferPool();
  resetUniformCache();
  clearPipelineCaches();
  setDevice(null);
}

console.log('attention-online-constants.test: ok');
