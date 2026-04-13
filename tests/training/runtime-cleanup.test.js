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

globalThis.GPUMapMode = {
  READ: 1 << 0,
  WRITE: 1 << 1,
};

const { configurePerfGuards } = await import('../../src/gpu/perf-guards.js');
const { setDevice } = await import('../../src/gpu/device.js');
const {
  acquireBuffer,
  destroyBufferPool,
  getBufferPool,
  releaseBuffer,
  uploadData,
} = await import('../../src/memory/buffer-pool.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { AutogradTape } = await import('../../src/experimental/training/autograd.js');
const {
  attentionBackwardCpu,
  buildAttentionSoftmaxCache,
} = await import('../../src/experimental/training/attention-backward.js');
const { AdamOptimizer } = await import('../../src/experimental/training/optimizer.js');
const { runAdam } = await import('../../src/gpu/kernels/backward/adam.js');
const { runAttentionBackward } = await import('../../src/gpu/kernels/backward/attention_backward.js');
const { runMatmul } = await import('../../src/gpu/kernels/matmul.js');
const { LoraAdapter } = await import('../../src/experimental/training/lora.js');
const { createTokenBatchTensors } = await import('../../src/experimental/training/datasets/token-batch.js');

class FakeBuffer {
  constructor({ size, usage, initialBytes = null, mapReject = false }) {
    this.size = size;
    this.usage = usage;
    this.mapReject = mapReject;
    this.destroyed = false;
    this.unmapped = false;
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

  async mapAsync() {
    if (this.mapReject) {
      throw new Error('map failed');
    }
  }

  getMappedRange(offset = 0, size = this.size - offset) {
    const bytes = this.ensureBytes(offset + size);
    return bytes.slice(offset, offset + size).buffer;
  }

  unmap() {
    this.unmapped = true;
  }

  destroy() {
    this.destroyed = true;
  }
}

function copyBytes(source, sourceOffset, target, targetOffset, size) {
  const sourceBytes = source.ensureBytes ? source.ensureBytes(sourceOffset + size) : new Uint8Array(source.size);
  const targetBytes = target.ensureBytes ? target.ensureBytes(targetOffset + size) : new Uint8Array(target.size);
  targetBytes.set(sourceBytes.subarray(sourceOffset, sourceOffset + size), targetOffset);
}

function createFakeDevice(options = {}) {
  let createBufferCount = 0;
  let createBindGroupCount = 0;
  let submitCount = 0;
  let writeBufferCount = 0;
  const createdBuffers = [];

  const queue = {
    submit(commandBuffers) {
      submitCount += 1;
      if (options.submitThrowAt === submitCount) {
        throw new Error(`submit failed at ${submitCount}`);
      }
      for (const commandBuffer of commandBuffers) {
        for (const op of commandBuffer.ops ?? []) {
          copyBytes(op.source, op.sourceOffset, op.target, op.targetOffset, op.size);
        }
      }
    },
    writeBuffer(buffer, offset, data) {
      writeBufferCount += 1;
      if (options.writeBufferThrowAt === writeBufferCount) {
        throw new Error(`writeBuffer failed at ${writeBufferCount}`);
      }
      const bytes = buffer.ensureBytes(offset + data.byteLength);
      bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
    },
    onSubmittedWorkDone() {
      return Promise.resolve();
    },
  };

  return {
    createdBuffers,
    queue,
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize: 1 << 30,
      maxBufferSize: 1 << 30,
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
      if (options.createBindGroupThrowAt === createBindGroupCount) {
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
      createBufferCount += 1;
      if (options.createBufferThrowAt === createBufferCount) {
        throw new Error(`createBuffer failed at ${createBufferCount}`);
      }
      const buffer = new FakeBuffer({ size, usage });
      createdBuffers.push(buffer);
      return buffer;
    },
    createCommandEncoder() {
      const ops = [];
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        copyBufferToBuffer(source, sourceOffset, target, targetOffset, size) {
          ops.push({ source, sourceOffset, target, targetOffset, size });
        },
        finish() {
          return { ops };
        },
      };
    },
  };
}

function createExternalTensor(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  const bytes = new Uint8Array(data.buffer.slice(0));
  const buffer = new FakeBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    initialBytes: bytes,
  });
  return createTensor(buffer, 'f32', shape, label);
}

function assertPoolIsClean() {
  const stats = getBufferPool().getStats();
  assert.equal(stats.activeBuffers, 0);
}

function resetRuntimeState() {
  destroyBufferPool();
  setDevice(null);
}

configurePerfGuards({
  allowGPUReadback: true,
  trackSubmitCount: false,
  trackAllocations: false,
  logExpensiveOps: false,
  strictMode: false,
});

{
  const device = createFakeDevice({ submitThrowAt: 1 });
  setDevice(device, { platformConfig: null });

  const tape = new AutogradTape({ ops: {} });
  const maxChunkElements = 65535 * 256;
  const size = maxChunkElements + 1;
  const bytes = size * 4;
  const existing = createTensor(
    new FakeBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
    'f32',
    [size],
    'existing_grad'
  );
  const grad = createTensor(
    new FakeBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
    'f32',
    [size],
    'next_grad'
  );

  await assert.rejects(
    () => tape.accumulateLargeGradF32(existing, grad, size, [size]),
    /submit failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ createBindGroupThrowAt: 1 });
  setDevice(device, { platformConfig: null });

  const input = createExternalTensor([1, 2, 3, 4], [1, 4], 'matmul_input');
  const weight = new FakeBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    initialBytes: new Uint8Array(4 * 4),
  });

  await assert.rejects(
    () => runMatmul(input, weight, 1, 1, 4, { transposeB: false }),
    /createBindGroup failed at 1/
  );

  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ createBindGroupThrowAt: 1 });
  setDevice(device, { platformConfig: null });

  const q = createExternalTensor([1], [1, 1, 1], 'attn_q');
  const k = createExternalTensor([1], [1, 1, 1], 'attn_k');
  const v = createExternalTensor([1], [1, 1, 1], 'attn_v');
  const softmax = createExternalTensor([1], [1, 1, 1], 'attn_softmax');
  const gradOutput = createExternalTensor([1], [1, 1, 1], 'attn_grad_output');

  await assert.rejects(
    () => runAttentionBackward(q, k, v, softmax, gradOutput, {
      seqLen: 1,
      numHeads: 1,
      headDim: 1,
    }),
    /createBindGroup failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ submitThrowAt: 1 });
  setDevice(device, { platformConfig: null });

  const maxChunkElements = 65535 * 256;
  const size = maxChunkElements + 1;
  const bytes = size * 4;
  const paramsBuffer = acquireBuffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'adam_params');
  const gradsBuffer = acquireBuffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'adam_grads');
  const mBuffer = acquireBuffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'adam_m');
  const vBuffer = acquireBuffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'adam_v');
  const params = createTensor(paramsBuffer, 'f32', [size], 'adam_params');
  const grads = createTensor(gradsBuffer, 'f32', [size], 'adam_grads');
  const moment1 = createTensor(mBuffer, 'f32', [size], 'adam_m');
  const moment2 = createTensor(vBuffer, 'f32', [size], 'adam_v');

  await assert.rejects(
    () => runAdam(params, grads, moment1, moment2, {
      count: size,
      step: 1,
      lr: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
    }),
    /submit failed at 1/
  );

  releaseBuffer(paramsBuffer);
  releaseBuffer(gradsBuffer);
  releaseBuffer(mBuffer);
  releaseBuffer(vBuffer);
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 2 });
  setDevice(device, { platformConfig: null });

  const batch = {
    inputFlat: new Uint32Array([1, 2, 3]),
    targetFlat: new Uint32Array([4, 5, 6]),
    offsets: [0],
  };

  assert.throws(
    () => createTokenBatchTensors(batch),
    /writeBuffer failed at 2/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ createBufferThrowAt: 2 });
  setDevice(device, { platformConfig: null });

  const optimizer = new AdamOptimizer({});
  const paramBuffer = acquireBuffer(16, GPUBufferUsage.STORAGE, 'param');
  const param = createTensor(paramBuffer, 'f32', [4], 'param');

  assert.throws(
    () => optimizer.getState(param),
    /createBuffer failed at 2/
  );

  releaseBuffer(paramBuffer);
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ createBufferThrowAt: 2 });
  setDevice(device, { platformConfig: null });

  assert.throws(
    () => new LoraAdapter({ inDim: 4, outDim: 4, rank: 2, alpha: 8 }),
    /createBuffer failed at 2/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 1 });
  setDevice(device, { platformConfig: null });

  const q = createExternalTensor([1], [1, 1, 1], 'q');
  const k = createExternalTensor([1], [1, 1, 1], 'k');

  await assert.rejects(
    () => buildAttentionSoftmaxCache(q, k, { seqLen: 1, numHeads: 1, headDim: 1 }),
    /writeBuffer failed at 1/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

{
  const device = createFakeDevice({ writeBufferThrowAt: 2 });
  setDevice(device, { platformConfig: null });

  const q = createExternalTensor([1], [1, 1, 1], 'q');
  const k = createExternalTensor([1], [1, 1, 1], 'k');
  const v = createExternalTensor([1], [1, 1, 1], 'v');
  const gradOutput = createExternalTensor([1], [1, 1, 1], 'grad');

  await assert.rejects(
    () => attentionBackwardCpu(q, k, v, null, gradOutput, { seqLen: 1, numHeads: 1, headDim: 1 }),
    /writeBuffer failed at 2/
  );
  assertPoolIsClean();
  resetRuntimeState();
}

console.log('runtime-cleanup.test: ok');
