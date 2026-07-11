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
const { destroyBufferPool, getBufferPool, releaseBuffer } = await import('../../src/memory/buffer-pool.js');
const { loadBackwardRegistry } = await import('../../src/config/backward-registry-loader.js');
const { AutogradTape, OpType } = await import('../../src/experimental/training/autograd.js');

class FakeBuffer {
  constructor({ size, usage, label = null }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
    this.bytes = null;
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

const ORIGINAL_GPU_BUFFER = globalThis.GPUBuffer;
globalThis.GPUBuffer = FakeBuffer;

function createFakeDevice() {
  const dispatches = [];
  return {
    dispatches,
    queue: {
      submit() {},
      writeBuffer(buffer, offset, data) {
        const bytes = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        buffer.ensureBytes(offset + bytes.byteLength).set(bytes, offset);
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
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
    createBuffer({ size, usage, label }) {
      return new FakeBuffer({ size, usage, label });
    },
    createCommandEncoder({ label }) {
      return {
        beginComputePass({ label: passLabel }) {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups(x, y, z) {
              dispatches.push({ encoderLabel: label, passLabel, x, y, z });
            },
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

function makeTensor(size, dtype, shape, label) {
  return createTensor(
    new FakeBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label }),
    dtype,
    shape,
    label
  );
}

try {
  destroyBufferPool();
  const device = createFakeDevice();
  setDevice(device, { platformConfig: null });
  getBufferPool().configure({ enablePooling: false });

  const numTokens = 383;
  const vocabSize = 262144;
  const totalBytes = numTokens * vocabSize * 4;
  const logits = makeTensor(totalBytes, 'f32', [numTokens, vocabSize], 'logits');
  const softmax = makeTensor(totalBytes, 'f32', [numTokens, vocabSize], 'softmax');
  const targets = makeTensor(numTokens * 4, 'u32', [numTokens], 'targets');
  const loss = makeTensor(numTokens * 4, 'f32', [numTokens], 'loss');
  const gradLoss = makeTensor(numTokens * 4, 'f32', [numTokens], 'grad_loss');

  const tape = new AutogradTape(loadBackwardRegistry());
  tape.records.push({
    op: OpType.SOFTMAX,
    inputs: [logits],
    output: softmax,
    options: { rows: numTokens, cols: vocabSize },
  });
  tape.records.push({
    op: OpType.CROSS_ENTROPY,
    inputs: [softmax, targets],
    output: loss,
    options: { numTokens, vocabSize, logitsInput: logits },
  });

  const grads = await tape.backward(gradLoss);
  const gradLogits = grads.get(logits);
  assert.ok(gradLogits);
  assert.equal(grads.has(softmax), false);
  assert.equal(device.dispatches.length, 1);
  assert.equal(device.dispatches[0].passLabel, 'cross_entropy_backward_pass');
  assert.equal(device.dispatches[0].x, 65535);
  assert.equal(device.dispatches[0].y, 6);
  assert.equal(device.dispatches[0].z, 1);
  releaseBuffer(gradLogits.buffer);

  const reshapeInput = makeTensor(24, 'f32', [2, 3], 'reshape_input');
  const reshapeTape = new AutogradTape(loadBackwardRegistry());
  const reshapeOutput = await reshapeTape.record(
    OpType.RESHAPE,
    (input) => createTensor(input.buffer, input.dtype, [1, 2, 3], 'reshape_output'),
    [reshapeInput],
    { shape: [1, 2, 3] }
  );
  const reshapeGrad = makeTensor(24, 'f32', [1, 2, 3], 'reshape_grad');
  const reshapeGrads = await reshapeTape.backward(reshapeGrad);
  assert.deepEqual(reshapeGrads.get(reshapeInput).shape, [2, 3]);
  assert.equal(reshapeGrads.get(reshapeInput).buffer, reshapeGrad.buffer);
  assert.equal(reshapeOutput.buffer, reshapeInput.buffer);

  const ropeInput = makeTensor(32, 'f32', [2, 1, 4], 'rope_input');
  const ropeCos = makeTensor(16, 'f32', [2, 2], 'rope_cos');
  const ropeSin = makeTensor(16, 'f32', [2, 2], 'rope_sin');
  const ropeOutput = makeTensor(32, 'f32', [2, 1, 4], 'rope_output');
  const ropeGrad = makeTensor(32, 'f32', [2, 1, 4], 'rope_grad');
  const ropeTape = new AutogradTape(loadBackwardRegistry());
  ropeTape.records.push({
    op: OpType.ROPE,
    inputs: [ropeInput, ropeCos, ropeSin],
    output: ropeOutput,
    options: {
      seqLen: 2,
      numHeads: 1,
      headDim: 4,
      startPos: 0,
      stopGradInputs: [1, 2],
    },
  });
  const ropeGrads = await ropeTape.backward(ropeGrad);
  assert.deepEqual(ropeGrads.get(ropeInput).shape, [2, 1, 4]);
  assert.equal(ropeGrads.has(ropeCos), false);
  assert.equal(ropeGrads.has(ropeSin), false);
  releaseBuffer(ropeGrads.get(ropeInput).buffer);
  destroyBufferPool();
  setDevice(null);
} finally {
  globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
}
