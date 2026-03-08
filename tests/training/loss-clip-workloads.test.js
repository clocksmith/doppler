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

const { setDevice } = await import('../../src/gpu/device.js');
const {
  acquireBuffer,
  destroyBufferPool,
  getBufferPool,
  releaseBuffer,
  uploadData,
} = await import('../../src/memory/buffer-pool.js');
const { createTensor } = await import('../../src/gpu/tensor.js');
const { AutogradTape } = await import('../../src/training/autograd.js');
const { clipGradients } = await import('../../src/training/clip.js');
const { normalizeTrainingWorkloadPack } = await import('../../src/training/workloads.js');

class FakeBuffer {
  constructor({ size, usage, initialBytes = null }) {
    this.size = size;
    this.usage = usage;
    this.destroyed = false;
    this.unmapped = false;
    this.bytes = initialBytes ? new Uint8Array(initialBytes) : new Uint8Array(size);
  }

  ensureBytes(minSize = this.size) {
    if (this.bytes.length < minSize) {
      const next = new Uint8Array(minSize);
      next.set(this.bytes.subarray(0, this.bytes.length));
      this.bytes = next;
    }
    return this.bytes;
  }

  async mapAsync() {}

  getMappedRange(offset = 0, size = this.size - offset) {
    return this.ensureBytes(offset + size).slice(offset, offset + size).buffer;
  }

  unmap() {
    this.unmapped = true;
  }

  destroy() {
    this.destroyed = true;
  }
}

function copyBytes(source, sourceOffset, target, targetOffset, size) {
  const sourceBytes = source.ensureBytes(sourceOffset + size);
  const targetBytes = target.ensureBytes(targetOffset + size);
  targetBytes.set(sourceBytes.subarray(sourceOffset, sourceOffset + size), targetOffset);
}

function createFakeDevice() {
  return {
    lost: new Promise(() => {}),
    queue: {
      submit(commandBuffers) {
        for (const commandBuffer of commandBuffers) {
          for (const op of commandBuffer.ops ?? []) {
            copyBytes(op.source, op.sourceOffset, op.target, op.targetOffset, op.size);
          }
        }
      },
      writeBuffer(buffer, offset, data) {
        const bytes = buffer.ensureBytes(offset + data.byteLength);
        bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
      },
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
    createBuffer({ size, usage }) {
      return new FakeBuffer({ size, usage });
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
    destroy() {},
  };
}

function makeF32Tensor(values, shape, label) {
  const data = new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label);
}

try {
  {
    destroyBufferPool();
    const device = createFakeDevice();
    setDevice(device, { platformConfig: null });
    getBufferPool().configure({ enablePooling: false });
    const pool = getBufferPool();
    const retained = pool.acquire(32, GPUBufferUsage.STORAGE, 'retained_logits');
    const output = { buffer: new FakeBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }), shape: [1], dtype: 'f32' };
    const grad = { buffer: new FakeBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }), shape: [1], dtype: 'f32' };
    const tape = new AutogradTape({ ops: {} });

    await tape.record('softmax', async () => output, [], { retainBuffers: [retained] });
    assert.equal(pool.isActiveBuffer(retained), true);
    await tape.backward(grad);
    assert.equal(pool.isActiveBuffer(retained), false);
    destroyBufferPool();
    setDevice(null);
  }

  {
    destroyBufferPool();
    const device = createFakeDevice();
    setDevice(device, { platformConfig: null });
    getBufferPool().configure({ enablePooling: false });

    const grads = new Map([
      ['weight', makeF32Tensor([3, 4], [2], 'clip_grad')],
    ]);

    const result = await clipGradients(grads, {
      training: {
        gradientClipping: { maxNorm: 0 },
        gradient: { maxNorm: 0.5 },
      },
    });

    assert.equal(result.clipped_event_count, 0);
    for (const grad of grads.values()) {
      releaseBuffer(grad.buffer);
    }
    destroyBufferPool();
    setDevice(null);
  }

  {
    assert.throws(
      () => normalizeTrainingWorkloadPack({
        schemaVersion: 1,
        id: 'distill-smoke',
        description: 'missing explicit kind',
        seed: 1,
        trainingSchemaVersion: 1,
        trainingTests: ['distill-stage-a'],
      }, { label: 'workload' }),
      /workload\.kind is required/
    );

    assert.throws(
      () => normalizeTrainingWorkloadPack({
        schemaVersion: 1,
        kind: 'lora',
        id: 'lora-smoke',
        description: 'missing explicit optimizer fields',
        claimBoundary: 'boundary',
        seed: 1,
        baseModelId: 'toy',
        studentModelId: null,
        teacherModelId: null,
        datasetId: 'toy',
        datasetPath: 'toy.jsonl',
        evalDatasets: [],
        trainingSchemaVersion: 1,
        checkpointEvery: 1,
        selectionMetric: 'accuracy',
        selectionGoal: 'max',
        surfaceSupport: 'node',
        training: {
          optimizer: {
            lr: 0.001,
            beta2: 0.999,
            eps: 1e-8,
            weightDecay: 0,
            scheduler: {
              enabled: false,
              type: 'constant',
              warmupSteps: 0,
              stepSize: 1,
              gamma: 1,
              totalSteps: 1,
              minLr: 0,
            },
          },
          batchSize: 1,
          accumSteps: 1,
          steps: 1,
          precision: {
            activations: 'f32',
            gradients: 'f32',
            loraParams: 'f32',
          },
          gradientClipping: {
            maxNorm: 1,
          },
        },
        lora: {
          datasetFormat: 'toy_linear_classification_jsonl',
          taskType: 'classification',
          adapter: {
            rank: 2,
            alpha: 4,
            dropout: 0,
            targetModules: ['q_proj'],
          },
          freeze: {
            encoder: false,
            prior: false,
            decoder: false,
            base: true,
            lora: false,
          },
          export: {
            enabled: true,
            atCheckpoints: true,
            select: 'best',
            id: 'toy',
            name: 'Toy',
            format: 'manifest_json',
          },
          activation: {
            enabled: false,
            autoActivate: false,
            smokePrompt: null,
          },
        },
      }, { label: 'workload' }),
      /workload\.training\.optimizer\.type is required/
    );
  }
} finally {
  destroyBufferPool();
  setDevice(null);
}

console.log('loss-clip-workloads.test: ok');
