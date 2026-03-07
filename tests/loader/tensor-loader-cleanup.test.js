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
const { destroyBufferPool, getBufferPool } = await import('../../src/memory/buffer-pool.js');
const {
  loadQ4KFused,
  loadQ6K,
  loadBF16,
  loadFloat,
} = await import('../../src/loader/tensors/tensor-loader.js');
const { loadEmbeddings } = await import('../../src/loader/embedding-loader.js');
const { loadFinalWeights } = await import('../../src/loader/final-weights-loader.js');

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

function createFakeDevice(options = {}) {
  const createdBuffers = [];
  return {
    createdBuffers,
    features: new Set(options.features || []),
    limits: {
      maxBufferSize: 1 << 20,
      maxStorageBufferBindingSize: 1 << 20,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
    },
    queue: {
      submit() {},
      writeBuffer() {
        if (options.writeBufferThrows) {
          throw new Error('writeBuffer failed');
        }
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    createBindGroup(descriptor) {
      return descriptor;
    },
    createBuffer({ size, usage }) {
      const buffer = new FakeBuffer({ size, usage });
      createdBuffers.push(buffer);
      return buffer;
    },
    ...(options.withShaders ? {
      createShaderModule() {
        return {
          async getCompilationInfo() {
            return { messages: [] };
          },
        };
      },
      createComputePipeline() {
        return {
          getBindGroupLayout() {
            return {};
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
      createPipelineLayout(descriptor) {
        return descriptor;
      },
      createBindGroupLayout(descriptor) {
        return descriptor;
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
    } : {}),
  };
}

function resetRuntimeState(device) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

async function flushDeferredDestroy() {
  await Promise.resolve();
  await Promise.resolve();
}

function assertPoolClean() {
  assert.equal(getBufferPool().getStats().activeBuffers, 0);
}

const PIPELINE_FAILURE = /shader-f16|createShaderModule|createComputePipeline|createComputePipelineAsync/;

{
  const device = createFakeDevice({ writeBufferThrows: true });
  resetRuntimeState(device);

  await assert.rejects(
    () => loadQ4KFused(new Uint8Array([1, 2, 3, 4]), { size: 4, shape: [2, 2] }, 'q4k_fused'),
    /writeBuffer failed/
  );
  await flushDeferredDestroy();
  assertPoolClean();
  assert.equal(device.createdBuffers[0]?.destroyed, true);
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  await assert.rejects(
    () => loadQ6K(
      new Uint8Array(210),
      { size: 210, shape: [1, 256], dtype: 'Q6_K', role: 'matmul' },
      'q6k_weight'
    ),
    PIPELINE_FAILURE
  );
  await flushDeferredDestroy();
  assertPoolClean();
  assert.equal(device.createdBuffers[0]?.destroyed, true);
}

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  await assert.rejects(
    () => loadBF16(
      new Uint8Array([1, 0, 2, 0]),
      { size: 4, shape: [2], dtype: 'BF16', role: 'embedding' },
      'bf16_weight',
      { gpuCapabilities: { hasF16: false } }
    ),
    PIPELINE_FAILURE
  );
  await flushDeferredDestroy();
  assertPoolClean();
  assert.equal(device.createdBuffers[0]?.destroyed, true);
}

{
  const device = createFakeDevice({ writeBufferThrows: true });
  resetRuntimeState(device);

  await assert.rejects(
    () => loadFloat(
      new Uint8Array([1, 0, 2, 0]),
      { size: 4, shape: [2], dtype: 'F16', role: 'embedding' },
      'float_weight',
      { allowF32UpcastNonMatmul: true }
    ),
    /writeBuffer failed/
  );
  await flushDeferredDestroy();
  assertPoolClean();
  assert.equal(device.createdBuffers[0]?.destroyed, true);
}

{
  resetRuntimeState(null);
  delete globalThis.GPUBuffer;

  const tensorLocations = new Map([
    ['embed.weight', { role: 'embedding', group: 'embed', shape: [2, 2], dtype: 'F32' }],
  ]);
  const embeddings = await loadEmbeddings({
    tensorLocations,
    loadTensor: async () => new Float32Array([1, 2, 3, 4]),
    shouldStreamLargeWeight: () => false,
    resolveWeightLayout: () => 'row',
    gpuBuffers: new Set(),
    keepF32Weights: false,
    preserveF32Embeddings: false,
  });
  assert.ok(embeddings instanceof Float32Array);
}

{
  resetRuntimeState(null);
  delete globalThis.GPUBuffer;

  const tensorLocations = new Map([
    ['norm.weight', { role: 'norm', group: 'head', shape: [2], dtype: 'F32' }],
    ['lm_head.weight', { role: 'lm_head', group: 'head', shape: [2, 2], dtype: 'F32' }],
  ]);
  const lookup = new Map([
    ['norm.weight', new Float32Array([1, 2])],
    ['lm_head.weight', new Float32Array([1, 2, 3, 4])],
  ]);
  const result = await loadFinalWeights({
    tensorLocations,
    tieWordEmbeddings: false,
    loadTensor: async (name) => lookup.get(name) || null,
    shouldStreamLargeWeight: () => false,
    needsNormWeightOffset: () => false,
    resolveWeightLayout: () => 'row',
    embeddings: null,
    gpuBuffers: new Set(),
    keepF32Weights: false,
    normOffsetDebugLogged: false,
  });
  assert.ok(result.finalNorm instanceof Float32Array);
  assert.ok(result.lmHead instanceof Float32Array);
}

resetRuntimeState(null);
console.log('tensor-loader-cleanup.test: ok');
