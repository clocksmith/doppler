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
  loadTensorToGPU,
  loadQ6K,
  loadBF16,
  loadFloat,
} = await import('../../src/loader/tensors/tensor-loader.js');
const { loadEmbeddings } = await import('../../src/loader/embedding-loader.js');
const { loadFinalWeights } = await import('../../src/loader/final-weights-loader.js');
const { DopplerLoader } = await import('../../src/loader/doppler-loader.js');
const { getWeightDtype } = await import('../../src/gpu/weight-buffer.js');

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

globalThis.GPUBuffer = FakeBuffer;

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

{
  const device = createFakeDevice({ features: ['shader-f16', 'subgroups'] });
  resetRuntimeState(device);
  const location = { dtype: 'Q4_K_M', size: 144, shape: [1, 256], role: 'matmul' };
  const borrowed = new FakeBuffer({ size: 144, usage: GPUBufferUsage.STORAGE });
  const result = await loadTensorToGPU(borrowed, location, 'packed_only_q_proj', {
    useFusedQ4K: true, q4kLayout: 'row', keepF32Weights: false,
    q4kMaterializationMode: 'mixed', q4kFusedRoles: ['q_proj'], q4kPackedOnlyRoles: ['q_proj'],
    consumerRole: 'q_proj',
  });
  assert.equal(location.role, 'matmul', 'Consumer binding never rewrites artifact classification.');
  assert.equal(getWeightDtype(result.data), 'q4k');
  assert.equal(result.data.buffer, borrowed);
  assert.equal(device.createdBuffers.length, 0, 'Packed-only dispatch must allocate no dense shadow.');
  assertPoolClean();
  const loader = new DopplerLoader();
  loader.setQ4KConfig({useFusedQ4K:true, q4kLayout:'row', keepF32Weights:false,
    q4kMaterializationMode:'mixed', q4kFusedRoles:['q_proj'], q4kPackedOnlyRoles:['q_proj']});
  loader.tensorLocations.set('generic-q4', location);
  loader._loadingConfig = {allowF32UpcastNonMatmul:true};
  loader.gpuCapabilities = {hasF16:true, hasSubgroups:true};
  loader._shouldStreamUploadToGPU = () => false;
  loader._assembleShardData = async () => borrowed;
  const throughLoader = await loader._loadTensor('generic-q4', true, false, 'q_proj');
  assert.equal(throughLoader.buffer, borrowed, 'Loader facade must retain the consumer-slot binding.');
  assert.equal(device.createdBuffers.length, 0);
}

// Reachable fallback precision is retained, not replaced by role-wide packed policy.
{
  const device = createFakeDevice({ features: ['shader-f16', 'subgroups'], withShaders: true });
  resetRuntimeState(device);
  const location = { dtype: 'Q4_K_M', size: 144, shape: [1, 256], role: 'matmul' };
  const config = {
    useFusedQ4K: true, q4kLayout: 'row', keepF32Weights: false,
    q4kMaterializationMode: 'mixed', q4kPackedOnlyRoles: ['q_proj'], consumerRole: 'q_proj',
    representationRequirement: { mode: 'both', complete: true, dtypes: Object.freeze(['f16', 'f32', 'q4k']) },
  };
  const result = await loadTensorToGPU(new Uint8Array(144), location, 'fallback_q_proj', config);
  assert.deepEqual(Object.keys(result.data.materializations).sort(), ['f16', 'f32', 'q4k']);
  assert.equal(result.allocatedBuffers.length, 3);
  assert.notEqual(result.data.materializations.f16.buffer, result.data.materializations.f32.buffer);
  const dense = await loadTensorToGPU(new Uint8Array(144), location, 'dense_q_proj', {
    ...config, representationRequirement: { mode: 'dense-only', complete: true, dtypes: ['f16', 'f32'] },
  });
  assert.deepEqual(Object.keys(dense.data.materializations).sort(), ['f16', 'f32']);
  assert.equal(dense.allocatedBuffers.length, 2);
  const float = await loadTensorToGPU(new Uint8Array(1024), { ...location, dtype: 'F32', size: 1024 }, 'float_q_proj', {
    ...config, representationRequirement: { mode: 'dense-only', complete: true, dtypes: ['f16', 'f32'] },
  });
  assert.deepEqual(Object.keys(float.data.materializations).sort(), ['f16', 'f32']);
  assert.equal(float.allocatedBuffers.length, 2);
  // An embedding alias may feed a packed output head as well as dense gather.
  const { compileWeightRepresentationPlan } = await import('../../src/loader/weight-representation-plan.js');
  const embeddingLocation = { ...location, shardIndex: 0, offset: 0, role: 'embedding' };
  const aliasPlan = compileWeightRepresentationPlan(new Map([
    ['embedding', embeddingLocation], ['head', { ...embeddingLocation, role: 'lm_head' }],
  ]), { architecture: { numLayers: 1 } }, [{ id: 'alias', postLayer: [
    { op: 'lm_head', kernel: 'fused_matmul_q4.wgsl', entry: 'main_multicol' },
    { op: 'lm_head_prefill', kernel: 'matmul_f16w_f32a_tiled.wgsl', entry: 'main' },
  ], prefill: { steps: [] }, decode: { steps: [] } }], config);
  const embedding = await loadTensorToGPU(new Uint8Array(144), embeddingLocation, 'embedding', {
    ...config, representationRequirement: aliasPlan.embedding,
  });
  assert.deepEqual(Object.keys(embedding.data.materializations).sort(), ['f16', 'f32', 'q4k']);
  assert.equal(embeddingLocation.role, 'embedding');
}

// Failure after the first dense copy must retire it and the owned packed upload.
{
  const device = createFakeDevice({ features: ['shader-f16', 'subgroups'], withShaders: true });
  let submissions = 0;
  device.queue.submit = () => { if (++submissions === 2) throw new Error('second precision failed'); };
  resetRuntimeState(device);
  await assert.rejects(() => loadTensorToGPU(new Uint8Array(144), {
    dtype: 'Q4_K_M', size: 144, shape: [1, 256], role: 'matmul',
  }, 'failed_fallback', {
    useFusedQ4K: true, q4kLayout: 'row', keepF32Weights: false,
    representationRequirement: { mode: 'both', complete: true, dtypes: ['f16', 'f32', 'q4k'] },
  }), /second precision failed/);
  await flushDeferredDestroy();
  assertPoolClean();
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
  const device = createFakeDevice({ withShaders: true, features: ['shader-f16'] });
  resetRuntimeState(device);

  const result = await loadBF16(
    new Uint8Array([1, 0, 2, 0]),
    { size: 4, shape: [1, 2], dtype: 'BF16', role: 'matmul' },
    'bf16_matmul_packed',
    {
      gpuCapabilities: { hasF16: true },
      keepF32Weights: false,
      keepBF16Weights: true,
    }
  );

  assert.equal(result.data.dtype, 'bf16');
  assert.ok(result.data.buffer.size >= 4);
  assert.equal(result.allocatedBuffers.length, 1);
  assert.equal(result.allocatedBuffers[0], result.data.buffer);
  resetRuntimeState(null);
}

{
  const device = createFakeDevice({ withShaders: true, features: ['shader-f16'] });
  resetRuntimeState(device);

  const result = await loadBF16(
    new Uint8Array([1, 0, 2, 0]),
    { size: 4, shape: [2], dtype: 'BF16', role: 'norm' },
    'bf16_norm_f32',
    {
      gpuCapabilities: { hasF16: true },
      keepF32Weights: false,
      keepBF16Weights: true,
    }
  );

  assert.equal(getWeightDtype(result.data), 'f32');
  assert.ok(result.data.size >= 8);
  assert.equal(result.allocatedBuffers.length, 1);
  assert.equal(result.allocatedBuffers[0], result.data);
  resetRuntimeState(null);
}

{
  const device = createFakeDevice({ withShaders: true, features: ['shader-f16'] });
  resetRuntimeState(device);

  const result = await loadBF16(
    new Uint8Array([1, 0, 2, 0]),
    { size: 4, shape: [1, 2], dtype: 'BF16', role: 'matmul' },
    'bf16_matmul_f16',
    {
      gpuCapabilities: { hasF16: true },
      keepF32Weights: false,
    }
  );

  assert.equal(result.data.dtype, 'f16');
  assert.ok(result.data.buffer.size >= 4);
  assert.equal(result.allocatedBuffers.length, 1);
  assert.equal(result.allocatedBuffers[0], result.data.buffer);
  resetRuntimeState(null);
}

{
  const device = createFakeDevice({ withShaders: true, features: ['shader-f16'] });
  resetRuntimeState(device);

  const result = await loadBF16(
    new Uint8Array([1, 0, 2, 0]),
    { size: 4, shape: [1, 2], dtype: 'BF16', role: 'matmul' },
    'bf16_matmul_f32',
    {
      gpuCapabilities: { hasF16: true },
      keepF32Weights: true,
    }
  );

  assert.equal(result.data.dtype, 'f32');
  assert.ok(result.data.buffer.size >= 8);
  assert.equal(result.allocatedBuffers.length, 1);
  assert.equal(result.allocatedBuffers[0], result.data.buffer);
  resetRuntimeState(null);
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
  await assert.rejects(() => loadEmbeddings({
    tensorLocations,
    loadTensor: async () => new Float32Array([1, 2, 3, 4]),
    shouldStreamLargeWeight: () => false,
    resolveWeightLayout: () => 'row',
    gpuBuffers: new Set(),
    keepF32Weights: false,
    preserveF32Embeddings: false,
  }), /must materialize GPU weights for token gather/);
}

{
  resetRuntimeState(null);
  delete globalThis.GPUBuffer;

  const tensorLocations = new Map([
    ['embed.weight', { role: 'embedding', group: 'embed', shape: [2, 2], dtype: 'F16' }],
  ]);
  const toGpuCalls = [];
  await assert.rejects(() => loadEmbeddings({
    tensorLocations,
    loadTensor: async (_name, toGPU) => {
      toGpuCalls.push(toGPU);
      return new Float32Array([1, 2, 3, 4]);
    },
    shouldStreamLargeWeight: () => false,
    resolveWeightLayout: () => 'row',
    gpuBuffers: new Set(),
    keepF32Weights: false,
    preserveF32Embeddings: false,
    hostHasShaderF16: false,
  }), /must materialize GPU weights for token gather/);
  assert.deepEqual(toGpuCalls, [true]);
}

{
  resetRuntimeState(null);
  delete globalThis.GPUBuffer;

  const tensorLocations = new Map([
    ['embed.weight', {
      role: 'embedding',
      group: 'embed',
      shape: [4, 4],
      dtype: 'F16',
      sourceTransform: {
        kind: 'litert_axis_blocked_dequant',
        sourceDtype: 'INT2',
        targetDtype: 'F16',
      },
    }],
  ]);
  const toGpuCalls = [];
  await assert.rejects(() => loadEmbeddings({
    tensorLocations,
    loadTensor: async (_name, toGPU) => {
      toGpuCalls.push(toGPU);
      return new Float32Array(16);
    },
    shouldStreamLargeWeight: () => false,
    loadShardRange: async () => new ArrayBuffer(0),
    resolveWeightLayout: () => 'row',
    gpuBuffers: new Set(),
    keepF32Weights: false,
    preserveF32Embeddings: false,
    hostHasShaderF16: true,
  }), /must materialize GPU weights for token gather/);
  assert.deepEqual(toGpuCalls, [true], 'source transforms must materialize before GPU gather');
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
