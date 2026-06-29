import assert from 'node:assert/strict';

const { loadFinalWeights } = await import('../../src/loader/final-weights-loader.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');

function createFakeGpuBuffer(size) {
  return {
    __dopplerFakeGPUBuffer: true,
    size,
    usage: 0,
    destroy() {},
    mapAsync() {
      return Promise.resolve();
    },
    getMappedRange() {
      return new ArrayBuffer(size);
    },
    unmap() {},
  };
}

{
  const tensorLocations = new Map([
    ['model.language_model.norm.weight', { role: 'norm', shape: [2], dtype: 'F32' }],
    ['lm_head.weight', { role: 'lm_head', group: 'head', shape: [2, 2], dtype: 'F32' }],
  ]);

  const lookup = new Map([
    ['model.language_model.norm.weight', new Float32Array([1, 2])],
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

  assert.equal(result.finalNorm, lookup.get('model.language_model.norm.weight'));
  assert.equal(result.lmHead, lookup.get('lm_head.weight'));
  assert.equal(result.normOffsetDebugLogged, false);
}

// DiffusionGemma self-conditioning is loaded only when explicitly enabled
{
  const tensorLocations = new Map([
    ['model.language_model.norm.weight', { role: 'norm', group: 'head', shape: [2], dtype: 'F32' }],
    ['lm_head.weight', { role: 'lm_head', group: 'head', shape: [2, 2], dtype: 'F32' }],
  ]);

  await assert.rejects(
    () => loadFinalWeights({
      tensorLocations,
      tieWordEmbeddings: false,
      diffusionGemmaSelfConditioning: true,
      loadTensor: async (name) => {
        if (name === 'model.language_model.norm.weight') return new Float32Array([1, 2]);
        if (name === 'lm_head.weight') return new Float32Array([1, 2, 3, 4]);
        if (name.endsWith('pre_norm.weight')) return new Float32Array([1, 2]);
        return null;
      },
      shouldStreamLargeWeight: () => false,
      needsNormWeightOffset: () => false,
      resolveWeightLayout: () => 'row',
      embeddings: null,
      gpuBuffers: new Set(),
      keepF32Weights: false,
      normOffsetDebugLogged: false,
    }),
    /self-conditioning weights are incomplete/
  );
}

// Missing final norm throws
{
  const tensorLocations = new Map([
    ['lm_head.weight', { role: 'lm_head', group: 'head', shape: [2, 2], dtype: 'F32' }],
  ]);

  await assert.rejects(
    () => loadFinalWeights({
      tensorLocations,
      tieWordEmbeddings: false,
      loadTensor: async () => null,
      shouldStreamLargeWeight: () => false,
      needsNormWeightOffset: () => false,
      resolveWeightLayout: () => 'row',
      embeddings: null,
      gpuBuffers: new Set(),
      keepF32Weights: false,
      normOffsetDebugLogged: false,
    }),
    /Final norm not found/
  );
}

// Missing LM head with tieWordEmbeddings=false throws
{
  const tensorLocations = new Map([
    ['model.language_model.norm.weight', { role: 'norm', group: 'head', shape: [2], dtype: 'F32' }],
  ]);

  await assert.rejects(
    () => loadFinalWeights({
      tensorLocations,
      tieWordEmbeddings: false,
      modelType: 'transformer',
      loadTensor: async (name) => name.includes('norm') ? new Float32Array([1, 2]) : null,
      shouldStreamLargeWeight: () => false,
      needsNormWeightOffset: () => false,
      resolveWeightLayout: () => 'row',
      embeddings: null,
      gpuBuffers: new Set(),
      keepF32Weights: false,
      normOffsetDebugLogged: false,
    }),
    /LM head not found/
  );
}

// tieWordEmbeddings=true uses embeddings as fallback
{
  const tensorLocations = new Map([
    ['model.language_model.norm.weight', { role: 'norm', group: 'head', shape: [2], dtype: 'F32' }],
  ]);
  const fakeEmbeddings = new Float32Array([10, 20, 30, 40]);

  const result = await loadFinalWeights({
    tensorLocations,
    tieWordEmbeddings: true,
    loadTensor: async (name) => name.includes('norm') ? new Float32Array([1, 2]) : null,
    shouldStreamLargeWeight: () => false,
    needsNormWeightOffset: () => false,
    resolveWeightLayout: () => 'row',
    embeddings: fakeEmbeddings,
    gpuBuffers: new Set(),
    keepF32Weights: false,
    normOffsetDebugLogged: false,
  });

  assert.equal(result.finalNorm instanceof Float32Array, true);
  assert.equal(result.lmHead, fakeEmbeddings);
}

// tieWordEmbeddings=true reuses a matching dense LM-head alias before streaming
{
  const tensorLocations = new Map([
    ['model.language_model.norm.weight', { role: 'norm', group: 'head', shape: [2], dtype: 'F32' }],
    ['model.language_model.lm_head.weight', { role: 'lm_head', group: 'head', shape: [2, 2], dtype: 'F16' }],
  ]);
  const fakeEmbeddings = createWeightBuffer(
    createFakeGpuBuffer(8),
    'f16',
    'row',
    [2, 2],
    'model.language_model.embed_tokens.weight'
  );
  const loadedNames = [];
  const streamNames = [];

  const result = await loadFinalWeights({
    tensorLocations,
    tieWordEmbeddings: true,
    loadTensor: async (name) => {
      loadedNames.push(name);
      if (name.includes('norm')) return new Float32Array([1, 2]);
      throw new Error(`dense tied LM head should not be loaded separately: ${name}`);
    },
    shouldStreamLargeWeight: (name) => {
      streamNames.push(name);
      return true;
    },
    needsNormWeightOffset: () => false,
    resolveWeightLayout: () => 'row',
    embeddings: fakeEmbeddings,
    gpuBuffers: new Set(),
    keepF32Weights: false,
    normOffsetDebugLogged: false,
  });

  assert.equal(result.lmHead, fakeEmbeddings);
  assert.deepEqual(loadedNames, ['model.language_model.norm.weight']);
  assert.deepEqual(streamNames, []);
}

// Packed Q4 tied LM heads remain explicit materialized heads
{
  const tensorLocations = new Map([
    ['model.language_model.norm.weight', { role: 'norm', group: 'head', shape: [2], dtype: 'F32' }],
    ['model.language_model.lm_head.weight', { role: 'lm_head', group: 'head', shape: [2, 2], dtype: 'Q4_K_M' }],
  ]);
  const fakeEmbeddings = createWeightBuffer(
    createFakeGpuBuffer(8),
    'f16',
    'row',
    [2, 2],
    'model.language_model.embed_tokens.weight'
  );
  const explicitLmHead = createWeightBuffer(
    createFakeGpuBuffer(144),
    'q4k',
    'row',
    [2, 2],
    'model.language_model.lm_head.weight'
  );
  const loadedNames = [];

  const result = await loadFinalWeights({
    tensorLocations,
    tieWordEmbeddings: true,
    loadTensor: async (name) => {
      loadedNames.push(name);
      if (name.includes('norm')) return new Float32Array([1, 2]);
      if (name.includes('lm_head')) return explicitLmHead;
      return null;
    },
    shouldStreamLargeWeight: () => false,
    needsNormWeightOffset: () => false,
    resolveWeightLayout: () => 'row',
    embeddings: fakeEmbeddings,
    gpuBuffers: new Set(),
    keepF32Weights: false,
    normOffsetDebugLogged: false,
  });

  assert.equal(result.lmHead, explicitLmHead);
  assert.deepEqual(
    loadedNames,
    ['model.language_model.norm.weight', 'model.language_model.lm_head.weight']
  );
}

// Embedding models may omit LM head entirely
{
  const tensorLocations = new Map([
    ['model.language_model.norm.weight', { role: 'norm', group: 'head', shape: [2], dtype: 'F32' }],
  ]);

  const result = await loadFinalWeights({
    tensorLocations,
    tieWordEmbeddings: false,
    modelType: 'embedding',
    loadTensor: async (name) => name.includes('norm') ? new Float32Array([1, 2]) : null,
    shouldStreamLargeWeight: () => false,
    needsNormWeightOffset: () => false,
    resolveWeightLayout: () => 'row',
    embeddings: null,
    gpuBuffers: new Set(),
    keepF32Weights: false,
    normOffsetDebugLogged: false,
  });

  assert.equal(result.finalNorm instanceof Float32Array, true);
  assert.equal(result.lmHead, null);
}

// normOffsetDebugLogged set when needsNormWeightOffset returns true
{
  const tensorLocations = new Map([
    ['model.language_model.norm.weight', { role: 'norm', group: 'head', shape: [2], dtype: 'F32' }],
    ['lm_head.weight', { role: 'lm_head', group: 'head', shape: [2, 2], dtype: 'F32' }],
  ]);

  const lookup = new Map([
    ['model.language_model.norm.weight', new Float32Array([1, 2])],
    ['lm_head.weight', new Float32Array([1, 2, 3, 4])],
  ]);

  const result = await loadFinalWeights({
    tensorLocations,
    tieWordEmbeddings: false,
    loadTensor: async (name) => lookup.get(name) || null,
    shouldStreamLargeWeight: () => false,
    needsNormWeightOffset: () => true,
    resolveWeightLayout: () => 'row',
    embeddings: null,
    gpuBuffers: new Set(),
    keepF32Weights: false,
    normOffsetDebugLogged: false,
  });

  assert.equal(result.normOffsetDebugLogged, true);
}

{
  const tensorLocations = new Map([
    ['model.language_model.norm.weight', { role: 'norm', group: 'head', shape: [2], dtype: 'F32' }],
    ['embedding_postprocessor.projections.0.weight', { role: 'lm_head', group: 'embedding_postprocessor', shape: [2, 2], dtype: 'F32' }],
  ]);

  const lookup = new Map([
    ['model.language_model.norm.weight', new Float32Array([1, 2])],
    ['embedding_postprocessor.projections.0.weight', new Float32Array([1, 0, 0, 1])],
  ]);

  const result = await loadFinalWeights({
    tensorLocations,
    tieWordEmbeddings: true,
    modelType: 'embedding',
    embeddingPostprocessor: {
      poolingMode: 'mean',
      includePrompt: true,
      projections: [{
        weightTensor: 'embedding_postprocessor.projections.0.weight',
        biasTensor: null,
        inputSize: 2,
        outputSize: 2,
        activation: 'identity',
      }],
      normalize: 'l2',
    },
    loadTensor: async (name) => lookup.get(name) || null,
    shouldStreamLargeWeight: () => false,
    needsNormWeightOffset: () => false,
    resolveWeightLayout: () => 'row',
    embeddings: new Float32Array([10, 20, 30, 40]),
    gpuBuffers: new Set(),
    keepF32Weights: false,
    normOffsetDebugLogged: false,
  });

  assert.equal(result.embeddingPostprocessor?.poolingMode, 'mean');
  assert.equal(result.embeddingPostprocessor?.projections.length, 1);
  assert.deepEqual(
    Array.from(result.embeddingPostprocessor?.projections[0].weight ?? []),
    [1, 0, 0, 1]
  );
}

console.log('final-norm-selection.test: ok');
