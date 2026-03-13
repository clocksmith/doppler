import assert from 'node:assert/strict';

const { loadFinalWeights } = await import('../../src/loader/final-weights-loader.js');

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

console.log('final-norm-selection.test: ok');
