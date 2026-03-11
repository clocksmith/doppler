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
}

console.log('final-norm-selection.test: ok');
