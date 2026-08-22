import assert from 'node:assert/strict';
import { createModelIR, hashModelIR, validateModelIR } from '../../src/config/model-ir.js';

const params = {
  modelId: 'test-model', architecture: 'transformer', vocabSize: 16, hiddenSize: 8, numLayers: 1,
  sourceIdentity: { manifestArtifactId: 'manifest', manifestHash: `sha256:${'1'.repeat(64)}` },
  tensorRoles: { weight: { role: 'matmul', shape: [8, 8], semanticDtype: 'f32' } },
  layers: [{ index: 0, type: 'global-attention' }],
  attentionGeometry: { numHeads: 1, numKvHeads: 1, headDim: 8 },
  normalization: { type: 'rmsnorm', eps: 1e-6 },
  rope: { dimension: 8, baseFreq: 10000 },
  ffn: { type: 'gelu', intermediateSize: 16 },
  outputTopology: { headType: 'causal-lm', tieWeights: false },
  phases: ['prefill', 'decode'],
};

const ir = createModelIR(params);
assert.deepEqual(validateModelIR(ir), { ok: true, errors: [] });
assert.match(hashModelIR(ir), /^sha256:[0-9a-f]{64}$/);
assert.notEqual(hashModelIR(ir), hashModelIR({ ...ir, hiddenSize: 16 }));
assert.throws(() => createModelIR({ ...params, attentionGeometry: undefined }), /attentionGeometry/);
assert.throws(() => createModelIR({ ...params, tensorRoles: {} }), /tensorRoles/);

console.log('✔ model-ir.test.js passed');
