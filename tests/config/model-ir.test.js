import assert from 'node:assert/strict';
import { createModelIR, hashModelIR, validateModelIR } from '../../src/config/model-ir.js';

const ir = createModelIR({
  modelId: 'test-qwen-model',
  architecture: 'qwen3',
  vocabSize: 151936,
  hiddenSize: 2048,
  numLayers: 24,
  attentionGeometry: {
    numHeads: 16,
    numKvHeads: 4,
    headDim: 128,
  },
  normalization: {
    type: 'rmsnorm',
    eps: 1e-6,
  },
  ffn: {
    type: 'swiglu',
    intermediateSize: 5504,
  },
  outputTopology: {
    headType: 'causal-lm',
    tieWeights: false,
  },
  phases: ['prefill', 'decode'],
});

const validation = validateModelIR(ir);
assert.equal(validation.ok, true);
assert.equal(validation.errors.length, 0);

const hash = hashModelIR(ir);
assert.match(hash, /^sha256:[0-9a-f]{64}$/);

// Verify that mutating a property alters the hash deterministically
const ir2 = { ...ir, hiddenSize: 4096 };
const hash2 = hashModelIR(ir2);
assert.notEqual(hash, hash2);

console.log('✔ model-ir.test.js passed');
