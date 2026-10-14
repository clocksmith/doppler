// Attention observation must decode paired activations by their declared representation.
// Regression: the full-attention probes passed only dtype, so a two-F32 Q/K tensor was read as
// scalar F32 words (hi, lo, hi, lo, ...) and reported half of its logical elements.
import assert from 'node:assert/strict';

const { observeAttentionRoPE } = await import('../../src/inference/pipelines/text/attention/rope-observation.js');
const { OperatorEventEmitter } = await import('../../src/inference/pipelines/text/operator-events.js');
const { createDefaultCaptureConfig } = await import('../../src/debug/capture-policy.js');

const numTokens = 2, numHeads = 2, numKVHeads = 1, headDim = 4;
function pairedTensor(count, seed) {
  const components = new Float32Array(count * 2);
  const logical = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const high = Math.fround(seed + i * 0.25);
    const low = Math.fround((i + 1) * 2 ** -30);
    components[2 * i] = high;
    components[2 * i + 1] = low;
    logical[i] = high + low;
  }
  return { tensor: { buffer: components, dtype: 'f32', precision: 'two-f32', components: 2, elementStrideBytes: 8 }, logical };
}

const query = pairedTensor(numTokens * numHeads * headDim, 1);
const key = pairedTensor(numTokens * numKVHeads * headDim, -3);
const emitter = new OperatorEventEmitter({});
const state = {
  debugProbes: [],
  operatorDiagnostics: {
    enabled: true, emitter,
    captureConfig: { ...createDefaultCaptureConfig(), enabled: true, defaultLevel: 'none',
      targetOpIds: ['layer.3.attn.q_rope', 'layer.3.attn.k_rope'], targetLevel: 'full' },
  },
};

await observeAttentionRoPE({ state, recorder: null, ropeApplied: true, disableRoPE: false,
  qTensor: query.tensor, kTensor: key.tensor, layerIdx: 3, numTokens, numHeads, numKVHeads, headDim });

const captures = Object.fromEntries(emitter.getTimeline().map(row => [row.opId, row.capture]));
for (const [opId, expected, width] of [['layer.3.attn.q_rope', query.logical, numHeads * headDim],
  ['layer.3.attn.k_rope', key.logical, numKVHeads * headDim]]) {
  const capture = captures[opId];
  assert.ok(capture, `${opId} was captured`);
  assert.deepEqual(capture.shape, [numTokens, width]);
  assert.equal(capture.precision, 'two-f32');
  assert.deepEqual(capture.data, Array.from(expected), `${opId} decodes hi + lo per logical element`);
}

// Scalar tensors keep their scalar observation.
const scalarEmitter = new OperatorEventEmitter({});
const scalarState = { debugProbes: [], operatorDiagnostics: { ...state.operatorDiagnostics, emitter: scalarEmitter } };
const scalarQuery = Float32Array.from({ length: numTokens * numHeads * headDim }, (_, i) => i);
await observeAttentionRoPE({ state: scalarState, recorder: null, ropeApplied: true, disableRoPE: false,
  qTensor: { buffer: scalarQuery, dtype: 'f32' }, kTensor: null, layerIdx: 3, numTokens, numHeads, numKVHeads, headDim });
const [scalarCapture] = scalarEmitter.getTimeline().map(row => row.capture);
assert.equal(scalarCapture.precision, undefined);
assert.deepEqual(scalarCapture.data, Array.from(scalarQuery));

console.log('attention-probe-representation.test: ok');
