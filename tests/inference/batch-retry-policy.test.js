import assert from 'node:assert/strict';
import {
  assertBatchRetryInPlaceAllowed,
  hasUnrewindableDecodeState,
} from '../../src/inference/pipelines/text/generator/batch-retry-policy.js';

const attentionOnly = { linearAttentionRuntime: { layers: new Map() }, convLayerStates: new Map() };
const linear = { linearAttentionRuntime: { layers: new Map([[0, {}]]) }, convLayerStates: new Map() };
const conv = { linearAttentionRuntime: { layers: new Map() }, convLayerStates: new Map([[0, {}]]) };
const submitted = Object.assign(new Error('Timestamp query capacity did not cover every compute pass.'), { decodeStateAdvanced: true });
const unsubmitted = new Error('Batch size exceeds decode ring capacity.');

assert.equal(hasUnrewindableDecodeState(attentionOnly), false);
assert.equal(hasUnrewindableDecodeState(linear), true);
assert.equal(hasUnrewindableDecodeState(conv), true);
assert.equal(hasUnrewindableDecodeState(null), false);

// Failures before any submission, or on attention-only models, keep the existing retry.
assert.doesNotThrow(() => assertBatchRetryInPlaceAllowed(linear, unsubmitted));
assert.doesNotThrow(() => assertBatchRetryInPlaceAllowed(attentionOnly, submitted));
for (const state of [linear, conv]) {
  assert.throws(() => assertBatchRetryInPlaceAllowed(state, submitted), (error) => {
    assert.match(error.message, /recurrent decode state already advanced/);
    assert.equal(error.cause, submitted);
    return true;
  });
}
console.log('batch-retry-policy.test: ok');
