import assert from 'node:assert/strict';
import {
  assertBatchRetryInPlaceAllowed,
  assertShortBatchContinuationAllowed,
  hasUnrewindableDecodeState,
} from "../../src/inference/pipelines/text/generator/decode/batch-retry-policy.js";

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
// A short non-stop batch may continue on attention-only models (KV is rewritten by
// position) but never on recurrent models, whose state ran ahead of the delivered tokens.
assert.doesNotThrow(() => assertShortBatchContinuationAllowed(attentionOnly, 3, 8));
assert.doesNotThrow(() => assertShortBatchContinuationAllowed(linear, 8, 8));
for (const state of [linear, conv]) {
  assert.throws(() => assertShortBatchContinuationAllowed(state, 3, 8), /cannot be rewound/);
}
// The guard above is unreachable today because linear-attention models never get the
// hot-vocabulary batch (the only producer of short non-stop batches). Pin that rule.
const { isBatchDecodeEnabled } = await import('../../src/inference/pipelines/text/execution/execution-plan.js');
const batchFacts = { batchSize: 4, useGPU: true, gpuSamplingAvailable: true, disableMultiTokenDecode: false,
  disableCommandBatching: false, isBdpaPagedLayout: false, finitenessFallbackWindowOpen: false };
assert.equal(isBatchDecodeEnabled({ ...batchFacts, hasLinearAttentionLayers: true }), true);
assert.equal(isBatchDecodeEnabled({ ...batchFacts, hasLinearAttentionLayers: true, hasRangeBackedPerLayerInputs: true }), false);
console.log('batch-retry-policy.test: ok');
