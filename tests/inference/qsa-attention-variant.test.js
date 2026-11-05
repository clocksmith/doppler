import assert from 'node:assert/strict';
import { selectRuleValue } from '../../src/rules/rule-registry.js';
import { resolveAttentionPlanForDispatch } from '../../src/inference/pipelines/text/attention/plan.js';
import { runQsaIndexedAttention } from '../../src/inference/pipelines/text/indexed-attention/attention-runner.js';

// Contract: QSA-indexed layers select the qsaIndexed attention core only for a
// contiguous KV cache, ordinary layers keep their variants, and the runner
// fails closed before dispatch when its selection or cache contract is absent.
const variant = (input) => selectRuleValue('inference', 'attention', 'attentionKernelVariant',
  { coldQuantMode: 'none', ...input });
assert.equal(variant({ kvLayout: 'contiguous', numTokens: 7, indexedSelection: true }), 'qsaIndexed');
assert.equal(variant({ kvLayout: 'contiguous', numTokens: 1, indexedSelection: true }), 'qsaIndexed');
assert.equal(variant({ kvLayout: 'tiered', numTokens: 1, indexedSelection: true }), 'qsaIndexedRequiresContiguousKV');
assert.equal(variant({ kvLayout: 'contiguous', numTokens: 7, indexedSelection: false }), 'contiguous');
assert.equal(variant({ kvLayout: 'bdpa', numTokens: 1 }), 'bdpa');

const plan = (indexedSelection, layout = 'contiguous') => resolveAttentionPlanForDispatch({
  config: { isPrefill: true, layerIdx: 3, numTokens: 4, numHeads: 24, numKVHeads: 2, headDim: 256, hiddenSize: 2560,
    currentSeqLen: 0, queryScale: 1, attentionOutputGate: true, indexedSelection },
  kvState: { kvLayout: layout, kvLenForAttention: 4, coldQuantMode: 'none' },
  inputDtype: 'f32', activationDtype: 'f32', projectionDtype: 'f32', kvDtype: 'f16', outputProjectionInputDtype: 'f32',
  outputDtype: 'f32', session: {}, capabilities: { hasF16: true, hasSubgroups: true }, fusion: {},
  dispatchParams: { cachedKDtype: 'f16', cachedVDtype: 'f16' },
});
assert.equal(plan(true).attention.implementation, 'qsaIndexed');
assert.equal(plan(false).attention.implementation, 'contiguous');
assert.equal(plan(true, 'tiered').attention.implementation, 'qsaIndexedRequiresContiguousKV');

const base = {
  state: { qsa: { layerIdx: 3, numTokens: 4, visibleRows: 10, selection: {}, meta: {}, maxSelected: 2051 } },
  qTensor: { dtype: 'f32' }, cachedKTensor: { buffer: {} }, cachedVTensor: { buffer: {} }, cachedKDtype: 'f16', cachedVDtype: 'f16',
  kvState: { kvLayout: 'contiguous', kvStart: 0, kvLenForAttention: 10 }, numTokens: 4, numHeads: 24, numKVHeads: 2, headDim: 256,
  attnScale: 1 / 16, attnSoftcap: 0, effectiveSlidingWindow: null, causalForAttention: true, layerIdx: 3,
};
for (const [patch, pattern] of [
  [{ state: { qsa: null } }, /no indexer selection/],
  [{ layerIdx: 7 }, /no indexer selection/],
  [{ numTokens: 5 }, /covers 4 queries/],
  [{ cachedKDtype: 'f32' }, /f16 KV cache/],
  [{ kvState: { kvLayout: 'paged', kvStart: 0, kvLenForAttention: 10 } }, /contiguous KV layout/],
  [{ kvState: { kvLayout: 'contiguous', kvStart: 0, kvLenForAttention: 9 } }, /holds 9 rows/],
  [{ qTensor: { dtype: 'f16' } }, /f32 queries/],
  [{ attnSoftcap: 30 }, /softcapping/],
  [{ effectiveSlidingWindow: 128 }, /sliding-window/],
  [{ causalForAttention: false }, /causal/],
]) {
  await assert.rejects(runQsaIndexedAttention(null, { ...base, ...patch }), pattern, JSON.stringify(patch));
}
console.log('qsa-attention-variant.test: ok');
