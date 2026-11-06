import assert from 'node:assert/strict';
import { selectRuleValue } from '../../src/rules/rule-registry.js';
import { resolveAttentionPlanForDispatch } from '../../src/inference/pipelines/text/attention/plan.js';
import { runQsaIndexedAttention } from '../../src/inference/pipelines/text/indexed-attention/attention-runner.js';
import { resolveRecordedAttentionDispatch } from '../../src/inference/pipelines/text/attention/interpreter/immediate.js';
import { setDevice } from '../../src/gpu/device.js';

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

// Exercise the live handoff, not only the rule/plan in isolation: the prepared
// resource in AttentionState must survive construction of both dispatch inputs.
setDevice({ features: new Set(['shader-f16']), limits: {}, createBindGroup() {}, queue: { submit() {} } });
for (const numTokens of [1, 2304]) {
  const cached = { __dopplerFakeGPUBuffer: true, size: 2305 * 2 * 256 * 2 };
  const input = { buffer: cached, dtype: 'f32', shape: [numTokens, 2560] };
  const kvState = { kvLayout: 'contiguous', kvLenForAttention: 2305, startPosForMask: 2305 - numTokens,
    coldQuantMode: 'none', cachedK: cached, cachedV: cached, hasCache: true };
  const options = {
    attentionActivationDtype: 'f32', attentionPrecisionContract: { resolvedKvCacheDtype: 'f16' },
    config: { numHeads: 24, numKVHeads: 2, headDim: 256, hiddenSize: 2560, numTokens, layerIdx: 3,
      currentSeqLen: 2305 - numTokens, queryScale: 1, causalAttention: true, attentionOutputGate: false },
    input, normed: input, isPrefill: numTokens > 1, numTokens, numHeads: 24, numKVHeads: 2,
    headDim: 256, hiddenSize: 2560, layerIdx: 3, kvCacheDtype: 'f16', kvState,
    kTensor: { buffer: cached, dtype: 'f16', shape: [numTokens, 512] },
    vTensor: { buffer: cached, dtype: 'f16', shape: [numTokens, 512] },
    qTensor: { buffer: cached, dtype: 'f32', shape: [numTokens, 6144] },
    matmulOutputDtype: 'f32', oProjInputDtype: 'f32', oProjOutputDtype: 'f32', layerWeights: {},
    runtimeSession: {}, state: { kvCache: { kvDtype: 'f16' },
      qsa: { layerIdx: 3, numTokens, visibleRows: 2305, selection: cached, meta: cached, maxSelected: 2051 },
      observationContext: { receiptPolicy: 'off' }, stats: {} },
  };
  const dispatched = resolveRecordedAttentionDispatch(options);
  assert.equal(dispatched.attentionKernelVariant, 'qsaIndexed', 'live dispatch must consume its prepared QSA selection');
  assert.equal(dispatched.attentionPlan.attention.implementation, 'qsaIndexed');
  const ordinary = resolveRecordedAttentionDispatch({ ...options, state: { ...options.state, qsa: null } });
  assert.equal(ordinary.attentionKernelVariant, 'contiguous', 'ordinary attention retains its existing route');
}
setDevice(null);

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
