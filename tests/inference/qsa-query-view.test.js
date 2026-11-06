import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const target = new URL('../../src/inference/pipelines/text/indexed-attention/attention-runner.js', import.meta.url).href;
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier === '../../../../gpu/kernels/qsa-indexer.js') {
    return { url: `data:text/javascript,${encodeURIComponent('export const recordIndexedAttention = async (_recorder, query) => query;')}`, shortCircuit: true };
  }
  return next(specifier, context);
} });
try {
  const { runQsaIndexedAttention } = await import(target);
  const recorder = { beginComputePass() {} };
  for (const numTokens of [1, 15, 2304]) {
    const buffer = { __dopplerFakeGPUBuffer: true, size: numTokens * 24 * 256 * 4 };
    const options = { state: { qsa: { layerIdx: 3, numTokens, visibleRows: 2305, selection: {}, meta: {}, maxSelected: 2051 } },
      cachedKTensor: { buffer: {} }, cachedVTensor: { buffer: {} }, cachedKDtype: 'f16', cachedVDtype: 'f16',
      kvState: { kvLayout: 'contiguous', kvStart: 0, kvLenForAttention: 2305 }, numTokens, numHeads: 24, numKVHeads: 2,
      headDim: 256, attnScale: 1 / 16, attnSoftcap: 0, effectiveSlidingWindow: null, causalForAttention: true, layerIdx: 3 };
    for (const shape of [[numTokens * 24, 256], [numTokens, 6144]]) {
      const result = await runQsaIndexedAttention(recorder, { ...options, qTensor: { buffer, dtype: 'f32', shape } });
      assert.equal(result.buffer, buffer, 'query adaptation must retain the exact upstream buffer');
      assert.deepEqual(result.shape, [numTokens, 6144]);
    }
    await assert.rejects(runQsaIndexedAttention(recorder, { ...options,
      qTensor: { buffer, dtype: 'f32', shape: [6144, numTokens] } }), /token-major query\/head geometry/);
  }
} finally { hooks.deregister(); }
console.log('qsa query view: canonical and normalized-head views preserve bytes; unrelated geometry rejects');
