import assert from 'node:assert/strict';

import {
  resolveAttentionPrecisionContract,
  isAttentionKvDtypeExplicit,
} from '../../src/inference/pipelines/text/attention/precision-contract.js';

const kernelPath = {
  id: 'test-path',
  name: 'Test Path',
  activationDtype: 'f32',
  kvDtype: 'f16',
  decode: {
    steps: [
      {
        op: 'attention',
        kernel: 'attention_decode_online_f16kv.wgsl',
        entry: 'main',
        precision: { activationDtype: 'f32', kvDtype: 'f16', outputDtype: 'f32' },
      },
    ],
  },
  prefill: {
    steps: [
      {
        op: 'attention',
        kernel: 'attention_streaming_f16kv.wgsl',
        entry: 'main',
        precision: { kvDtype: 'f16' },
      },
    ],
  },
};

const contract = resolveAttentionPrecisionContract(
  { layerIdx: 0, isPrefill: false, kernelPath },
  { kvCache: { kvDtype: 'f16' } }
);

assert.equal(contract.explicitInputDtype, 'f32');
assert.equal(contract.explicitOutputDtype, 'f32');
assert.equal(contract.explicitKvDtype, 'f16');
assert.equal(contract.resolvedActivationDtype, 'f32');
assert.equal(contract.resolvedOutputDtype, 'f32');
assert.equal(contract.resolvedKvCacheDtype, 'f16');
assert.equal(isAttentionKvDtypeExplicit(contract, 'f16'), true);
assert.equal(isAttentionKvDtypeExplicit(contract, 'f32'), false);

const explicitConfigContract = resolveAttentionPrecisionContract(
  { layerIdx: 0, isPrefill: true, kernelPath: null, kvDtype: 'f16' },
  { kvCache: { kvDtype: 'f16' } }
);
assert.equal(explicitConfigContract.explicitKvDtype, 'f16');
assert.equal(explicitConfigContract.resolvedActivationDtype, null);

assert.throws(
  () => resolveAttentionPrecisionContract(
    { layerIdx: 0, isPrefill: false, kernelPath },
    { kvCache: { kvDtype: 'f32' } }
  ),
  /attention precision declares kvDtype="f16"/
);

console.log('attention-kv-precision-contract.test: ok');
