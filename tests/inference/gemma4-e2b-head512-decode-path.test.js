import assert from 'node:assert/strict';

import { resolveAttentionPlanForTest } from '../../src/gpu/kernels/attention.js';

const kernelPath = {
  id: 'gemma4-e2b-head512-decode',
  decode: {
    steps: [
      { op: 'attention', kernel: 'attention_decode_online_f16kv.wgsl', entry: 'main' },
    ],
  },
  prefill: {
    steps: [],
  },
};

const plan = resolveAttentionPlanForTest(
  1,
  64,
  512,
  8,
  'f16',
  'f32',
  32768,
  { hasSubgroups: true, hasF16: true },
  0,
  false,
  kernelPath
);

assert.equal(plan.tier, 'subgroup');
assert.equal(plan.variant, 'decode_online_f16kv');

console.log('gemma4-e2b-head512-decode-path.test: ok');
