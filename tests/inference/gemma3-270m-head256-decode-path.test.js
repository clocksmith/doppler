import assert from 'node:assert/strict';

import { resolveAttentionPlanForTest } from '../../src/gpu/kernels/attention.js';

const kernelPath = {
  id: 'gemma3-270m-head256-decode',
  name: 'gemma3-270m-head256-decode',
  activationDtype: 'f32',
  decode: {
    steps: [
      { op: 'attention', kernel: 'attention_decode_online_head256_f16kv.wgsl', entry: 'main' },
    ],
  },
  prefill: {
    steps: [
      { op: 'attention', kernel: 'attention_head256_f16kv.wgsl', entry: 'main' },
    ],
  },
};

const plan = resolveAttentionPlanForTest(
  1,
  512,
  256,
  4,
  'f16',
  'f32',
  32768,
  { hasSubgroups: true, hasF16: true },
  0,
  false,
  kernelPath
);

assert.equal(plan.tier, 'subgroup');
assert.equal(plan.variant, 'decode_online_head256_f16kv');

assert.throws(
  () => resolveAttentionPlanForTest(
    1,
    512,
    512,
    8,
    'f16',
    'f32',
    32768,
    { hasSubgroups: true, hasF16: true },
    0,
    false,
    kernelPath
  ),
  /requires headDim == 256/
);

console.log('gemma3-270m-head256-decode-path.test: ok');
