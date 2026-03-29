import assert from 'node:assert/strict';

import { resolveAttentionPlanForTest } from '../../src/gpu/kernels/attention.js';

const kernelPath = {
  id: 'test-head256-prefill',
  name: 'test-head256-prefill',
  activationDtype: 'f32',
  decode: {
    steps: [
      { op: 'attention', kernel: 'attention_decode_online_f16kv.wgsl', entry: 'main' },
    ],
  },
  prefill: {
    steps: [
      { op: 'attention', kernel: 'attention_head256_f16kv.wgsl', entry: 'main' },
    ],
  },
};

const plan = resolveAttentionPlanForTest(
  183,
  183,
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

assert.equal(plan.tier, 'tiled_small');
assert.equal(plan.variant, 'prefill_head256_f16kv');

console.log('gemma3-1b-head256-prefill-path.test: ok');
