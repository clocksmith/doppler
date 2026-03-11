import assert from 'node:assert/strict';

import { resolveAttentionPlanForTest } from '../../src/gpu/kernels/attention.js';
import { resolveKernelPath } from '../../src/config/kernel-path-loader.js';

const kernelPath = resolveKernelPath('gemma3-f16-fused-f32a-online-streamingprefill');

const plan = resolveAttentionPlanForTest(
  15,
  15,
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

assert.equal(plan.tier, 'streaming');
assert.equal(plan.variant, 'prefill_streaming_f16kv');

console.log('gemma3-1b-streaming-prefill-path.test: ok');
