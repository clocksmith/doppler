import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { resolveAttentionPlanForTest } from '../../src/gpu/kernels/attention.js';

const kernelPathPath = path.join(
  process.cwd(),
  'src/config/presets/kernel-paths/gemma3-f16-fused-f32a-online-streamingprefill.json'
);
const kernelPath = JSON.parse(fs.readFileSync(kernelPathPath, 'utf8'));

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
