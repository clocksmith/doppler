import assert from 'node:assert/strict';

import { classifyHostedCapabilitySkip } from '../../tools/ci-browser-opfs-registry-smoke.js';

const shaderF16Skip = classifyHostedCapabilitySkip(
  new Error('KernelPath "gemma3-q4k-dequant-f16a-online" requires unsupported GPU features (shader-f16).')
);
assert.deepEqual(shaderF16Skip, {
  code: 'HOSTED_BROWSER_CAPABILITY_SKIP',
  reason: 'KernelPath "gemma3-q4k-dequant-f16a-online" requires unsupported GPU features (shader-f16).',
});

const adapterSkip = classifyHostedCapabilitySkip('No suitable GPU adapter found for WebGPU.');
assert.deepEqual(adapterSkip, {
  code: 'HOSTED_BROWSER_CAPABILITY_SKIP',
  reason: 'No suitable GPU adapter found for WebGPU.',
});

assert.equal(classifyHostedCapabilitySkip(new Error('HTTP 404')), null);

console.log('browser opfs smoke capability skip contract ok');
