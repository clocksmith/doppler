import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { resolveCapabilityKernelPathRef } = await import('../../src/inference/pipelines/text/kernel-path-auto-select.js');
const { selectRuleValue } = await import('../../src/rules/rule-registry.js');

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'model',
    { hasSubgroups: true }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a-online');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'model',
    { hasSubgroups: false }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'runtime',
    { hasSubgroups: true }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'config',
    { hasSubgroups: true }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-f16-fused-f32a-online',
    'model',
    { hasSubgroups: true }
  );
  assert.equal(selected, 'gemma3-f16-fused-f32a-online');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-f16-fused-f32a-online',
    'runtime',
    { hasSubgroups: true }
  );
  assert.equal(selected, 'gemma3-f16-fused-f32a-online');
}

{
  const selected = selectRuleValue(
    'inference',
    'kernelPath',
    'finitenessFallback',
    { kernelPathId: 'gemma3-f16-fused-f16a-online' }
  );
  assert.equal(selected, 'gemma3-f16-fused-f32a-online');
}

{
  const selected = selectRuleValue(
    'inference',
    'kernelPath',
    'finitenessFallback',
    { kernelPathId: 'gemma3-q4k-dequant-f16a-online' }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a-online');
}

{
  const selected = selectRuleValue(
    'inference',
    'kernelPath',
    'finitenessFallback',
    { kernelPathId: 'gemma2-q4k-dequant-f16a' }
  );
  assert.equal(selected, 'gemma2-q4k-fused-f32a');
}

{
  const selected = selectRuleValue(
    'inference',
    'kernelPath',
    'finitenessFallback',
    { kernelPathId: 'unknown-path' }
  );
  assert.equal(selected, null);
}

console.log('kernel-path-auto-select.test: ok');
