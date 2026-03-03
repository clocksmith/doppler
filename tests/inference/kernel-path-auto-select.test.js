import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { resolveCapabilityKernelPathRef } = await import('../../src/inference/pipelines/text/kernel-path-auto-select.js');
const { selectRuleValue } = await import('../../src/rules/rule-registry.js');
const { setActiveKernelPath, getKernelPathStrict } = await import('../../src/config/kernel-path-loader.js');

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'model',
    { hasSubgroups: true }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'model',
    { hasSubgroups: true },
    {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest'],
    }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a-online');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'model',
    { hasSubgroups: false },
    {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest'],
    }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'runtime',
    { hasSubgroups: true },
    {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest', 'config'],
    }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a-online');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'config',
    { hasSubgroups: true },
    {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest', 'config'],
    }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a-online');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'execution-v0',
    { hasSubgroups: true },
    {
      mode: 'capability-aware',
      sourceScope: ['execution-v0'],
    }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a-online');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f32a',
    'execution-v0',
    { hasSubgroups: true },
    {
      mode: 'capability-aware',
      sourceScope: ['config'],
    }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma2-q4k-fused-f32a',
    'config',
    { hasSubgroups: false },
    {
      mode: 'capability-aware',
      sourceScope: ['config'],
    }
  );
  assert.equal(selected, 'gemma2-q4k-dequant-f32a');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f16a-online',
    'config',
    { hasSubgroups: false },
    {
      mode: 'capability-aware',
      sourceScope: ['config'],
    }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'lfm2-q4k-dequant-f32a-online',
    'config',
    { hasSubgroups: false },
    {
      mode: 'capability-aware',
      sourceScope: ['config'],
    }
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
  assert.equal(selected, 'gemma3-q4k-dequant-f32a');
}

{
  const selected = selectRuleValue(
    'inference',
    'kernelPath',
    'finitenessFallback',
    { kernelPathId: 'lfm2-q4k-dequant-f32a-online' }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f32a');
}

{
  const selected = selectRuleValue(
    'inference',
    'kernelPath',
    'finitenessFallback',
    { kernelPathId: 'gemma2-q4k-dequant-f16a' }
  );
  assert.equal(selected, 'gemma2-q4k-dequant-f32a');
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

{
  setActiveKernelPath(null, 'none', { mode: 'locked', sourceScope: ['model'], onIncompatible: 'error' });
  assert.equal(getKernelPathStrict(), true);
  setActiveKernelPath(null, 'none', { mode: 'capability-aware', sourceScope: ['config'], onIncompatible: 'remap' });
  assert.equal(getKernelPathStrict(), true);
}

console.log('kernel-path-auto-select.test: ok');
