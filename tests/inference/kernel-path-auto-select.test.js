import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { resolveCapabilityKernelPathRef } = await import('../../src/inference/pipelines/text/kernel-path-auto-select.js');

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f16a',
    'model',
    { hasSubgroups: true }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f16a-online');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f16a',
    'model',
    { hasSubgroups: false }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f16a');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f16a',
    'runtime',
    { hasSubgroups: true }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f16a');
}

{
  const selected = resolveCapabilityKernelPathRef(
    'gemma3-q4k-dequant-f16a',
    'config',
    { hasSubgroups: true }
  );
  assert.equal(selected, 'gemma3-q4k-dequant-f16a');
}

console.log('kernel-path-auto-select.test: ok');
