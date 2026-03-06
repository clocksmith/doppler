import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { getRuleSet, selectRuleValue } from '../../src/gpu/kernels/rule-registry.js';
import { runReLU, recordReLU, runRepeatChannels, recordRepeatChannels } from '../../src/gpu/kernels/index.js';

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(TEST_DIR, '..', '..');
const KERNEL_REGISTRY_PATH = path.join(REPO_ROOT, 'src', 'config', 'kernels', 'registry.json');

const kernelRegistry = JSON.parse(await fs.readFile(KERNEL_REGISTRY_PATH, 'utf8'));

assert.equal(typeof runReLU, 'function');
assert.equal(typeof recordReLU, 'function');
assert.equal(typeof runRepeatChannels, 'function');
assert.equal(typeof recordRepeatChannels, 'function');

assert.deepEqual(getRuleSet('relu', 'variant'), [
  { match: { dtype: 'f16' }, value: 'default_f16' },
  { match: {}, value: 'default' },
]);
assert.equal(selectRuleValue('relu', 'variant', { dtype: 'f32' }), 'default');
assert.equal(selectRuleValue('relu', 'variant', { dtype: 'f16' }), 'default_f16');

assert.deepEqual(getRuleSet('repeatChannels', 'variant'), [
  { match: { dtype: 'f16' }, value: 'default_f16' },
  { match: {}, value: 'default' },
]);
assert.equal(selectRuleValue('repeatChannels', 'variant', { dtype: 'f32' }), 'default');
assert.equal(selectRuleValue('repeatChannels', 'variant', { dtype: 'f16' }), 'default_f16');

assert.deepEqual(kernelRegistry.operations.relu?.variants?.default, {
  wgsl: 'relu.wgsl',
  entryPoint: 'main',
  workgroup: [256, 1, 1],
  requires: [],
  outputDtype: 'f32',
});
assert.deepEqual(kernelRegistry.operations.relu?.variants?.default_f16, {
  wgsl: 'relu_f16.wgsl',
  entryPoint: 'main',
  workgroup: [256, 1, 1],
  requires: ['shader-f16'],
  outputDtype: 'f16',
});

assert.deepEqual(kernelRegistry.operations.repeat_channels?.variants?.default, {
  wgsl: 'repeat_channels.wgsl',
  entryPoint: 'main',
  workgroup: [256, 1, 1],
  requires: [],
  outputDtype: 'f32',
});
assert.deepEqual(kernelRegistry.operations.repeat_channels?.variants?.default_f16, {
  wgsl: 'repeat_channels_f16.wgsl',
  entryPoint: 'main',
  workgroup: [256, 1, 1],
  requires: ['shader-f16'],
  outputDtype: 'f16',
});

console.log('diffusion-kernel-wiring.test: ok');
