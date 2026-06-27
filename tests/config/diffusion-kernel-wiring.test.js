import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { getRuleSet, selectRuleValue } from '../../src/gpu/kernels/rule-registry.js';
import { setDevice } from '../../src/gpu/device.js';
import { runReLU, recordReLU, runRepeatChannels, recordRepeatChannels } from '../../src/gpu/kernels/index.js';
import { registerRuleGroup } from '../../src/rules/rule-registry.js';

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(TEST_DIR, '..', '..');
const KERNEL_REGISTRY_PATH = path.join(REPO_ROOT, 'src', 'config', 'kernels', 'registry.json');

const kernelRegistry = JSON.parse(await fs.readFile(KERNEL_REGISTRY_PATH, 'utf8'));

function createFakeDevice(features = []) {
  return {
    lost: new Promise(() => {}),
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    features: new Set(features),
    limits: {
      maxStorageBufferBindingSize: 1 << 20,
      maxBufferSize: 1 << 20,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 8,
      maxUniformBufferBindingSize: 65536,
    },
    createBuffer({ size, usage }) {
      return {
        size,
        usage,
        destroy() {},
      };
    },
    createBindGroup() {
      return {};
    },
    createShaderModule() {
      return {};
    },
    createCommandEncoder() {
      return {};
    },
    destroy() {},
  };
}

const unitPlatformConfig = {
  platform: {
    id: 'unit-platform',
    detection: {
      vendor: 'unit-vendor',
      architecture: 'unit-arch',
      device: 'unit-device',
    },
  },
};

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

registerRuleGroup('kernels', 'unitKernelContext', {
  variant: [
    {
      match: { hasF16: true, platformId: 'unit-platform' },
      value: 'base-context',
    },
    {
      match: { hasF16: false },
      value: 'override-context',
    },
    {
      match: {},
      value: 'fallback',
    },
  ],
});

try {
  setDevice(createFakeDevice(['shader-f16']), { platformConfig: unitPlatformConfig });
  assert.equal(selectRuleValue('unitKernelContext', 'variant', {}), 'base-context');
  assert.equal(
    selectRuleValue('unitKernelContext', 'variant', { hasF16: false }),
    'override-context'
  );

  setDevice(createFakeDevice(), { platformConfig: unitPlatformConfig });
  assert.equal(selectRuleValue('unitKernelContext', 'variant', {}), 'override-context');
} finally {
  setDevice(null, { platformConfig: null });
}

assert.deepEqual(kernelRegistry.operations.relu?.variants?.default, {
  wgsl: 'relu.wgsl',
  entryPoint: 'main',
  workgroup: [256, 1, 1],
  requires: [],
  outputDtype: 'f32',
  reachability: {
    status: 'selectable',
    inlineConfigs: [],
    ruleChains: ['relu.rules.json#variant'],
    wgslOnDisk: true,
  },
});
assert.deepEqual(kernelRegistry.operations.relu?.variants?.default_f16, {
  wgsl: 'relu_f16.wgsl',
  entryPoint: 'main',
  workgroup: [256, 1, 1],
  requires: ['shader-f16'],
  outputDtype: 'f16',
  reachability: {
    status: 'selectable',
    inlineConfigs: [],
    ruleChains: ['relu.rules.json#variant'],
    wgslOnDisk: true,
  },
});

assert.deepEqual(kernelRegistry.operations.repeat_channels?.variants?.default, {
  wgsl: 'repeat_channels.wgsl',
  entryPoint: 'main',
  workgroup: [256, 1, 1],
  requires: [],
  outputDtype: 'f32',
  reachability: {
    status: 'selectable',
    inlineConfigs: [],
    ruleChains: ['repeat-channels.rules.json#variant'],
    wgslOnDisk: true,
  },
});
assert.deepEqual(kernelRegistry.operations.repeat_channels?.variants?.default_f16, {
  wgsl: 'repeat_channels_f16.wgsl',
  entryPoint: 'main',
  workgroup: [256, 1, 1],
  requires: ['shader-f16'],
  outputDtype: 'f16',
  reachability: {
    status: 'selectable',
    inlineConfigs: [],
    ruleChains: ['repeat-channels.rules.json#variant'],
    wgslOnDisk: true,
  },
});

console.log('diffusion-kernel-wiring.test: ok');
