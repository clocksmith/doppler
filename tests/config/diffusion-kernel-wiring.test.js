import assert from 'node:assert/strict';

import { selectRuleValue } from '../../src/gpu/kernels/rule-registry.js';
import { setDevice } from '../../src/gpu/device.js';
import { runReLU, recordReLU } from '../../src/gpu/kernels/relu.js';
import { runRepeatChannels, recordRepeatChannels } from '../../src/gpu/kernels/repeat_channels.js';
import { registerRuleGroup } from '../../src/rules/rule-registry.js';

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

assert.equal(selectRuleValue('relu', 'variant', { dtype: 'f32' }), 'default');
assert.equal(selectRuleValue('relu', 'variant', { dtype: 'f16' }), 'default_f16');

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

console.log('diffusion-kernel-wiring.test: ok');
