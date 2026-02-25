import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { selectRuleValue } = await import('../../src/rules/rule-registry.js');
const { getKernelConfig } = await import('../../src/gpu/kernels/kernel-configs.js');

const f16Variant = selectRuleValue('kernels', 'splitQkv', 'variant', {
  outputDtype: 'f16',
});
assert.equal(f16Variant, 'f16');
const f16Config = getKernelConfig('split_qkv', f16Variant);
assert.equal(f16Config.shaderFile, 'split_qkv_f16.wgsl');
assert.deepEqual(f16Config.requires, ['shader-f16']);

const f32Variant = selectRuleValue('kernels', 'splitQkv', 'variant', {
  outputDtype: 'f32',
});
assert.equal(f32Variant, 'default');
const f32Config = getKernelConfig('split_qkv', f32Variant);
assert.equal(f32Config.shaderFile, 'split_qkv.wgsl');

console.log('split_qkv_variant_parity.test: ok');
