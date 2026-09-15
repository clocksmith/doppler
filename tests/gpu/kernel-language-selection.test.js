import assert from 'node:assert/strict';
import { getKernelConfig } from '../../src/config/kernel-registry-contract.js';
import { getKernelWgslRequirements, hasRequiredFeatures } from '../../src/gpu/kernels/feature-check.js';
import { createShaderSourceScope, runWithShaderSourceScope } from '../../src/gpu/kernels/shader-source-scope.js';
import { resolveAttentionVariant } from '../../src/gpu/kernels/attention/plan.js';
import { selectRuleValue } from '../../src/gpu/kernels/rule-registry.js';

const config = getKernelConfig('rmsnorm_stats', 'subgroups');
const arithmeticOnly = { hasSubgroups: true, hasF16: false };
const identified = { ...arithmeticOnly, wgslLanguageFeatures: ['subgroup_id'] };
function statsVariant(caps) {
  return selectRuleValue('rmsnorm', 'statsVariant', {
    canUseSubgroups: hasRequiredFeatures(config.requires, caps, getKernelWgslRequirements(config)),
  });
}
assert.equal(statsVariant(arithmeticOnly), 'workgroup');
assert.equal(statsVariant(identified), 'subgroups');
const attentionVariant = caps => resolveAttentionVariant('subgroup', true, false, false, 2, 64, 128, false, caps, 16384);
assert.notEqual(attentionVariant(arithmeticOnly), 'decode_subgroup');
assert.equal(attentionVariant(identified), 'decode_subgroup');

const legacyScope = createShaderSourceScope(new Map([
  [config.shaderFile, 'enable subgroups; @compute @workgroup_size(32) fn main() {}'],
]));
await runWithShaderSourceScope(legacyScope, () => {
  assert.deepEqual(getKernelWgslRequirements(config), [], 'a registry update cannot rewrite old signed shader requirements');
  assert.equal(statsVariant(arithmeticOnly), 'subgroups');
});
assert.deepEqual(getKernelWgslRequirements(config), ['subgroup_id']);
console.log('kernel-language-selection.test: ok');
