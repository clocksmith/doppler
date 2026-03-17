import assert from 'node:assert/strict';

import { selectRuleValue } from '../../src/rules/rule-registry.js';

// === moeMixtral rule group is registered and selectable ===

{
  const vendorProfile = selectRuleValue('kernels', 'moeMixtral', 'vendorQuirkProfile', {
    vendor: 'apple',
  });
  assert.equal(vendorProfile.preferVec4Dequant, false);
  assert.equal(vendorProfile.dequantTileShape, 'scalar');
  assert.equal(vendorProfile.routerWorkgroupSize, 256);
  assert.equal(vendorProfile.maxTokensPerExpertScale, 1.0);
}

// === Intel/AMD vendor gets conservative scaling ===

{
  const vendorProfile = selectRuleValue('kernels', 'moeMixtral', 'vendorQuirkProfile', {
    vendor: 'intel',
  });
  assert.equal(vendorProfile.routerWorkgroupSize, 128);
  assert.equal(vendorProfile.maxTokensPerExpertScale, 0.85);
}

// === Unknown vendor hits catch-all ===

{
  const vendorProfile = selectRuleValue('kernels', 'moeMixtral', 'vendorQuirkProfile', {
    vendor: 'unknown',
  });
  assert.equal(vendorProfile.routerWorkgroupSize, 128);
  assert.equal(vendorProfile.maxTokensPerExpertScale, 1.0);
}

// === routerTopKVariant resolves for mixtral with subgroups ===

{
  const variant = selectRuleValue('kernels', 'moeMixtral', 'routerTopKVariant', {
    modelType: 'mixtral',
    hasF16: true,
    hasSubgroups: true,
    routerDtype: 'f32',
  });
  assert.equal(variant, 'softmax_topk_f32_subgroup');
}

// === routerTopKVariant resolves for mixtral without subgroups ===

{
  const variant = selectRuleValue('kernels', 'moeMixtral', 'routerTopKVariant', {
    modelType: 'mixtral',
    hasF16: true,
    hasSubgroups: false,
    routerDtype: 'f32',
  });
  assert.equal(variant, 'softmax_topk_f32');
}

// === routerTopKVariant catch-all for mixtral ===

{
  const variant = selectRuleValue('kernels', 'moeMixtral', 'routerTopKVariant', {
    modelType: 'mixtral',
  });
  assert.equal(variant, 'softmax_topk_f32');
}

// === dequantVariant resolves q4k with subgroups ===

{
  const variant = selectRuleValue('kernels', 'moeMixtral', 'dequantVariant', {
    modelType: 'mixtral',
    weightsDtype: 'q4k',
    hasF16: true,
    hasSubgroups: true,
    outputDtype: 'f32',
  });
  assert.equal(variant, 'q4k_expert_dequant_f32_subgroup');
}

// === dequantVariant resolves q4k f16 output ===

{
  const variant = selectRuleValue('kernels', 'moeMixtral', 'dequantVariant', {
    modelType: 'mixtral',
    weightsDtype: 'q4k',
    hasF16: true,
    outputDtype: 'f16',
  });
  assert.equal(variant, 'q4k_expert_dequant_f16');
}

// === dequantVariant resolves q4k catch-all ===

{
  const variant = selectRuleValue('kernels', 'moeMixtral', 'dequantVariant', {
    modelType: 'mixtral',
    weightsDtype: 'q4k',
  });
  assert.equal(variant, 'q4k_expert_dequant_f32');
}

// === dequantVariant resolves f16 passthrough ===

{
  const variant = selectRuleValue('kernels', 'moeMixtral', 'dequantVariant', {
    modelType: 'mixtral',
    weightsDtype: 'f16',
    hasF16: true,
    outputDtype: 'f16',
  });
  assert.equal(variant, 'f16_expert_passthrough');
}

// === dequantVariant resolves f16 upcast catch-all ===

{
  const variant = selectRuleValue('kernels', 'moeMixtral', 'dequantVariant', {
    modelType: 'mixtral',
  });
  assert.equal(variant, 'f16_expert_upcast_f32');
}

// === resolveMixtralKernelPathProfile returns both fields ===

{
  const { resolveMixtralKernelPathProfile } = await import(
    '../../src/inference/pipelines/text/moe-shape-validator.js'
  );
  const profile = await resolveMixtralKernelPathProfile({
    hasF16: true,
    hasSubgroups: true,
    routerDtype: 'f32',
    weightsDtype: 'q4k',
    outputDtype: 'f32',
  });
  assert.equal(profile.routerTopK, 'softmax_topk_f32_subgroup');
  assert.equal(profile.dequantExpert, 'q4k_expert_dequant_f32_subgroup');
}

// === resolveMixtralKernelPathProfile without subgroups ===

{
  const { resolveMixtralKernelPathProfile } = await import(
    '../../src/inference/pipelines/text/moe-shape-validator.js'
  );
  const profile = await resolveMixtralKernelPathProfile({
    hasF16: true,
    hasSubgroups: false,
    routerDtype: 'f32',
    weightsDtype: 'q4k',
    outputDtype: 'f32',
  });
  assert.equal(profile.routerTopK, 'softmax_topk_f32');
  assert.equal(profile.dequantExpert, 'q4k_expert_dequant_f32');
}

// === Generic softmax.rules.json topkVariant has explicit Mixtral entries ===
// This is the ACTUAL kernel dispatch rule — runSoftmaxTopK() at softmax.js:75
// uses this, not the MoE-specific routerTopKVariant metadata above.

{
  const variant = selectRuleValue('kernels', 'softmax', 'topkVariant', {
    modelType: 'mixtral',
    inputDtype: 'f32',
    weightsDtype: 'f32',
    hasF16: true,
    hasSubgroups: true,
  });
  assert.equal(variant, 'fused',
    'mixtral f32 must get explicit "fused" from softmax rules, not fall through to catch-all');
}

{
  const variant = selectRuleValue('kernels', 'softmax', 'topkVariant', {
    modelType: 'mixtral',
    inputDtype: 'f16',
    weightsDtype: 'f16',
    hasF16: true,
    hasSubgroups: true,
  });
  assert.equal(variant, 'fused_f16_w16',
    'mixtral f16 must get explicit "fused_f16_w16" from softmax rules');
}

// === Vendor profile: routerWorkgroupSize and preferVec4Dequant are structural ===
// These fields have no runtime consumers today on either GPT-OSS or Mixtral.
// They are preserved for structural parity with GPT-OSS vendor profile shape.
// dequantTileShape and maxTokensPerExpertScale ARE consumed at runtime.

{
  const vendorProfile = selectRuleValue('kernels', 'moeMixtral', 'vendorQuirkProfile', {
    vendor: 'nvidia',
  });
  assert.ok('routerWorkgroupSize' in vendorProfile, 'structural field present');
  assert.ok('preferVec4Dequant' in vendorProfile, 'structural field present');
  assert.ok('dequantTileShape' in vendorProfile, 'active runtime field present');
  assert.ok('maxTokensPerExpertScale' in vendorProfile, 'active runtime field present');
}

console.log('moe-mixtral-runtime-wiring.test: ok');
