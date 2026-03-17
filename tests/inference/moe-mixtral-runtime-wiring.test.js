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

console.log('moe-mixtral-runtime-wiring.test: ok');
