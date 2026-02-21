import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { selectRuleValue } = await import('../../src/rules/rule-registry.js');

const routerVariant = selectRuleValue('kernels', 'softmax', 'topkVariant', {
  modelType: 'gpt-oss',
  hasF16: true,
  hasSubgroups: true,
  inputDtype: 'f16',
  weightsDtype: 'f16',
});
assert.equal(routerVariant, 'gptoss_router_topk');

const dequantVariant = selectRuleValue('kernels', 'dequant', 'mxfp4ExpertVariant', {
  modelType: 'gpt-oss',
  hasF16: true,
  hasSubgroups: true,
  outputDtype: 'f16',
  dequantTileShape: 'vec4',
  groupSize: 32,
});
assert.equal(dequantVariant, 'mxfp4_expert_fused_f16');

const fallbackVariant = selectRuleValue('kernels', 'dequant', 'mxfp4ExpertVariant', {
  outputDtype: 'f16',
});
assert.equal(fallbackVariant, 'mxfp4_expert_f16');

console.log('gptoss_moe_selection.test: ok');
