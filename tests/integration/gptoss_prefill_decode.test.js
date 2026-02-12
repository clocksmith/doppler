import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { resolvePreset } = await import('../../src/config/loader.js');
const { getRegistry } = await import('../../src/config/kernels/registry.js');
const { resolveGptOssKernelPathProfile } = await import('../../src/inference/pipelines/text/moe-shape-validator.js');

const preset = resolvePreset('gpt_oss');
assert.ok(preset, 'gpt_oss preset must resolve');
assert.equal(preset.modelType, 'gpt-oss');
assert.equal(preset.inference.moe.kernelProfileId, 'gpt-oss-moe-v1');
assert.equal(preset.inference.moe.shapeConstraints.hiddenSizeDivisor, 32);
assert.equal(preset.inference.rope.ropeTheta, 150000);

const registry = await getRegistry();
assert.ok(registry.operations.topk.variants.gptoss_router_topk, 'gptoss_router_topk kernel variant must exist');
assert.ok(registry.operations.dequant.variants.mxfp4_expert_fused_f16, 'fused mxfp4 expert kernel variant must exist');

const profile = await resolveGptOssKernelPathProfile({
  hasF16: true,
  hasSubgroups: true,
  routerDtype: 'f16',
  weightsDtype: 'f16',
  outputDtype: 'f16',
  groupSize: 32,
  tileShape: 'vec4',
});
assert.equal(profile.routerTopK, 'gptoss_router_topk');
assert.equal(profile.dequantExpert, 'mxfp4_expert_fused_f16');

console.log('gptoss_prefill_decode.test: ok');
