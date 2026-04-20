import assert from 'node:assert/strict';
import fs from 'node:fs';

const { compileExecutionV1 } = await import('../../src/inference/pipelines/text/execution-v1.js');
const { getLayerSteps } = await import('../../src/config/kernel-path-loader.js');

const conversionConfig = JSON.parse(
  fs.readFileSync(
    new URL('../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json', import.meta.url),
    'utf8'
  )
);

const manifestInference = {
  schema: 'doppler.execution/v1',
  session: conversionConfig.session,
  layerPattern: conversionConfig.inference.layerPattern,
  execution: conversionConfig.execution,
};

const compiled = compileExecutionV1({
  manifestInference,
  modelId: conversionConfig.output.modelBaseId,
  numLayers: conversionConfig.inference.layerPattern.layerTypes.length,
  runtimeCompute: {
    activationDtype: 'f16',
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['model', 'manifest', 'config'],
    allowSources: ['model', 'manifest', 'config'],
    onIncompatible: 'remap',
  },
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    maxWorkgroupStorageSize: 32768,
  },
  platform: {
    id: 'metal',
    vendor: 'apple',
    architecture: 'metal-3',
  },
});

assert.deepEqual(
  compiled.appliedTransforms,
  ['useQwenF16PrimaryMatmuls']
);

const kernelPath = compiled.runtimeInferencePatch.kernelPath;
assert.ok(kernelPath, 'execution-v1 compile should build an inline kernel path');
assert.equal(compiled.runtimeInferencePatch.compute?.activationDtype, 'f32');
assert.equal(kernelPath.activationDtype, 'f32');
assert.equal(kernelPath.kvDtype, 'f16');

assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'q_proj')?.kernel,
  'fused_matmul_q4_multicol_f16a.wgsl'
);
assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'gate_proj')?.kernel,
  'fused_matmul_q4_multicol_f16a.wgsl'
);
assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'gate_proj')?.precision?.inputDtype,
  'f16'
);
assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'gate_proj')?.precision?.outputDtype,
  'f16'
);
assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'down_proj')?.kernel,
  'fused_matmul_q4.wgsl'
);
assert.equal(
  kernelPath.prefill.steps.find((step) => step.op === 'q_proj')?.kernel,
  'fused_matmul_q4_widetile.wgsl'
);
assert.equal(
  kernelPath.prefill.steps.find((step) => step.op === 'attention')?.kernel,
  'attention_head256_f16kv.wgsl'
);
assert.equal(
  kernelPath.prefill.steps.find((step) => step.op === 'attention')?.precision?.activationDtype,
  undefined
);
assert.equal(
  kernelPath.prefill.steps.find((step) => step.op === 'attention')?.precision?.kvDtype,
  'f16'
);
assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'lm_head')?.kernel,
  'matmul_gemv_subgroup_f16a.wgsl'
);

const linearDecodeSteps = getLayerSteps(kernelPath, 0, 'decode');
const linearDecodeOProj = linearDecodeSteps.find((step) => step.op === 'o_proj');
assert.equal(linearDecodeOProj?.kernel, 'fused_matmul_q4.wgsl');
assert.equal(linearDecodeOProj?.entry, 'main_gemv');
assert.equal(linearDecodeOProj?.precision?.inputDtype, 'f32');
assert.equal(linearDecodeOProj?.precision?.outputDtype, 'f32');

const fullDecodeSteps = getLayerSteps(kernelPath, 3, 'decode');
const fullDecodeOProj = fullDecodeSteps.find((step) => step.op === 'o_proj');
assert.equal(fullDecodeOProj?.kernel, 'fused_matmul_q4.wgsl');
assert.equal(fullDecodeOProj?.precision?.inputDtype, 'f32');
assert.equal(fullDecodeOProj?.precision?.outputDtype, 'f32');

const linearPrefillSteps = getLayerSteps(kernelPath, 0, 'prefill');
const linearPrefillOProj = linearPrefillSteps.find((step) => step.op === 'o_proj');
assert.equal(linearPrefillOProj?.kernel, 'fused_matmul_q4_widetile.wgsl');
assert.equal(linearPrefillOProj?.precision?.inputDtype, 'f32');
assert.equal(linearPrefillOProj?.precision?.outputDtype, 'f32');

const fullPrefillSteps = getLayerSteps(kernelPath, 3, 'prefill');
const fullPrefillOProj = fullPrefillSteps.find((step) => step.op === 'o_proj');
assert.equal(fullPrefillOProj?.kernel, 'fused_matmul_q4_widetile.wgsl');
assert.equal(fullPrefillOProj?.precision?.inputDtype, 'f32');
assert.equal(fullPrefillOProj?.precision?.outputDtype, 'f32');

const compiledWithFallbackPlan = compileExecutionV1({
  manifestInference,
  modelId: conversionConfig.output.modelBaseId,
  numLayers: conversionConfig.inference.layerPattern.layerTypes.length,
  runtimeCompute: {
    activationDtype: 'f16',
    rangeAwareSelectiveWidening: {
      enabled: true,
      includeNonFinite: true,
      absThreshold: 65500,
      onTrigger: 'fallback-plan',
    },
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['model', 'manifest', 'config'],
    allowSources: ['model', 'manifest', 'config'],
    onIncompatible: 'remap',
  },
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    maxWorkgroupStorageSize: 32768,
  },
  platform: {
    id: 'metal',
    vendor: 'apple',
    architecture: 'metal-3',
  },
});

assert.ok(compiledWithFallbackPlan.fallbackKernelPath, 'f16 Qwen compile should build a finiteness fallback kernel path');
assert.equal(compiledWithFallbackPlan.runtimeInferencePatch.compute?.activationDtype, 'f32');
assert.equal(compiledWithFallbackPlan.fallbackKernelPath.activationDtype, 'f32');
const fallbackDecodeAttention = compiledWithFallbackPlan.fallbackKernelPath.decode.steps.find((step) => step.op === 'attention');
assert.ok(fallbackDecodeAttention, 'fallback kernel path should include decode attention');
if (fallbackDecodeAttention.kernel.includes('_f16kv')) {
  assert.equal(fallbackDecodeAttention.precision?.activationDtype, 'f32');
  assert.equal(compiledWithFallbackPlan.fallbackKernelPath.kvDtype, 'f16');
}

console.log('qwen-execution-v1-linear-decode-f16.test: ok');
