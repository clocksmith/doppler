import assert from 'node:assert/strict';
import fs from 'node:fs';

const { compileExecutionV1 } = await import('../../src/inference/pipelines/text/execution-v1.js');
const { getLayerSteps } = await import('../../src/config/kernel-path-loader.js');

const conversionConfig = JSON.parse(
  fs.readFileSync(
    new URL('../../src/config/conversion/gemma4/gemma-4-31b-it-text-q4k-ehf16-af32.json', import.meta.url),
    'utf8'
  )
);

const profile = JSON.parse(
  fs.readFileSync(
    new URL('../../src/config/runtime/profiles/gemma4-31b-f16-activations-probe.json', import.meta.url),
    'utf8'
  )
);

const manifestInference = {
  schema: 'doppler.execution/v1',
  session: conversionConfig.session,
  layerPattern: conversionConfig.inference.layerPattern,
  execution: conversionConfig.execution,
};

assert.equal(profile.model, conversionConfig.output.modelBaseId);
assert.equal(profile.runtime.inference.compute.activationDtype, 'f16');
assert.deepEqual(profile.runtime.inference.largeWeights.gpuResidentOverrides, []);
assert.equal(profile.runtime.inference.session.compute.defaults.activationDtype, 'f16');
assert.equal(profile.runtime.inference.session.compute.defaults.mathDtype, 'f16');
assert.equal(profile.runtime.inference.session.compute.defaults.accumDtype, 'f16');
assert.equal(profile.runtime.inference.session.compute.defaults.outputDtype, 'f16');
assert.equal(profile.runtime.inference.session.kvcache.kvDtype, 'f16');
assert.deepEqual(profile.runtime.inference.session.decodeLoop, {
  batchSize: 3,
  readbackInterval: 1,
  stopCheckMode: 'batch',
});

const compiled = compileExecutionV1({
  manifestInference,
  modelId: conversionConfig.output.modelBaseId,
  numLayers: conversionConfig.inference.layerPattern.layerTypes.length,
  headDim: 512,
  runtimeSession: profile.runtime.inference.session,
  runtimeCompute: profile.runtime.inference.compute,
  kernelPathPolicy: profile.runtime.inference.kernelPathPolicy,
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    maxWorkgroupStorageSize: 32768,
  },
  platform: {
    id: 'rdna3',
    vendor: 'amd',
    architecture: 'rdna-3',
  },
});

assert.deepEqual(
  compiled.appliedTransforms,
  ['useGemma431BTextF16Activations']
);

const kernelPath = compiled.runtimeInferencePatch.kernelPath;
assert.ok(kernelPath, 'Gemma 4 31B f16 compile should build an inline kernel path');
assert.equal(compiled.runtimeInferencePatch.compute?.activationDtype, 'f16');
assert.equal(compiled.runtimeInferencePatch.session?.compute?.defaults?.activationDtype, 'f16');
assert.equal(compiled.runtimeInferencePatch.session?.compute?.defaults?.mathDtype, 'f16');
assert.equal(compiled.runtimeInferencePatch.session?.compute?.defaults?.accumDtype, 'f16');
assert.equal(compiled.runtimeInferencePatch.session?.compute?.defaults?.outputDtype, 'f16');
assert.equal(kernelPath.activationDtype, 'f16');
assert.equal(kernelPath.kvDtype, 'f16');

const embed = kernelPath.preLayer.find((step) => step.op === 'embed');
assert.equal(embed?.kernel, 'gather_f16_vec4_f16_out.wgsl');
assert.equal(embed?.entry, 'gather_vec4_f16_out');
assert.equal(embed?.precision?.inputDtype, 'f16');
assert.equal(embed?.precision?.outputDtype, 'f16');

const decodeQ = kernelPath.decode.steps.find((step) => step.op === 'q_proj');
assert.equal(decodeQ?.kernel, 'fused_matmul_q4_multicol_f16a.wgsl');
assert.equal(decodeQ?.precision?.inputDtype, 'f16');
assert.equal(decodeQ?.precision?.outputDtype, 'f16');
assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'attention')?.kernel,
  'attention_decode_online_f16.wgsl'
);

const slidingPrefillSteps = getLayerSteps(kernelPath, 0, 'prefill');
assert.equal(
  slidingPrefillSteps.find((step) => step.op === 'q_proj')?.kernel,
  'fused_matmul_q4_batched_f16acc_f16a.wgsl'
);
assert.equal(
  slidingPrefillSteps.find((step) => step.op === 'attention')?.kernel,
  'attention_small_f16.wgsl'
);

const fullPrefillSteps = getLayerSteps(kernelPath, 5, 'prefill');
assert.equal(
  fullPrefillSteps.find((step) => step.op === 'q_proj')?.kernel,
  'fused_matmul_q4_batched_f16acc_f16a.wgsl'
);
assert.equal(
  fullPrefillSteps.find((step) => step.op === 'attention')?.kernel,
  'attention_head512_f16.wgsl'
);

assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'final_norm')?.kernel,
  'rmsnorm_f16.wgsl'
);
assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'lm_head')?.kernel,
  'matmul_gemv_subgroup_f16a.wgsl'
);
assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'lm_head_prefill')?.kernel,
  'matmul_f16_tiled.wgsl'
);
assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'sample')?.kernel,
  'sample_f16.wgsl'
);

for (const step of compiled.resolvedSteps.all) {
  assert.ok(
    !String(step.kernel).includes('_f16kv'),
    `Gemma 4 31B f16 lane should not keep f16kv attention in resolved step ${step.id}`
  );
}

console.log('gemma4-31b-f16-activations.test: ok');
