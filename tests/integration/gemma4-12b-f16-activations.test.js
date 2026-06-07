import assert from 'node:assert/strict';
import fs from 'node:fs';

const { compileExecutionV1 } = await import('../../src/inference/pipelines/text/execution-v1.js');
const { getLayerSteps } = await import('../../src/config/kernel-path-loader.js');

const AF32_MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-af32';
const AF16_MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-af16';

const af32ConversionConfig = JSON.parse(
  fs.readFileSync(
    new URL(`../../src/config/conversion/gemma4/${AF32_MODEL_ID}.json`, import.meta.url),
    'utf8'
  )
);

const conversionConfig = JSON.parse(
  fs.readFileSync(
    new URL(`../../src/config/conversion/gemma4/${AF16_MODEL_ID}.json`, import.meta.url),
    'utf8'
  )
);

assert.equal(conversionConfig.output.modelBaseId, AF16_MODEL_ID);
assert.equal(
  conversionConfig.manifest?.artifactIdentity?.weightPackId,
  af32ConversionConfig.manifest?.artifactIdentity?.weightPackId
);
assert.notEqual(
  conversionConfig.manifest?.artifactIdentity?.manifestVariantId,
  af32ConversionConfig.manifest?.artifactIdentity?.manifestVariantId
);
assert.equal(conversionConfig.quantization.computePrecision, 'f16');

const manifestInference = {
  schema: 'doppler.execution/v1',
  session: conversionConfig.session,
  layerPattern: conversionConfig.inference.layerPattern,
  execution: conversionConfig.execution,
};

const runtimeSession = {
  ...conversionConfig.session,
  compute: {
    ...conversionConfig.session.compute,
    defaults: {
      activationDtype: 'f16',
      mathDtype: 'f16',
      accumDtype: 'f16',
      outputDtype: 'f16',
    },
  },
  kvcache: {
    ...conversionConfig.session.kvcache,
    kvDtype: 'f16',
  },
};

const compiled = compileExecutionV1({
  manifestInference,
  modelId: AF16_MODEL_ID,
  numLayers: conversionConfig.inference.layerPattern.layerTypes.length,
  headDim: 512,
  runtimeSession,
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest'],
    allowSources: ['manifest'],
    onIncompatible: 'remap',
  },
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
  ['useGemma412BTextF16Activations']
);

const kernelPath = compiled.runtimeInferencePatch.kernelPath;
assert.ok(kernelPath, 'Gemma 4 12B f16 compile should build an inline kernel path');
assert.equal(compiled.runtimeInferencePatch.compute?.activationDtype, 'f16');
assert.equal(compiled.runtimeInferencePatch.session?.compute?.defaults?.activationDtype, 'f16');
assert.equal(compiled.runtimeInferencePatch.session?.compute?.defaults?.mathDtype, 'f16');
assert.equal(compiled.runtimeInferencePatch.session?.compute?.defaults?.accumDtype, 'f16');
assert.equal(compiled.runtimeInferencePatch.session?.compute?.defaults?.outputDtype, 'f16');
assert.equal(kernelPath.activationDtype, 'f16');
assert.equal(kernelPath.kvDtype, 'f16');

assert.equal(
  kernelPath.preLayer.find((step) => step.op === 'embed')?.kernel,
  'gather_f16_vec4_f16_out.wgsl'
);
assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'q_proj')?.kernel,
  'fused_matmul_q4.wgsl'
);
assert.deepEqual(
  kernelPath.decode.steps.find((step) => step.op === 'q_proj')?.precision,
  { inputDtype: 'f32', outputDtype: 'f32' }
);
assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'attention')?.kernel,
  'attention_decode_online_f16kv.wgsl'
);
assert.deepEqual(
  kernelPath.decode.steps.find((step) => step.op === 'attention')?.precision,
  { kvDtype: 'f16', activationDtype: 'f32', outputDtype: 'f32' }
);

const slidingPrefillSteps = getLayerSteps(kernelPath, 0, 'prefill');
assert.equal(
  slidingPrefillSteps.find((step) => step.op === 'q_proj')?.kernel,
  'fused_matmul_q4_widetile.wgsl'
);
assert.deepEqual(
  slidingPrefillSteps.find((step) => step.op === 'q_proj')?.precision,
  { inputDtype: 'f32', outputDtype: 'f32' }
);
assert.equal(
  slidingPrefillSteps.find((step) => step.op === 'attention')?.kernel,
  'attention_head256_f16kv.wgsl'
);
assert.deepEqual(
  slidingPrefillSteps.find((step) => step.op === 'attention')?.precision,
  { kvDtype: 'f16', activationDtype: 'f32', outputDtype: 'f32' }
);

const fullPrefillSteps = getLayerSteps(kernelPath, 5, 'prefill');
assert.equal(
  fullPrefillSteps.find((step) => step.op === 'q_proj')?.kernel,
  'fused_matmul_q4_widetile.wgsl'
);
assert.deepEqual(
  fullPrefillSteps.find((step) => step.op === 'q_proj')?.precision,
  { inputDtype: 'f32', outputDtype: 'f32' }
);
assert.equal(
  fullPrefillSteps.find((step) => step.op === 'k_proj')?.kernel,
  'fused_matmul_q4_widetile.wgsl'
);
assert.equal(
  fullPrefillSteps.find((step) => step.op === 'v_proj')?.kernel,
  'fused_matmul_q4_widetile.wgsl'
);
assert.equal(
  fullPrefillSteps.find((step) => step.op === 'o_proj')?.kernel,
  'fused_matmul_q4_widetile_f16a.wgsl'
);
assert.deepEqual(
  fullPrefillSteps.find((step) => step.op === 'o_proj')?.precision,
  { inputDtype: 'f16', outputDtype: 'f16' }
);
assert.equal(
  fullPrefillSteps.find((step) => step.op === 'down_proj')?.kernel,
  'fused_matmul_q4_widetile_f16a.wgsl'
);
assert.deepEqual(
  fullPrefillSteps.find((step) => step.op === 'down_proj')?.precision,
  { inputDtype: 'f16', outputDtype: 'f16' }
);
assert.equal(
  fullPrefillSteps.find((step) => step.op === 'attention')?.kernel,
  'attention_head512_f16kv.wgsl'
);
assert.deepEqual(
  fullPrefillSteps.find((step) => step.op === 'attention')?.precision,
  { kvDtype: 'f16', activationDtype: 'f32', outputDtype: 'f32' }
);

assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'final_norm')?.kernel,
  'rmsnorm.wgsl'
);
assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'lm_head')?.kernel,
  'matmul_gemv_subgroup.wgsl'
);
assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'lm_head_prefill')?.kernel,
  'matmul_f16w_f32a.wgsl'
);
assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'sample')?.kernel,
  'sample.wgsl'
);

for (const step of compiled.resolvedSteps.all) {
  assert.ok(
    step.kernel !== 'attention_head512_f16.wgsl',
    `Gemma 4 12B f16 lane should avoid pure-f16 head512 attention in resolved step ${step.id}`
  );
  assert.ok(
    step.kernel !== 'attention_small_f16.wgsl',
    `Gemma 4 12B f16 lane should avoid pure-f16 sliding attention in resolved step ${step.id}`
  );
}

assert.ok(
  compiled.resolvedSteps.all.some((step) => (
    step.kernel === 'attention_head512_f16kv.wgsl'
    && step.precision?.activationDtype === 'f32'
    && step.precision?.kvDtype === 'f16'
  )),
  'Gemma 4 12B f16 lane should keep head512 prefill on the f32-Q/f16-KV boundary'
);

console.log('gemma4-12b-f16-activations.test: ok');
