import assert from 'node:assert/strict';
import fs from 'node:fs';

const { compileExecutionV1 } = await import('../../src/inference/pipelines/text/execution-v1.js');

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
  runtimeSession: conversionConfig.session,
  runtimeCompute: {
    activationDtype: 'f32',
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

// Transform chain: prefill attention + prefill dense remap + decode GEMV
assert.deepEqual(
  compiled.appliedTransforms,
  ['useHead256PrefillAttention', 'remapQ4KPrefillToDense', 'remapQ4KDecodeToGemv']
);

const kernelPath = compiled.runtimeInferencePatch.kernelPath;
assert.ok(kernelPath, 'execution-v1 compile should build an inline kernel path');

// All decode projections should use GEMV subgroup (f32 activations, f16 weights)
for (const op of ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']) {
  assert.equal(
    kernelPath.decode.steps.find((step) => step.op === op)?.kernel,
    'matmul_gemv_subgroup.wgsl',
    `decode ${op} should use matmul_gemv_subgroup.wgsl`
  );
}

// Prefill projections should use dense matmul (from remapQ4KPrefillToDense)
assert.equal(
  kernelPath.prefill.steps.find((step) => step.op === 'q_proj')?.kernel,
  'matmul_f16w_f32a.wgsl'
);

// Prefill attention should be the head256 variant (from useHead256PrefillAttention)
assert.equal(
  kernelPath.prefill.steps.find((step) => step.op === 'attention')?.kernel,
  'attention_head256_f16kv.wgsl'
);

// GEMV decode projections should NOT have precision overrides — f32 in/out is
// the natural contract for the GEMV f32-activation path
assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'q_proj')?.precision,
  undefined,
  'GEMV decode q_proj should have no precision override'
);
assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'gate_proj')?.precision,
  undefined,
  'GEMV decode gate_proj should have no precision override'
);

// lm_head should remain as manifest-original GEMV subgroup
assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'lm_head')?.kernel,
  'matmul_gemv_subgroup.wgsl'
);

console.log('qwen-execution-v1-linear-decode-f16.test: ok');
