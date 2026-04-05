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

// Fused Q4 decode and prefill are now the primary execution graph —
// no capability transforms needed on a capable GPU.
assert.deepEqual(compiled.appliedTransforms, []);

const kernelPath = compiled.runtimeInferencePatch.kernelPath;
assert.ok(kernelPath, 'execution-v1 compile should build an inline kernel path');

// All decode projections should use fused Q4 multicol (primary path)
for (const op of ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']) {
  assert.equal(
    kernelPath.decode.steps.find((step) => step.op === op)?.kernel,
    'fused_matmul_q4.wgsl',
    `decode ${op} should use fused_matmul_q4.wgsl`
  );
}

// Prefill projections should use fused batched Q4 matmul (primary path)
assert.equal(
  kernelPath.prefill.steps.find((step) => step.op === 'q_proj')?.kernel,
  'fused_matmul_q4_batched.wgsl'
);

// Prefill attention should use the streaming attention path (primary path)
assert.equal(
  kernelPath.prefill.steps.find((step) => step.op === 'attention')?.kernel,
  'attention_streaming_f16kv.wgsl'
);

// Fused Q4 decode FFN projections should keep manifest precision defaults
assert.equal(
  kernelPath.decode.steps.find((step) => step.op === 'gate_proj')?.precision,
  undefined,
  'Q4 decode gate_proj should have no precision override'
);

// lm_head should remain as manifest-original GEMV subgroup
assert.equal(
  kernelPath.postLayer.find((step) => step.op === 'lm_head')?.kernel,
  'matmul_gemv_subgroup.wgsl'
);

// AMD Vulkan should also resolve the same kernel path (no platform-specific transforms)
const compiledAmd = compileExecutionV1({
  manifestInference,
  modelId: conversionConfig.output.modelBaseId,
  numLayers: conversionConfig.inference.layerPattern.layerTypes.length,
  runtimeSession: conversionConfig.session,
  runtimeCompute: { activationDtype: 'f32' },
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
    id: 'vulkan',
    vendor: 'amd',
    architecture: 'rdna3',
  },
});

assert.deepEqual(compiledAmd.appliedTransforms, [],
  'AMD Vulkan should also get zero transforms (fused Q4 is the primary path)');
assert.equal(
  compiledAmd.runtimeInferencePatch.kernelPath.decode.steps.find((s) => s.op === 'q_proj')?.kernel,
  'fused_matmul_q4.wgsl',
  'AMD Vulkan decode q_proj should use fused_matmul_q4.wgsl'
);

console.log('qwen-execution-v1-linear-decode-f16.test: ok');
