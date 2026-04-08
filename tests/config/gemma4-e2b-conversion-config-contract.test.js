import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

const configPath = path.join(
  repoRoot,
  'src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json'
);
const config = JSON.parse(await fs.readFile(configPath, 'utf8'));

assert.equal(config.output?.baseDir, 'models/local');
assert.equal(config.output?.modelBaseId, 'gemma-4-e2b-it-q4k-ehf16-af32');
assert.equal(config.output?.textOnly, false);

assert.equal(config.quantization?.weights, 'q4k');
assert.equal(config.quantization?.embeddings, 'f16');
assert.equal(config.quantization?.lmHead, 'f16');
assert.equal(config.quantization?.computePrecision, 'f32');
assert.equal(config.quantization?.q4kLayout, 'row');

assert.equal(config.inference?.attention?.queryPreAttnScalar, 1);
assert.equal(config.inference?.attention?.slidingWindow, 512);
assert.equal(config.inference?.attention?.valueNorm, true);
assert.equal(config.inference?.normalization?.rmsNormWeightOffset, false);
assert.equal(config.inference?.ffn?.useDoubleWideMlp, true);
assert.equal(config.inference?.chatTemplate?.type, 'gemma4');
assert.equal(config.inference?.rope?.partialRotaryFactor, 0.25);
assert.equal(config.inference?.rope?.ropeLocalPartialRotaryFactor, null);
assert.equal(config.inference?.rope?.ropeInterleaved, false);
assert.equal(config.inference?.rope?.ropeFrequencyBaseDim, 512);
assert.equal(config.inference?.rope?.ropeLocalFrequencyBaseDim, null);
assert.equal(config.inference?.rope?.ropeScalingType, null);
assert.equal(config.inference?.output?.finalLogitSoftcapping, 30);
assert.equal(config.inference?.output?.embeddingPostprocessor, null);

const expanded = expandExecutionV1(config.execution);
assert.ok(expanded.length > 0, 'execution must expand to at least one step');
assert.equal(config.execution?.kernels?.attn_decode?.kernel, 'attention_streaming_f16kv.wgsl');
assert.equal(config.execution?.kernels?.attn_decode?.entry, 'main');
assert.equal(config.execution?.kernels?.attn_decode?.precision?.kvDtype, 'f16');
assert.equal(config.execution?.kernels?.attn_stream?.kernel, 'attention_streaming_f16kv.wgsl');
assert.equal(config.execution?.kernels?.attn_stream?.precision?.kvDtype, 'f16');
assert.equal(config.execution?.kernels?.gemv?.kernel, 'matmul_gemv_subgroup.wgsl');
assert.equal(config.execution?.kernels?.tiled?.kernel, 'matmul_f16w_f32a.wgsl');
assert.equal(config.execution?.kernels?.final_norm_stable?.kernel, 'rmsnorm.wgsl');
assert.equal(config.execution?.kernels?.final_norm_stable?.precision?.inputDtype, 'f32');
assert.equal(config.execution?.kernels?.final_norm_stable?.precision?.outputDtype, 'f32');
assert.equal(config.execution?.kernels?.lm_head_gemv_stable?.kernel, 'matmul_gemv_subgroup.wgsl');
assert.equal(config.execution?.kernels?.lm_head_gemv_stable?.precision?.inputDtype, 'f32');
assert.equal(config.execution?.kernels?.lm_head_gemv_stable?.precision?.outputDtype, 'f32');
assert.equal(config.execution?.kernels?.lm_head_prefill_stable?.kernel, 'matmul_f16w_f32a.wgsl');
assert.equal(config.execution?.kernels?.lm_head_prefill_stable?.precision?.inputDtype, 'f32');
assert.equal(config.execution?.kernels?.lm_head_prefill_stable?.precision?.outputDtype, 'f32');
assert.equal(config.execution?.kernels?.sample?.kernel, 'sample.wgsl');

assert.equal(config.session?.compute?.defaults?.activationDtype, 'f32');
assert.equal(config.session?.compute?.defaults?.mathDtype, 'f32');
assert.equal(config.session?.compute?.defaults?.accumDtype, 'f32');
assert.equal(config.session?.compute?.defaults?.outputDtype, 'f32');
assert.equal(config.session?.kvcache?.kvDtype, 'f16');
assert.equal(config.session?.decodeLoop?.batchSize, 8);
assert.equal(config.session?.decodeLoop?.stopCheckMode, 'batch');
assert.equal(config.session?.decodeLoop?.readbackInterval, 8);
assert.equal(config.session?.decodeLoop?.readbackMode, 'overlapped');
assert.equal(config.session?.decodeLoop?.ringTokens, 2);
assert.equal(config.session?.decodeLoop?.ringStop, 1);
assert.equal(config.session?.decodeLoop?.ringStaging, 2);
assert.equal(config.session?.decodeLoop?.disableCommandBatching, false);
assert.equal(config.session?.perLayerInputs?.materialization, 'range_backed');
assert.equal(config.session?.perLayerInputs?.hotCache?.mode, 'prepared_tokens');
assert.equal(config.session?.perLayerInputs?.hotCache?.maxTokens, 4096);
assert.equal(config.session?.perLayerInputs?.hotCache?.maxBytes, 268435456);
assert.equal(config.session?.perLayerInputs?.hotCache?.outputDtype, 'f32');

const manifestInference = {
  schema: 'doppler.execution/v1',
  ...config.inference,
  session: config.session,
  execution: config.execution,
};

const f16Primary = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
  },
  platform: {
    id: 'test-f16',
    vendor: 'test',
    architecture: 'test',
  },
  runtimeCompute: {
    rangeAwareSelectiveWidening: {
      enabled: true,
      includeNonFinite: true,
      onTrigger: 'error',
      absThreshold: 65500,
    },
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model', 'config'],
    onIncompatible: 'remap',
  },
});

assert.equal(f16Primary.session.compute.defaults.activationDtype, 'f32');
assert.equal(f16Primary.session.kvcache.kvDtype, 'f16');
assert.equal(
  f16Primary.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.kernel,
  'attention_streaming_f16kv.wgsl'
);
assert.equal(
  f16Primary.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.precision?.kvDtype,
  'f16'
);
assert.equal(
  f16Primary.runtimeInferencePatch?.kernelPath?.prefill?.steps?.find((step) => step.op === 'attention')?.precision?.kvDtype,
  'f16'
);

const runtimeF16Primary = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  runtimeSession: {
    ...config.session,
    compute: {
      ...config.session.compute,
      defaults: {
        ...config.session.compute.defaults,
        activationDtype: 'f16',
        mathDtype: 'f16',
        outputDtype: 'f16',
      },
    },
  },
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
  },
  platform: {
    id: 'test-f16-runtime',
    vendor: 'test',
    architecture: 'test',
  },
  runtimeCompute: {
    rangeAwareSelectiveWidening: {
      enabled: true,
      includeNonFinite: true,
      onTrigger: 'fallback-plan',
      absThreshold: 65500,
    },
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model', 'config'],
    onIncompatible: 'remap',
  },
});

assert.deepEqual(runtimeF16Primary.appliedTransforms, ['narrowToF16Activations']);
assert.equal(runtimeF16Primary.session.compute.defaults.activationDtype, 'f16');
assert.equal(runtimeF16Primary.session.kvcache.kvDtype, 'f16');
assert.equal(
  runtimeF16Primary.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'q_proj')?.kernel,
  'matmul_gemv_subgroup_f16a.wgsl'
);
assert.equal(
  runtimeF16Primary.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.kernel,
  'attention_streaming_f16.wgsl'
);
assert.equal(
  runtimeF16Primary.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.precision?.kvDtype,
  'f16'
);
assert.equal(
  runtimeF16Primary.runtimeInferencePatch?.kernelPath?.prefill?.steps?.find((step) => step.op === 'q_proj')?.kernel,
  'matmul_f16.wgsl'
);
assert.equal(
  runtimeF16Primary.runtimeInferencePatch?.kernelPath?.postLayer?.find((step) => step.op === 'final_norm')?.kernel,
  'rmsnorm.wgsl'
);
assert.equal(
  runtimeF16Primary.runtimeInferencePatch?.kernelPath?.postLayer?.find((step) => step.op === 'final_norm')?.precision?.inputDtype,
  'f32'
);
assert.equal(
  runtimeF16Primary.runtimeInferencePatch?.kernelPath?.postLayer?.find((step) => step.op === 'final_norm')?.precision?.outputDtype,
  'f32'
);
assert.equal(
  runtimeF16Primary.runtimeInferencePatch?.kernelPath?.postLayer?.find((step) => step.op === 'lm_head')?.kernel,
  'matmul_gemv_subgroup.wgsl'
);
assert.equal(
  runtimeF16Primary.runtimeInferencePatch?.kernelPath?.postLayer?.find((step) => step.op === 'lm_head')?.precision?.inputDtype,
  'f32'
);
assert.equal(
  runtimeF16Primary.runtimeInferencePatch?.kernelPath?.postLayer?.find((step) => step.op === 'lm_head_prefill')?.kernel,
  'matmul_f16w_f32a.wgsl'
);

const finitenessFallback = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
  },
  platform: {
    id: 'test-f16-fallback',
    vendor: 'test',
    architecture: 'test',
  },
  runtimeCompute: {
    rangeAwareSelectiveWidening: {
      enabled: true,
      includeNonFinite: true,
      onTrigger: 'fallback-plan',
      absThreshold: 65500,
    },
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model', 'config'],
    onIncompatible: 'remap',
  },
});

assert.equal(finitenessFallback.fallbackKernelPath, null);

const widenedFallback = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  capabilities: {
    hasSubgroups: true,
    hasF16: false,
    hasSubgroupsF16: false,
  },
  platform: {
    id: 'test-f32-fallback',
    vendor: 'test',
    architecture: 'test',
  },
  runtimeCompute: {
    rangeAwareSelectiveWidening: {
      enabled: true,
      includeNonFinite: true,
      onTrigger: 'error',
      absThreshold: 65500,
    },
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model', 'config'],
    onIncompatible: 'remap',
  },
});

assert.deepEqual(widenedFallback.appliedTransforms, ['widenToF32Activations']);
assert.equal(widenedFallback.session.compute.defaults.activationDtype, 'f32');
assert.equal(widenedFallback.session.kvcache.kvDtype, 'f32');
assert.equal(
  widenedFallback.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.kernel,
  'attention_streaming.wgsl'
);

console.log('gemma4-e2b-conversion-config-contract.test: ok');
