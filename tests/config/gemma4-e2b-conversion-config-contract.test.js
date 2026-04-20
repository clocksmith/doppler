import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { getLayerSteps } from '../../src/config/kernel-path-loader.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

const configPath = path.join(
  repoRoot,
  'src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json'
);
const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
const int4PleConfigPath = path.join(
  repoRoot,
  'src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json'
);
const int4PleConfig = JSON.parse(await fs.readFile(int4PleConfigPath, 'utf8'));

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
assert.equal(config.execution?.kernels?.attn_decode?.kernel, 'attention_decode_online_f16kv.wgsl');
assert.equal(config.execution?.kernels?.attn_decode?.entry, 'main');
assert.equal(config.execution?.kernels?.attn_decode?.precision?.kvDtype, 'f16');
assert.equal(config.execution?.kernels?.attn_small?.kernel, 'attention_small_f16kv.wgsl');
assert.equal(config.execution?.kernels?.attn_small?.precision?.kvDtype, 'f16');
assert.equal(config.execution?.kernels?.attn_stream?.kernel, 'attention_streaming_f16kv.wgsl');
assert.equal(config.execution?.kernels?.attn_stream?.precision?.kvDtype, 'f16');
assert.equal(config.execution?.kernels?.attn_head256?.kernel, 'attention_head256_f16kv.wgsl');
assert.equal(config.execution?.kernels?.attn_head256?.precision?.kvDtype, 'f16');
assert.equal(config.execution?.kernels?.attn_head512?.kernel, 'attention_head512_f16kv.wgsl');
assert.equal(config.execution?.kernels?.attn_head512?.precision?.kvDtype, 'f16');
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
assert.equal(
  config.execution?.postLayer?.find((step) => step[0] === 'lm_head')?.[1],
  'lm_head_gemv_stable'
);

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
assert.equal(config.session?.retainQ4KMaterialization, false);
assert.equal(config.session?.perLayerInputs?.materialization, 'gpu_split_tables');
assert.equal(config.session?.perLayerInputs?.hotCache?.mode, 'prepared_tokens');
assert.equal(config.session?.perLayerInputs?.hotCache?.maxTokens, 4096);
assert.equal(config.session?.perLayerInputs?.hotCache?.maxBytes, 268435456);
assert.equal(config.session?.perLayerInputs?.hotCache?.outputDtype, 'f32');

assert.equal(int4PleConfig.output?.baseDir, 'models/local');
assert.equal(int4PleConfig.output?.modelBaseId, 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple');
assert.equal(int4PleConfig.quantization?.weights, 'q4k');
assert.equal(int4PleConfig.quantization?.embeddings, 'f16');
assert.equal(int4PleConfig.quantization?.lmHead, 'f16');
assert.equal(int4PleConfig.quantization?.perLayerEmbeddings, 'int4_per_row');
assert.equal(int4PleConfig.session?.retainQ4KMaterialization, false);
// INT4 PLE rows stay range-backed; conversion rejects split-table materialization
// for packed INT4 PLE tensors.
assert.equal(int4PleConfig.session?.perLayerInputs?.materialization, 'range_backed');

const manifestInference = {
  schema: 'doppler.execution/v1',
  ...config.inference,
  session: config.session,
  execution: config.execution,
};
const modelHeadDim = 256;

const f16Primary = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  headDim: modelHeadDim,
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
  'attention_decode_online_f16kv.wgsl'
);
assert.equal(
  f16Primary.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.precision?.kvDtype,
  'f16'
);
assert.equal(
  f16Primary.runtimeInferencePatch?.kernelPath?.prefill?.steps?.find((step) => step.op === 'attention')?.precision?.kvDtype,
  'f16'
);
assert.equal(
  getLayerSteps(f16Primary.runtimeInferencePatch?.kernelPath, 0, 'prefill')
    .find((step) => step.op === 'attention')?.kernel,
  'attention_head256_f16kv.wgsl'
);
assert.equal(
  getLayerSteps(f16Primary.runtimeInferencePatch?.kernelPath, 4, 'prefill')
    .find((step) => step.op === 'attention')?.kernel,
  'attention_head512_f16kv.wgsl'
);

const appleRetainDisabled = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  headDim: modelHeadDim,
  runtimeSession: {
    retainQ4KMaterialization: true,
  },
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
  },
  platform: {
    id: 'apple-m3',
    vendor: 'apple',
    architecture: 'metal-3',
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

assert.deepEqual(appleRetainDisabled.appliedTransforms, ['disableRetainQ4KMaterialization']);
assert.equal(appleRetainDisabled.session.retainQ4KMaterialization, false);
assert.equal(
  appleRetainDisabled.runtimeInferencePatch.session?.retainQ4KMaterialization,
  false
);

const amdRetainAllowed = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  headDim: modelHeadDim,
  runtimeSession: {
    retainQ4KMaterialization: true,
  },
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
  },
  platform: {
    id: 'amd-rdna3',
    vendor: 'amd',
    architecture: 'rdna3',
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

assert.deepEqual(amdRetainAllowed.appliedTransforms, []);
assert.equal(amdRetainAllowed.session.retainQ4KMaterialization, true);
assert.equal(
  amdRetainAllowed.runtimeInferencePatch.session?.retainQ4KMaterialization,
  true
);

assert.throws(
  () => compileExecutionV1({
    manifestInference,
    modelId: config.output.modelBaseId,
    numLayers: 35,
    headDim: modelHeadDim,
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
  }),
  /attention_head256_f16kv\.wgsl" requires activationDtype="f32" and kvcache\.kvDtype="f16"/
);

const finitenessFallback = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  headDim: modelHeadDim,
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
  headDim: modelHeadDim,
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
  'attention_decode.wgsl'
);

console.log('gemma4-e2b-conversion-config-contract.test: ok');
