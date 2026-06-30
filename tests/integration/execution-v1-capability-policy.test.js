import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const {
  applyExecutionV1RuntimeConfig,
  compileExecutionV1,
} = await import('../../src/inference/pipelines/text/execution-v1.js');
const {
  applyModelBatchingRuntimeDefaults,
  resolveKernelPathState,
} = await import('../../src/inference/pipelines/text/model-load.js');

const kernelRegistry = JSON.parse(readFileSync('src/config/kernels/registry.json', 'utf8'));
const shaderF16KernelFiles = new Set();
for (const operation of Object.values(kernelRegistry.operations ?? {})) {
  for (const variant of Object.values(operation.variants ?? {})) {
    if (Array.isArray(variant.requires) && variant.requires.includes('shader-f16')) {
      shaderF16KernelFiles.add(variant.wgsl);
    }
  }
}

function collectKernelPathSteps(path) {
  return [
    ...(path?.decode?.steps ?? []),
    ...(path?.prefill?.steps ?? []),
    ...(path?.layerOverrides?.flatMap((override) => [
      ...(override?.steps ?? []),
      ...(override?.decode?.steps ?? []),
      ...(override?.prefill?.steps ?? []),
    ]) ?? []),
  ];
}

function assertNoShaderF16KernelSteps(path, label) {
  const blocked = collectKernelPathSteps(path)
    .filter((step) => shaderF16KernelFiles.has(step.kernel))
    .map((step) => `${step.op}:${step.kernel}`);
  assert.deepEqual(blocked, [], label);
}

const SUBGROUP_MANIFEST = {
  schema: 'doppler.execution/v1',
  session: {
    compute: {
      defaults: {
        activationDtype: 'f16',
        mathDtype: 'f16',
        accumDtype: 'f32',
        outputDtype: 'f16',
      },
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 1,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      disableCommandBatching: false,
    },
  },
  execution: {
    inlineKernelPath: true,
    kernels: {
      proj: {
        kernel: 'matmul_gemv_subgroup_f16a.wgsl',
        entry: 'main_vec4',
        digest: 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      },
      attn: {
        kernel: 'attention_decode_online_f16.wgsl',
        entry: 'main',
        digest: 'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
      },
    },
    decode: [
      ['q_proj', 'proj', 'layer.{L}.self_attn.q_proj'],
      ['attention', 'attn'],
    ],
    prefill: [
      ['q_proj', 'proj', 'layer.{L}.self_attn.q_proj'],
    ],
    preLayer: [],
    postLayer: [],
    policies: {
      unsupportedPrecision: 'error',
      dtypeTransition: 'require_cast_step',
      unresolvedKernel: 'error',
    },
  },
};

assert.throws(
  () => compileExecutionV1({
    manifestInference: SUBGROUP_MANIFEST,
    modelId: 'test-subgroup-model',
    numLayers: 1,
    kernelPathPolicy: {
      mode: 'locked',
      sourceScope: ['manifest', 'model'],
      onIncompatible: 'error',
    },
    capabilities: {
      hasSubgroups: false,
      hasF16: true,
      hasSubgroupsF16: false,
    },
    platform: {
      id: 'test-gpu',
      vendor: 'test',
      architecture: 'test',
    },
  }),
  /capability transforms required/
);

const gemma3Config = JSON.parse(
  readFileSync('src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json', 'utf8')
);

const gemma3NoF16 = compileExecutionV1({
  manifestInference: {
    ...gemma3Config.inference,
    schema: 'doppler.execution/v1',
    session: gemma3Config.session,
    execution: gemma3Config.execution,
  },
  modelId: 'gemma-3-1b-it-q4k-ehf16-af32',
  numLayers: 26,
  headDim: 256,
  capabilities: {
    hasSubgroups: false,
    hasF16: false,
    hasSubgroupsF16: false,
  },
  platform: {
    id: 'test-gpu',
    vendor: 'test',
    architecture: 'test',
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model'],
    onIncompatible: 'remap',
  },
});

const gemma3DecodeAttention = gemma3NoF16.runtimeInferencePatch.kernelPath.decode.steps.find(
  (step) => step.op === 'attention'
);
assert.equal(gemma3DecodeAttention?.kernel, 'attention_streaming.wgsl');
assert.equal(gemma3DecodeAttention?.entry, 'main');

// Lane integrity: af32 manifest on hasF16=false GPU. The manifest declares
// f32 activations + f16 KV (mixed lane); widenToF32Activations on hasF16=false
// forces KV to f32 too. The integrity stamp must reflect the executed KV
// dtype, not the declared one — this is exactly the case receipts need to
// honestly surface.
assert.ok(gemma3NoF16.laneIntegrity, 'compileExecutionV1 returns laneIntegrity');
assert.equal(gemma3NoF16.laneIntegrity.declared.activationDtype, 'f32');
assert.equal(gemma3NoF16.laneIntegrity.declared.kvDtype, 'f16');
assert.equal(gemma3NoF16.laneIntegrity.executed.activationDtype, 'f32');
assert.equal(gemma3NoF16.laneIntegrity.executed.kvDtype, 'f32');
assert.equal(gemma3NoF16.laneIntegrity.status, 'transformed',
  'fullF32 widening on hasF16=false flips KV dtype f16→f32 — must show as transformed');
assert.ok(Array.isArray(gemma3NoF16.laneIntegrity.transforms));
assert.ok(gemma3NoF16.laneIntegrity.transforms.includes('widenToF32Activations'));

const qwen08Config = JSON.parse(
  readFileSync('src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json', 'utf8')
);

const qwen08NoF16 = compileExecutionV1({
  manifestInference: {
    ...qwen08Config.inference,
    schema: 'doppler.execution/v1',
    session: qwen08Config.session,
    execution: qwen08Config.execution,
  },
  modelId: qwen08Config.output.modelBaseId,
  numLayers: qwen08Config.inference.layerPattern.layerTypes.length,
  headDim: 256,
  capabilities: {
    hasSubgroups: true,
    hasF16: false,
    hasSubgroupsF16: false,
  },
  platform: {
    id: 'swiftshader',
    vendor: 'google',
    architecture: 'swiftshader',
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model'],
    onIncompatible: 'remap',
  },
});

assert.ok(qwen08NoF16.appliedTransforms.includes('widenToF32Activations'),
  'Qwen 0.8B no-f16 compile must apply the manifest capability widening transform');
assert.equal(qwen08NoF16.laneIntegrity.declared.kvDtype, 'f16');
assert.equal(qwen08NoF16.laneIntegrity.executed.kvDtype, 'f32',
  'Qwen 0.8B no-f16 compile must widen f16 KV to f32');
assert.equal(
  qwen08NoF16.runtimeInferencePatch.kernelPath.prefill.steps.find((step) => step.op === 'q_proj')?.kernel,
  'fused_matmul_q4_batched_multicol_shared.wgsl',
  'Qwen 0.8B no-f16 prefill q_proj must remap off the shader-f16 WideTile kernel'
);
assert.equal(
  qwen08NoF16.runtimeInferencePatch.kernelPath.prefill.steps.find((step) => step.op === 'gate_proj')?.kernel,
  'fused_matmul_q4_batched_multicol_shared.wgsl',
  'Qwen 0.8B no-f16 prefill gate_proj must remap off the shader-f16 WideTile kernel'
);
assertNoShaderF16KernelSteps(
  qwen08NoF16.runtimeInferencePatch.kernelPath,
  'Qwen 0.8B no-f16 compile must not emit shader-f16 kernel path steps'
);

{
  const qwen08Manifest = {
    modelId: qwen08Config.output.modelBaseId,
    architecture: {
      numLayers: qwen08Config.inference.layerPattern.layerTypes.length,
      headDim: 256,
    },
    inference: {
      ...qwen08Config.inference,
      schema: 'doppler.execution/v1',
      session: qwen08Config.session,
      execution: qwen08Config.execution,
    },
  };
  const runtimeOverrides = {
    inference: {
      kernelPathPolicy: {
        mode: 'capability-aware',
        sourceScope: ['manifest', 'model'],
        onIncompatible: 'remap',
      },
    },
  };
  const phase1 = applyExecutionV1RuntimeConfig({
    runtimeConfig: structuredClone(runtimeOverrides),
    runtimeOverrides,
    manifest: qwen08Manifest,
    capabilities: {
      hasSubgroups: true,
      hasF16: false,
      hasSubgroupsF16: false,
    },
    platform: {
      id: 'swiftshader',
      vendor: 'google',
      architecture: 'swiftshader',
    },
  });
  const phase2Runtime = applyModelBatchingRuntimeDefaults(
    phase1.runtimeConfig,
    qwen08Manifest,
    null,
    runtimeOverrides
  );
  assert.equal(
    phase2Runtime.inference.session.kvcache.kvDtype,
    'f32',
    'Qwen 0.8B no-f16 batching defaults must preserve the execution-v1 f32 KV session'
  );
  assert.equal(
    phase2Runtime.inference.session.compute.defaults.outputDtype,
    'f32',
    'Qwen 0.8B no-f16 batching defaults must preserve the execution-v1 f32 output session'
  );
  assert.doesNotThrow(
    () => resolveKernelPathState({
      manifest: qwen08Manifest,
      runtimeConfig: phase2Runtime,
      runtimeOverrides,
      modelConfig: { kernelPath: null },
      kernelCapabilities: {
        hasSubgroups: true,
        hasF16: false,
        hasSubgroupsF16: false,
      },
    }),
    'Qwen 0.8B no-f16 kernel-path dtype validation must see the execution-v1 f32 session'
  );
}

// Lane integrity: synthetic af16-style manifest dispatched on a GPU lacking
// shader-f16. The capability resolver must install widenToF32Activations and
// the lane integrity stamp must record the declared→executed delta so receipts
// honestly reflect that an af16-modelId session ran on f32 shaders.
const F16_MANIFEST = {
  schema: 'doppler.execution/v1',
  session: {
    compute: {
      defaults: {
        activationDtype: 'f16',
        mathDtype: 'f16',
        accumDtype: 'f16',
        outputDtype: 'f16',
      },
    },
    kvcache: {
      kvDtype: 'f16',
      layout: 'contiguous',
      pageSize: 256,
      tiering: { mode: 'off' },
    },
    decodeLoop: {
      batchSize: 1,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      disableCommandBatching: false,
    },
  },
  execution: {
    inlineKernelPath: false,
    kernels: {
      norm: {
        kernel: 'rmsnorm_f16.wgsl',
        entry: 'main',
        digest: 'sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
      },
    },
    decode: [
      ['rms_norm', 'norm', 'layer.{L}.input_norm'],
    ],
    prefill: [
      ['rms_norm', 'norm', 'layer.{L}.input_norm'],
    ],
    preLayer: [],
    postLayer: [],
    policies: {
      unsupportedPrecision: 'error',
      dtypeTransition: 'require_cast_step',
      unresolvedKernel: 'error',
    },
  },
};

const widenedF16 = compileExecutionV1({
  manifestInference: F16_MANIFEST,
  modelId: 'synthetic-af16-test',
  numLayers: 1,
  capabilities: {
    hasSubgroups: true,
    hasF16: false,
    hasSubgroupsF16: false,
  },
  platform: { id: 'test-gpu', vendor: 'test', architecture: 'test' },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model'],
    onIncompatible: 'remap',
  },
});

assert.ok(widenedF16.appliedTransforms.includes('widenToF32Activations'),
  'capability resolver must widen f16 activations on hasF16=false');
assert.equal(widenedF16.laneIntegrity.declared.activationDtype, 'f16',
  'declared lane reflects manifest activation dtype');
assert.equal(widenedF16.laneIntegrity.executed.activationDtype, 'f32',
  'executed lane reflects post-widening dispatch dtype');
assert.equal(widenedF16.laneIntegrity.declared.kvDtype, 'f16');
assert.equal(widenedF16.laneIntegrity.executed.kvDtype, 'f32',
  'fullF32 widening forces kvDtype to f32');
assert.equal(widenedF16.laneIntegrity.status, 'transformed',
  'laneIntegrity.status flags capability-driven dtype divergence for receipts');

const f16PrimaryWithoutFallbackPolicy = compileExecutionV1({
  manifestInference: F16_MANIFEST,
  modelId: 'synthetic-af16-test',
  numLayers: 1,
  headDim: 256,
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
  },
  platform: { id: 'test-gpu', vendor: 'test', architecture: 'test' },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model'],
    onIncompatible: 'remap',
  },
});

assert.equal(
  f16PrimaryWithoutFallbackPolicy.fallbackKernelPath,
  null,
  'f16 primary path must not compile fallback metadata without explicit fallback-plan policy'
);

const gemma4W4A16QatConfig = JSON.parse(
  readFileSync('src/config/conversion/gemma4/gemma-4-12b-it-text-w4a16-ct-ehf16-af16.json', 'utf8')
);

const gemma4W4A16QatWithFallbackPolicy = compileExecutionV1({
  manifestInference: {
    ...gemma4W4A16QatConfig.inference,
    schema: 'doppler.execution/v1',
    session: gemma4W4A16QatConfig.session,
    execution: gemma4W4A16QatConfig.execution,
  },
  modelId: gemma4W4A16QatConfig.output.modelBaseId,
  numLayers: 48,
  headDim: 256,
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
  },
  platform: { id: 'test-gpu', vendor: 'test', architecture: 'test' },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model'],
    onIncompatible: 'remap',
  },
  runtimeCompute: {
    rangeAwareSelectiveWidening: {
      enabled: true,
      includeNonFinite: true,
      absThreshold: 65500,
      onTrigger: 'fallback-plan',
    },
  },
});

assert.ok(
  gemma4W4A16QatWithFallbackPolicy.fallbackKernelPath,
  'Gemma 4 12B W4A16 QAT fallback-plan policy must compile an alternate finiteness kernel path'
);
assert.equal(gemma4W4A16QatWithFallbackPolicy.fallbackKernelPath.activationDtype, 'f32');
assert.equal(gemma4W4A16QatWithFallbackPolicy.fallbackKernelPath.kvDtype, 'f16');
assert.equal(
  gemma4W4A16QatWithFallbackPolicy.runtimeInferencePatch.kernelPath.finitenessFallbackKernelPathId,
  gemma4W4A16QatWithFallbackPolicy.fallbackKernelPath.id,
  'primary inline path must point at the explicit finiteness fallback path id'
);

console.log('execution-v1-capability-policy.test: ok');
