import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const { compileExecutionV1 } = await import('../../src/inference/pipelines/text/execution-v1.js');

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
    runtimeCompute: {
      rangeAwareSelectiveWidening: {
        enabled: true,
        includeNonFinite: true,
        onTrigger: 'error',
        absThreshold: 65500,
      },
    },
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

console.log('execution-v1-capability-policy.test: ok');
