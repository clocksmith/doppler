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

console.log('execution-v1-capability-policy.test: ok');
