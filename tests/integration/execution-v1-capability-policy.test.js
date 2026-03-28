import assert from 'node:assert/strict';

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

console.log('execution-v1-capability-policy.test: ok');
