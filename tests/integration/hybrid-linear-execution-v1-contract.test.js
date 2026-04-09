import assert from 'node:assert/strict';

const { compileExecutionV1 } = await import('../../src/inference/pipelines/text/execution-v1.js');

const DIGEST = 'sha256:0000000000000000000000000000000000000000000000000000000000000000';

function createSession() {
  return {
    compute: {
      defaults: {
        activationDtype: 'f32',
        mathDtype: 'f32',
        accumDtype: 'f32',
        outputDtype: 'f32',
      },
    },
    kvcache: {
      kvDtype: 'f16',
    },
  };
}

function compileHybridExecution(execution) {
  return compileExecutionV1({
    manifestInference: {
      schema: 'doppler.execution/v1',
      session: createSession(),
      layerPattern: {
        type: 'custom',
        layerTypes: ['linear_attention', 'full_attention'],
      },
      execution,
    },
    modelId: 'unit-qwen-hybrid',
    numLayers: 2,
    runtimeSession: createSession(),
    runtimeCompute: { activationDtype: 'f32' },
    kernelPathPolicy: {
      mode: 'capability-aware',
      sourceScope: ['manifest', 'model', 'config'],
      allowSources: ['manifest', 'model', 'config'],
      onIncompatible: 'remap',
    },
  });
}

const unsafeHybridExecution = {
  inlineKernelPath: true,
  kernels: {
    dense_decode: {
      kernel: 'matmul_gemv_subgroup.wgsl',
      entry: 'main_multicol',
      digest: DIGEST,
    },
    dense_prefill: {
      kernel: 'matmul_f16w_f32a.wgsl',
      entry: 'main',
      digest: DIGEST,
    },
  },
  preLayer: [],
  decode: [
    ['q_proj', 'dense_decode', 'layer.{L}.self_attn.q_proj'],
    ['o_proj', 'dense_decode', 'layer.{L}.self_attn.o_proj'],
  ],
  prefill: [
    ['q_proj', 'dense_prefill', 'layer.{L}.self_attn.q_proj'],
    ['o_proj', 'dense_prefill', 'layer.{L}.self_attn.o_proj'],
  ],
  postLayer: [],
};

assert.throws(
  () => compileHybridExecution(unsafeHybridExecution),
  /Hybrid linear-attention model "unit-qwen-hybrid" cannot apply a global non-Q4/
);

const isolatedHybridExecution = {
  inlineKernelPath: true,
  kernels: {
    dense_decode: {
      kernel: 'matmul_gemv_subgroup.wgsl',
      entry: 'main_multicol',
      digest: DIGEST,
    },
    dense_prefill: {
      kernel: 'matmul_f16w_f32a.wgsl',
      entry: 'main',
      digest: DIGEST,
    },
    fused_decode: {
      kernel: 'fused_matmul_q4.wgsl',
      entry: 'main_multicol',
      digest: DIGEST,
    },
    fused_prefill: {
      kernel: 'fused_matmul_q4_batched.wgsl',
      entry: 'main',
      digest: DIGEST,
    },
  },
  preLayer: [],
  decode: [
    ['q_proj', 'dense_decode', 'layer.{L}.self_attn.q_proj'],
    ['o_proj', 'dense_decode', 'layer.{L}.self_attn.o_proj'],
    {
      layers: [0],
      steps: [
        ['qkv_proj', 'fused_decode', 'layer.{L}.linear_attn.in_proj_qkv'],
        ['linear_out_proj', 'fused_decode', 'layer.{L}.linear_attn.out_proj'],
      ],
    },
  ],
  prefill: [
    ['q_proj', 'dense_prefill', 'layer.{L}.self_attn.q_proj'],
    ['o_proj', 'dense_prefill', 'layer.{L}.self_attn.o_proj'],
    {
      layers: [0],
      steps: [
        ['qkv_proj', 'fused_prefill', 'layer.{L}.linear_attn.in_proj_qkv'],
        ['linear_out_proj', 'fused_prefill', 'layer.{L}.linear_attn.out_proj'],
      ],
    },
  ],
  postLayer: [],
};

assert.doesNotThrow(() => compileHybridExecution(isolatedHybridExecution));

console.log('hybrid-linear-execution-v1-contract.test: ok');
