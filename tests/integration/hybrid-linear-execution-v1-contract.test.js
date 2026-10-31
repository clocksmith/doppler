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

function compileHybridExecution(execution, weightDtype = null) {
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
    weightDtype,
    runtimeSession: createSession(),
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

assert.doesNotThrow(
  () => compileHybridExecution(unsafeHybridExecution, 'f16'),
  'A global F16 path is safe when every stored projection weight is F16.'
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

// Native GGUF: global projection programs are judged against the declared block format.
const ggufProgram = (entry, format, bytes) => ({ kernel: 'matmul_q4_0.wgsl', entry, digest: DIGEST,
  constants: { WORKGROUP_SIZE: 128, COLS_PER_WG: 4, THREADS_PER_COL_GEMV: 32, TILE_SIZE: 16,
    BLOCK_VALUES: 32, BLOCK_BYTES: bytes, QUANT_FORMAT: format } });
const nativeHybrid = (format, bytes) => ({
  inlineKernelPath: true,
  kernels: { native_decode: ggufProgram('gemv', format, bytes), native_prefill: ggufProgram('prefill', format, bytes) },
  preLayer: [],
  decode: [['q_proj', 'native_decode', 'layer.{L}.self_attn.q_proj'], ['o_proj', 'native_decode', 'layer.{L}.self_attn.o_proj']],
  prefill: [['q_proj', 'native_prefill', 'layer.{L}.self_attn.q_proj'], ['o_proj', 'native_prefill', 'layer.{L}.self_attn.o_proj']],
  postLayer: [],
});
assert.doesNotThrow(() => compileHybridExecution(nativeHybrid(0, 18), 'q4_0'),
  'A global Q4_0 program matches Q4_0 linear-attention storage.');
assert.throws(() => compileHybridExecution(nativeHybrid(2, 34), 'q4_0'),
  /cannot apply a global non-Q4/, 'A global Q8_0 program would be bound to Q4_0 linear-attention weights.');

const pairedHybrid = structuredClone(unsafeHybridExecution);
for (const key of ['dense_decode', 'dense_prefill']) pairedHybrid.kernels[key] = {
  kernel: 'matmul_paired.wgsl', entry: 'main', digest: DIGEST,
  constants: { WEIGHT_F16: false, WEIGHT_Q4K: true, SCALAR_OUTPUT: false },
};
assert.doesNotThrow(() => compileHybridExecution(pairedHybrid, 'q4k'),
  'Declared paired Q4 reads the same retained Q4 weight encoding.');
const scalarWeightHybrid = structuredClone(pairedHybrid);
scalarWeightHybrid.kernels.dense_decode.constants.WEIGHT_Q4K = false;
assert.throws(() => compileHybridExecution(scalarWeightHybrid, 'q4k'), /global non-Q4/,
  'The same shader with F32 weight constants must not count as Q4 coverage.');
console.log('hybrid-linear-execution-v1-contract.test: ok');
