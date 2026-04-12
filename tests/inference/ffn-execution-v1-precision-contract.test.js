import assert from 'node:assert/strict';

const {
  buildInlineKernelPath,
  buildLayerPipelineFromExecution,
} = await import('../../src/inference/pipelines/text/execution-runtime-builders.js');
const { getKernelPathMatmulPrecision } = await import('../../src/config/kernel-path-loader.js');
const {
  resolveGateUpPathMode,
  resolveDenseFFNMatmulStepDtype,
  resolveDenseFFNFusedPathDtypes,
  canUseNativeF16FusedGateUp,
} = await import('../../src/inference/pipelines/text/ffn/dense.js');

{
  const pipeline = buildLayerPipelineFromExecution(
    [
      {
        section: 'layer',
        phase: 'decode',
        op: 'ffn',
        src: 'state',
        dst: 'state',
      },
    ],
    { ffnDtypeFallback: 'f16' }
  );
  const step = pipeline?.steps?.[0] ?? null;
  assert.ok(step, 'expected one fused ffn pipeline step');
  assert.equal(step.inputDtype, 'f16');
  assert.equal(step.outputDtype, 'f16');
}

{
  const pipeline = buildLayerPipelineFromExecution(
    [
      {
        section: 'layer',
        phase: 'decode',
        op: 'ffn',
        src: 'state',
        dst: 'state',
        precision: {
          inputDtype: 'f32',
          outputDtype: 'f32',
        },
      },
    ],
    { ffnDtypeFallback: 'f16' }
  );
  const step = pipeline?.steps?.[0] ?? null;
  assert.ok(step, 'expected one explicit fused ffn pipeline step');
  assert.equal(step.inputDtype, 'f32');
  assert.equal(step.outputDtype, 'f32');
}

{
  const fusedFfnKernelPath = buildInlineKernelPath(
    [
      {
        section: 'layer',
        phase: 'decode',
        op: 'ffn',
        kernel: 'gelu.wgsl',
        entry: 'main',
        src: 'state',
        dst: 'state',
        layers: 'all',
      },
      {
        section: 'layer',
        phase: 'prefill',
        op: 'ffn',
        kernel: 'gelu.wgsl',
        entry: 'main',
        src: 'state',
        dst: 'state',
        layers: 'all',
      },
    ],
    {
      compute: {
        defaults: {
          activationDtype: 'f16',
        },
      },
      kvcache: {
        kvDtype: 'f16',
      },
    },
    'gemma4-inline',
    1
  );

  assert.equal(
    fusedFfnKernelPath?.decode?.steps?.[0]?.precision?.outputDtype,
    'f16',
    'inline kernel path should stamp fused ffn output precision from the session activation lane'
  );
  assert.equal(
    getKernelPathMatmulPrecision('ffn_down', 'decode', 0, fusedFfnKernelPath)?.outputDtype,
    'f16',
    'ffn_down precision should inherit the fused ffn step precision when no explicit down step exists'
  );

  assert.equal(
    resolveDenseFFNMatmulStepDtype({
      role: 'ffn_down',
      phase: 'decode',
      layerIdx: 0,
      kernelPath: fusedFfnKernelPath,
      fallback: 'f32',
      field: 'outputDtype',
    }),
    'f16',
    'fused FFN step precision should keep ffn_down on the f16 activation lane'
  );
}

{
  const explicitDownPrecisionKernelPath = {
    id: 'explicit-down-precision',
    decode: {
      steps: [
        {
          op: 'ffn_down',
          kernel: 'matmul_f16w_f32a.wgsl',
          entry: 'main',
          precision: {
            outputDtype: 'f32',
          },
        },
      ],
    },
    prefill: {
      steps: [],
    },
  };

  assert.equal(
    resolveDenseFFNMatmulStepDtype({
      role: 'ffn_down',
      phase: 'decode',
      layerIdx: 0,
      kernelPath: explicitDownPrecisionKernelPath,
      fallback: 'f16',
      field: 'outputDtype',
      ffnStepPrecision: {
        inputDtype: 'f16',
        outputDtype: 'f16',
      },
    }),
    'f32',
    'explicit role precision must override the fused ffn step fallback'
  );
}

{
  assert.equal(
    canUseNativeF16FusedGateUp({
      inputDtype: 'f16',
      gateDtype: 'q4k',
      hasF16: true,
    }),
    true,
    'Q4K f16-activation fused FFN should consume the explicit f16 gate/up input without an extra widening cast'
  );
  assert.equal(
    canUseNativeF16FusedGateUp({
      inputDtype: 'f16',
      gateDtype: 'q4k',
      hasF16: false,
    }),
    false,
    'Q4K f16-activation fused FFN still requires shader-f16 support'
  );
  assert.equal(
    canUseNativeF16FusedGateUp({
      inputDtype: 'f32',
      gateDtype: 'q4k',
      hasF16: true,
    }),
    false,
    'Q4K fused FFN should only stay native on the f16 lane when the declared gate/up input is already f16'
  );
}

{
  const fusedQwenLikeKernelPath = {
    id: 'fused-qwen-like',
    decode: {
      steps: [
        {
          op: 'gate_proj',
          kernel: 'matmul_gemv_subgroup_f16a.wgsl',
          entry: 'main_multicol',
          precision: {
            inputDtype: 'f16',
            outputDtype: 'f16',
          },
        },
        {
          op: 'up_proj',
          kernel: 'matmul_gemv_subgroup_f16a.wgsl',
          entry: 'main_multicol',
          precision: {
            inputDtype: 'f16',
            outputDtype: 'f16',
          },
        },
        {
          op: 'down_proj',
          kernel: 'matmul_gemv_subgroup_f16a.wgsl',
          entry: 'main_multicol',
          precision: {
            inputDtype: 'f16',
            outputDtype: 'f16',
          },
        },
      ],
    },
    prefill: {
      steps: [
        {
          op: 'gate_proj',
          kernel: 'fused_matmul_q4_batched_f16a.wgsl',
          entry: 'main_batched_f16a',
          precision: {
            inputDtype: 'f16',
            outputDtype: 'f16',
          },
        },
        {
          op: 'up_proj',
          kernel: 'fused_matmul_q4_batched_f16a.wgsl',
          entry: 'main_batched_f16a',
          precision: {
            inputDtype: 'f16',
            outputDtype: 'f16',
          },
        },
        {
          op: 'down_proj',
          kernel: 'fused_matmul_q4_batched_f16a.wgsl',
          entry: 'main_batched_f16a',
          precision: {
            inputDtype: 'f16',
            outputDtype: 'f16',
          },
        },
      ],
    },
  };

  assert.deepEqual(
    resolveDenseFFNFusedPathDtypes({
      phase: 'prefill',
      layerIdx: 0,
      kernelPath: fusedQwenLikeKernelPath,
      fallbackInputDtype: 'f32',
      fallbackOutputDtype: 'f32',
    }),
    {
      fusedGateUpInputDtype: 'f16',
      fusedGateUpOutputDtype: 'f32',
      downInputDtype: 'f16',
    },
    'fused FFN path should infer f16 gate/up input and f16 down input even when ffn_gate_up precision is implicit'
  );
  assert.equal(
    resolveGateUpPathMode({
      phase: 'prefill',
      layerIdx: 0,
      kernelPath: fusedQwenLikeKernelPath,
    }),
    'split',
    'explicit FFN step precision should force the split gate/up path'
  );
}

console.log('ffn-execution-v1-precision-contract.test: ok');
