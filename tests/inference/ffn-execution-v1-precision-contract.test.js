import assert from 'node:assert/strict';

const {
  buildInlineKernelPath,
  buildLayerPipelineFromExecution,
} = await import('../../src/inference/pipelines/text/execution-runtime-builders.js');
const { getKernelPathMatmulPrecision } = await import('../../src/config/kernel-path-loader.js');
const { resolveDenseFFNMatmulStepDtype } = await import('../../src/inference/pipelines/text/ffn/dense.js');

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

console.log('ffn-execution-v1-precision-contract.test: ok');
