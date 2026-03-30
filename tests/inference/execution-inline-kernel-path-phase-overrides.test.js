import assert from 'node:assert/strict';

const { buildInlineKernelPath } = await import('../../src/inference/pipelines/text/execution-runtime-builders.js');
const { getLayerSteps, getKernelPathAttentionVariant } = await import('../../src/config/kernel-path-loader.js');

const session = {
  compute: {
    defaults: {
      activationDtype: 'f32',
    },
  },
  kvcache: {
    kvDtype: 'f16',
  },
};

const steps = [
  {
    section: 'layer',
    phase: 'decode',
    op: 'attention',
    kernel: 'attention_decode_online_f16kv.wgsl',
    entry: 'main',
    layers: 'all',
  },
  {
    section: 'layer',
    phase: 'prefill',
    op: 'attention',
    kernel: 'attention_streaming_f16kv.wgsl',
    entry: 'main',
    layers: 'all',
  },
  {
    section: 'layer',
    phase: 'decode',
    op: 'q_proj',
    kernel: 'fused_matmul_q4.wgsl',
    entry: 'main_multicol',
    weights: 'layer.{L}.self_attn.q_proj',
    precision: {
      outputDtype: 'f16',
    },
    layers: [1],
  },
];

const kernelPath = buildInlineKernelPath(steps, session, 'phase-override-test', 2);

assert.ok(kernelPath?.layerOverrides?.length === 1);
assert.equal(Array.isArray(kernelPath.layerOverrides[0].steps), false);
assert.equal(kernelPath.layerOverrides[0].layers[0], 1);
assert.equal(getLayerSteps(kernelPath, 1, 'decode').find((step) => step.op === 'q_proj')?.precision?.outputDtype, 'f16');
assert.equal(getLayerSteps(kernelPath, 1, 'prefill').find((step) => step.op === 'q_proj'), undefined);
assert.equal(
  getLayerSteps(kernelPath, 1, 'prefill').find((step) => step.op === 'attention')?.kernel,
  'attention_streaming_f16kv.wgsl'
);
assert.equal(
  getLayerSteps(kernelPath, 1, 'decode').find((step) => step.op === 'attention')?.kernel,
  'attention_decode_online_f16kv.wgsl'
);
assert.equal(getKernelPathAttentionVariant('prefill', 1, kernelPath), 'prefill_streaming_f16kv');
assert.equal(getKernelPathAttentionVariant('decode', 1, kernelPath), 'decode_online_f16kv');

console.log('execution-inline-kernel-path-phase-overrides.test: ok');
