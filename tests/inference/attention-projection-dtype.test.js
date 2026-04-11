import assert from 'node:assert/strict';

import {
  resolveAttentionProjectionOutputDtype,
  resolveProjectionMatmulDtype,
  shouldForceF32AttentionProjectionForRoPE,
} from '../../src/inference/pipelines/text/attention/projections.js';

assert.equal(
  shouldForceF32AttentionProjectionForRoPE({
    attentionInputDtype: 'f16',
    headDim: 128,
    rotaryDim: 32,
    interleaved: true,
  }),
  true
);

assert.equal(
  shouldForceF32AttentionProjectionForRoPE({
    attentionInputDtype: 'f16',
    headDim: 128,
    rotaryDim: 32,
    interleaved: false,
  }),
  true
);

assert.equal(
  shouldForceF32AttentionProjectionForRoPE({
    attentionInputDtype: 'f16',
    headDim: 128,
    rotaryDim: 128,
    interleaved: false,
  }),
  false
);

assert.equal(
  resolveAttentionProjectionOutputDtype('f16', { forceF32: true }),
  'f32'
);

assert.equal(
  resolveAttentionProjectionOutputDtype('f16', { forceF32: false }),
  'f16'
);

assert.equal(
  resolveAttentionProjectionOutputDtype('f32', { forceF32: true }),
  'f32'
);

const splitKernelPath = {
  prefill: {
    steps: [
      { op: 'q_proj', precision: { inputDtype: 'f16', outputDtype: 'f16' } },
      { op: 'k_proj', precision: { inputDtype: 'f16', outputDtype: 'f16' } },
      { op: 'v_proj', precision: { inputDtype: 'f16', outputDtype: 'f16' } },
    ],
  },
};

assert.equal(
  resolveProjectionMatmulDtype({
    useFusedQKV: false,
    phase: 'prefill',
    layerIdx: 0,
    kernelPath: splitKernelPath,
    precisionField: 'outputDtype',
    fallbackDtype: 'f32',
  }),
  'f16'
);

const fusedKernelPath = {
  decode: {
    steps: [
      { op: 'qkv_proj', precision: { inputDtype: 'f16', outputDtype: 'f16' } },
    ],
  },
};

assert.equal(
  resolveProjectionMatmulDtype({
    useFusedQKV: true,
    phase: 'decode',
    layerIdx: 0,
    kernelPath: fusedKernelPath,
    precisionField: 'outputDtype',
    fallbackDtype: 'f32',
  }),
  'f16'
);

assert.throws(
  () => resolveProjectionMatmulDtype({
    useFusedQKV: false,
    phase: 'prefill',
    layerIdx: 0,
    kernelPath: {
      prefill: {
        steps: [
          { op: 'q_proj', precision: { outputDtype: 'f16' } },
          { op: 'k_proj', precision: { outputDtype: 'f32' } },
          { op: 'v_proj', precision: { outputDtype: 'f16' } },
        ],
      },
    },
    precisionField: 'outputDtype',
    fallbackDtype: 'f32',
  }),
  /conflicting outputDtype values/
);

console.log('attention-projection-dtype.test: ok');
