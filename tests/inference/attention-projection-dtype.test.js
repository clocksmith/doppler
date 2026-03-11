import assert from 'node:assert/strict';

import {
  resolveAttentionProjectionOutputDtype,
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

console.log('attention-projection-dtype.test: ok');
