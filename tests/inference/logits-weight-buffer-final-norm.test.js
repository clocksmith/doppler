import assert from 'node:assert/strict';

import { getLogitsWeights } from '../../src/inference/pipelines/text/generator-helpers.js';

const finalNormBuffer = {
  buffer: { __dopplerFakeGPUBuffer: true, size: 8, usage: 0 },
  dtype: 'f16',
  layout: 'row',
  shape: Object.freeze([4]),
  label: 'final_norm',
};

const lmHeadBuffer = {
  buffer: { __dopplerFakeGPUBuffer: true, size: 32, usage: 0 },
  dtype: 'f16',
  layout: 'row',
  shape: Object.freeze([4, 4]),
  label: 'lm_head',
};

const weights = new Map([
  ['final_norm', finalNormBuffer],
  ['lm_head', lmHeadBuffer],
]);

const resolved = getLogitsWeights({ weights });

assert.equal(resolved.finalNorm, finalNormBuffer);
assert.equal(resolved.lmHead, lmHeadBuffer);

console.log('logits-weight-buffer-final-norm.test: ok');
