import assert from 'node:assert/strict';

import { extractTokenEmbeddingsFromHidden } from '../../src/inference/pipelines/text/generator-runtime.js';

const hidden = new Float32Array([1, 3, 2, 6]);
const weights = new Float32Array([1, 1]);
const baseConfig = {
  normalizationType: 'layernorm',
  finalNormBiasTensor: null,
  rmsNormEps: 1e-5,
  rmsNormWeightOffset: false,
};

const biasless = extractTokenEmbeddingsFromHidden(hidden, 2, 2, weights, baseConfig);
assert.ok(Math.abs(biasless[0] + 0.999995) < 1e-5);
assert.ok(Math.abs(biasless[1] - 0.999995) < 1e-5);
assert.ok(Math.abs(biasless[2] + 0.99999875) < 1e-5);
assert.ok(Math.abs(biasless[3] - 0.99999875) < 1e-5);

assert.throws(
  () => extractTokenEmbeddingsFromHidden(
    hidden,
    2,
    2,
    weights,
    { ...baseConfig, finalNormBiasTensor: 'model.norm.bias' }
  ),
  /declares bias tensor "model\.norm\.bias" but it was not loaded/
);

const biased = extractTokenEmbeddingsFromHidden(
  hidden,
  2,
  2,
  weights,
  { ...baseConfig, finalNormBiasTensor: 'model.norm.bias' },
  new Float32Array([0.25, -0.5])
);
assert.ok(Math.abs(biased[0] + 0.749995) < 1e-5);
assert.ok(Math.abs(biased[1] - 0.499995) < 1e-5);

console.log('layernorm-bias-contract.test: ok');
