import assert from 'node:assert/strict';

import { finalizeLogits } from '../../src/inference/pipelines/text/logits/utils.js';

const config = {
  logitOutputScale: 0.5,
  finalLogitSoftcapping: 20,
};
const transformed = await finalizeLogits(
  new Float32Array([40, -40]), 1, 2, 2, config, null
);
assert.ok(Math.abs(transformed[0] - (20 * Math.tanh(1))) < 1e-5);
assert.ok(Math.abs(transformed[1] + (20 * Math.tanh(1))) < 1e-5);

const padded = await finalizeLogits(
  new Float32Array([4]), 1, 1, 2, { logitOutputScale: 0.25, finalLogitSoftcapping: null }, null
);
assert.deepEqual(Array.from(padded), [1, -Infinity]);

await assert.rejects(
  () => finalizeLogits(
    new Float32Array([1]), 1, 1, 1, { logitOutputScale: 0, finalLogitSoftcapping: null }, null
  ),
  /logitOutputScale must be a positive finite number/
);

console.log('logit-output-transform.test: ok');
