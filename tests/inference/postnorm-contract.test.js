import assert from 'node:assert/strict';

import {
  postNormContractMatchesBase,
  resolvePostNormContract,
} from '../../src/inference/pipelines/text/normalization-contract.js';

assert.deepEqual(resolvePostNormContract({
  rmsNormEps: 1e-5,
  rmsNormWeightOffset: true,
  postNormEps: 1e-8,
  postNormWeightOffset: true,
}), {
  postNormEps: 1e-8,
  postNormWeightOffset: true,
});
assert.equal(postNormContractMatchesBase({
  rmsNormEps: 1e-5,
  rmsNormWeightOffset: true,
  postNormEps: 1e-8,
  postNormWeightOffset: true,
}), false);
assert.equal(postNormContractMatchesBase({
  rmsNormEps: 1e-5,
  rmsNormWeightOffset: true,
  postNormEps: null,
  postNormWeightOffset: null,
}), true);
assert.throws(
  () => resolvePostNormContract({
    rmsNormEps: 1e-5,
    rmsNormWeightOffset: false,
    postNormEps: 0,
  }),
  /postNormEps must be null or a positive finite number/
);

console.log('postnorm-contract.test: ok');
