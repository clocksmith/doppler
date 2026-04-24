import assert from 'node:assert/strict';

import { resolveLayerScalarValue } from '../../src/inference/pipelines/text/layer.js';

assert.equal(resolveLayerScalarValue(null), 1);
assert.equal(resolveLayerScalarValue(undefined), 1);
assert.equal(resolveLayerScalarValue(new Float32Array([0.125])), 0.125);

assert.throws(
  () => resolveLayerScalarValue(new Float32Array()),
  /layer_scalar must be CPU-resident Float32Array data/
);

assert.throws(
  () => resolveLayerScalarValue(new Float32Array([Number.NaN])),
  /layer_scalar must be finite/
);

console.log('layer-scalar-contract.test: ok');
