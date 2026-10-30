import assert from 'node:assert/strict';
import { getVectorElementCount } from '../../src/loader/layer-weight-shape.js';
assert.equal(getVectorElementCount([48], 'router scale'), 48);
assert.equal(getVectorElementCount([2, 3], 'router scale'), 6);
for (const shape of [null, [], [0], [-1], [1.5], [NaN]]) {
  assert.throws(() => getVectorElementCount(shape, 'router scale'), /router scale/);
}
console.log('layer-weight-shape.test: ok');
