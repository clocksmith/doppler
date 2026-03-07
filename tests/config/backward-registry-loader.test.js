import assert from 'node:assert/strict';

import { loadBackwardRegistry } from '../../src/config/backward-registry-loader.js';

const registry = loadBackwardRegistry();

assert.equal(Object.isFrozen(registry), true);
assert.equal(Object.isFrozen(registry.ops), true);
assert.equal(Object.isFrozen(registry.ops.matmul), true);

assert.throws(
  () => {
    registry.ops.matmul.backward = 'mutated_backward';
  },
  /read only|Cannot assign to read only property/i
);

const reloaded = loadBackwardRegistry();
assert.equal(reloaded.ops.matmul.backward, 'matmul_backward');

console.log('backward-registry-loader.test: ok');
