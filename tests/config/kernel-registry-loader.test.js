import assert from 'node:assert/strict';

import {
  clearRegistryCache,
  getRegistry,
} from '../../src/config/kernels/registry.js';

clearRegistryCache();
const registry = await getRegistry();

assert.equal(Object.isFrozen(registry), true);
assert.equal(Object.isFrozen(registry.operations), true);
assert.equal(Object.isFrozen(registry.operations.matmul), true);
assert.equal(Object.isFrozen(registry.operations.matmul.variants.f16), true);

assert.throws(
  () => {
    registry.operations.matmul.variants.f16.wgsl = 'mutated.wgsl';
  },
  /read only|Cannot assign to read only property/i
);

const reloaded = await getRegistry();
assert.equal(reloaded.operations.matmul.variants.f16.wgsl, 'matmul_f16.wgsl');

console.log('kernel-registry-loader.test: ok');
