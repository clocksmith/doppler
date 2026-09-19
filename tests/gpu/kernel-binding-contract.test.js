import assert from 'node:assert/strict';
import { createKernelBindingEntries } from '../../src/gpu/kernels/kernel-bindings.js';
import { KERNEL_CONFIGS } from '../../src/config/kernel-registry-contract.js';
import { resolveKernelConfig } from '../../src/config/schema/kernel-registry.schema.js';
import { getRegistry, resolveKernelConfig as resolveLoaded } from '../../src/config/kernels/registry.js';

const registry = await getRegistry();
for (const operation of ['matmul', 'sample', 'rmsnorm_stats']) {
  for (const [variant, config] of Object.entries(KERNEL_CONFIGS[operation])) {
    const resources = Object.fromEntries(config.bindings.map((binding) => [binding.name, { buffer: {}, offset: binding.index * 256, size: 32 }]));
    const entries = createKernelBindingEntries(config, resources);
    for (const binding of config.bindings) {
      assert.equal(entries.find((entry) => entry.binding === binding.index).resource, resources[binding.name]);
      const missing = { ...resources };
      delete missing[binding.name];
      assert.throws(() => createKernelBindingEntries(config, missing), /requires a GPUBuffer/);
    }
    assert.deepEqual(resolveLoaded(operation, variant).bindings, config.bindings, 'both registry surfaces share resolution');
  }
}
const config = { operation: 'test', variant: 'moved', bindings: [
  { name: 'input', index: 2, type: 'read-only-storage' },
  { name: 'output', index: 5, type: 'storage' },
  { name: 'optional', index: 7, type: 'read-only-storage', optional: true },
] };
assert.deepEqual(createKernelBindingEntries(config, { input: { buffer: {} }, output: { buffer: {} } }).map((entry) => entry.binding), [2, 5]);
assert.throws(() => resolveKernelConfig('matmul', 'invalid', registry.operations.matmul, {
  ...registry.operations.matmul.variants.f32, bindingsOverride: [],
}), /both bindings and bindingsOverride/);
console.log('kernel-binding-contract.test: named roles, moved/optional resources, resolver agreement passed');
