import assert from 'node:assert/strict';
import { createKernelRegistry, enterKernelRegistry } from '../../src/config/kernel-registry-contract.js';
import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';

const variant = (wgsl, constants = {}) => ({
  wgsl, entryPoint: 'main', workgroup: [1, 1, 1], wgslOverrides: constants,
});
const registry = createKernelRegistry({ extensions: {
  closure_projection: {
    baseBindings: [], baseUniforms: null,
    variants: { packed: { ...variant('closure_projection.wgsl'), variantMetadata: {
      mechanismKernels: [
        { operation: 'closure_scale', variant: 'first' },
        { operation: 'closure_scale', variant: 'second' },
      ],
    } } },
  },
  closure_scale: {
    baseBindings: [], baseUniforms: null,
    variants: {
      first: variant('closure_scale.wgsl', { FIRST: true }),
      second: variant('closure_scale.wgsl', { FIRST: false }),
    },
  },
} });
const declaration = (kernel, constants) => ({
  kernel, entry: 'main', digest: `sha256:${'a'.repeat(64)}`, ...(constants ? { constants } : {}),
});
const graph = {
  kernels: {
    projection: declaration('closure_projection.wgsl'),
    first: declaration('closure_scale.wgsl', { FIRST: true }),
    second: declaration('closure_scale.wgsl', { FIRST: false }),
  },
  decode: [['q_proj', 'projection']],
  mechanismKernels: ['first', 'second'],
};
const restore = enterKernelRegistry(registry);
try {
  assert.equal(expandExecutionV1(graph).length, 1, 'Mechanisms belong to the closure, not extra semantic steps.');
  for (const key of ['first', 'second']) {
    const missing = structuredClone(graph);
    missing.mechanismKernels = missing.mechanismKernels.filter(value => value !== key);
    assert.throws(() => expandExecutionV1(missing), /requires mechanism/, 'Unreferenced declarations cannot authorize dispatch.');
    const changed = structuredClone(graph);
    changed.kernels[key].constants.FIRST = !changed.kernels[key].constants.FIRST;
    assert.throws(() => expandExecutionV1(changed), /specialization constants/);
  }
  const omitted = structuredClone(graph);
  delete omitted.mechanismKernels;
  assert.throws(() => expandExecutionV1(omitted), /requires mechanism/);
  const wrongEntry = structuredClone(graph);
  wrongEntry.kernels.first.entry = 'unaccepted';
  assert.throws(() => expandExecutionV1(wrongEntry), /requires mechanism/);
  const wrongDigest = structuredClone(graph);
  wrongDigest.kernels.first.digest = 'unpinned';
  assert.throws(() => expandExecutionV1(wrongDigest), /digest must match/);
  const unknownConstant = structuredClone(graph);
  unknownConstant.kernels.first.constants.UNUSED = true;
  assert.throws(() => expandExecutionV1(unknownConstant), /specialization constants/);
  // A sealed older shader identity remains valid; source-byte verification is
  // performed by the existing accepted artifact provider, not the current registry.
  const older = structuredClone(graph);
  older.kernels.first.digest = `sha256:${'b'.repeat(64)}`;
  assert.equal(expandExecutionV1(older).length, 1);
} finally {
  restore();
}
console.log('execution-mechanism-closure.test: ok');
