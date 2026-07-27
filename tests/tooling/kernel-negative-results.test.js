import assert from 'node:assert/strict';
import registry from '../../benchmarks/kernels/negative-results.json' with { type: 'json' };
import {
  findKernelNegativeResults,
  validateKernelNegativeResults,
} from '../../src/tooling/kernel-negative-results.js';

const validated = validateKernelNegativeResults(registry);
assert.equal(validated.entries.length, 1);

const matches = findKernelNegativeResults(registry, {
  modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
  candidate: 'useQwenF16PrimaryMatmuls',
  adapterDigest: 'sha256:061268f4db627cde2ee6c032dd277a4d2c2c5c7b38ec63f171a64775deef931a',
  phase: 'decode',
});
assert.equal(matches.length, 1);
assert.equal(matches[0].decision.status, 'rejected');
assert.equal(matches[0].correctness.exactTokenParity, true);
assert.ok(matches[0].measurement.throughputDeltaPercent < 0);
assert.deepEqual(findKernelNegativeResults(registry, { modelId: 'unknown' }), []);

assert.throws(
  () => validateKernelNegativeResults({
    ...registry,
    entries: [
      registry.entries[0],
      registry.entries[0],
    ],
  }),
  /duplicate id/
);

console.log('kernel-negative-results.test: ok');
