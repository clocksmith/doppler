import assert from 'node:assert/strict';
import { lookupDraft } from '../../src/inference/pipelines/text/generator/speculative-runtime.js';
import { validateSelfSpeculationConfig } from '../../src/config/schema/speculation-self.schema.js';

assert.deepEqual(lookupDraft([1, 2, 3, 4, 1, 2], 3, 4, 1), [3, 4, 1]);
assert.deepEqual(lookupDraft([1, 2, 3, 4, 1, 2], 1, 4, 1), [3]);
assert.deepEqual(lookupDraft([1, 2, 9, 1, 2, 8, 1, 2], 2, 2, 1), [8, 1]);
assert.deepEqual(lookupDraft([1, 2, 3], 3, 4, 1), []);
assert.deepEqual(lookupDraft([1], 3, 4, 1), []);
assert.throws(() => lookupDraft([1], 0, 4, 1), /positive/);
assert.throws(() => lookupDraft([1], 3, Infinity, 1), /positive/);
const policy = { mode: 'lookup', tokens: 3, maxMatchTokens: 4, minMatchTokens: 3, maxRejectedBlocks: 1, verify: 'greedy', threshold: null, rollbackOnReject: true };
validateSelfSpeculationConfig(policy);
for (const invalid of [{ maxMatchTokens: 0 }, { maxMatchTokens: undefined }, { rollbackOnReject: false }, { threshold: 0.9 }, { tokens: 0 }, { verify: 'approximate' }]) {
  assert.throws(() => validateSelfSpeculationConfig({ ...policy, ...invalid }));
}
console.log('speculative-lookup: bounded proposal and policy checks passed');

assert.deepEqual(lookupDraft([1, 2, 3, 4, 1, 2], 3, 4, 3), []);
