import assert from 'node:assert/strict';
import { GENERATION_CONTRACT, resolveGenerationOptions, validateGenerationInput } from '../../src/config/generation-contract.js';
import { sample, applyPresencePenalty, applyRepetitionPenalty } from '../../src/inference/token-sampling.js';

const options = { maxTokens: 3, maxSeqLen: 16, temperature: 0, topP: 1, topK: 0,
  repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false };
const resolved = resolveGenerationOptions(options);
assert.equal(resolved.presencePenalty, 0);
assert.equal(Object.hasOwn(options, 'presencePenalty'), false);
assert(Object.isFrozen(resolved.suppressTokenIds));
assert(Object.isFrozen(GENERATION_CONTRACT.options));
for (const change of [{ maxTokens: 0 }, { maxSeqLen: 0 }, { topK: -1 }, { topK: 1.5 }, { temperature: null },
  { topP: 0 }, { presencePenalty: -1 }, { repetitionPenaltyWindow: -1 }, { repetitionPenaltyWindow: 4294967296 },
  { seed: -1 }, { suppressTokenIds: [4294967295] }, { unsupported: true }, { temperature: 1 }]) {
  assert.throws(() => resolveGenerationOptions({ ...options, ...change }), { code: 'DOPPLER_GENERATION_INVALID_REQUEST' });
}
for (const input of [{}, { prompt: ' ' }, { prompt: 'text', promptTokens: [0] }, { promptTokens: [] }, { promptTokens: [-1] }]) {
  assert.throws(() => validateGenerationInput(input), { code: 'DOPPLER_GENERATION_INVALID_REQUEST' });
}
validateGenerationInput({ promptTokens: [0, 2] });

for (const [window, expected] of [[0, [2, 1, -2]], [1, [6, 1, -2]], [2, [2, 1, -2]]]) {
  const logits = new Float32Array([6, 4, -2]);
  applyRepetitionPenalty(logits, [0, 0, 1], 2, window);
  applyPresencePenalty(logits, [0, 0, 1], 1, window);
  assert.deepEqual([...logits], expected);
}
// Nucleus selection consumes the normalized top-k distribution, matching the
// existing Capsule contract rather than retaining the legacy pipeline drift.
for (const seed of [0, 1, 2, 7, 42]) {
  assert.equal(sample(Float32Array.from([4, 3, 2, 1]), { temperature: 1, topK: 2, topP: 0.7, seed }), 0);
}
assert.equal(sample(Float32Array.from([NaN, 3, Infinity, 2]), { temperature: 1, topK: 0, topP: 1, seed: 0 }), 1);
assert.throws(() => sample(Float32Array.from([1, 2]), { temperature: 0, topK: 0, topP: 1, suppressTokenIds: [0, 1] }), /no finite candidate/);
console.log('token-sampling-contract: canonical validation, penalty window and filter order passed');
