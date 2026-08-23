import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const profileUrl = new URL(
  '../../src/config/runtime/profiles/qwen3-8-27b-source-parity.json',
  import.meta.url,
);
const profile = JSON.parse(await fs.readFile(profileUrl, 'utf8'));

assert.equal(profile.intent, 'verify');
assert.equal(profile.model, 'qwen3-8-27b-text-q4k-ehaf16');
assert.deepEqual(profile.runtime.shared.harness.referenceTranscript, {
  enabled: true,
  captureLogits: true,
  captureKvBytes: true,
});
assert.equal(profile.runtime.inference.prompt, 'The capital of France is');
assert.deepEqual(profile.runtime.inference.chatTemplate, {
  enabled: false,
  type: null,
  thinking: false,
});
assert.deepEqual(profile.runtime.inference.generation, {
  maxTokens: 128,
  disableMultiTokenDecode: true,
});
assert.deepEqual(profile.runtime.inference.sampling, {
  temperature: 0,
  topK: 1,
  topP: 1,
  repetitionPenalty: 1,
  repetitionPenaltyWindow: 1,
  greedyThreshold: 0,
  suppressSpecialTokens: false,
  suppressSpecialLikeTokens: false,
  suppressTokenIds: [],
});
assert.equal(profile.runtime.inference.batching.batchSize, 1);
assert.equal(profile.runtime.inference.batching.readbackInterval, 1);
assert.equal(profile.runtime.inference.batching.stopCheckMode, 'per-token');
assert.equal(profile.runtime.inference.session.decodeLoop.batchSize, 1);
assert.equal(profile.runtime.inference.session.decodeLoop.maxBatchDecodeTokens, 1);
assert.deepEqual(profile.runtime.inference.kernelPathPolicy, {
  mode: 'capability-aware',
  sourceScope: ['manifest'],
  allowSources: ['manifest'],
  onIncompatible: 'error',
});
assert.equal(
  profile.runtime.inference.executionPatch,
  undefined,
  'Source parity must qualify the manifest graph without a runtime execution patch.',
);

console.log('qwen38-source-parity-profile.test: ok');
