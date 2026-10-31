import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolvePrefillTokenChunkSize, shouldDisablePrefillCommandBatching } from '../../src/inference/pipelines/text/generator/prefill-policy.js';
import { _prefillInputIdsToLogits } from '../../src/inference/pipelines/text/generator/prefill-runtime.js';

const profile = JSON.parse(readFileSync(new URL('../../src/config/runtime/profiles/qwen3-6-35b-a3b-q4_0-candidate.json', import.meta.url)));
const state = { runtimeConfig: profile.runtime, modelConfig: { sessionSettings: { prefillTokenChunkSize: null } } };
assert.equal(resolvePrefillTokenChunkSize(state, 2287), 256,
  'native candidate must declare its bounded token work policy rather than inherit disabled chunking');
assert.equal(shouldDisablePrefillCommandBatching(state, {}, null), true);
const run = async (policy, count) => {
  const calls = [];
  let position = 0;
  const context = {
    _resolvePrefillTokenChunkSize: ids => resolvePrefillTokenChunkSize(policy, ids.length),
    _prefillToHidden: async ids => { calls.push({ kind: 'hidden', start: position, length: ids.length }); return ids.length; },
    _commitPrefillHiddenChunk: async length => { position += length; },
    _prefill: async ids => { calls.push({ kind: 'logits', start: position, length: ids.length }); return 'final-logits'; },
  };
  assert.equal(await _prefillInputIdsToLogits.call(context, Array.from({ length: count }, (_, i) => i), {}), 'final-logits');
  assert.equal(calls.reduce((sum, call) => sum + call.length, 0), count);
  return calls;
};
const chunked = await run(state, 2287);
assert.deepEqual(chunked.map(call => call.length), [256, 256, 256, 256, 256, 256, 256, 256, 239]);
assert.deepEqual(chunked.map(call => call.start), [0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048]);
assert.equal(chunked.at(-1).kind, 'logits');
for (const count of [1, 28, 255, 256, 257]) {
  const calls = await run(state, count);
  assert.ok(calls.every(call => call.length <= 256));
}
const historical = { runtimeConfig: { inference: { session: { prefillTokenChunkSize: null } } },
  modelConfig: { sessionSettings: { prefillTokenChunkSize: null } } };
assert.equal(resolvePrefillTokenChunkSize(historical, 2287), null);
assert.deepEqual(await run(historical, 2287), [{ kind: 'logits', start: 0, length: 2287 }]);
console.log(JSON.stringify({ status: 'passed', scope: 'CPU production planner geometry only; not GPU/full model',
  historical: { tokenChunk: null, invocationLengths: [2287] }, declaredCandidate: { tokenChunk: 256, invocationLengths: chunked.map(call => call.length) } }));
