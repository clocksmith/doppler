import test from 'node:test';
import assert from 'node:assert/strict';
import { measurePooledRequests } from './timing.js';

function fixture(tokens) {
  const calls = []; let index = 0, clock = 0;
  return { calls, now: () => clock,
    engine: { reset() { index = 0; calls.push('reset'); },
      async prefillTokens(ids) { calls.push(['prefill', ids]); clock += 5; },
      async forwardTokenIds(id) { calls.push(['forward', id]); clock += 2; return { ids: [tokens[index++]] }; } },
    device: { queue: { async onSubmittedWorkDone() { calls.push('fence'); clock += 1; } } } };
}
const contract = { shared: { maxNewTokens: 4, eosTokenIds: [99] } };
test('resident timing includes prefill and completion, keeps warmups, and stops naturally at EOS', async () => {
  const f = fixture([7, 99, 88]);
  const samples = await measurePooledRequests(f.engine, f.device, [1, 2, 3], contract,
    { diagnosticOnly: true, warmupRuns: 1, timedRuns: 2, expectedTokenIds: [7, 99] }, f.now);
  assert.equal(samples.length, 3); assert.deepEqual(samples.map(x => x.warmup), [true, false, false]);
  assert.equal(samples.every(x => x.firstTokenMs === 7 && x.completeMs === 10 && x.stopReason === 'eos'), true);
  assert.equal(f.calls.filter(x => Array.isArray(x) && x[0] === 'forward').length, 6);
  assert.deepEqual(f.calls.slice(0, 4), ['reset', 'fence', ['prefill', [1, 2]], ['forward', 3]]);
});
test('mismatched timed output fails with its attempted sample retained', async () => {
  const f = fixture([4, 3, 2, 1]);
  await assert.rejects(measurePooledRequests(f.engine, f.device, [1], contract,
    { diagnosticOnly: true, warmupRuns: 0, timedRuns: 1, expectedTokenIds: [4, 3, 2, 5] }, f.now), error => {
    assert.equal(error.completedSamples.length, 1); assert.equal(error.completedSamples[0].valid, false);
    assert.equal(error.completedSamples[0].stopReason, 'max-tokens'); return true;
  });
});
test('pilot refuses missing reference output and non-diagnostic admission', async () => {
  const f = fixture([99]);
  for (const timing of [{}, { diagnosticOnly: true, warmupRuns: 0, timedRuns: 1 }]) {
    await assert.rejects(measurePooledRequests(f.engine, f.device, [1], contract, timing, f.now));
  }
  assert.equal(f.calls.length, 0);
});
