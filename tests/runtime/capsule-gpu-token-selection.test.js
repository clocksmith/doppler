import assert from 'node:assert/strict';
import { createSessionController } from '../../src/client/runtime/session-controller.js';

const options = { promptTokens: [1], maxTokens: 3, maxSeqLen: 16, temperature: 0,
  topP: 1, topK: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false, seed: 0 };
const plan = { memoryLayout: {}, phases: { prefill: [{}], decode: [{}] }, tokenSelection: {} };
function fixture(makeResult) {
  const phases = [], released = [];
  let transientReleases = 0;
  const binder = { bindSlots() {}, writeSlot() {}, assertDeviceAvailable() {},
    releaseTransient() { transientReleases++; }, releaseAll() {} };
  const program = { reset() {}, getTokenContract: () => ({ eosTokenId: 4, padTokenId: null, stopTokenIds: [] }),
    releaseStepResult(result) { if (result) released.push(result); }, decodeTokens: ids => ids.join(''), close() {} };
  const executor = { async executePhase(phase, commands, request) {
    phases.push({ phase, context: structuredClone({ ...request.context, generationOptions: undefined }) });
    return { results: [await makeResult(phases.length)] };
  } };
  return { controller: createSessionController(executor, binder, program), phases, released,
    transientReleases: () => transientReleases };
}
const selected = id => ({ tokenId: id, vocabSize: 5, get logits() { throw new Error('must not read scores'); } });
const gpu = fixture(index => selected(index + 1));
const iterator = gpu.controller.generateTokens(plan, options);
assert.deepEqual((await iterator.next()).value, 2);
assert.deepEqual((await iterator.next()).value, 3);
assert.deepEqual((await iterator.next()).value, 4);
const completed = await iterator.next();
assert.equal(completed.done, true);
assert.equal(completed.value.completion.stopReason, 'eos-token');
assert.deepEqual(gpu.phases.map(row => row.phase), ['prefill', 'decode', 'decode']);
assert.deepEqual(gpu.phases[2].context.contextTokens, [1, 2, 3]);
assert.equal(gpu.released.length, 3);
assert.equal(gpu.transientReleases(), 1);

const legacy = fixture(() => ({ tokenId: 99, logits: [0, 1, 0, 0, 2] }));
const { tokenSelection, ...legacyPlan } = plan;
assert.deepEqual(await Array.fromAsync(legacy.controller.generateTokens(legacyPlan, options)), [4]);
for (const result of [{}, { tokenId: 1 }, { tokenId: -1, vocabSize: 5 }, { tokenId: 5, vocabSize: 5 }, { tokenId: 0.5, vocabSize: 5 }]) {
  const invalid = fixture(() => result);
  await assert.rejects(Array.fromAsync(invalid.controller.generateTokens(plan, options)), /invalid token result/);
  assert.equal(invalid.released.length, 1);
  assert.equal(invalid.transientReleases(), 1);
}
const abort = new AbortController();
const cancelled = fixture(() => { abort.abort(); return selected(2); });
await assert.rejects(cancelled.controller.generateTokens(plan, { ...options, signal: abort.signal }).next(), /aborted/);
assert.equal(cancelled.released.length, 1);
assert.equal(cancelled.transientReleases(), 1);
const interrupted = fixture(() => selected(2));
const stream = interrupted.controller.generateTokens(plan, options);
await stream.next(); await stream.return();
assert.equal(interrupted.phases.length, 1);
assert.equal(interrupted.transientReleases(), 1);
console.log('Capsule GPU token selection controller tests passed.');
