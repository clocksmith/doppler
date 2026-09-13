import assert from 'node:assert/strict';
import { createCapsuleOperationExecutor } from '../../src/client/runtime/capsule-operation-executor.js';
import { createCapsuleOperationAdapters } from '../../src/client/runtime/capsule-operation-adapters.js';
import { createSessionController } from '../../src/client/runtime/session-controller.js';
import { createCapsuleStreamAccumulator, capsuleOperationSnapshots } from '../../src/client/runtime/capsule-operation-stream.js';
import { hashCapsuleObservation } from '../../src/config/capsule-operation.js';
import { createCapsuleDeltaBudget } from '../../src/client/runtime/capsule-operation-deltas.js';
import { BundledTokenizer } from '../../src/inference/tokenizers/bundled.js';

const tokenizer = new BundledTokenizer({ vocabSize: 0, deferSpecialTokens: true, addBosToken: false, addEosToken: false });
tokenizer.load({ model: { type: 'BPE', vocab: { ...Object.fromEntries(Array.from({ length: 256 }, (_, i) => [`<0x${i.toString(16).padStart(2, '0')}>`, i])), '<eos>': 256 }, merges: [], byte_fallback: true }, pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false }, added_tokens: [{ id: 256, content: '<eos>', special: true }] });
const options = { maxTokens: 6, maxSeqLen: 100, temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false, stopSequences: [] };
const request = (version, overrides = {}) => ({ schema: `doppler.capsule-operation-request/v${version}`, operation: { name: 'generate', version: 1 }, input: { promptTokens: [1] }, options: { ...options, ...overrides }, assignment: null, limits: { maxInputBytes: 10000, maxOutputBytes: 10000, deadlineAt: Date.now() + 60000 } });
function engine(ids, contract = {}) {
  let phases = 0, released = 0, decodedIds = 0;
  const program = { reset() { phases = 0; }, getTokenContract: () => ({ eosTokenId: null, stopTokenIds: [], ...contract }),
    decodeTokens(values) { decodedIds += values.length; return tokenizer.decode(values, true, false); },
    createIncrementalDecoder: () => tokenizer.createIncrementalDecoder(), releaseStepResult(value) { if (value) released++; } };
  const controller = createSessionController({ async executePhase() {
    const logits = new Float32Array(256); logits.fill(-100); logits[ids[phases++]] = 100; return { results: [{ logits }] };
  } }, { bindSlots() {}, writeSlot() {}, releaseTransient() {}, assertDeviceAvailable() {} }, program);
  const adapters = createCapsuleOperationAdapters({ program, generate: (input, control) => controller.generateTokens({ phases: { prefill: [], decode: [] }, memoryLayout: {} }, input, control),
    embed: async ({ text }) => ({ embedding: new Float32Array([text.length, 1]), text }) });
  return { execute: createCapsuleOperationExecutor({ adapters, identity: { modelId: 'stream-fixture', targetPlanDigest: hashCapsuleObservation('plan') }, assertCurrent: async () => {} }),
    stats: () => ({ phases, released, decodedIds }) };
}
const collect = async stream => { const events = []; for await (const event of stream) events.push(event); return events; };
for (const [ids, stopSequences, contract] of [
  [[65, 0xc3, 0xa9, 66, 67, 68], ['é'], {}],
  [[65, 0xc3, 0xa9, 66, 67, 68], ['�'], {}],
  [[65, 66, 67, 68, 69, 70], ['BCD'], {}],
  [[65, 66, 67, 68, 69, 70], [], { eosTokenId: 67 }],
  [[65, 66, 67, 68, 69, 70], [], { stopTokenIds: [68] }],
  [[65, 66, 67, 68, 0xf0, 0x9f], [], {}],
]) {
  const oldEngine = engine(ids, contract), nextEngine = engine(ids, contract);
  const oldEvents = await collect(oldEngine.execute(request(1, { stopSequences })));
  const job = request(2, { stopSequences });
  const events = await collect(nextEngine.execute(job));
  assert.deepEqual(events.at(-1).output, oldEvents.at(-1).output);
  const accumulator = createCapsuleStreamAccumulator(job);
  for (const event of events) {
    accumulator.accept(JSON.parse(JSON.stringify(event)));
    if (event.status === 'partial') { assert(!Object.hasOwn(event, 'output')); assert(!Object.hasOwn(event, 'receipt')); }
  }
  assert.deepEqual(accumulator.finish().output, oldEvents.at(-1).output);
  assert.equal(nextEngine.stats().decodedIds, 0, 'v2 must never decode the accumulated IDs');
  assert.equal(nextEngine.stats().released, nextEngine.stats().phases);
  assert.throws(() => accumulator.accept(events[0]), /after completion/);
  const missing = createCapsuleStreamAccumulator(job);
  assert.throws(() => missing.accept(events[1]), /missing/);
  assert.throws(() => missing.finish(), /missing/);
  assert.throws(() => missing.accept(events[0]), /missing/);
  assert.throws(() => accumulator.finish(), /after completion/);
  const tampered = structuredClone(events[0]); tampered.delta.text += 'x';
  assert.throws(() => createCapsuleStreamAccumulator(job).accept(tampered), /digest/);
  const changed = structuredClone(events.at(-1)); changed.output.text += 'x';
  changed.receipt.outputHash = hashCapsuleObservation(changed.output);
  const { receiptDigest, ...receipt } = changed.receipt; changed.receipt.receiptDigest = hashCapsuleObservation(receipt);
  const { eventDigest, ...payload } = changed; changed.eventDigest = hashCapsuleObservation(payload);
  const forged = createCapsuleStreamAccumulator(job); events.slice(0, -1).forEach(event => forged.accept(event));
  assert.throws(() => forged.accept(changed), /reconstructed/);
}
{
  const run = engine([65, 66, 67, 68, 69, 70]), job = request(2);
  const abort = new AbortController();
  const stream = run.execute(job, { signal: abort.signal });
  const first = await stream.next();
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(run.stats().phases, 1, 'slow consumers cannot trigger speculative generation');
  const accumulator = createCapsuleStreamAccumulator(job); accumulator.accept(first.value);
  abort.abort(); await assert.rejects(stream.next(), /cancelled|abort/i);
  assert.throws(() => accumulator.finish(), /without completion/);
  assert.equal(run.stats().phases, run.stats().released);
}
{
  const job = request(2); job.limits.maxOutputBytes = 30;
  await assert.rejects(collect(engine([65, 66, 67, 68, 69, 70]).execute(job)), /maxOutputBytes/);
  const stream = engine([65, 66, 67, 68, 69, 70]).execute(request(2));
  await stream.next(); await stream.return();
}
{
  const job = { ...request(2), operation: { name: 'embed', version: 1 }, input: { texts: ['a', 'bb', 'ccc'], application: {} }, options: {} };
  const events = await collect(engine([]).execute(job));
  assert.deepEqual(events.slice(0, -1).map(event => event.delta.itemIndex), [0, 1, 2]);
  const accumulator = createCapsuleStreamAccumulator(job); events.forEach(event => accumulator.accept(event));
  assert.deepEqual(accumulator.finish().output.embeddings.map(item => item.embedding), [[1, 1], [2, 1], [3, 1]]);
  const snapshots = await collect(capsuleOperationSnapshots((async function* () { yield* events; })(), job));
  assert.equal(snapshots[0].output.embeddings.length, 1);
  assert.equal(snapshots.at(-1).output.embeddings.length, 3);
}
{
  const job = request(2, { maxTokens: 1 });
  const budget = createCapsuleDeltaBudget(job);
  budget.accept({ tokenIds: [65], text: 'A' });
  assert.throws(() => budget.accept({ tokenIds: [66], text: 'B' }), /maxTokens/);
  const run = createCapsuleOperationExecutor({ identity: {}, assertCurrent: async () => {}, adapters: {
    generate: { validate() {}, async *execute() {
      yield { delta: { tokenIds: [65, 66], text: 'AB' } };
    } },
  } });
  await assert.rejects(collect(run(job)), /maxTokens/);
  const embedBudget = createCapsuleDeltaBudget({ ...job, operation: { name: 'embed', version: 1 }, input: { texts: ['a'] } });
  embedBudget.accept({ itemIndex: 0, item: { embedding: [1] } });
  assert.throws(() => embedBudget.accept({ itemIndex: 1, item: { embedding: [1] } }), /texts count/);
}
console.log('capsule-incremental-stream.test: Unicode, stop boundaries, reconstruction, integrity, backpressure, cancellation, limits and embedding deltas passed');
