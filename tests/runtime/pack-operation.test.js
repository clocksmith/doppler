import assert from 'node:assert/strict';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import { createPackOperationExecutor } from '../../src/client/runtime/pack-operation-executor.js';
import { PACK_OPERATION_REQUEST_SCHEMA, hashPackObservation, normalizePackObservation } from '../../src/config/pack-operation.js';
import { TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY, createSignedPackFixture } from '../helpers/pack-v2-fixture.js';

// Signed artifact/operation contract test. The injected program is not model qualification.
const fixture = await createSignedPackFixture();
const generationOptions = { maxTokens: 2, maxSeqLen: 16, temperature: 0, topP: 1,
  topK: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 8, seed: 0, useChatTemplate: false };
const request = (name, input, options = {}) => ({ schema: PACK_OPERATION_REQUEST_SCHEMA,
  operation: { name, version: 1 }, input, options, assignment: { assignmentId: 'job', attempt: 1 },
  limits: { maxInputBytes: 10000, maxOutputBytes: 100000, deadlineAt: Date.now() + 60000 } });
const calls = [];
const program = {
  executionGraphHash: fixture.pack.program.executionGraphHash,
  tokenize() { return [1]; }, decodeTokens(tokens) { return tokens.join(','); },
  getTokenContract() { return { padTokenId: null, eosTokenId: null, stopTokenIds: [] }; },
  reset() {}, releaseStepResult() {}, async close() {},
  async executePhase() { return { logits: new Float32Array([0, 10]) }; },
  async embed(text, options) { calls.push({ text, options }); return { embedding: new Float32Array([0.5, 1]), seqLen: 1 }; },
  async encodeSequence(sequence, options) {
    calls.push({ sequence, options });
    return { pooledEmbedding: new Float32Array([0.5, 1]), tokenEmbeddings: null, logits: null };
  },
  async rerank() { return { schema: 'doppler_rerank_evidence/v1', inputHash: hashPackObservation('input'),
    outputHash: hashPackObservation('output'), backendIdentityHash: hashPackObservation('backend'),
    scores: [1, 0], ranking: [0, 1] }; },
};
const runtime = createDopplerRuntime({
  device: { getDevice: () => ({ limits: { maxBufferSize: 1024 },
    createBuffer: () => ({ destroy() {} }), createCommandEncoder() {}, queue: { writeBuffer() {} } }),
  getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
  artifactStore: fixture.artifactStore, trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
  programFactory: async () => program,
});
const session = await runtime.openPack(fixture.pack);
const collect = async (iterator) => { const values = []; for await (const value of iterator) values.push(value); return values; };
const jobs = [
  request('generate', { prompt: 'public question' }, generationOptions),
  request('embed', { texts: ['a', 'b'] }),
  request('rerank', { query: 'a', documents: ['a', 'b'], application: structuredClone(fixture.pack.release.application) }),
  request('encodeSequence', { sequence: 'ACD' }, { includeLogits: false, includeTokenEmbeddings: false }),
];
for (const job of jobs) {
  const events = await collect(session.executeOperation(job));
  assert.equal(events.filter((event) => event.status === 'completed').length, 1);
  let previous = null;
  for (const [index, event] of events.entries()) {
    const { eventDigest, ...payload } = event;
    assert.equal(eventDigest, hashPackObservation(payload));
    assert.equal(event.previousEventDigest, previous);
    assert.equal(event.eventIndex, index);
    assert.equal(event.requestHash, hashPackObservation(job));
    assert.equal(event.assignmentHash, hashPackObservation(job.assignment));
    assert.ok(Object.isFrozen(event.output));
    previous = eventDigest;
  }
  const completed = events.at(-1);
  const { receiptDigest, ...payload } = completed.receipt;
  assert.equal(receiptDigest, hashPackObservation(payload));
  assert.equal(payload.pack.envelopeDigest, session.packIdentity.envelopeDigest);
  assert.equal(payload.targetPlanDigest, session.selectedTargetPlanDigest);
  assert.equal(payload.outputHash, hashPackObservation(completed.output));
  assert.equal(payload.inputHash, hashPackObservation({ input: job.input, options: job.options }));
  assert.deepEqual(payload.operation, job.operation);
  assert.ok(payload.runtimeVersion);
  if (job.operation.name === 'generate') assert.deepEqual(completed.output, { text: '1,1', tokenIds: [1, 1] });
  if (job.operation.name === 'embed') assert.deepEqual(completed.output.embeddings[0].embedding, [0.5, 1]);
}

const mutable = structuredClone(jobs[3]);
const beforeMutation = hashPackObservation(mutable);
const frozenInvocation = session.executeOperation(mutable);
mutable.input.sequence = 'changed'; mutable.assignment.attempt = 9;
const original = (await collect(frozenInvocation)).at(-1);
assert.equal(original.requestHash, beforeMutation);
assert.equal(calls.at(-1).sequence, 'ACD');
for (const change of [
  (job) => { job.operation.name = 'unknown'; },
  (job) => { job.operation.version = 2; },
  (job) => { job.options.unknown = true; },
  (job) => { delete job.assignment; },
  (job) => { delete job.options.includeLogits; },
  (job) => { job.limits.maxInputBytes = 1; },
]) {
  const invalid = structuredClone(jobs[3]); change(invalid);
  assert.throws(() => session.executeOperation(invalid));
}
assert.throws(() => session.executeOperation(request('generate', { prompt: 'a', promptTokens: [] }, generationOptions)), /exactly one/);
assert.throws(() => session.executeOperation(request('generate', { promptTokens: 'wrong' }, generationOptions)), /token IDs/);
assert.throws(() => normalizePackObservation([, 1]), /finite, acyclic/);
assert.throws(() => normalizePackObservation({ vector: new Float32Array([NaN]) }), /finite, acyclic/);
const cycle = {}; cycle.self = cycle;
assert.throws(() => normalizePackObservation(cycle), /finite, acyclic/);

let cleaned = 0;
let abortAfterStep = null;
const execute = createPackOperationExecutor({ identity: { fixture: true }, assertCurrent: async () => {},
  adapters: { encodeSequence: { validate() {}, async *execute(job, signal) {
    try {
      signal.throwIfAborted();
      yield { delta: { item: 0 }, output: { value: 1 } };
      abortAfterStep?.abort(new Error('cancelled during result'));
      return { value: 2 };
    } finally { cleaned++; }
  } } } });
const iterator = execute(jobs[3]);
assert.equal((await iterator.next()).value.status, 'partial');
await assert.rejects(execute(jobs[3]).next(), /already active/);
await iterator.return();
assert.equal(cleaned, 1);
const cancellation = new AbortController();
const cancelled = execute(jobs[3], { signal: cancellation.signal });
await cancelled.next(); cancellation.abort(new Error('requester cancelled'));
await assert.rejects(cancelled.next(), /requester cancelled/);
assert.equal(cleaned, 2);
abortAfterStep = new AbortController();
const lateCancelled = execute(jobs[3], { signal: abortAfterStep.signal });
await lateCancelled.next();
await assert.rejects(lateCancelled.next(), /cancelled during result/);
abortAfterStep = null;
const expired = structuredClone(jobs[3]); expired.limits.deadlineAt = Date.now() - 1;
await assert.rejects(collect(execute(expired)), /deadline exceeded/);
const oversized = structuredClone(jobs[3]); oversized.limits.maxOutputBytes = 1;
await assert.rejects(collect(execute(oversized)), /maxOutputBytes/);
const cancelledBefore = new AbortController(); cancelledBefore.abort(new Error('before execution'));
await assert.rejects(collect(execute(jobs[3], { signal: cancelledBefore.signal })), /before execution/);
assert.equal((await collect(execute(jobs[3]))).at(-1).status, 'completed', 'failure releases the operation slot');

// Even an adapter with a fallible explicit return cannot issue a successful receipt first.
const cleanupFailure = createPackOperationExecutor({ identity: {}, assertCurrent: async () => {}, adapters: {
  encodeSequence: { validate() {}, execute: () => ({ next: async () => ({ done: true, value: {} }),
    return: async () => { throw new Error('cleanup failed'); } }) },
} });
await assert.rejects(cleanupFailure(jobs[3]).next(), /cleanup failed/);
await session.close();
await assert.rejects(collect(session.executeOperation(jobs[3])), /session is closed/);
console.log('✔ pack-operation.test.js passed (injected program; contract evidence only)');
