import assert from 'node:assert/strict';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import { createCapsuleOperationExecutor } from '../../src/client/runtime/capsule-operation-executor.js';
import { CAPSULE_OPERATION_REQUEST_SCHEMA, hashCapsuleObservation, normalizeCapsuleObservation } from '../../src/config/capsule-operation.js';
import { TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY, createSignedCapsuleFixture } from '../helpers/capsule-v2-fixture.js';

// Signed artifact/operation contract test. The injected program is not model qualification.
const fixture = await createSignedCapsuleFixture();
const generationOptions = { maxTokens: 2, maxSeqLen: 16, temperature: 0, topP: 1,
  topK: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 8, seed: 0, useChatTemplate: false };
const request = (name, input, options = {}) => ({ schema: CAPSULE_OPERATION_REQUEST_SCHEMA,
  operation: { name, version: 1 }, input, options, assignment: { assignmentId: 'job', attempt: 1 },
  limits: { maxInputBytes: 10000, maxOutputBytes: 100000, deadlineAt: Date.now() + 60000 } });
const calls = [];
const program = {
  executionGraphHash: fixture.capsule.program.executionGraphHash,
  tokenize() { return [1]; }, decodeTokens(tokens) { return tokens.join(','); },
  getTokenContract() { return { padTokenId: null, eosTokenId: null, stopTokenIds: [] }; },
  reset() {}, releaseStepResult() {}, async close() {},
  async executePhase() { return { logits: new Float32Array([0, 10]) }; },
  async embed(text, options) { calls.push({ text, options }); return { embedding: new Float32Array([0.5, 1]), seqLen: 1 }; },
  async encodeSequence(sequence, options) {
    calls.push({ sequence, options });
    return { pooledEmbedding: new Float32Array([0.5, 1]), tokenEmbeddings: null, logits: null };
  },
  async rerank() { return { schema: 'doppler_rerank_evidence/v1', inputHash: hashCapsuleObservation('input'),
    outputHash: hashCapsuleObservation('output'), backendIdentityHash: hashCapsuleObservation('backend'),
    scores: [1, 0], ranking: [0, 1] }; },
};
const createRuntime = (qualifiedFixture) => createDopplerRuntime({
  device: { getDevice: () => ({ limits: { maxBufferSize: 1024 },
    createBuffer: () => ({ destroy() {} }), createCommandEncoder() {}, queue: { writeBuffer() {} } }),
  getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
  artifactStore: qualifiedFixture.artifactStore, trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
  programFactory: async () => program,
});
const session = await createRuntime(fixture).openCapsule(fixture.capsule);
const sessions = [session];
const rerankFixture = await createSignedCapsuleFixture({ operation: 'rerank' });
const sequenceFixture = await createSignedCapsuleFixture({ operation: 'encodeSequence' });
const rerankSession = await createRuntime(rerankFixture).openCapsule(rerankFixture.capsule);
const sequenceSession = await createRuntime(sequenceFixture).openCapsule(sequenceFixture.capsule);
sessions.push(rerankSession, sequenceSession);
const collect = async (iterator) => { const values = []; for await (const value of iterator) values.push(value); return values; };
const jobs = [
  request('generate', { prompt: 'public question' }, generationOptions),
  request('embed', { texts: ['a', 'b'], application: structuredClone(fixture.capsule.release.application) }),
  request('rerank', { query: 'a', documents: ['a', 'b'], application: structuredClone(fixture.capsule.release.application) }),
  request('encodeSequence', { sequence: 'ACD' }, { includeLogits: false, includeTokenEmbeddings: false }),
];
for (const job of jobs) {
  if (job.operation.name !== 'generate') {
    const callsBefore = calls.length;
    await assert.rejects(collect(session.executeOperation(job)), /not qualified/);
    assert.equal(calls.length, callsBefore, 'generic operation may not bypass qualification');
  }
  // Full evidence-backed embedding integration is exercised in capsule-embedding.test.js.
  if (job.operation.name === 'embed') continue;
  const qualifiedSession = job.operation.name === 'generate' ? session
    : job.operation.name === 'rerank' ? rerankSession : sequenceSession;
  const events = await collect(qualifiedSession.executeOperation(job));
  assert.equal(events.filter((event) => event.status === 'completed').length, 1);
  let previous = null;
  for (const [index, event] of events.entries()) {
    const { eventDigest, ...payload } = event;
    assert.equal(eventDigest, hashCapsuleObservation(payload));
    assert.equal(event.previousEventDigest, previous);
    assert.equal(event.eventIndex, index);
    assert.equal(event.requestHash, hashCapsuleObservation(job));
    assert.equal(event.assignmentHash, hashCapsuleObservation(job.assignment));
    assert.ok(Object.isFrozen(event.output));
    previous = eventDigest;
  }
  const completed = events.at(-1);
  const { receiptDigest, ...payload } = completed.receipt;
  assert.equal(receiptDigest, hashCapsuleObservation(payload));
  assert.equal(payload.capsule.envelopeDigest, qualifiedSession.capsuleIdentity.envelopeDigest);
  assert.equal(payload.targetPlanDigest, qualifiedSession.selectedTargetPlanDigest);
  assert.equal(payload.outputHash, hashCapsuleObservation(completed.output));
  assert.equal(payload.inputHash, hashCapsuleObservation({ input: job.input, options: job.options }));
  assert.deepEqual(payload.operation, job.operation);
  assert.ok(payload.runtimeVersion);
  if (job.operation.name === 'generate') assert.deepEqual(completed.output, { text: '1,1', tokenIds: [1, 1] });
  if (job.operation.name === 'encodeSequence') {
    assert.equal(completed.output.receipt.assignmentHash, hashCapsuleObservation(job.assignment));
    assert.equal(completed.output.receipt.operation, 'encodeSequence');
  }
}

const mutable = structuredClone(jobs[3]);
const beforeMutation = hashCapsuleObservation(mutable);
const frozenInvocation = sequenceSession.executeOperation(mutable);
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
assert.throws(() => normalizeCapsuleObservation([, 1]), /finite, acyclic/);
assert.throws(() => normalizeCapsuleObservation({ vector: new Float32Array([NaN]) }), /finite, acyclic/);
const cycle = {}; cycle.self = cycle;
assert.throws(() => normalizeCapsuleObservation(cycle), /finite, acyclic/);

let cleaned = 0;
let abortAfterStep = null;
const execute = createCapsuleOperationExecutor({ identity: { fixture: true }, assertCurrent: async () => {},
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
const cleanupFailure = createCapsuleOperationExecutor({ identity: {}, assertCurrent: async () => {}, adapters: {
  encodeSequence: { validate() {}, execute: () => ({ next: async () => ({ done: true, value: {} }),
    return: async () => { throw new Error('cleanup failed'); } }) },
} });
await assert.rejects(cleanupFailure(jobs[3]).next(), /cleanup failed/);
for (const openSession of sessions) await openSession.close();
await assert.rejects(collect(session.executeOperation(jobs[3])), /session is closed/);
console.log('✔ capsule-operation.test.js passed (injected program; contract evidence only)');
