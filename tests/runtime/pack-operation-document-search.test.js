import assert from 'node:assert/strict';
import { createPackOperationExecutor } from '../../src/client/runtime/pack-operation-executor.js';
import { createPackOperationAdapters } from '../../src/client/runtime/pack-operation-adapters.js';
import { createPackProgramAdapter } from '../../src/client/runtime/pack-program-adapter.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

// Preserve the local document regressions against the shared upstream executor.
const executePackOperation = (context, request, control) => createPackOperationExecutor({
  identity: { pack: context.identity, targetId: context.targetId, targetPlanDigest: context.targetPlanDigest,
    artifactReceipts: context.artifactReceipts, releaseEventDigest: context.releaseEventDigest },
  assertCurrent: context.assertCurrent,
  adapters: createPackOperationAdapters({ program: context.program }),
})(request, control);

const collect = async (iterator) => {
  const events = [];
  for await (const event of iterator) events.push(event);
  return events;
};
const request = () => ({ schema: 'doppler.pack-operation-request/v1', operation: { name: 'embed', version: 1 },
  input: { texts: ['one', 'two'] }, options: {}, assignment: null,
  limits: { maxInputBytes: 1024, maxOutputBytes: 1024, deadlineAt: Date.now() + 10000 } });
let calls = 0;
let closed = false;
const context = { identity: { packId: 'fixture' }, targetId: 'fixture', targetPlanDigest: 'fixture',
  artifactReceipts: [], runtimeVersion: 'fixture', releaseEventDigest: null,
  assertCurrent: async () => { if (closed) throw new Error('closed'); },
  program: { embed: async (text) => { calls++; return { embedding: new Float32Array([text.length, 1]) }; } } };
const input = request();
const originalHash = computeCanonicalSha256(input);
const iterator = executePackOperation(context, input);
input.input.texts[0] = 'changed after invocation';
const events = await collect(iterator);
assert.equal(calls, 2);
assert.deepEqual(events.map((event) => event.status), ['partial', 'partial', 'completed']);
assert.deepEqual(events.at(-1).output.embeddings[0].embedding, [3, 1]);
assert.equal(events.at(-1).receipt.requestHash, originalHash);
for (const [index, event] of events.entries()) {
  const { eventDigest, ...payload } = event;
  assert.equal(eventDigest, computeCanonicalSha256(payload));
  assert.equal(event.eventIndex, index);
  assert.equal(event.previousEventDigest, index ? events[index - 1].eventDigest : null);
}
assert(Object.isFrozen(events.at(-1).output.embeddings[0].embedding));
const before = calls;
assert.throws(() => executePackOperation(context, { ...request(), options: { precision: 'f16' } }), /fields/);
assert.throws(() => executePackOperation(context, { ...request(), operation: { name: 'unknown', version: 1 } }), /Unsupported Pack operation/);
assert.throws(() => executePackOperation(context, { ...request(), input: { texts: [''] } }), /texts/);
const oversized = request(); oversized.limits.maxInputBytes = 1;
assert.throws(() => executePackOperation(context, oversized), /maxInputBytes/);
assert.equal(calls, before);
const expired = request(); expired.limits.deadlineAt = 1;
await assert.rejects(collect(executePackOperation(context, expired)), /deadline/);
const abort = new AbortController();
const cancelled = executePackOperation(context, request(), { signal: abort.signal });
assert.equal((await cancelled.next()).value.status, 'partial');
abort.abort(new Error('cancelled'));
await assert.rejects(cancelled.next(), /cancelled/);
const early = executePackOperation(context, request());
await early.next(); const earlyCalls = calls;
await early.return(); assert.equal(calls, earlyCalls, 'closing iterator never starts next document');
closed = true;
await assert.rejects(collect(executePackOperation(context, request())), /closed/);
closed = false;
await assert.rejects(collect(executePackOperation({ ...context, program: {
  embed: async () => ({ embedding: [NaN] }) } }, request())), /finite/);
const outputLimit = request(); outputLimit.limits.maxOutputBytes = 1;
await assert.rejects(collect(executePackOperation(context, outputLimit)), /maxOutputBytes/);
let vectorIndex = 0;
await assert.rejects(collect(executePackOperation({ ...context, program: {
  embed: async () => ({ embedding: ++vectorIndex === 1 ? [1, 2] : [1] }) } }, request())), /inconsistent vector dimensions/);
await assert.rejects(collect(executePackOperation({ ...context, program: {
  embed: async () => ({ embedding: [] }) } }, request())), /empty vector/);
let embedCalls = 0;
let transitions = [];
const handle = { manifest: { modelId: 'fixture' }, supportsEmbedding: false,
  advanced: { getStats: () => ({ executionPlan: { transitions } }) },
  embed: async () => { embedCalls++; return { embedding: [1, 2] }; } };
const programAdapter = createPackProgramAdapter(handle,
  { modelId: 'fixture', program: { executionGraphHash: 'fixture' } }, { phases: { prefill: [], decode: [] } });
await assert.rejects(programAdapter.embed('one', {}), /does not declare embedding/);
assert.equal(embedCalls, 0, 'undeclared embedding support never dispatches');
handle.supportsEmbedding = true;
assert.deepEqual(await programAdapter.embed('one', {}), { embedding: [1, 2] });
transitions = [{}];
await assert.rejects(programAdapter.embed('one', {}), /undeclared execution-plan transition/);
console.log('pack-operation-document-search.test.js passed (injected program; no physical inference claim)');
