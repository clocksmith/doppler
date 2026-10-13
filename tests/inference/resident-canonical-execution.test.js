import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { readFileSync } from 'node:fs';
import { resolveGenerationOptions } from '../../src/config/generation-contract.js';
import { canonicalGenerationIdentity } from '../../src/config/canonical-generation-policy.js';
import { createResidentTargetState, createResidentCursorMirror } from '../../src/inference/resident-target-state.js';
import { createResidentOutputState } from '../../src/inference/resident-output-state.js';
import { createBundledIncrementalDecoder } from '../../src/inference/tokenizers/bundled/incremental-decoder.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

// Simulate only numerical layers/readback. The real resident owns admission,
// sequencing, target sampling/decoding/cursors and cleanup. No GPU qualification.
let dispatches = 0, allocations = 0, releases = 0, closes = 0;
const rows = [3, 4, 5].map(id => { const row = new Float32Array(100).fill(-Infinity); row[id] = 0; return row; });
globalThis.canonicalResidentFixture = {
  execute: async (state, input) => {
    dispatches++;
    await globalThis.canonicalResidentFixture.pause?.();
    return state.index === 0 ? { activationBytes: new ArrayBuffer(input.numTokens * 16), timing: {} }
      : { logits: { logitsBuffer: rows[state.decodeStepCount], vocabSize: 100 }, timing: {} };
  },
  create: owner => { allocations++; return { state: { index: owner.modelPartition.index, currentSeqLen: 0, decodeStepCount: 0 }, close() { closes++; } }; },
  release: () => { releases++; },
};
const target = new URL('../../src/inference/pipelines/text/resident-partition.js', import.meta.url).href;
const modules = {
  './partition-execution.js': 'export const assertPartitionExecutionSupported = () => {}; export const executePartitionLayers = (...args) => canonicalResidentFixture.execute(...args);',
  './partition-attempt.js': 'export const createPartitionAttempt = owner => canonicalResidentFixture.create(owner);',
  '../../../memory/buffer-pool.js': 'export const readBuffer = async row => row.buffer; export const releaseBuffer = () => canonicalResidentFixture.release();',
};
const hooks = registerHooks({ resolve(name, context, next) {
  if (context.parentURL === target && modules[name]) return { url: 'data:text/javascript,' + encodeURIComponent(modules[name]), shortCircuit: true };
  return next(name, context);
} });
const { createResidentPartitionSession } = await import(target);
const plan = { schema: 'doppler.layer-partition-contract/v1', activationDtype: 'f32', hiddenSize: 4, vocabSize: 100,
  partitions: [{ layerRange: [0, 0] }, { layerRange: [1, 1] }] };
const planId = computeCanonicalSha256(plan), executionIdentity = 'sha256:' + 'a'.repeat(64);
const policy = { ...JSON.parse(readFileSync(new URL('../fixtures/canonical-generation-policy.json', import.meta.url))),
  maxGeneratedTokens: 3, materializedInputCapacity: 8, totalAcceptedTokenLimit: 9,
  temperature: 0, eosTokenIds: [], stopSequences: [], maxRawOutputUtf16: 128, maxVisibleOutputUtf16: 128 };
const decoder = () => createBundledIncrementalDecoder({ tokenForId: id => String(id), byteLevel: false,
  byteDecoder: new Map(), wordPiece: false, splitEveryCharacter: false, wordPiecePrefix: '##' });
const identity = attemptId => ({ modelId: 'model', modelIdentity: 'sha256:'+'b'.repeat(64), planId,
  threadId: 'thread', attemptId, participantA: 'left', participantB: 'right' });
const signal = new AbortController().signal;
async function open(index, selectedPolicy = policy, enabled = true) {
  const allocation = { model: { id: 'model', identity: identity('x').modelIdentity }, plan, planId, index,
    participantId: index === 0 ? 'left' : 'right', generation: resolveGenerationOptions({ maxTokens: 3, maxSeqLen: 8,
      temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false }),
    limits: { maxAttempts: 8, maxConcurrentAttempts: 2, maxTokens: 3, maxPromptTokens: 8, maxActivationBytes: 128, maxOutputCharacters: 128 },
    ...(enabled ? { canonicalGeneration: { policy: selectedPolicy, executionIdentity } } : {}) };
  const pipeline = { modelPartition: { plan, index }, gpuContext: null, modelConfig: {}, dopplerLoader: { gpuBuffers: [] } };
  const ownedDecoder = () => { assert.equal(index, 1, 'Only partition B creates mutable output decoders'); return decoder(); };
  return createResidentPartitionSession(pipeline, allocation, { tokenize: () => [1,2], createIncrementalDecoder: ownedDecoder,
    createRawIncrementalDecoder: ownedDecoder, getTokenContract() { throw Error('Legacy sampling must not run'); } }, async () => {});
}
try {
  for (const selectedPolicy of [{ ...policy, stopSequences: ['3'] }, { ...policy, eosTokenIds: [3] },
    { ...policy, materializedInputCapacity: 2 }]) {
    const generationId = canonicalGenerationIdentity(selectedPolicy, executionIdentity, 100).generationId;
    const targetState = createResidentTargetState([1,2], selectedPolicy, 100, generationId,
      createResidentOutputState(decoder(), decoder(), selectedPolicy, 100));
    const mirror = createResidentCursorMirror([1,2], selectedPolicy, 100, generationId);
    const transition = targetState.bootstrap(rows[0]);
    assert.deepEqual(mirror.acceptTransition(null, transition), transition);
    assert.deepEqual(mirror.getCursor(), transition.successor);
    assert(transition.successor.terminal);
  }
  const a = await open(0), b = await open(1), binding = identity('ordinary');
  assert.equal(typeof a.admitCanonicalPrompt, 'function', 'Resident must expose actual canonical admission');
  const prompt = [1, 2], generation = canonicalGenerationIdentity(policy, executionIdentity, 100);
  const independent = createResidentTargetState(prompt, policy, 100, generation.generationId,
    createResidentOutputState(decoder(), decoder(), policy, 100));
  for (const session of [a,b]) await session.admitCanonicalPrompt({ identity: binding, tokenIds: prompt, generationId: generation.generationId, signal });
  let previous = null, ca = null, cb = null;
  for (let step = 0; step < 3; step++) {
    const ids = step ? [previous.successor.pendingTokenId] : prompt;
    const baseCursorId = previous?.successor.cursorId ?? null;
    const common = { identity: binding, inputTokenCount: ids.length, tokenPosition: step ? step+1 : 0,
      step, baseCursorId, generationId: generation.generationId, signal };
    const first = await a.executeCanonicalGroup0({ ...common, tokenIds: ids, continuation: ca });
    await assert.rejects(a.executeCanonicalGroup0({ ...common, tokenIds: ids, continuation: first.continuation }), /pending|order|commit/i);
    const activation = { ...first.activationTensor, tensorData: new Float32Array(first.activationTensor.data) };
    const second = await b.executeCanonicalGroup1({ ...common, inputTokenIds: ids, activation, continuation: cb });
    const expected = step ? independent.advance(previous.successor.cursorId, rows[step]) : independent.bootstrap(rows[step]);
    assert.deepEqual(second.transition, expected, 'resident result must be the real canonical transition');
    assert.deepEqual(second.transition.acceptedTokenIds, [3 + step]);
    assert.equal(second.transition.successor.materializedTokenCount, 2 + step);
    assert.equal(second.transition.successor.acceptedGeneratedTokenCount, 1 + step);
    assert.equal(second.transition.successor.pendingTokenId, 3 + step);
    assert(!('acknowledgments' in second), 'runtime must not invent host authentication');
    assert.deepEqual(await a.commitCanonicalTransition({ identity: binding, baseCursorId, transition: second.transition, signal }), expected);
    previous = second.transition; ca = first.continuation; cb = second.continuation;
  }
  assert(previous.successor.terminal); assert.equal(previous.successor.stopReason, 'max-tokens');
  assert.equal(dispatches, 6); assert.equal(allocations, 2); assert.equal(releases, 3);
  for (const session of [a,b]) { await session.closeAttempt({ identity: binding }); await session.close(); }
  assert.equal(closes, 2);
  for (const selectedPolicy of [{ ...policy, maxGeneratedTokens: 0 }, { ...policy, totalAcceptedTokenLimit: 2 }]) {
    const empty = await open(0, selectedPolicy), before = dispatches;
    const id = canonicalGenerationIdentity(selectedPolicy, executionIdentity, 100).generationId;
    const result = await empty.admitCanonicalPrompt({ identity: identity('empty'), tokenIds: [1,2], generationId: id, signal });
    assert.equal(result.schema, 'doppler.admission-only/v1'); assert.equal(dispatches, before);
    await empty.close();
  }
  const legacy = await open(0, policy, false);
  await assert.rejects(legacy.admitCanonicalPrompt({ identity: identity('legacy'), tokenIds: [1,2], generationId: generation.generationId, signal }), /canonical|Canonical/);
  await legacy.close();
  const guarded = await open(1), guardedIdentity = identity('guarded');
  await assert.rejects(guarded.admitCanonicalPrompt({ identity: guardedIdentity, tokenIds: [1,2],
    generationId: 'sha256:'+'c'.repeat(64), signal }), /allocation mismatch/);
  await guarded.admitCanonicalPrompt({ identity: guardedIdentity, tokenIds: [1,2], generationId: generation.generationId, signal });
  const common = { identity: guardedIdentity, step: 0, tokenPosition: 0, inputTokenCount: 2,
    inputTokenIds: [1,2], continuation: null, generationId: generation.generationId, baseCursorId: null, signal,
    activation: { shape: [1,2,4], dtype: 'f32', seqOffset: 0, step: 0, tensorData: new Float32Array(8) } };
  for (const changed of [{ generationId: 'sha256:'+'c'.repeat(64) }, { baseCursorId: 'sha256:'+'d'.repeat(64) }, { inputTokenIds: [2,1] }]) {
    const before = dispatches;
    await assert.rejects(guarded.executeCanonicalGroup1({ ...common, ...changed }), /stale cursor|changed generation/);
    assert.equal(dispatches, before, 'changed admission/cursor/input rejects before dispatch');
  }
  await assert.rejects(guarded.executeGroup1({ ...common, generation: {}, maxTokens: 3 }), /mode/);
  const first = await guarded.executeCanonicalGroup1(common);
  assert.deepEqual(first.transition, createResidentTargetState([1,2], policy, 100, generation.generationId,
    createResidentOutputState(decoder(), decoder(), policy, 100)).bootstrap(rows[0]));
  await guarded.close();
  for (const selectedPolicy of [{ ...policy, materializedInputCapacity: 2 }, { ...policy, totalAcceptedTokenLimit: 3 }]) {
    const terminal = await open(1, selectedPolicy);
    const generationId = canonicalGenerationIdentity(selectedPolicy, executionIdentity, 100).generationId;
    await terminal.admitCanonicalPrompt({ identity: guardedIdentity, tokenIds: [1,2], generationId, signal });
    const result = await terminal.executeCanonicalGroup1({ ...common, generationId });
    assert.equal(result.transition.successor.stopReason, 'context-limit');
    const before = dispatches;
    await assert.rejects(terminal.executeCanonicalGroup1({ ...common, generationId, step: 1, tokenPosition: 2,
      inputTokenCount: 1, inputTokenIds: [3], continuation: result.continuation,
      baseCursorId: result.transition.successor.cursorId }), /retired|order/);
    assert.equal(dispatches, before);
    await terminal.close();
  }
  const changedDecision = await open(0);
  await changedDecision.admitCanonicalPrompt({ identity: guardedIdentity, tokenIds: [1,2], generationId: generation.generationId, signal });
  const aRequest = { ...common, tokenIds: [1,2] }; delete aRequest.activation; delete aRequest.inputTokenIds;
  await assert.rejects(changedDecision.commitCanonicalTransition({ identity: guardedIdentity, baseCursorId: null,
    transition: first.transition, signal }), /no settled numerical step/);
  await changedDecision.executeCanonicalGroup0(aRequest);
  await assert.rejects(changedDecision.commitCanonicalTransition({ identity: guardedIdentity, baseCursorId: null,
    transition: { ...first.transition, acceptedTokenIds: [4] }, signal }), /differs/);
  await assert.rejects(changedDecision.commitCanonicalTransition({ identity: guardedIdentity, baseCursorId: null,
    transition: first.transition, signal }), /no settled numerical step/);
  await changedDecision.close();
  const cancelled = await open(0), abort = new AbortController();
  await cancelled.admitCanonicalPrompt({ identity: guardedIdentity, tokenIds: [1,2], generationId: generation.generationId, signal });
  const entered = Promise.withResolvers(), release = Promise.withResolvers();
  globalThis.canonicalResidentFixture.pause = () => { entered.resolve(); return release.promise; };
  const pending = cancelled.executeCanonicalGroup0({ ...aRequest, signal: abort.signal });
  await entered.promise; const priorCloses = closes;
  const settlement = cancelled.closeAttempt({ identity: guardedIdentity });
  await Promise.resolve(); assert.equal(closes, priorCloses, 'submitted state stays owned until execution settles');
  release.resolve(); await assert.rejects(pending, /closed|abort/i); await settlement;
  assert.equal(closes, priorCloses + 1);
  delete globalThis.canonicalResidentFixture.pause;
  await cancelled.close();
  for (const bad of [{ ...policy, maxGeneratedTokens: 4 }, { ...policy, materializedInputCapacity: 9 },
    { ...policy, maxRawOutputUtf16: 129 }]) await assert.rejects(open(0, bad), /exceeds/);
} finally { hooks.deregister(); delete globalThis.canonicalResidentFixture; }
console.log('resident canonical execution: actual transition, two-stage state, empty admission and legacy isolation passed');
