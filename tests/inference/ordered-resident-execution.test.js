import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createOrderedLayerPartitionPlan, hashLayerPartitionPlan } from '../../src/inference/pipelines/text/layer-partition-contract.js';
import { hashPartitionGeneration } from '../../src/partitions.js';
import { resolveResidentPartitionAllocation } from '../../src/inference/pipelines/text/resident-partition-contract.js';
import { resolveGenerationOptions } from '../../src/config/generation-contract.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { createBundledIncrementalDecoder } from '../../src/inference/tokenizers/bundled/incremental-decoder.js';

// Inject only numerical execution and GPU readback. Real resident admission,
// private attempts, sampling, decoding, continuations and close paths run here.
let dispatches = 0, closes = 0, decoderOwners = [], created = [];
globalThis.orderedResidentFixture = {
  create(owner) {
    const state = { index: owner.modelPartition.index, plan: owner.modelPartition.plan, currentSeqLen: 0, decodeStepCount: 0 };
    created.push(state);
    return { state, close() { closes++; } };
  },
  async execute(state, input) {
    dispatches++;
    await globalThis.orderedResidentFixture.pause?.(state);
    if (state.index === 0) assert(input.tokenIds && !input.activationBytes);
    else assert(input.activationBytes && !input.tokenIds);
    if (state.index < state.plan.stageCount - 1) {
      return { activationBytes: new ArrayBuffer(input.numTokens * 4 * state.plan.activationElementStrideBytes), timing: {} };
    }
    const logits = new Float32Array(16).fill(-100); logits[3 + state.decodeStepCount] = 1;
    return { logits: { logitsBuffer: logits, vocabSize: 16 }, timing: {} };
  },
};
const target = new URL('../../src/inference/pipelines/text/resident-partition.js', import.meta.url).href;
const substitutions = {
  './partition-execution.js': 'export const assertPartitionExecutionSupported = () => {}; export const executePartitionLayers = (...args) => orderedResidentFixture.execute(...args);',
  './partition-attempt.js': 'export const createPartitionAttempt = owner => orderedResidentFixture.create(owner);',
  // Fixture owners load no model; real stage preparation is covered by resident-pipeline-preparation.test.js.
  './resident-pipeline-preparation.js': 'export const prepareResidentPartitionPipelines = async () => ({ schema: "doppler.resident-pipeline-preparation/v1", complete: true, count: 0, durationMs: 0, pipelines: [], lazy: [] });',
  '../../../memory/buffer-pool.js': 'export const readBuffer = async row => row.buffer; export const releaseBuffer = () => {};',
};
const hooks = registerHooks({ resolve(name, context, next) {
  if (context.parentURL === target && substitutions[name]) return { url: 'data:text/javascript,' + encodeURIComponent(substitutions[name]), shortCircuit: true };
  return next(name, context);
} });
const { createResidentPartitionSession } = await import(target);
const signal = new AbortController().signal;
const generation = resolveGenerationOptions({ maxTokens: 2, maxSeqLen: 8, temperature: 0, topK: 1, topP: 1,
  repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false });
const generationDigest = hashPartitionGeneration(generation);
assert.equal(generationDigest, computeCanonicalSha256(generation));
assert.equal(hashPartitionGeneration({ ...generation, seed: undefined }), generationDigest);
assert.throws(() => hashPartitionGeneration({}), /explicit/);
assert.notEqual(hashPartitionGeneration({ ...generation, seed: 0 }), generationDigest);
assert.throws(() => hashPartitionGeneration({ ...generation, seed: null }));
assert.notEqual(hashPartitionGeneration({ ...generation, temperature: 0 }), hashPartitionGeneration({ ...generation, temperature: 1, seed: 0 }));
assert.throws(() => hashPartitionGeneration({ maxTokens: -1 }));
const dimensions = { modelId: 'ordered-fixture', numLayers: 28, hiddenSize: 4, vocabSize: 16 };
const model = { id: dimensions.modelId, identity: 'sha256:' + 'a'.repeat(64) };
const limits = { maxAttempts: 20, maxConcurrentAttempts: 2, maxTokens: 2, maxPromptTokens: 4, maxActivationBytes: 128, maxOutputCharacters: 128 };
async function setup(splitLayers, activationPrecision = 'f32') {
  const plan = createOrderedLayerPartitionPlan({ ...dimensions, splitLayers, activationPrecision });
  const planId = hashLayerPartitionPlan(plan);
  const stages = plan.partitions.map(({ index }) => ({ index, participantId: `peer-${index}`, transportId: `transport-${index}`,
    residentId: `resident-${index}`, executionId: 'sha256:' + 'b'.repeat(64) }));
  const identity = attemptId => ({ schema: 'doppler.resident-partition-attempt/v2', modelId: model.id, modelIdentity: model.identity,
    planId, threadId: `thread-${attemptId}`, attemptId, requesterId: 'requester', placementGeneration: 0, stages });
  const sessions = [];
  for (let index = 0; index < plan.stageCount; index++) {
    const allocation = resolveResidentPartitionAllocation({ modelId: model.id, architecture: dimensions }, model.identity,
      { model, plan, planId, index, participantId: stages[index].participantId, generation, limits });
    assert.throws(() => resolveResidentPartitionAllocation({ modelId: model.id, architecture: dimensions }, model.identity,
      { ...allocation, canonicalGeneration: {} }), /not implemented/);
    const pipeline = { modelPartition: { plan, index }, gpuContext: null, modelConfig: {}, dopplerLoader: { gpuBuffers: [] } };
    const session = await createResidentPartitionSession(pipeline, allocation, {
      tokenize: () => [1, 2], getTokenContract() { assert.equal(index, plan.stageCount - 1); return {}; },
      createIncrementalDecoder() {
        decoderOwners.push(index); assert.equal(index, plan.stageCount - 1);
        return createBundledIncrementalDecoder({ tokenForId: id => String(id), byteLevel: false,
          byteDecoder: new Map(), wordPiece: false, splitEveryCharacter: false, wordPiecePrefix: '##' });
      },
    }, async () => {});
    assert.equal(session.getDescriptor().schema, 'doppler.resident-partition/v2');
    assert.equal(session.getDescriptor().stageCount, plan.stageCount);
    sessions.push(session);
  }
  return { plan, sessions, identity };
}
function input(identity, partitionIndex, step = 0, continuation = null, activation) {
  return { identity, partitionIndex, step, tokenPosition: step ? 2 : 0, inputTokenCount: step ? 1 : 2,
    inputTokenIds: step ? [3] : [1, 2], generation, generationDigest, maxTokens: 2, continuation, activation, signal };
}
function decoded(result) {
  const { data, ...metadata } = result.activationTensor;
  return { ...metadata, tensorData: new Float32Array(data) };
}
try {
  for (const cuts of [[14], [3, 19], [1, 8, 24], [1, 3, 8, 14, 18, 27]]) {
    for (const precision of ['f32', 'two-f32']) {
      const { plan, sessions, identity } = await setup(cuts, precision), id = identity('whole');
      assert.deepEqual((await sessions[0].tokenize({ identity: id, messages: 'prompt', signal })).tokenIds, [1, 2]);
      const continuations = sessions.map(() => null);
      for (let step = 0; step < 2; step++) {
        let activation;
        for (let index = 0; index < sessions.length; index++) {
          const result = await sessions[index].executeStage(input(id, index, step, continuations[index], activation));
          assert.deepEqual(result.identity, id); assert.equal(result.partitionIndex, index); assert.equal(result.step, step);
          continuations[index] = result.continuation;
          if (index < sessions.length - 1) { assert(!('tokenId' in result)); activation = decoded(result); }
          else { assert.equal(result.tokenId, 3 + step); assert.equal(result.done, step === 1); }
        }
      }
      for (const session of sessions) {
        await assert.rejects(session.executeGroup0(input(id, 0)), /Legacy/);
        await session.closeAttempt({ identity: id }); await session.close();
      }
      assert(created.slice(-plan.stageCount).every(state => state.currentSeqLen === 3 && state.decodeStepCount === 2));
    }
  }
  const { sessions, identity } = await setup([7, 14, 21]);
  const base = input(identity('guard'), 0), before = dispatches;
  for (const changed of [{ partitionIndex: 1 }, { generationDigest: 'changed' }, { continuation: {} },
    { identity: { ...base.identity, modelIdentity: 'other' } },
    { identity: { ...base.identity, stages: base.identity.stages.slice().reverse() } },
    { inputTokenCount: 20 }, { activation: {} }]) await assert.rejects(sessions[0].executeStage({ ...base, ...changed }));
  assert.equal(dispatches, before);
  const first = await sessions[0].executeStage(base);
  await assert.rejects(sessions[0].executeStage({ ...base, identity: { ...base.identity, requesterId: 'other' } }), /collision/);
  await assert.rejects(sessions[0].executeStage({ ...base, identity: { ...base.identity, placementGeneration: 1 } }), /collision/);
  const middle = input(identity('guard'), 1, 0, null, decoded(first));
  await assert.rejects(sessions[1].executeStage({ ...middle, activation: undefined }), /role/);
  await assert.rejects(sessions[1].executeStage({ ...middle, activation: { ...middle.activation, step: 4 } }), /shape/);
  await sessions[0].closeAttempt({ identity: identity('guard') });
  // Independent attempts on the same resident cannot reuse each other's continuation.
  const a = await sessions[0].executeStage(input(identity('a'), 0));
  const b = await sessions[0].executeStage(input(identity('b'), 0));
  await assert.rejects(sessions[0].executeStage(input(identity('a'), 0, 1, b.continuation)), /continuation/);
  await sessions[0].closeAttempt({ identity: identity('a') });
  const result = await sessions[0].executeStage(input(identity('b'), 0, 1, b.continuation));
  assert(result.activationTensor);
  await sessions[0].closeAttempt({ identity: identity('b') });
  // Closing an in-flight attempt waits for numerical settlement, without cancelling the resident.
  const entered = Promise.withResolvers(), release = Promise.withResolvers();
  globalThis.orderedResidentFixture.pause = () => { entered.resolve(); return release.promise; };
  const pending = sessions[0].executeStage(input(identity('cancel'), 0)); await entered.promise;
  const prior = closes, closing = sessions[0].closeAttempt({ identity: identity('cancel') });
  await Promise.resolve(); assert.equal(closes, prior);
  release.resolve(); await assert.rejects(pending, /closed|abort/i); await closing;
  assert.equal(closes, prior + 1); delete globalThis.orderedResidentFixture.pause;
  for (const session of sessions) await session.close();
  assert.equal(closes, created.length, 'Every private numerical state is released once.');
} finally { hooks.deregister(); delete globalThis.orderedResidentFixture; }
console.log('ordered resident execution passed: real stage dispatch admission, final-only sampling, private attempts and cancellation; numerical work injected');
