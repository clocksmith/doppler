import assert from 'node:assert/strict';
import { createResidentPartitionSession } from '../../src/inference/pipelines/text/resident-partition.js';
import { scopePipelineShaders } from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import { getSharedDeviceState } from '../../src/gpu/device-state.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { resolveGenerationOptions } from '../../src/config/generation-contract.js';

const plan = { schema: 'doppler.layer-partition-contract/v1', activationDtype: 'f32', vocabSize: 8,
  partitions: [{ layerRange: [0, 0] }, { layerRange: [1, 1] }] };
const planId = computeCanonicalSha256(plan);
const allocation = { model: { id: 'model', identity: 'sha256:model' }, plan, planId, index: 0,
  participantId: 'left', generation: resolveGenerationOptions({ maxTokens: 1, maxSeqLen: 16, temperature: 0, topK: 1, topP: 1,
    repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false }), limits: { maxAttempts: 1, maxConcurrentAttempts: 1, maxPromptTokens: 4 } };
const identity = attemptId => ({ modelId: 'model', modelIdentity: 'sha256:model', planId,
  threadId: attemptId, attemptId, participantA: 'left', participantB: 'right' });

async function sessionWith(tokenize) {
  const device = {};
  let closes = 0, openingCacheCloses = 0;
  const pipeline = { useGPU: true, modelPartition: { plan, index: 0 },
    kvCache: { destroy() { openingCacheCloses++; } },
    modelConfig: { useMoE: false, numKvSharedLayers: 0, hiddenSizePerLayerInput: null,
      decodeStrategy: 'incremental', causalAttention: true },
    executionPlanState: { primaryPlan: { activationDtype: 'f32', finitenessGuardEnabled: false } },
    runtimeConfig: { inference: { session: { usePostFfnNextInputRMSNormPairFusion: false } } },
    gpuContext: null, dopplerLoader: { gpuBuffers: [] } };
  scopePipelineShaders(pipeline);
  const session = await createResidentPartitionSession(pipeline, allocation,
    { tokenize, createIncrementalDecoder() {}, getTokenContract() {} }, async () => { closes++; });
  assert.equal(openingCacheCloses, 1, 'Resident attempts must not retain the unused opening cache');
  assert.equal(pipeline.kvCache, null);
  assert.deepEqual(session.getRecoveryCapabilities(), { schema: 'doppler.resident-recovery/v1',
    inputReplay: false, checkpointExport: false, checkpointImport: false });
  assert.throws(() => { session.getRecoveryCapabilities().checkpointImport = true; }, TypeError);
  return { session, device, pipeline, get closes() { return closes; } };
}

let resolveTokens;
let enteredTokenization;
const entered = new Promise(resolve => { enteredTokenization = resolve; });
const delayed = new Promise(resolve => { resolveTokens = resolve; });
const active = await sessionWith(() => { enteredTokenization(); return delayed; });
const signal = new AbortController().signal;
const tokenization = active.session.tokenize({ identity: identity('pending'), messages: 'prompt', signal });
await entered;
const settled = active.session.closeAttempt({ identity: identity('pending') });
let closeSettled = false;
settled.then(() => { closeSettled = true; });
await Promise.resolve();
assert.equal(closeSettled, false, 'attempt closure must await tokenization');
resolveTokens([1]);
await assert.rejects(tokenization, /Resident attempt closed/);
await settled;
await active.session.closeAttempt({ identity: identity('pending') });
await assert.rejects(active.session.tokenize({ identity: identity('pending'), messages: 'prompt', signal }), /retired/);
await active.session.close();
assert.equal(active.closes, 1);
assert.equal(active.session.getDescriptor().ready, false);
assert.equal(active.session.getRecoveryCapabilities().inputReplay, false);
for (const id of ['unseen-a', 'unseen-b']) await active.session.closeAttempt({ identity: identity(id) });
await active.session.closeAttempt({ identity: identity('pending') });

const failed = await sessionWith(async () => { throw new Error('tokenizer failed'); });
await assert.rejects(failed.session.tokenize({ identity: identity('failure'), messages: 'prompt', signal }), /tokenizer failed/);
await failed.session.closeAttempt({ identity: identity('failure') });
await failed.session.close();
assert.equal(failed.closes, 1);

const lost = await sessionWith(() => [1]);
lost.pipeline.gpuContext = { device: lost.device };
getSharedDeviceState().lostDevices.add(lost.device);
assert.equal(lost.session.getDescriptor().ready, false);
assert.equal(lost.session.getRecoveryCapabilities().checkpointImport, false);
await assert.rejects(lost.session.tokenize({ identity: identity('lost'), messages: 'prompt', signal }), /device was lost/);
await lost.session.closeAttempt({ identity: identity('lost') });
await lost.session.close();
assert.equal(lost.closes, 1);

const textOnly = await sessionWith(() => [1]);
for (const messages of [[null], [{ role: 'user', content: 'text', images: ['image'] }]]) {
  await assert.rejects(textOnly.session.tokenize({ identity: identity('invalid'), messages, signal }), /text-only/);
}
// Rejected multimodal input never reserves the sole attempt slot.
await textOnly.session.tokenize({ identity: identity('valid'), messages: 'prompt', signal });
await textOnly.session.close();

// A new process/session has no state behind another resident's continuation.
const replacement = await sessionWith(() => [1]);
const oldContinuation = { nonce: 'previous-resident', step: 1, position: 1 };
const request = { identity: identity('interrupted'), tokenIds: [1], inputTokenCount: 1,
  maxTokens: 1, generation: allocation.generation, signal, continuation: oldContinuation };
await assert.rejects(replacement.session.executeGroup0({ ...request, step: 1, tokenPosition: 1 }), /out of order/);
await assert.rejects(replacement.session.executeGroup0({ ...request, step: 0, tokenPosition: 0 }), /continuation mismatch/);
await replacement.session.close();
const restarted = await sessionWith(() => [1]);
assert.deepEqual((await restarted.session.tokenize({ identity: identity('new-attempt'), messages: 'prompt', signal })).tokenIds, [1]);
await restarted.session.close();
