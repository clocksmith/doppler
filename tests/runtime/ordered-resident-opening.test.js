// Signed public-opener contract proof with synthetic program execution; no GPU qualification.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createDopplerRun } from '../../src/client/runtime/composition-root.js';
import { createTargetPlanV2 } from '../../src/config/target-plan.js';
import { createInitialExecutionIdentity } from '../../src/config/initial-execution-identity.js';
import { createOrderedLayerPartitionPlan, hashLayerPartitionPlan } from '../../src/partitions.js';
import { resolveResidentPartitionAllocation } from '../../src/inference/pipelines/text/resident-partition-contract.js';
import { computeCanonicalSha256 as hash } from '../../src/formats/canonical-hash.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';

const digest = value => `sha256:${value.repeat(64)}`;
async function fixture(count = 3) {
  const manifest = { modelId: 'capsule-test-model', architecture: { numLayers: count, hiddenSize: 4, vocabSize: 8 } };
  const plan = createOrderedLayerPartitionPlan({ modelId: manifest.modelId, ...manifest.architecture,
    splitLayers: Array.from({ length: count - 1 }, (_, i) => i + 1), activationPrecision: 'two-f32' });
  const initial = createInitialExecutionIdentity({ executionGraphHash: digest('3'), resolvedGraphHash: digest('4'),
    kernelClosure: [{ moduleId: 'main', file: 'main.wgsl', entry: 'main', digest: digest('6') }],
    dtypeLane: { activation: 'f32', output: 'f32', kv: 'f32', math: 'f32', accumulation: 'f32' }, fusionSet: [],
    kvLayout: { layout: 'contiguous', kvDtype: 'f32' }, memoryPolicy: { kvcache: { layout: 'contiguous', kvDtype: 'f32' } },
    executionPlanDigest: digest('5'), runtimeEngine: { resolvedRuntimeSchema: 'doppler.resolved-runtime-session/v1' } });
  const base = await createSignedCapsuleFixture({ manifest, initialExecutionIdentity: initial });
  const modelIdentity = base.capsule.artifacts.find(a => a.artifactId === 'manifest').hash;
  const contract = { schema: 'doppler.resident-qualification-contract/v2', manifestIdentity: modelIdentity,
    executionGraphHash: initial.executionGraphHash, partitionPlan: plan, partitionPlanHash: hashLayerPartitionPlan(plan),
    referenceHash: digest('7'), evaluatorHash: digest('8'), tolerance: 0.001, comparedSteps: 1,
    stages: plan.partitions.map((_, index) => ({ index, surface: 'browser-webgpu', providerIdentityHash: hash({ index, synthetic: true }) })) };
  const target = createTargetPlanV2({ ...base.targetPlan, qualification: contract.stages.map(stage => ({
    operation: 'residentPartition', status: 'passed', surface: stage.surface, partitionIndex: stage.index,
    partitionPlanHash: contract.partitionPlanHash, evidenceArtifactId: 'evidence',
    evidenceHash: base.capsule.artifacts.find(a => a.artifactId === 'evidence').hash,
    transcriptHash: digest('9'), comparedSteps: 1, providerIdentityHash: stage.providerIdentityHash, orderedContract: contract,
  })) });
  const signed = await createSignedCapsuleFixture({ manifest, targetPlans: [target] });
  const state = { calls: [], closes: 0, reads: 0, descriptorChange: {}, omit: null, observed: initial,
    onLoad: null, onStage: null, closeError: null };
  const allocation = index => ({ model: { id: manifest.modelId, identity: modelIdentity }, plan,
    planId: hashLayerPartitionPlan(plan), index, participantId: `stage-${index}`,
    generation: { maxTokens: 1, maxSeqLen: 8, temperature: 0, topK: 1, topP: 1,
      repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false },
    limits: { maxTokens: 1, maxPromptTokens: 4, maxActivationBytes: 128, maxOutputCharacters: 100,
      maxAttempts: 2, maxConcurrentAttempts: 1, maxPreparationMs: 60000 } });
  const runtime = createDopplerRun({
    device: { getProfile: () => ({ surface: 'browser-webgpu', maxBufferSize: 1024 }),
      getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }) },
    trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
    artifactStore: { ...signed.artifactStore, async readArtifact(a) { state.reads++; return signed.artifactStore.readArtifact(a); } },
    resolveResidentPartitionAllocation,
    async programFactory({ options }) {
      const assigned = options.residentPartition;
      const resident = {
        getDescriptor: () => ({ schema: 'doppler.resident-partition/v2', ready: true, stageCount: count,
          modelId: assigned.model.id, modelIdentity: assigned.model.identity, planId: assigned.planId,
          index: assigned.index, layerRange: assigned.plan.partitions[assigned.index].layerRange,
          generationDigest: hash(assigned.generation), ...state.descriptorChange }),
        getRecoveryCapabilities: () => ({ schema: 'doppler.resident-recovery/v1', inputReplay: false,
          checkpointExport: false, checkpointImport: false }),
        tokenize: async request => ({ identity: request.identity, tokenIds: [1] }),
        executeStage: async request => { state.calls.push(request); return state.onStage?.(request) ?? { partitionIndex: assigned.index }; },
        closeAttempt: async request => { state.calls.push({ closeAttempt: request }); }, close: async () => {},
      };
      if (state.omit) delete resident[state.omit];
      state.onLoad?.(options);
      return { getInitialExecutionIdentity: () => state.observed, residentPartition: resident,
        async close() { state.closes++; if (state.closeError) throw state.closeError; } };
    },
  });
  return { state, allocation, open: (index = 0, options = {}) => runtime.openCapsule(signed.capsule,
    { residentPartition: allocation(index), ...options }) };
}

for (const count of [2, 3, 7]) test(`signed ordered ${count}-stage facade admits every exact stage and exposes executeStage`, async () => {
  const f = await fixture(count);
  for (let index = 0; index < count; index++) {
    const session = await f.open(index);
    const resident = session.residentPartition;
    assert.equal(resident.getDescriptor().schema, 'doppler.resident-partition/v2');
    assert.equal(resident.getDescriptor().stageCount, count);
    assert.equal(resident.getDescriptor().index, index);
    const request = { signal: new AbortController().signal, partitionIndex: index, identity: { attemptId: 'one' } };
    assert.deepEqual(await resident.executeStage(request), { partitionIndex: index });
    assert.equal(f.state.calls.at(-1).signal, request.signal);
    await assert.rejects(resident.executeGroup0(request), /Legacy.*ordered|two-stage/i);
    await assert.rejects(resident.executeGroup1(request), /Legacy.*ordered|two-stage/i);
    await assert.rejects(resident.executeCanonicalGroup0(request), /not allocated/);
    await resident.closeAttempt({ identity: request.identity });
    await resident.close();
    assert.equal(resident.getDescriptor().ready, false);
    await assert.rejects(resident.executeStage(request), /closed/);
  }
  assert.equal(f.state.closes, count);
});

test('ordered opening rejects incompatible descriptors and missing executor after acquisition and closes each program', async () => {
  const f = await fixture();
  for (const change of [{ schema: 'doppler.resident-partition/v1' }, { stageCount: undefined }, { stageCount: 2 },
    { stageCount: '3' }, { ready: false }, { index: 1 }, { layerRange: [0, 3] }, { planId: digest('f') },
    { modelIdentity: digest('e') }, { generationDigest: digest('d') }]) {
    f.state.descriptorChange = change;
    const before = f.state.closes;
    await assert.rejects(f.open(), /descriptor differs/);
    assert.equal(f.state.closes, before + 1);
  }
  f.state.descriptorChange = {};
  f.state.omit = 'executeStage';
  await assert.rejects(f.open(), /missing required partition methods/);
  assert(f.state.reads > 0, 'real signed artifact acquisition preceded program validation');
});

test('cancellation after program acquisition and invalid descriptor cleanup failure retain their actual cause', async () => {
  const f = await fixture();
  const controller = new AbortController();
  const cancelled = new Error('cancelled while opening');
  f.state.onLoad = () => controller.abort(cancelled);
  await assert.rejects(f.open(0, { signal: controller.signal }), error => error === cancelled);
  assert.equal(f.state.closes, 1);
  f.state.onLoad = null;
  f.state.descriptorChange = { stageCount: 99 };
  f.state.closeError = new Error('cleanup unsettled');
  await assert.rejects(f.open(), error => error instanceof AggregateError
    && /descriptor differs/.test(error.cause.message) && error.errors.includes(f.state.closeError));
  assert.equal(f.state.closes, 2);
});

test('ordered execution snapshots caller data, forwards cancellation and rejects changed execution identity before publication', async () => {
  const f = await fixture();
  const session = await f.open();
  const request = { signal: new AbortController().signal, identity: { attemptId: 'original' } };
  const pending = session.residentPartition.executeStage(request);
  request.identity.attemptId = 'mutated';
  await pending;
  assert.equal(f.state.calls[0].identity.attemptId, 'original');
  f.state.onStage = () => { f.state.observed = { ...f.state.observed, digest: digest('f') }; return { output: 'must not publish' }; };
  await assert.rejects(session.residentPartition.executeStage(request), /identity/i);
  assert.equal(f.state.calls.length, 2);
  await assert.rejects(session.residentPartition.executeStage(request), /identity/i);
  assert.equal(f.state.calls.length, 2, 'pre-dispatch identity guard prevents execution');
  await session.close();
});

test('ordered plan mutation rejects before program acquisition', async () => {
  const f = await fixture();
  const allocation = f.allocation(0);
  allocation.plan = structuredClone(allocation.plan);
  allocation.plan.partitions[0].layerRange[1]++;
  await assert.rejects(f.open(0, { residentPartition: allocation }), /plan|partition/i);
  assert.equal(f.state.closes, 0);
});

test('ordered cancellation remains the original signal and attempt cleanup remains callable', async () => {
  const f = await fixture();
  const session = await f.open();
  const controller = new AbortController();
  const cancelled = new Error('cancel this attempt');
  let entered;
  const ready = new Promise(resolve => { entered = resolve; });
  f.state.onStage = request => new Promise((resolve, reject) => {
    assert.equal(request.signal, controller.signal);
    request.signal.addEventListener('abort', () => reject(request.signal.reason), { once: true });
    entered();
  });
  const pending = session.residentPartition.executeStage({ signal: controller.signal, identity: { attemptId: 'cancel-me' } });
  await ready;
  controller.abort(cancelled);
  await assert.rejects(pending, error => error === cancelled);
  await session.residentPartition.closeAttempt({ identity: { attemptId: 'cancel-me' } });
  assert.deepEqual(f.state.calls.at(-1), { closeAttempt: { identity: { attemptId: 'cancel-me' } } });
  await session.close();
  assert.equal(f.state.closes, 1);
});
