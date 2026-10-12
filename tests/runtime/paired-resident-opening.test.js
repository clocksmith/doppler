// Synthetic admission proof only: test authority, synthetic shader/model and no GPU.
import assert from 'node:assert/strict';
import { createDopplerRun } from '../../src/client/runtime/composition-root.js';
import { resolveCapsuleProgramLoadOptions } from '../../src/client/model-host/index.js';
import { createTargetPlanV2 } from '../../src/config/target-plan.js';
import { createInitialExecutionIdentityV2 } from '../../src/config/initial-execution-identity.js';
import { createLayerPartitionPlan, hashLayerPartitionPlan } from '../../src/inference/pipelines/text/layer-partition-contract.js';
import { resolveResidentPartitionAllocation } from '../../src/inference/pipelines/text/resident-partition-contract.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';

const digest = x => `sha256:${x.repeat(64)}`;
const manifest = { modelId: 'capsule-test-model', architecture: { numLayers: 2, hiddenSize: 2, vocabSize: 8 } };
const plan = createLayerPartitionPlan({ modelId: manifest.modelId, ...manifest.architecture, activationPrecision: 'two-f32' });
const identityFor = precision => createInitialExecutionIdentityV2({
  executionGraphHash: digest('3'), resolvedGraphHash: digest('4'),
  kernelClosure: [{ moduleId: 'main', file: 'main.wgsl', entry: 'main', digest: digest('6') }],
  dtypeLane: { activation: 'f32', output: 'f32', kv: 'f16', math: 'f32', accumulation: 'f32' },
  fusionSet: [], kvLayout: { layout: 'contiguous', kvDtype: 'f16' },
  memoryPolicy: { kvcache: { layout: 'contiguous', kvDtype: 'f16' } }, executionPlanDigest: digest('5'),
  runtimeEngine: { resolvedRuntimeSchema: 'doppler.resolved-runtime-session/v1' },
  programLoadPolicy: { schema: 'doppler.capsule-program-load-policy/v2', runtimeConfig: { inference: {
    session: { intermediatePrecision: precision, activationDtype: 'f32', linearAttentionStatePrecision: 'two-f32' },
    compute: { outputDtype: 'f32' }, generation: { disableMultiTokenDecode: true },
  } } },
});
const identity = identityFor('two-f32');
const base = await createSignedCapsuleFixture({ manifest, initialExecutionIdentity: identity });
const target = createTargetPlanV2({ ...base.targetPlan, qualification: [...base.targetPlan.qualification, ...[0, 1].map(index => ({
  surface: 'test-webgpu', status: 'passed', operation: 'residentPartition',
  evidenceArtifactId: 'evidence', evidenceHash: base.capsule.artifacts.find(x => x.artifactId === 'evidence').hash,
  transcriptHash: digest('7'), partitionPlanHash: hashLayerPartitionPlan(plan), partitionIndex: index, comparedSteps: 1,
}))] });
const fixture = await createSignedCapsuleFixture({ manifest, targetPlans: [target] });
let observedIdentity = identity, closeCount = 0, loadedPolicy;
const runtime = createDopplerRun({
  device: { getProfile: () => ({ surface: 'test-webgpu', maxBufferSize: 1024 }), getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }) },
  trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY }, artifactStore: fixture.artifactStore,
  resolveResidentPartitionAllocation,
  async programFactory({ targetPlan, options }) {
    loadedPolicy = resolveCapsuleProgramLoadOptions(targetPlan).runtimeConfig;
    assert.equal(loadedPolicy.inference.session.intermediatePrecision, 'two-f32');
    const allocation = options.residentPartition;
    assert.equal(allocation.plan.schema, 'doppler.layer-partition-contract/v2');
    assert.equal(allocation.plan.activationPrecision, 'two-f32');
    assert.equal(allocation.plan.activationComponents, 2);
    assert.equal(allocation.plan.activationElementStrideBytes, 8);
    return { getInitialExecutionIdentity: () => observedIdentity, close: async () => { closeCount++; }, residentPartition: {
      getDescriptor: () => ({ schema: 'doppler.resident-partition/v1', ready: true,
        modelId: allocation.model.id, modelIdentity: allocation.model.identity, planId: allocation.planId,
        index: allocation.index, layerRange: allocation.plan.partitions[allocation.index].layerRange,
        generationDigest: computeCanonicalSha256(allocation.generation) }),
      getRecoveryCapabilities: () => ({ schema: 'doppler.resident-recovery/v1', inputReplay: false, checkpointExport: false, checkpointImport: false }),
      tokenize: async () => {}, executeGroup0: async () => {}, executeGroup1: async () => {}, closeAttempt: async () => {}, close: async () => {},
    } };
  },
});
const allocation = index => ({ model: { id: manifest.modelId, identity: fixture.capsule.artifacts.find(x => x.artifactId === 'manifest').hash },
  plan, planId: hashLayerPartitionPlan(plan), index, participantId: index ? 'B' : 'A',
  generation: { maxTokens: 1, maxSeqLen: 8, temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false },
  limits: { maxTokens: 1, maxPromptTokens: 4, maxActivationBytes: 64, maxOutputCharacters: 100, maxAttempts: 2, maxConcurrentAttempts: 1 } });
for (const index of [0, 1]) {
  const session = await runtime.openCapsule(fixture.capsule, { residentPartition: allocation(index) });
  assert.equal(session.residentPartition.getDescriptor().index, index);
  loadedPolicy.inference.session.intermediatePrecision = 'f32';
  assert.equal(session.selectedPlan.initialExecutionIdentity.programLoadPolicy.runtimeConfig.inference.session.intermediatePrecision, 'two-f32', 'load options cannot mutate signed policy');
  await session.close();
}
observedIdentity = identityFor('f32');
const before = closeCount;
await assert.rejects(runtime.openCapsule(fixture.capsule, { residentPartition: allocation(0) }), /execution identity|programLoadPolicy/i);
assert.equal(closeCount, before + 1, 'scalar substitution closes the unadmitted program');
const altered = structuredClone(allocation(1));
altered.plan.activationElementStrideBytes = 4;
await assert.rejects(runtime.openCapsule(fixture.capsule, { residentPartition: altered }), /exact plan digest|match the model manifest/);
console.log('synthetic signed paired opening preserves both partition policies and rejects scalar/stride substitution');
