import assert from 'node:assert/strict';
import { createDopplerRun } from '../../src/client/runtime/composition-root.js';
import { createTargetPlanV2 } from '../../src/config/target-plan.js';
import { createInitialExecutionIdentity } from '../../src/config/initial-execution-identity.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';

const fixture = await createSignedCapsuleFixture();
let releaseProfile;
const profile = new Promise(resolve => { releaseProfile = resolve; });
let observed;
const runtime = createDopplerRun({
  device: { async getProfile() { await profile; return { surface: 'test-webgpu', maxBufferSize: 1024 }; },
    getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }) },
  trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
  artifactStore: fixture.artifactStore,
  resolveResidentPartitionAllocation(_manifest, _hash, allocation) { observed = allocation; return allocation; },
  async programFactory() { throw new Error('Unqualified partition must never load a program.'); },
});
const allocation = { planId: `sha256:${'a'.repeat(64)}`, index: 0, plan: { partitions: [{ layerRange: [0, 1] }] } };
const expected = structuredClone(allocation);
const opening = runtime.openCapsule(fixture.capsule, { residentPartition: allocation });
allocation.plan.partitions[0].layerRange[0] = 9;
releaseProfile();
await assert.rejects(opening, /no signed resident partition qualification/);
assert.deepEqual(observed, expected, 'direct opening snapshots allocation before asynchronous verification');

const withoutValidator = createDopplerRun({ ...runtime.ports,
  trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
  artifactStore: fixture.artifactStore,
  async programFactory() { throw new Error('Resident opening must not proceed without allocation validation.'); } });
await assert.rejects(withoutValidator.openCapsule(fixture.capsule, { residentPartition: expected }),
  /allocation validator port/);

const digest = character => `sha256:${character.repeat(64)}`;
const executionIdentity = createInitialExecutionIdentity({
  executionGraphHash: digest('3'), resolvedGraphHash: digest('4'),
  kernelClosure: [{ moduleId: 'main', file: 'main.wgsl', entry: 'main', digest: digest('6') }],
  dtypeLane: { activation: 'f32', output: 'f32', kv: 'f32', math: 'f32', accumulation: 'f32' },
  fusionSet: [], kvLayout: { layout: 'contiguous', kvDtype: 'f32' },
  memoryPolicy: { kvcache: { layout: 'contiguous', kvDtype: 'f32' } },
  executionPlanDigest: digest('5'), runtimeEngine: { resolvedRuntimeSchema: 'doppler.resolved-runtime-session/v1' },
});
const base = await createSignedCapsuleFixture({ initialExecutionIdentity: executionIdentity });
const qualifiedPlan = createTargetPlanV2({ ...base.targetPlan, qualification: [...base.targetPlan.qualification, {
  surface: 'test-webgpu', status: 'passed', operation: 'residentPartition',
  evidenceArtifactId: 'evidence', evidenceHash: base.capsule.artifacts.find(row => row.artifactId === 'evidence').hash,
  transcriptHash: digest('7'), partitionPlanHash: expected.planId, partitionIndex: 0, comparedSteps: 1,
}] });
const signed = await createSignedCapsuleFixture({ targetPlans: [qualifiedPlan] });
const assigned = { ...expected, model: { id: signed.capsule.modelId,
  identity: signed.capsule.artifacts.find(row => row.artifactId === 'manifest').hash }, generation: { maxTokens: 1 } };
const resident = { getDescriptor: () => ({ schema: 'doppler.resident-partition/v1', ready: true, modelId: assigned.model.id,
  modelIdentity: assigned.model.identity, planId: assigned.planId, index: assigned.index,
  layerRange: assigned.plan.partitions[0].layerRange, generationDigest: computeCanonicalSha256(assigned.generation) }),
  getRecoveryCapabilities: () => ({ schema: 'doppler.resident-recovery/v1', inputReplay: false,
    checkpointExport: false, checkpointImport: false }),
  tokenize: async () => {}, executeGroup0: async () => {}, executeGroup1: async () => {},
  closeAttempt: async () => {}, close: async () => {} };
let closedPrograms = 0;
const authorized = createDopplerRun({
  device: { getProfile: () => ({ surface: 'test-webgpu', maxBufferSize: 1024 }),
    getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }) },
  trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY }, artifactStore: signed.artifactStore,
  resolveResidentPartitionAllocation: () => assigned,
  async programFactory() { return { getInitialExecutionIdentity: () => executionIdentity,
    residentPartition: resident, close: async () => { closedPrograms++; } }; },
});
const session = await authorized.openCapsule(signed.capsule, { residentPartition: assigned });
assert.equal(session.residentPartition.getDescriptor().planId, assigned.planId);
assert.deepEqual(session.residentPartition.getRecoveryCapabilities(), resident.getRecoveryCapabilities());
assert(Object.isFrozen(session.residentPartition.getRecoveryCapabilities()));
await session.close();
assert.equal(closedPrograms, 1);
assert.equal(session.residentPartition.getDescriptor().ready, false);
await assert.rejects(session.residentPartition.tokenize({ signal: new AbortController().signal }), /closed/);

async function rejectOpening(pattern) {
  const before = closedPrograms;
  await assert.rejects(authorized.openCapsule(signed.capsule, { residentPartition: assigned }), pattern);
  assert.equal(closedPrograms, before + 1, 'failed resident admission must close its loaded program');
}
const descriptor = resident.getDescriptor;
for (const change of [{ schema: 'doppler.resident-partition/v2' }, { ready: 'true' }, { ready: false },
  { modelId: 'other' }, { modelIdentity: digest('e') }, { planId: digest('f') }, { index: 1 },
  { layerRange: [0, 2] }, { generationDigest: digest('a') }]) {
  resident.getDescriptor = () => ({ ...descriptor(), ...change });
  await rejectOpening(/descriptor differs/);
}
resident.getDescriptor = descriptor;
for (const method of Object.keys(resident)) {
  const saved = resident[method];
  resident[method] = undefined;
  await rejectOpening(/missing required partition methods/);
  resident[method] = saved;
}
const capabilities = resident.getRecoveryCapabilities;
for (const change of [{ schema: 'doppler.resident-recovery/v2' }, { inputReplay: true },
  { checkpointExport: true }, { checkpointImport: true }, { inputReplay: undefined }]) {
  resident.getRecoveryCapabilities = () => ({ ...capabilities(), ...change });
  await rejectOpening(/recovery capabilities are unsupported/);
}
resident.getRecoveryCapabilities = capabilities;
