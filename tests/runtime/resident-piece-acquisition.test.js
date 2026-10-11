import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createDopplerRun } from '../../src/capsule-runtime.js';
import { createResidentPartitionFactory } from '../../src/partitions.js';
import { createVerifiedCapsuleArtifactStore } from '../../src/client/runtime/verified-capsule-artifact-store.js';
import { buildCapsuleV2, signCapsuleV2, validateCapsuleV2 } from '../../src/config/capsule-v2.js';
import { buildCapsuleV3, signCapsuleV3, validateCapsuleV3 } from '../../src/config/capsule-v3.js';
import { createTargetPlanV2 } from '../../src/config/target-plan.js';
import { createInitialExecutionIdentity } from '../../src/config/initial-execution-identity.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY, TEST_CAPSULE_PRIVATE_KEY } from '../helpers/capsule-v2-fixture.js';
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const digest = value => `sha256:${value.repeat(64)}`;
const signer = { authority: TEST_CAPSULE_AUTHORITY, publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY, privateKeyJwk: TEST_CAPSULE_PRIVATE_KEY };
const weights = Uint8Array.from([1, 2, 3, 4]);
const manifest = { modelId: 'capsule-test-model', hashAlgorithm: 'sha256', shards: [
  { index: 0, filename: 'weights.bin', size: 4, hash: sha(weights).slice(7) }] };
const identity = createInitialExecutionIdentity({ executionGraphHash: digest('3'), resolvedGraphHash: digest('4'),
  kernelClosure: [{ moduleId: 'main', file: 'main.wgsl', entry: 'main', digest: digest('6') }],
  dtypeLane: { activation: 'f32', output: 'f32', kv: 'f32', math: 'f32', accumulation: 'f32' },
  fusionSet: [], kvLayout: { layout: 'contiguous', kvDtype: 'f32' },
  memoryPolicy: { kvcache: { layout: 'contiguous', kvDtype: 'f32' } },
  executionPlanDigest: digest('5'), runtimeEngine: { resolvedRuntimeSchema: 'doppler.resolved-runtime-session/v1' } });
const base = await createSignedCapsuleFixture({ manifest, initialExecutionIdentity: identity });
const plan = createTargetPlanV2({ ...base.targetPlan, qualification: [...base.targetPlan.qualification, {
  surface: 'test-webgpu', status: 'passed', operation: 'residentPartition', evidenceArtifactId: 'evidence',
  evidenceHash: base.capsule.artifacts.find(row => row.artifactId === 'evidence').hash,
  transcriptHash: digest('7'), partitionPlanHash: digest('8'), partitionIndex: 0, comparedSteps: 1 }] });
const index = { schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(encode(manifest)), maxPieceBytes: 2,
  files: [{ path: 'weights.bin', size: 4, pieces: [0, 2].map(offset => ({ offset, size: 2, identity: sha(weights.slice(offset, offset + 2)) })) }] };
async function fixture(changeIndex = () => {}, changeCapsule = () => {}) {
  const record = structuredClone(index); changeIndex(record);
  const bytes = encode(record), source = new Map(base.artifactBytes); source.set('pieces', bytes);
  const draft = { ...base.capsule, targetPlans: [plan], program: { ...base.capsule.program, pieceIndexArtifactId: 'pieces' },
    artifacts: [...structuredClone(base.capsule.artifacts), { artifactId: 'pieces', role: 'piece-index', path: 'pieces.json', hash: sha(bytes), sizeBytes: bytes.length }] };
  changeCapsule(draft);
  return { capsule: await signCapsuleV2(buildCapsuleV2(draft), signer), source };
}
const qualified = await fixture();
assert(validateCapsuleV2(qualified.capsule).ok);
assert(validateCapsuleV3(await signCapsuleV3(buildCapsuleV3(qualified.capsule), signer)).ok);
const allocation = { model: { id: manifest.modelId, identity: sha(encode(manifest)) },
  planId: digest('8'), index: 0, plan: { partitions: [{ layerRange: [0, 1] }] }, generation: { maxTokens: 1 } };
function runtime(f, { index: partitionIndex = 0, readRange, cleanup = async () => {}, failProgram = false } = {}) {
  const reads = [], ranges = [];
  const store = { async readArtifact(artifact) { reads.push(artifact.artifactId); return f.source.get(artifact.artifactId); },
    async readArtifactRange(artifact, offset, length, options) {
      ranges.push([artifact.artifactId, offset, length]); options.signal.throwIfAborted();
      return readRange ? readRange(artifact, offset, length) : f.source.get(artifact.artifactId).slice(offset, offset + length);
    } };
  const assigned = { ...allocation, index: partitionIndex };
  const resident = { getDescriptor: () => ({ schema: 'doppler.resident-partition/v1', ready: true,
    modelId: manifest.modelId, modelIdentity: allocation.model.identity, planId: allocation.planId, index: 0,
    layerRange: [0, 1], generationDigest: computeCanonicalSha256(allocation.generation) }),
    getRecoveryCapabilities: () => ({ schema: 'doppler.resident-recovery/v1', inputReplay: false, checkpointExport: false, checkpointImport: false }),
    tokenize: async () => {}, executeGroup0: async () => {}, executeGroup1: async () => {}, closeAttempt: async () => {}, close: cleanup };
  const run = createDopplerRun({ device: { getProfile: () => ({ surface: 'test-webgpu', maxBufferSize: 1024 }),
    getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }) },
    trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY }, artifactStore: store,
    resolveResidentPartitionAllocation: () => assigned,
    async programFactory({ artifactStore }) {
      assert.deepEqual([...await artifactStore.readArtifactRange(f.capsule.artifacts.find(row => row.artifactId === 'weights'), 0, 1)], [1]);
      return { getInitialExecutionIdentity: () => failProgram ? null : identity, residentPartition: resident, close: cleanup };
    } });
  return { run, store, reads, ranges, open: options => run.openCapsule(f.capsule, { ...options, residentPartition: assigned }) };
}
const active = runtime(qualified), session = await active.open();
assert.deepEqual(active.ranges, [['weights', 0, 2]]);
assert(!active.reads.includes('weights'), 'unrelated weight pieces were not acquired');
assert.equal(session.verification.artifactVerificationScope, 'resident-pieces');
assert(!session.verification.artifactReceipts.some(row => row.artifactId === 'weights'), 'no invented whole-artifact receipt');
assert.equal(session.verification.pieceVerification.verifiedBytes, 2);
assert.deepEqual(session.verification.pieceVerification.ranges, [{ path: 'weights.bin', offset: 0, size: 2, identity: sha(weights.slice(0, 2)) }]);
await session.close();
const unsplit = runtime(qualified), ordinary = await unsplit.run.openCapsule(qualified.capsule);
assert(unsplit.reads.includes('weights'));
assert.equal(ordinary.verification.artifactVerificationScope, undefined);
assert.equal(ordinary.verification.artifactReceipts.length, qualified.capsule.artifacts.length);
await ordinary.close();
const rejected = runtime(qualified, { index: 1 });
await assert.rejects(rejected.open(), /no signed resident partition qualification/);
assert.deepEqual(rejected.ranges, []);
assert(!rejected.reads.includes('weights'));
await assert.rejects(runtime(qualified, { readRange: () => new Uint8Array(2) }).open(), /integrity/);
await assert.rejects(runtime(qualified, { readRange: () => [1, 2] }).open(), /integrity/);
await assert.rejects(runtime(qualified).open({ maxAcquisitionChunkBytes: 1 }), /acquisition limit|Invalid Capsule loading/);
for (const change of [value => { value.schema = 'doppler.verified-pieces/v2'; },
  value => { value.manifestIdentity = digest('0'); }, value => { value.files[0].pieces[1].offset = 1; },
  value => { value.files[0].pieces.pop(); }, value => { value.files[0].pieces[1].size = 3; },
  value => { value.files.push(structuredClone(value.files[0])); }, value => { value.files[0].size = 5; }]) {
  const f = await fixture(change), r = runtime(f);
  await assert.rejects(r.open(), /Piece|piece/); assert.deepEqual(r.ranges, []);
}
await assert.rejects(fixture(() => {}, value => { value.program.pieceIndexArtifactId = 'weights'; }), /piece-index/);
await assert.rejects(fixture(() => {}, value => { value.artifacts.push(value.artifacts[0]); }), /duplicate/);
const changed = await fixture(); changed.source.set('pieces', encode({ invalid: true }));
await assert.rejects(runtime(changed).open(), /hash or size mismatch/);
const badManifest = await fixture(() => {}, value => { value.artifacts.find(row => row.artifactId === 'weights').hash = digest('0'); });
await assert.rejects(runtime(badManifest).open(), /manifest.*artifact disagree/);
const failure = runtime(qualified, { failProgram: true, cleanup: async () => { throw new Error('cleanup-sentinel'); } });
await assert.rejects(failure.open(), error => error instanceof AggregateError && error.errors.some(row => row.message === 'cleanup-sentinel'));
const whole = createVerifiedCapsuleArtifactStore(qualified.capsule, active.store);
await whole.hashArtifact(qualified.capsule.artifacts.find(row => row.artifactId === 'weights'));
assert(active.reads.includes('weights'), 'ordinary full-hash API still reads and verifies the complete artifact');
await whole.close();
console.log('resident-piece-acquisition: signed selective opening, honest receipts, gates, corruption and cleanup passed');

const stalledSource = runtime(qualified, { readRange: () => new Promise(() => {}) });
const closingStore = createVerifiedCapsuleArtifactStore(qualified.capsule, stalledSource.store);
await closingStore.enableResidentPieces();
let settled = false;
const stalled = closingStore.readArtifactRange(qualified.capsule.artifacts.find(row => row.artifactId === 'weights'), 0, 1)
  .then(() => { throw new Error('Stalled source must not resolve'); }, error => { settled = true; return error; });
while (!stalledSource.ranges.length) await new Promise(resolve => setTimeout(resolve, 0));
await closingStore.close();
assert.equal(settled, true);
assert.equal(closingStore.getPieceVerification().activeReadBytes, 0);
assert.match((await stalled).message, /closed/);

const relativePath = await fixture(() => {}, value => { value.artifacts.find(row => row.artifactId === 'weights').path = './weights.bin'; });
const relativeSession = await runtime(relativePath).open();
await relativeSession.close();

const publicOpening = runtime(qualified);
const publicFactory = createResidentPartitionFactory({
  openCapsule: (capsule, options) => publicOpening.run.openCapsule(capsule, options), capsuleOptions: {},
});
const publicResident = await publicFactory.openResidentPartition({
  model: { ...allocation.model, capsule: qualified.capsule, generation: allocation.generation },
  plan: allocation.plan, planId: allocation.planId, index: 0, participantId: 'fixture-contributor',
  limits: { maxGpuBufferBytes: 1024 }, signal: new AbortController().signal,
});
assert.equal(publicResident.getDescriptor().ready, true);
assert.deepEqual(publicOpening.ranges, [['weights', 0, 2]]);
await publicResident.close();
console.log('public resident factory preserves signed selective acquisition');
