import assert from 'node:assert/strict';
import { generateKeyPairSync } from 'node:crypto';
import { createDopplerRuntime } from '../../src/pack-runtime.js';
import { createPackProgramAdapter } from '../../src/client/runtime/pack-program-adapter.js';
import { createModelHandle } from '../../src/client/runtime/model-session.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { createSignedPackFixture, TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY } from '../helpers/pack-v2-fixture.js';
import { migratePackV2, getPackIdentity, signPackReleaseEvent } from '../../src/pack.js';
import { PACK_OPERATION_REQUEST_SCHEMA, hashPackObservation } from '../../src/config/pack-operation.js';

// Connected public Pack session -> actual program adapter -> actual evidence handle.
// The pipeline and device are synthetic. This is not physical embedding qualification.
const manifest = {
  modelId: 'pack-test-model', modelType: 'embedding', architecture: { hiddenSize: 4 },
  inference: { output: { embeddingPostprocessor: {
    poolingMode: 'last', includePrompt: true, projections: [], normalize: 'l2',
  } } },
};

async function openFixture(options = {}) {
  const fixture = await createSignedPackFixture({ operation: 'embed', manifest, ...options });
  let pack = fixture.pack;
  let trustedSigners = { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY };
  let sessionOptions = {};
  let application = fixture.pack.release.application;
  let releaseEvent = null;
  let persisted = false;
  if (options.packV3) {
    const keys = generateKeyPairSync('ed25519');
    const signer = { authority: 'embedding-release-test',
      privateKeyJwk: keys.privateKey.export({ format: 'jwk' }), publicKeyJwk: keys.publicKey.export({ format: 'jwk' }) };
    const migrated = await migratePackV2(pack, { trustedSigners, signer });
    pack = migrated.pack;
    trustedSigners = { [signer.authority]: signer.publicKeyJwk };
    const { schema, semanticRoot, envelopeDigest } = getPackIdentity(pack);
    const release = structuredClone(migrated.release);
    release.application.applicationId = 'different-application-selected-by-release-event';
    application = release.application;
    releaseEvent = await signPackReleaseEvent({ pack: { schema, semanticRoot, envelopeDigest },
      sequence: 1, previousEventDigest: null, issuedAtUtc: '2026-09-01T00:00:00.000Z',
      expiresAtUtc: '2026-10-01T00:00:00.000Z', action: 'eligible', release,
      migratedFrom: migrated.migratedFrom, nextSigner: null }, signer);
    sessionOptions = { releaseEvents: [releaseEvent], releaseTrustedSigners: trustedSigners,
      releasePolicy: { now: '2026-09-05T00:00:00.000Z', minimumSequence: 1, checkpoint: { sequence: 0, digest: null } },
      persistReleaseCheckpoint: checkpoint => { assert.equal(checkpoint.digest, releaseEvent.digest); persisted = true; } };
  }
  let calls = 0;
  let closes = 0;
  let releaseDevice;
  let duringEmbed = null;
  let afterEvidence = null;
  let transitions = [];
  const gpu = { createBuffer() {}, createCommandEncoder() {},
    lost: new Promise(resolve => { releaseDevice = resolve; }) };
  const pipeline = {
    manifest: structuredClone(options.manifest ?? manifest), isLoaded: true,
    resolvedRuntimeSession: { id: `sha256:${'1'.repeat(64)}` },
    getStats() { return { executionPlan: { transitions } }; },
    getKernelCapabilities() {
      return { adapterInfo: { vendor: 'synthetic' }, hasF16: false,
        hasSubgroups: false, maxBufferSize: 1024, deviceEpoch: 0 };
    },
    async embed(text, executionOptions) {
      calls += 1;
      await duringEmbed?.(text, executionOptions);
      return { embedding: new Float32Array([0, 0, 0, 1]), tokens: [1, 2], seqLen: 2, embeddingMode: 'last' };
    },
    async unload() { closes += 1; },
  };
  const handle = createModelHandle(pipeline, { modelId: fixture.pack.modelId,
    manifestHash: fixture.pack.artifacts.find(artifact => artifact.artifactId === 'manifest').hash });
  const runtime = createDopplerRuntime({
    device: { getDevice: () => gpu, getProfile: () => ({ surface: 'test-webgpu',
      hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
    artifactStore: fixture.artifactStore,
    trustedSigners,
    async programFactory({ pack, targetPlan }) {
      if (options.packV3) assert.equal(persisted, true, 'checkpoint persistence precedes program loading');
      const adapter = createPackProgramAdapter(handle, pack, targetPlan);
      return { ...adapter, async embed(text, executionOptions) {
        const evidence = await adapter.embed(text, executionOptions);
        afterEvidence?.(evidence);
        return evidence;
      } };
    },
  });
  return { session: await runtime.openPack(pack, sessionOptions), fixture, pipeline, releaseEvent,
    request: { application: structuredClone(application), text: 'Local document search.' },
    get calls() { return calls; }, get closes() { return closes; },
    set duringEmbed(callback) { duringEmbed = callback; },
    set afterEvidence(callback) { afterEvidence = callback; },
    set transitions(value) { transitions = value; },
    loseDevice() { releaseDevice({ reason: 'destroyed', message: 'synthetic loss' }); },
  };
}

const test = await openFixture();
const controller = new AbortController();
test.duringEmbed = (text, options) => {
  assert.equal(text, test.request.text);
  assert.equal(options.signal, controller.signal);
};
const result = await test.session.embed({ ...test.request, options: { signal: controller.signal } });
assert.deepEqual(result.embedding, [0, 0, 0, 1]);
assert.equal(result.receipt.operation, 'embed');
assert.deepEqual(result.receipt.pack, test.session.packIdentity);
assert.equal(result.receipt.targetPlanDigest, test.session.selectedTargetPlanDigest);
assert.equal(result.receipt.inputHash, computeCanonicalSha256(test.request));
assert.equal(result.receipt.outputHash, computeCanonicalSha256({ embedding: result.embedding,
  tokens: result.tokens, seqLen: result.seqLen, embeddingMode: result.embeddingMode }));
const { receiptDigest, ...payload } = result.receipt;
assert.equal(receiptDigest, computeCanonicalSha256(payload));
assert.equal(result.receipt.artifactReceipts.length, test.fixture.pack.artifacts.length);
assert.equal(Object.isFrozen(result.embedding), true);
assert.equal(Object.isFrozen(result.receipt.application), true);
test.duringEmbed = null;

const collect = async iterator => { const events = []; for await (const event of iterator) events.push(event); return events; };
const operationRequest = (application) => ({ schema: PACK_OPERATION_REQUEST_SCHEMA,
  operation: { name: 'embed', version: 1 }, input: { texts: ['first', 'second'], application },
  options: {}, assignment: { jobId: 'embedding-batch', attempt: 1 },
  limits: { maxInputBytes: 10000, maxOutputBytes: 100000, deadlineAt: Date.now() + 60000 } });
const batch = operationRequest(test.request.application);
const batchEvents = await collect(test.session.executeOperation(batch));
assert.deepEqual(batchEvents.map(event => event.status), ['partial', 'partial', 'completed']);
const batchResult = batchEvents.at(-1);
assert.equal(batchResult.receipt.assignmentHash, hashPackObservation(batch.assignment));
assert.equal(batchResult.receipt.outputHash, hashPackObservation(batchResult.output));
for (const [index, item] of batchResult.output.embeddings.entries()) {
  assert.equal(item.receipt.operation, 'embed');
  assert.equal(item.receipt.inputHash, computeCanonicalSha256({ application: test.request.application, text: batch.input.texts[index] }));
  assert.deepEqual(item.embedding, result.embedding);
}
const callsBeforeInvalid = test.calls;
await assert.rejects(collect(test.session.executeOperation(operationRequest({ ...test.request.application,
  applicationId: 'unapproved' }))), /application identity/);
const missingApplication = operationRequest(test.request.application);
delete missingApplication.input.application;
assert.throws(() => test.session.executeOperation(missingApplication), /application binding/);
assert.equal(test.calls, callsBeforeInvalid);

for (const [request, message] of [
  [null, /request as an object/],
  [{ ...test.request, text: '' }, /non-empty string/],
  [{ ...test.request, application: { ...test.request.application, applicationId: 'other' } }, /application identity/],
  [{ ...test.request, application: { ...test.request.application, extraAuthority: true } }, /application identity/],
  [{ ...test.request, sequence: 'MKT' }, /undeclared fields/],
  [{ ...test.request, options: { embeddingMode: 'mean' } }, /only signal/],
  [{ ...test.request, options: { signal: {} } }, /AbortSignal/],
]) await assert.rejects(test.session.embed(request), message);
controller.abort(new Error('cancel before execution'));
await assert.rejects(test.session.embed({ ...test.request, options: { signal: controller.signal } }), /cancel before execution/);
assert.equal(test.calls, callsBeforeInvalid, 'invalid input or cancellation must fail before the pipeline');

const during = new AbortController();
test.duringEmbed = () => during.abort(new Error('cancel during execution'));
await assert.rejects(test.session.embed({ ...test.request, options: { signal: during.signal } }), /cancel during execution/);
test.duringEmbed = null;

for (const mutate of [
  evidence => { evidence.schema = 'doppler_sequence_evidence/v1'; },
  evidence => { evidence.embedding = [0, 1]; },
  evidence => { evidence.embedding[0] = NaN; },
  evidence => { evidence.embedding = new Array(4); },
  evidence => { evidence.tokens = new Array(2); },
  evidence => { evidence.tokens = [-1]; },
  evidence => { evidence.seqLen = 1; },
  evidence => { evidence.embeddingMode = 'mean'; },
  evidence => { evidence.inputHash = `sha256:${'2'.repeat(64)}`; },
  evidence => { evidence.outputHash = `sha256:${'2'.repeat(64)}`; },
  evidence => { evidence.resolution.resolvedArtifactVariantId = `sha256:${'2'.repeat(64)}`; },
  evidence => { evidence.backendIdentityHash = `sha256:${'2'.repeat(64)}`; },
  evidence => { delete evidence.resolution.schema; },
  evidence => { delete evidence.executionIdentity.schema; },
  evidence => { evidence.executionIdentity.activeAdapter = 'changed'; },
]) {
  test.afterEvidence = mutate;
  await assert.rejects(test.session.embed(test.request), /Malformed Pack|evidence does not match/);
}
test.afterEvidence = null;
test.afterEvidence = evidence => { evidence.outputHash = `sha256:${'2'.repeat(64)}`; };
await assert.rejects(collect(test.session.executeOperation(batch)), /evidence does not match/);
test.afterEvidence = null;
const batchCancellation = new AbortController();
test.duringEmbed = () => batchCancellation.abort(new Error('cancel batch during first item'));
const callsBeforeBatchCancel = test.calls;
await assert.rejects(collect(test.session.executeOperation(batch, { signal: batchCancellation.signal })), /cancel batch during first item/);
assert.equal(test.calls, callsBeforeBatchCancel + 1, 'cancellation must not execute another item');
test.duringEmbed = null;
test.transitions = [{ reason: 'undeclared' }];
await assert.rejects(test.session.embed(test.request), /undeclared execution-plan transition/);
test.transitions = [];

let heldEvidence;
const mutableRequest = structuredClone(test.request);
test.duringEmbed = () => { mutableRequest.application.applicationId = 'mutated'; mutableRequest.text = 'changed'; };
test.afterEvidence = evidence => { heldEvidence = evidence; };
const detached = await test.session.embed(mutableRequest);
heldEvidence.embedding[0] = 99;
heldEvidence.executionIdentity.activeAdapter = 'changed after return';
assert.equal(detached.receipt.inputHash, computeCanonicalSha256(test.request));
assert.equal(detached.embedding[0], 0);
assert.equal(detached.receipt.executionIdentity.activeAdapter, null);

test.loseDevice();
await Promise.resolve();
const beforeLoss = test.calls;
await assert.rejects(test.session.embed(test.request), error => error.code === 'DOPPLER_GPU_DEVICE_LOST');
assert.equal(test.calls, beforeLoss);
await test.session.close();
await test.session.close();
assert.equal(test.closes, 1);
await assert.rejects(test.session.embed(test.request), /session is closed/);

for (const operation of ['generate', 'rerank', 'encodeSequence']) {
  const other = await openFixture({ operation });
  await assert.rejects(other.session.embed(other.request), /not qualified.*embed/);
  await assert.rejects(collect(other.session.executeOperation(operationRequest(other.request.application))), /not qualified.*embed/);
  assert.equal(other.calls, 0);
  await other.session.close();
}

for (const mutate of [
  value => { value.modelType = 'text'; },
  value => { value.inference.output.embeddingPostprocessor = null; },
  value => { value.inference.output.embeddingPostprocessor.includePrompt = false; },
  value => { value.architecture.hiddenSize = null; },
  value => { value.inference.output.embeddingPostprocessor.projections = [{ inputSize: 3, outputSize: 2 }]; },
]) {
  const invalidManifest = structuredClone(manifest);
  mutate(invalidManifest);
  const invalid = await openFixture({ manifest: invalidManifest });
  await assert.rejects(invalid.session.embed(invalid.request), /manifest|embeddingPostprocessor|projection/);
  assert.equal(invalid.calls, 0);
  await invalid.session.close();
}

const projectedManifest = structuredClone(manifest);
projectedManifest.inference.output.embeddingPostprocessor.projections = [{
  inputSize: 4, outputSize: 2, weightTensor: 'projection.weight', biasTensor: null, activation: 'identity',
}];
const projected = await openFixture({ manifest: projectedManifest });
projected.pipeline.embed = async () => ({ embedding: new Float32Array([0, 1]), tokens: [1, 2], seqLen: 2, embeddingMode: 'last' });
assert.deepEqual((await projected.session.embed(projected.request)).embedding, [0, 1]);
await projected.session.close();

const v3 = await openFixture({ packV3: true });
await assert.rejects(v3.session.embed({ ...v3.request, application: v3.fixture.pack.release.application }), /application identity/);
assert.equal(v3.calls, 0, 'old v2 application authority must not replace the selected v3 release event');
const released = await v3.session.embed(v3.request);
assert.equal(released.receipt.pack.schema, 'doppler.pack/v3');
assert.equal(released.receipt.releaseEventDigest, v3.releaseEvent.digest);
assert.deepEqual(released.receipt.application, v3.request.application);
await assert.rejects(collect(v3.session.executeOperation(operationRequest(v3.fixture.pack.release.application))), /application identity/);
const v3Batch = (await collect(v3.session.executeOperation(operationRequest(v3.request.application)))).at(-1);
assert.equal(v3Batch.receipt.releaseEventDigest, v3.releaseEvent.digest);
assert.equal(v3Batch.output.embeddings[0].receipt.releaseEventDigest, v3.releaseEvent.digest);
await v3.session.close();
console.log('pack-embedding.test: passed (synthetic pipeline/device; actual session, adapter and evidence handle)');
