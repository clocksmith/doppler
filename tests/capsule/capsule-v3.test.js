import assert from 'node:assert/strict';
import { generateKeyPairSync } from 'node:crypto';
import { buildCapsuleV3, signCapsuleV3, migrateCapsuleV2, getCapsuleIdentity, validateCapsule, verifyCapsule, signCapsuleReleaseEvent, verifyCapsuleReleaseEvents } from '../../src/capsule.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';
import { createDopplerRuntime } from '../../src/capsule-runtime.js';

const keys = generateKeyPairSync('ed25519');
const signer = { authority: 'release-test', privateKeyJwk: keys.privateKey.export({ format: 'jwk' }), publicKeyJwk: keys.publicKey.export({ format: 'jwk' }) };
const trustedSigners = { [signer.authority]: signer.publicKeyJwk };
const fixture = await createSignedCapsuleFixture({ operation: 'encodeSequence' });
const original = JSON.stringify(fixture.capsule);
const migrated = await migrateCapsuleV2(fixture.capsule, { trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY }, signer });
const capsule = migrated.capsule;
assert.equal(JSON.stringify(fixture.capsule), original);
assert.equal(validateCapsule(fixture.capsule).ok, true);
assert.equal(validateCapsule(capsule).ok, true);
assert.equal(validateCapsule({ ...capsule, release: fixture.capsule.release }).ok, false);
assert.equal(validateCapsule({ ...capsule, createdAtUtc: '2026-01-01T00:00:00.000Z' }).ok, false);
assert.equal(buildCapsuleV3({ ...fixture.capsule, createdAtUtc: '2099-01-01T00:00:00.000Z', release: {} }).semanticRoot, capsule.semanticRoot);
assert.equal(validateCapsule({ ...capsule, modelId: 'changed' }).ok, false);
const { schema, semanticRoot, envelopeDigest } = getCapsuleIdentity(capsule);
const params = {
  capsule: { schema, semanticRoot, envelopeDigest }, sequence: 1, previousEventDigest: null,
  issuedAtUtc: '2026-09-01T00:00:00.000Z', expiresAtUtc: '2026-10-01T00:00:00.000Z',
  action: 'eligible', release: migrated.release, migratedFrom: migrated.migratedFrom, nextSigner: null,
};
const eligible = await signCapsuleReleaseEvent(params, signer);
const policy = { now: '2026-09-04T00:00:00.000Z', minimumSequence: 1, checkpoint: { sequence: 0, digest: null } };
const verify = (history, extra = {}) => verifyCapsuleReleaseEvents(history, { capsule, trustedSigners, policy, ...extra });
const qualified = await verify([eligible]);
assert.equal(qualified.checkpoint.digest, eligible.digest);
await assert.rejects(verify([]), /history/);
await assert.rejects(verify([{ ...eligible, action: 'promoted' }]), /digest/);
await assert.rejects(verify([eligible], { policy: { ...policy, now: params.expiresAtUtc } }), /expired/);
await assert.rejects(verify([eligible], { policy: { ...policy, minimumSequence: 2 } }), /rolled back/);
await assert.rejects(verify([eligible], { policy: { ...policy, checkpoint: { sequence: 1, digest: `sha256:${'0'.repeat(64)}` } } }), /checkpoint/);
const secondKeys = generateKeyPairSync('ed25519');
const second = { authority: signer.authority, privateKeyJwk: secondKeys.privateKey.export({ format: 'jwk' }), publicKeyJwk: secondKeys.publicKey.export({ format: 'jwk' }) };
const rotation = await signCapsuleReleaseEvent({ ...params, nextSigner: second.publicKeyJwk }, signer);
const promoted = await signCapsuleReleaseEvent({ ...params, sequence: 2, previousEventDigest: rotation.digest, action: 'promoted' }, second);
await verify([rotation, promoted]);
await assert.rejects(verify([promoted]), /gap/);
const unsignedRotation = await signCapsuleReleaseEvent({ ...params, sequence: 2, previousEventDigest: eligible.digest }, second);
await assert.rejects(verify([eligible, unsignedRotation]), /Untrusted/);
const quarantined = await signCapsuleReleaseEvent({ ...params, sequence: 2, previousEventDigest: eligible.digest, action: 'quarantined' }, signer);
await assert.rejects(verify([eligible, quarantined]), /blocked/);
const rollback = await signCapsuleReleaseEvent({ ...params, sequence: 3, previousEventDigest: quarantined.digest, action: 'rollback-authorized' }, signer);
await verify([eligible, quarantined, rollback]);
const revoked = await signCapsuleReleaseEvent({ ...params, sequence: 2, previousEventDigest: eligible.digest, action: 'revoked' }, signer);
const reactivated = await signCapsuleReleaseEvent({ ...params, sequence: 3, previousEventDigest: revoked.digest, action: 'rollback-authorized' }, signer);
await assert.rejects(verify([eligible, revoked, reactivated]), /blocked/);
const alternate = await signCapsuleV3(buildCapsuleV3(capsule), second);
assert.equal(alternate.semanticRoot, capsule.semanticRoot);
await assert.rejects(verify([eligible], { capsule: alternate }), /envelope/);

let executions = 0;
let closes = 0;
let persisted = false;
const runtime = createDopplerRuntime({
  device: { getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }), getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
  artifactStore: fixture.artifactStore, trustedSigners,
  programFactory: async ({ artifactStore }) => {
    assert.equal(persisted, true);
    const artifact = capsule.artifacts[0];
    const first = await artifactStore.readArtifact(artifact);
    first.fill(0);
    assert.notDeepEqual(await artifactStore.readArtifact(artifact), first);
    return { encodeSequence: async () => { executions += 1; return { pooledEmbedding: new Float32Array([1, 2]), phase: { elapsed: executions } }; }, close: async () => { closes += 1; } };
  },
});
const options = { releaseEvents: [eligible], releaseTrustedSigners: trustedSigners, releasePolicy: policy, persistReleaseCheckpoint: async (checkpoint) => { assert.equal(checkpoint.digest, eligible.digest); persisted = true; } };
await assert.rejects(runtime.openCapsule(capsule, { ...options, persistReleaseCheckpoint: undefined }), /persistReleaseCheckpoint/);
await assert.rejects(runtime.openCapsule(capsule, { ...options, acceptedTargetPlanDigests: [] }), /not accepted/);
const session = await runtime.openCapsule(capsule, options);
const before = JSON.stringify(session.verification.capsule);
const first = await session.encodeSequence('MKT', { includeTokenEmbeddings: false });
const repeat = await session.encodeSequence('MKT', { includeTokenEmbeddings: false });
assert.equal(first.receipt.outputHash, repeat.receipt.outputHash, 'timings are not semantic outputs');
assert.equal(first.receipt.capsule.envelopeDigest, envelopeDigest);
assert.equal(first.receipt.artifactReceipts.length, capsule.artifacts.length);
const abort = AbortSignal.abort(new Error('cancelled'));
await assert.rejects(session.encodeSequence('MKT', { signal: abort }), /cancelled/);
assert.equal(executions, 2);
assert.equal(JSON.stringify(session.verification.capsule), before);
await session.close();
assert.equal(closes, 1);
await assert.rejects(session.encodeSequence('MKT'), /closed/);

const corruptRuntime = createDopplerRuntime({
  device: { getProfile: () => ({ surface: 'test-webgpu', maxBufferSize: 1024 }) }, trustedSigners,
  artifactStore: { hashArtifact: fixture.artifactStore.hashArtifact, readArtifact: async (artifact) => new Uint8Array(artifact.sizeBytes) },
  programFactory: async () => { throw new Error('must not execute corrupt bytes'); },
});
await assert.rejects(corruptRuntime.openCapsule(capsule, options), /hash or size mismatch/);
let deniedReads = 0;
let denialCheckpoint = null;
const deniedRuntime = createDopplerRuntime({
  device: { getProfile() { throw new Error('Denied metadata must precede device inspection.'); } }, trustedSigners,
  artifactStore: { async readArtifact() { deniedReads += 1; throw new Error('Denied metadata must not fetch artifacts.'); } },
  async programFactory() { throw new Error('Denied metadata must not create a program.'); },
});
await assert.rejects(deniedRuntime.openCapsule(capsule, { ...options, releaseEvents: [eligible, revoked],
  persistReleaseCheckpoint(checkpoint) { denialCheckpoint = checkpoint; } }), /blocked/);
assert.equal(deniedReads, 0);
assert.equal(denialCheckpoint.digest, revoked.digest);
await assert.rejects(verifyCapsule(capsule, {
  ...options, trustedSigners,
  artifactStore: { hashArtifact: fixture.artifactStore.hashArtifact, readArtifact: async (artifact) => new Uint8Array(artifact.sizeBytes) },
}), /hash mismatch/);
console.log('✔ capsule-v3.test.js passed');
