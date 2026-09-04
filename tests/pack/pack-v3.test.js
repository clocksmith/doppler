import assert from 'node:assert/strict';
import { generateKeyPairSync } from 'node:crypto';
import { buildPackV3, signPackV3, migratePackV2, getPackIdentity, validatePack, verifyPack, signPackReleaseEvent, verifyPackReleaseEvents } from '../../src/pack.js';
import { createSignedPackFixture, TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY } from '../helpers/pack-v2-fixture.js';
import { createDopplerRuntime } from '../../src/pack-runtime.js';

const keys = generateKeyPairSync('ed25519');
const signer = { authority: 'release-test', privateKeyJwk: keys.privateKey.export({ format: 'jwk' }), publicKeyJwk: keys.publicKey.export({ format: 'jwk' }) };
const trustedSigners = { [signer.authority]: signer.publicKeyJwk };
const fixture = await createSignedPackFixture();
const original = JSON.stringify(fixture.pack);
const migrated = await migratePackV2(fixture.pack, { trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY }, signer });
const pack = migrated.pack;
assert.equal(JSON.stringify(fixture.pack), original);
assert.equal(validatePack(fixture.pack).ok, true);
assert.equal(validatePack(pack).ok, true);
assert.equal(validatePack({ ...pack, release: fixture.pack.release }).ok, false);
assert.equal(validatePack({ ...pack, createdAtUtc: '2026-01-01T00:00:00.000Z' }).ok, false);
assert.equal(buildPackV3({ ...fixture.pack, createdAtUtc: '2099-01-01T00:00:00.000Z', release: {} }).semanticRoot, pack.semanticRoot);
assert.equal(validatePack({ ...pack, modelId: 'changed' }).ok, false);
const { schema, semanticRoot, envelopeDigest } = getPackIdentity(pack);
const params = {
  pack: { schema, semanticRoot, envelopeDigest }, sequence: 1, previousEventDigest: null,
  issuedAtUtc: '2026-09-01T00:00:00.000Z', expiresAtUtc: '2026-10-01T00:00:00.000Z',
  action: 'eligible', release: migrated.release, migratedFrom: migrated.migratedFrom, nextSigner: null,
};
const eligible = await signPackReleaseEvent(params, signer);
const policy = { now: '2026-09-04T00:00:00.000Z', minimumSequence: 1, checkpoint: { sequence: 0, digest: null } };
const verify = (history, extra = {}) => verifyPackReleaseEvents(history, { pack, trustedSigners, policy, ...extra });
const qualified = await verify([eligible]);
assert.equal(qualified.checkpoint.digest, eligible.digest);
await assert.rejects(verify([]), /history/);
await assert.rejects(verify([{ ...eligible, action: 'promoted' }]), /digest/);
await assert.rejects(verify([eligible], { policy: { ...policy, now: params.expiresAtUtc } }), /expired/);
await assert.rejects(verify([eligible], { policy: { ...policy, minimumSequence: 2 } }), /rolled back/);
await assert.rejects(verify([eligible], { policy: { ...policy, checkpoint: { sequence: 1, digest: `sha256:${'0'.repeat(64)}` } } }), /checkpoint/);
const secondKeys = generateKeyPairSync('ed25519');
const second = { authority: signer.authority, privateKeyJwk: secondKeys.privateKey.export({ format: 'jwk' }), publicKeyJwk: secondKeys.publicKey.export({ format: 'jwk' }) };
const rotation = await signPackReleaseEvent({ ...params, nextSigner: second.publicKeyJwk }, signer);
const promoted = await signPackReleaseEvent({ ...params, sequence: 2, previousEventDigest: rotation.digest, action: 'promoted' }, second);
await verify([rotation, promoted]);
await assert.rejects(verify([promoted]), /gap/);
const unsignedRotation = await signPackReleaseEvent({ ...params, sequence: 2, previousEventDigest: eligible.digest }, second);
await assert.rejects(verify([eligible, unsignedRotation]), /Untrusted/);
const quarantined = await signPackReleaseEvent({ ...params, sequence: 2, previousEventDigest: eligible.digest, action: 'quarantined' }, signer);
await assert.rejects(verify([eligible, quarantined]), /blocked/);
const rollback = await signPackReleaseEvent({ ...params, sequence: 3, previousEventDigest: quarantined.digest, action: 'rollback-authorized' }, signer);
await verify([eligible, quarantined, rollback]);
const revoked = await signPackReleaseEvent({ ...params, sequence: 2, previousEventDigest: eligible.digest, action: 'revoked' }, signer);
const reactivated = await signPackReleaseEvent({ ...params, sequence: 3, previousEventDigest: revoked.digest, action: 'rollback-authorized' }, signer);
await assert.rejects(verify([eligible, revoked, reactivated]), /blocked/);
const alternate = await signPackV3(buildPackV3(pack), second);
assert.equal(alternate.semanticRoot, pack.semanticRoot);
await assert.rejects(verify([eligible], { pack: alternate }), /envelope/);

let executions = 0;
let closes = 0;
let persisted = false;
const runtime = createDopplerRuntime({
  device: { getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }), getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
  artifactStore: fixture.artifactStore, trustedSigners,
  programFactory: async ({ artifactStore }) => {
    assert.equal(persisted, true);
    const artifact = pack.artifacts[0];
    const first = await artifactStore.readArtifact(artifact);
    first.fill(0);
    assert.notDeepEqual(await artifactStore.readArtifact(artifact), first);
    return { encodeSequence: async () => { executions += 1; return { pooledEmbedding: new Float32Array([1, 2]), phase: { elapsed: executions } }; }, close: async () => { closes += 1; } };
  },
});
const options = { releaseEvents: [eligible], releaseTrustedSigners: trustedSigners, releasePolicy: policy, persistReleaseCheckpoint: async (checkpoint) => { assert.equal(checkpoint.digest, eligible.digest); persisted = true; } };
await assert.rejects(runtime.openPack(pack, { ...options, persistReleaseCheckpoint: undefined }), /persistReleaseCheckpoint/);
await assert.rejects(runtime.openPack(pack, { ...options, acceptedTargetPlanDigests: [] }), /not accepted/);
const session = await runtime.openPack(pack, options);
const before = JSON.stringify(session.verification.pack);
const first = await session.encodeSequence('MKT', { includeTokenEmbeddings: false });
const repeat = await session.encodeSequence('MKT', { includeTokenEmbeddings: false });
assert.equal(first.receipt.outputHash, repeat.receipt.outputHash, 'timings are not semantic outputs');
assert.equal(first.receipt.pack.envelopeDigest, envelopeDigest);
assert.equal(first.receipt.artifactReceipts.length, pack.artifacts.length);
const abort = AbortSignal.abort(new Error('cancelled'));
await assert.rejects(session.encodeSequence('MKT', { signal: abort }), /cancelled/);
assert.equal(executions, 2);
assert.equal(JSON.stringify(session.verification.pack), before);
await session.close();
assert.equal(closes, 1);
await assert.rejects(session.encodeSequence('MKT'), /closed/);

const corruptRuntime = createDopplerRuntime({
  device: {}, trustedSigners,
  artifactStore: { hashArtifact: fixture.artifactStore.hashArtifact, readArtifact: async (artifact) => new Uint8Array(artifact.sizeBytes) },
  programFactory: async () => { throw new Error('must not execute corrupt bytes'); },
});
await assert.rejects(corruptRuntime.openPack(pack, options), /hash or size mismatch/);
await assert.rejects(verifyPack(pack, {
  ...options, trustedSigners,
  artifactStore: { hashArtifact: fixture.artifactStore.hashArtifact, readArtifact: async (artifact) => new Uint8Array(artifact.sizeBytes) },
}), /hash mismatch/);
console.log('✔ pack-v3.test.js passed');
