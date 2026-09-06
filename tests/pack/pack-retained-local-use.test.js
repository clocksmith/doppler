import assert from 'node:assert/strict';
import { generateKeyPairSync } from 'node:crypto';
import { migratePackV2, getPackIdentity, signPackReleaseEvent, verifyPack, verifyPackReleaseEvents } from '../../src/pack.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { createDopplerRuntime } from '../../src/pack-runtime.js';
import { createSignedPackFixture, TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY } from '../helpers/pack-v2-fixture.js';

// Real signatures, verifier, persistence and runtime; synthetic GPU/program, not physical inference.
const keys = generateKeyPairSync('ed25519');
const signer = { authority: 'recipient-test', privateKeyJwk: keys.privateKey.export({ format: 'jwk' }), publicKeyJwk: keys.publicKey.export({ format: 'jwk' }) };
const trustedSigners = { [signer.authority]: signer.publicKeyJwk };
const fixture = await createSignedPackFixture({ operation: 'encodeSequence' });
const migrated = await migratePackV2(fixture.pack, { trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY }, signer });
const pack = migrated.pack;
const { schema, semanticRoot, envelopeDigest } = getPackIdentity(pack);
const params = { pack: { schema, semanticRoot, envelopeDigest }, sequence: 1, previousEventDigest: null,
  issuedAtUtc: '2026-09-01T00:00:00.000Z', expiresAtUtc: '2026-09-02T00:00:00.000Z',
  action: 'eligible', release: migrated.release, migratedFrom: migrated.migratedFrom, nextSigner: null };
const eligible = await signPackReleaseEvent(params, signer);
const checkpoint = { sequence: 1, digest: eligible.digest };
const policy = { now: '2026-09-03T00:00:00.000Z', minimumSequence: 1, checkpoint,
  retainedLocalUse: { schema: 'doppler.pack-retained-local-use/v1', pack: params.pack,
    releaseEventDigest: eligible.digest, applicationDigest: computeCanonicalSha256(params.release.application),
    acceptedAtUtc: '2026-09-01T12:00:00.000Z', acknowledgeUnseenRevocations: true } };
const verify = (history = [eligible], override = policy) => verifyPackReleaseEvents(history, { pack, trustedSigners, policy: override });
const authorized = await verify();
assert.equal(authorized.authorization.mode, 'retained-local');
assert.equal(authorized.authorization.eventExpired, true);
assert.equal(authorized.authorization.unseenRevocations, 'unknown');
assert.deepEqual(authorized.authorization.retainedLocalUse, policy.retainedLocalUse);
assert.ok(Object.isFrozen(authorized.authorization.retainedLocalUse.pack));
assert.equal(authorized.release.revocation.failClosedAfterExpiry, true, 'signed managed policy is not rewritten');
const { retainedLocalUse, ...managed } = policy;
await assert.rejects(verify([eligible], managed), /expired/);
await assert.rejects(verify([eligible], { ...policy, now: params.expiresAtUtc, retainedLocalUse: undefined }), /expired/);
assert.equal((await verify([eligible], { ...managed, now: retainedLocalUse.acceptedAtUtc })).authorization.mode, 'managed');
assert.equal((await verify([eligible], { ...policy, now: retainedLocalUse.acceptedAtUtc })).authorization.mode, 'retained-local', 'local scope applies even before expiry');

const badDigest = `sha256:${'0'.repeat(64)}`;
for (const change of [
  p => { p.retainedLocalUse = null; },
  p => { p.retainedLocalUse = true; },
  p => { p.retainedLocalUse.schema = 'unknown'; },
  p => { p.retainedLocalUse.allowDelegation = true; },
  p => { delete p.retainedLocalUse.applicationDigest; },
  p => { p.retainedLocalUse.applicationDigest = badDigest; },
  p => { p.retainedLocalUse.pack.semanticRoot = badDigest; },
  p => { p.retainedLocalUse.pack.envelopeDigest = badDigest; },
  p => { p.retainedLocalUse.pack.schema = 'doppler.pack/v2'; },
  p => { p.retainedLocalUse.pack.extra = true; },
  p => { p.retainedLocalUse.releaseEventDigest = badDigest; },
  p => { p.checkpoint = { sequence: 0, digest: null }; },
  p => { p.checkpoint.digest = badDigest; },
  p => { p.minimumSequence = 2; },
  p => { p.retainedLocalUse.acceptedAtUtc = 'invalid'; },
  p => { p.retainedLocalUse.acceptedAtUtc = '2026-08-31T00:00:00.000Z'; },
  p => { p.retainedLocalUse.acceptedAtUtc = params.expiresAtUtc; },
  p => { p.now = params.issuedAtUtc; },
  p => { p.retainedLocalUse.acknowledgeUnseenRevocations = false; },
  p => { p.allowExpired = true; },
]) {
  const invalid = structuredClone(policy); change(invalid);
  await assert.rejects(verify([eligible], invalid));
}
await assert.rejects(verify([]), /history/);
await assert.rejects(verify([{ ...eligible, action: 'promoted' }]), /digest/);
await assert.rejects(verifyPack(fixture.pack, { trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
  artifactStore: fixture.artifactStore, releasePolicy: policy }), /requires Pack v3/);

let writes = [];
let opens = 0;
let executions = 0;
let closes = 0;
const createRuntime = (artifactStore = fixture.artifactStore) => createDopplerRuntime({
  device: { getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }),
    getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
  artifactStore, trustedSigners,
  programFactory: async () => {
    assert.equal(writes.at(-1)?.digest, eligible.digest, 'persist before program creation');
    opens++;
    return { encodeSequence: async () => { executions++; return { pooledEmbedding: new Float32Array([1, 2]) }; }, close: async () => { closes++; } };
  },
});
const runtime = createRuntime();
const options = { releaseEvents: [eligible], releaseTrustedSigners: trustedSigners, releasePolicy: policy,
  persistReleaseCheckpoint: async value => { writes.push(value); } };
await assert.rejects(runtime.openPack(pack, { ...options, persistReleaseCheckpoint: undefined }), /persistReleaseCheckpoint/);
await assert.rejects(runtime.openPack(pack, { ...options, persistReleaseCheckpoint: async () => { throw new Error('store unavailable'); } }), /store unavailable/);
assert.equal(opens, 0);
const session = await runtime.openPack(pack, options);
const result = await session.encodeSequence('MKT');
const { receiptDigest, ...payload } = result.receipt;
assert.equal(receiptDigest, computeCanonicalSha256(payload));
assert.deepEqual(result.receipt.releaseAuthorization, authorized.authorization);
await assert.rejects(session.encodeSequence('MKT', { assignment: { id: 'remote' } }), /delegated/);
await assert.rejects(session.forecast({ assignmentHash: badDigest }), /delegated/);
const request = { schema: 'doppler.pack-operation-request/v1', operation: { name: 'encodeSequence', version: 1 },
  input: { sequence: 'MKT' }, options: { includeLogits: false, includeTokenEmbeddings: false }, assignment: { id: 'remote' },
  limits: { maxInputBytes: 10000, maxOutputBytes: 10000, deadlineAt: Date.now() + 60000 } };
await assert.rejects(session.executeOperation(request).next(), /delegated/);
assert.equal(executions, 1);
const local = session.executeOperation({ ...request, assignment: null });
const completed = await local.next();
assert.equal(completed.value.receipt.releaseAuthorization.mode, 'retained-local');
await local.return();
await session.close();
assert.equal(closes, 1);
await assert.rejects(session.encodeSequence('MKT'), /closed/);
await assert.rejects(createRuntime({ readArtifact: async artifact => new Uint8Array(artifact.sizeBytes) }).openPack(pack, options), /hash or size mismatch/);
assert.equal(opens, 1);

for (const action of ['blocked', 'quarantined', 'revoked', 'superseded']) {
  const denied = await signPackReleaseEvent({ ...params, sequence: 2, previousEventDigest: eligible.digest, action }, signer);
  writes = [];
  await assert.rejects(runtime.openPack(pack, { ...options, releaseEvents: [eligible, denied] }), /blocked/);
  assert.deepEqual(writes, [{ sequence: 2, digest: denied.digest }], 'authenticated denial advances durable checkpoint');
  await assert.rejects(runtime.openPack(pack, { ...options, releasePolicy: { ...policy, minimumSequence: 2, checkpoint: writes[0] } }), /rolled back/);
  await assert.rejects(runtime.openPack(pack, { ...options, releaseEvents: [eligible, denied],
    persistReleaseCheckpoint: async () => { throw new Error('disk full'); } }), error => {
    assert.ok(error instanceof AggregateError);
    assert.match(error.errors[0].message, /blocked/);
    assert.match(error.errors[1].message, /disk full/);
    return true;
  });
}
const revoked = await signPackReleaseEvent({ ...params, sequence: 2, previousEventDigest: eligible.digest, action: 'revoked' }, signer);
const rollback = await signPackReleaseEvent({ ...params, sequence: 3, previousEventDigest: revoked.digest, action: 'rollback-authorized' }, signer);
await assert.rejects(verify([eligible, revoked, rollback]), /blocked/);
const promoted = await signPackReleaseEvent({ ...params, sequence: 2, previousEventDigest: eligible.digest, action: 'promoted' }, signer);
await assert.rejects(verify([eligible, promoted]), /previously persisted/, 'a retained decision cannot implicitly accept a new release event');
writes = [];
for (const history of [[{ ...eligible, action: 'blocked' }], [promoted], [eligible, { ...revoked, signature: eligible.signature }]]) {
  await assert.rejects(runtime.openPack(pack, { ...options, releaseEvents: history }));
}
assert.equal(writes.length, 0, 'invalid signatures, digests or noncontiguous history cannot poison durable state');
assert.equal(opens, 1);
console.log('✔ pack-retained-local-use.test.js passed');
