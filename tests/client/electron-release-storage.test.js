import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { generateKeyPairSync } from 'node:crypto';
import {
  createDocumentSearchCheckpointStore, createDocumentSearchReleaseStore,
  prepareDocumentSearchReleaseOptions,
} from '../../examples/electron-document-search/release-storage.js';
import { createSignedCapsuleFixture } from '../helpers/capsule-v2-fixture.js';
import { buildCapsuleV3, signCapsuleV3, getCapsuleIdentity, signCapsuleReleaseEvent } from 'doppler-gpu/capsule';
import { createElectronReleaseStateCoordinator } from 'doppler-gpu/electron';
import { createDopplerRuntime } from 'doppler-gpu';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-release-storage-'));
try {
  const filename = path.join(directory, 'checkpoint.json');
  const store = createDocumentSearchCheckpointStore(filename);
  const fixture = await createSignedCapsuleFixture({ operation: 'rerank' });
  const keys = generateKeyPairSync('ed25519');
  const signer = { authority: 'release-storage-fixture',
    publicKeyJwk: keys.publicKey.export({ format: 'jwk' }), privateKeyJwk: keys.privateKey.export({ format: 'jwk' }) };
  const trustedSigners = { [signer.authority]: signer.publicKeyJwk };
  const capsule = await signCapsuleV3(buildCapsuleV3(fixture.capsule), signer);
  const { schema, semanticRoot, envelopeDigest } = getCapsuleIdentity(capsule);
  const params = { capsule: { schema, semanticRoot, envelopeDigest }, sequence: 1, previousEventDigest: null,
    issuedAtUtc: '2026-09-01T00:00:00.000Z', expiresAtUtc: '2026-10-01T00:00:00.000Z',
    action: 'eligible', release: fixture.capsule.release, migratedFrom: null, nextSigner: null };
  const first = await signCapsuleReleaseEvent(params, signer);
  const second = await signCapsuleReleaseEvent({ ...params, sequence: 2, previousEventDigest: first.digest, action: 'promoted' }, signer);
  const base = { capsule, releaseEvents: [first], releaseTrustedSigners: trustedSigners, checkpointStore: store,
    minimumSequence: 1, now: '2026-09-06T00:00:00.000Z' };
  assert.equal(await store.load(), null);
  const context = await prepareDocumentSearchReleaseOptions(base);
  assert.equal(await store.load(), null, 'reviewing history is not persistence or activation');
  await assert.rejects(context.persistReleaseCheckpoint({ sequence: 2, digest: first.digest }), /differs/);
  let created = 0;
  const runtime = createDopplerRuntime({ trustedSigners, artifactStore: fixture.artifactStore,
    device: { getDevice: () => ({ createBuffer() {}, createCommandEncoder() {} }),
      getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
    programFactory: async () => {
      assert.deepEqual(await store.load(), { sequence: 1, digest: first.digest });
      created += 1;
      return { close: async () => {} };
    } });
  const session = await runtime.openCapsule(capsule, context);
  await session.close();
  assert.equal(created, 1, 'checkpoint persistence precedes model creation');
  const moduleUrl = pathToFileURL(path.resolve('examples/electron-document-search/release-storage.js')).href;
  const reopened = JSON.parse(execFileSync(process.execPath, ['--input-type=module', '-e',
    `import {createDocumentSearchCheckpointStore} from ${JSON.stringify(moduleUrl)}; console.log(JSON.stringify(await createDocumentSearchCheckpointStore(${JSON.stringify(filename)}).load()));`], { encoding: 'utf8' }));
  assert.deepEqual(reopened, { sequence: 1, digest: first.digest }, 'separate process retains the checkpoint');
  await context.persistReleaseCheckpoint(reopened);

  const advance = await prepareDocumentSearchReleaseOptions({ ...base, releaseEvents: [first, second], minimumSequence: 2 });
  await advance.persistReleaseCheckpoint({ sequence: 2, digest: second.digest });
  await assert.rejects(context.persistReleaseCheckpoint(reopened), /changed concurrently/);
  await assert.rejects(prepareDocumentSearchReleaseOptions({ ...base, minimumSequence: 2 }), /rolled back/);
  await assert.rejects(prepareDocumentSearchReleaseOptions({ ...base, releaseEvents: [first, second], minimumSequence: 2,
    now: params.expiresAtUtc }), /expired/);
  assert.equal((await store.load()).sequence, 2);
  const retainedLocalUse = { schema: 'doppler.capsule-retained-local-use/v1', capsule: params.capsule,
    releaseEventDigest: second.digest, applicationDigest: computeCanonicalSha256(params.release.application),
    acceptedAtUtc: base.now, acknowledgeUnseenRevocations: true };
  const retained = await prepareDocumentSearchReleaseOptions({ ...base, releaseEvents: [first, second],
    minimumSequence: 2, now: params.expiresAtUtc, retainedLocalUse });
  assert.deepEqual(retained.releasePolicy.retainedLocalUse, retainedLocalUse);
  retainedLocalUse.acknowledgeUnseenRevocations = false;
  assert.equal(retained.releasePolicy.retainedLocalUse.acknowledgeUnseenRevocations, true, 'decision is snapshotted');

  const rejectedStore = { async load() { return null; }, async compareAndSwap() { return false; } };
  const rejected = await prepareDocumentSearchReleaseOptions({ ...base, checkpointStore: rejectedStore });
  await assert.rejects(runtime.openCapsule(capsule, rejected), /changed concurrently/);
  assert.equal(created, 1, 'a failed checkpoint write cannot start another model');

  const uncertainStore = createDocumentSearchCheckpointStore(path.join(directory, 'uncertain.json'));
  const uncertain = await prepareDocumentSearchReleaseOptions({ ...base, checkpointStore: uncertainStore });
  const originalOpen = fs.open;
  const syncFailure = new Error('injected directory synchronization failure');
  let closeAlsoFails = false;
  fs.open = async (...args) => {
    const handle = await originalOpen(...args);
    if (args[0] !== directory) return handle;
    return { async sync() { throw syncFailure; }, async close() {
      await handle.close();
      if (closeAlsoFails) throw new Error('secondary directory close failure');
    } };
  };
  try {
    await assert.rejects(uncertain.persistReleaseCheckpoint(reopened), error => error === syncFailure);
    await assert.rejects(uncertain.persistReleaseCheckpoint(reopened), error => error === syncFailure,
      'an identical visible record cannot bypass a failed durability barrier');
    closeAlsoFails = true;
    await assert.rejects(uncertain.persistReleaseCheckpoint(reopened), error => error === syncFailure,
      'cleanup must preserve the original synchronization failure');
  } finally { fs.open = originalOpen; }
  await uncertain.persistReleaseCheckpoint(reopened);
  assert.deepEqual(await uncertainStore.load(), reopened);

  const raceFile = path.join(directory, 'race.json');
  const attempts = await Promise.all([0, 1].map(() => createDocumentSearchCheckpointStore(raceFile)
    .compareAndSwap(0, { sequence: 1, digest: first.digest })));
  assert.deepEqual(attempts.sort(), [false, true]);
  await assert.rejects(store.compareAndSwap(2, { sequence: 1, digest: first.digest }), /advance/);
  await assert.rejects(store.compareAndSwap(2, { sequence: 3, digest: 'invalid' }), /checkpoint/);
  await fs.writeFile(`${filename}.lock`, 'retained crash lock');
  assert.equal(await store.compareAndSwap(2, { sequence: 3, digest: second.digest }), false);
  assert.equal(await fs.readFile(`${filename}.lock`, 'utf8'), 'retained crash lock');
  await fs.unlink(`${filename}.lock`);
  const corrupt = path.join(directory, 'corrupt.json');
  await fs.writeFile(corrupt, '{invalid');
  const corruptStore = createDocumentSearchCheckpointStore(corrupt);
  await assert.rejects(corruptStore.load());
  await assert.rejects(corruptStore.compareAndSwap(0, { sequence: 1, digest: first.digest }));
  assert.equal(await fs.readFile(corrupt, 'utf8'), '{invalid');
  await assert.rejects(fs.stat(`${corrupt}.lock`), { code: 'ENOENT' });
  const linked = path.join(directory, 'linked.json');
  await fs.symlink(filename, linked);
  await assert.rejects(createDocumentSearchCheckpointStore(linked).load());

  const releaseStore = createDocumentSearchReleaseStore(path.join(directory, 'release.json'));
  const options = { stateStore: releaseStore, verifyReleaseDecision: () => false, verifyRevocationSnapshot: () => false };
  const main = createElectronReleaseStateCoordinator(options);
  await main.installCandidate({ capsuleId: capsule.capsuleId, semanticRoot: capsule.semanticRoot, path: 'capsule.json' }, first.digest);
  const restarted = createElectronReleaseStateCoordinator({ ...options,
    stateStore: createDocumentSearchReleaseStore(path.join(directory, 'release.json')) });
  assert.equal((await restarted.load()).candidate.capsule.semanticRoot, capsule.semanticRoot);
  await assert.rejects(restarted.resolveCurrent(), /no active Capsule/);
  assert.deepEqual((await fs.readdir(directory)).filter(name => name.endsWith('.tmp') || name.endsWith('.lock')), []);
  const revoked = await signCapsuleReleaseEvent({ ...params, sequence: 3, previousEventDigest: second.digest, action: 'revoked' }, signer);
  await assert.rejects(prepareDocumentSearchReleaseOptions({ ...base, minimumSequence: 2,
    releaseEvents: [first, second, { ...revoked, signature: first.signature }] }), /signature|digest/);
  assert.equal((await store.load()).sequence, 2, 'forged denial cannot poison the store');
  await assert.rejects(prepareDocumentSearchReleaseOptions({ ...base, minimumSequence: 2,
    releaseEvents: [first, second, revoked] }), /blocked/);
  assert.deepEqual(await store.load(), { sequence: 3, digest: revoked.digest }, 'reviewed denial survives process restart');
  await assert.rejects(prepareDocumentSearchReleaseOptions({ ...base, minimumSequence: 3,
    releaseEvents: [first, second], retainedLocalUse: retained.releasePolicy.retainedLocalUse }), /rolled back/);
  await assert.rejects(prepareDocumentSearchReleaseOptions({ ...base, checkpointStore: rejectedStore,
    releaseEvents: [first, second, revoked] }), error => error instanceof AggregateError
      && /blocked/.test(error.errors[0].message) && /concurrently/.test(error.errors[1].message));
} finally { await fs.rm(directory, { recursive: true, force: true }); }
console.log('electron-release-storage.test: ok (real filesystem; synthetic Capsule execution, no adoption claim)');
