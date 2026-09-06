import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createDocumentSearchReleaseStore } from '../../examples/electron-document-search/release-storage.js';
import { registerDocumentSearchReleaseMain } from '../../examples/electron-document-search/main.js';
import { exposeDocumentSearchReleaseBridge } from '../../examples/electron-document-search/preload.js';
import { createDocumentSearchRenderer } from '../../examples/electron-document-search/renderer.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';
import {
  RELEASE_DECISION_SCHEMA, signProductionReleaseEvidence, verifyProductionReleaseEvidenceSignature,
} from '../../src/config/production-release-evidence.js';
import { ELECTRON_REVOCATION_SNAPSHOT_SCHEMA } from 'doppler-gpu/electron';

const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-durable-episode-'));
try {
const sha = (character) => `sha256:${character.repeat(64)}`;
const signer = {
  authority: 'electron-episode-fixture',
  publicKeyJwk: TEST_CAPSULE_PUBLIC_KEY,
  privateKeyJwk: { ...TEST_CAPSULE_PUBLIC_KEY, d: 'WQi2FHRfw0jZxl_IXiMp5TAuehMfssojWd2Oj3WaUKU' },
};
const first = await createSignedCapsuleFixture({ operation: 'rerank' });
const nextRelease = structuredClone(first.capsule.release);
nextRelease.lifecycle.releaseVersion = '2.0.0';
nextRelease.lifecycle.supersedes = { capsuleId: first.capsule.capsuleId, semanticRoot: first.capsule.semanticRoot };
nextRelease.lifecycle.failedUpgrade.previousCapsuleId = first.capsule.capsuleId;
nextRelease.lifecycle.failedUpgrade.previousSemanticRoot = first.capsule.semanticRoot;
const second = await createSignedCapsuleFixture({ release: nextRelease, operation: 'rerank' });
assert.notEqual(first.capsule.semanticRoot, second.capsule.semanticRoot);
const fixtures = [first, second];
const references = fixtures.map(({ capsule }, index) => ({
  capsuleId: capsule.capsuleId, semanticRoot: capsule.semanticRoot, path: `capsules/revision-${index}.json`,
}));
let handler;
let bridge;
let now = '2026-09-04T00:00:00.000Z';
const stateStore = createDocumentSearchReleaseStore(path.join(directory, 'release.json'));
const mainOptions = {
  stateStore,
  authorizeRequest: () => true, // Synthetic trusted application event; sender denial has its own regression.
  now: () => now,
  verifyReleaseDecision: (record) => verifyProductionReleaseEvidenceSignature(record, { [signer.authority]: signer.publicKeyJwk }),
  verifyRevocationSnapshot: (record) => verifyProductionReleaseEvidenceSignature(record, { [signer.authority]: signer.publicKeyJwk }),
  ipcMain: { handle(_channel, value) { handler = value; } },
};
let coordinator = registerDocumentSearchReleaseMain(mainOptions);
exposeDocumentSearchReleaseBridge(
  { exposeInMainWorld(name, value) { assert.equal(name, 'dopplerRelease'); bridge = value; } },
  { invoke(_channel, request) { return handler({}, request); } },
);
const revocation = { ...first.capsule.release.revocation, authorityId: signer.authority };
async function decision(index) {
  const capsule = references[index];
  return signProductionReleaseEvidence({
    schema: RELEASE_DECISION_SCHEMA, releaseId: `electron-episode-${index}`, productionReleaseDigest: sha('1'),
    capsule: { ...capsule, envelopeDigest: sha('2') },
    eligibility: 'eligible', reasons: [], applicationGateReceipts: [], fleetReceipts: [], knownExclusions: [],
    previousRelease: { releaseId: 'previous', capsuleSemanticRoot: first.capsule.semanticRoot },
    rollback: { releaseId: 'previous', capsuleSemanticRoot: first.capsule.semanticRoot, authority: 'customer' },
    revocation, activationAuthority: 'customer', selfPromotionAllowed: false,
    createdAtUtc: now, digest: '', signature: null,
  }, signer);
}
async function snapshot(sequence, revokedSemanticRoots, fields = {}) {
  return signProductionReleaseEvidence({
    schema: ELECTRON_REVOCATION_SNAPSHOT_SCHEMA, authorityId: signer.authority,
    policyDigest: revocation.policyDigest, sequence, revokedSemanticRoots,
    issuedAtUtc: '2026-09-04T00:00:00.000Z', expiresAtUtc: '2026-09-05T00:00:00.000Z',
    ...fields,
    digest: '', signature: null,
  }, signer);
}
let executions = 0;
let closed = 0;
const renderer = createDocumentSearchRenderer(bridge, {
  device: {
    getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }),
    getDevice: () => ({ createBuffer() {}, createCommandEncoder() {}, queue: {} }),
  },
  trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
  capsuleSource: { async fetchCapsule(path) { return fixtures[references.findIndex((ref) => ref.path === path)].capsule; } },
  artifactStore: first.artifactStore,
  async programFactory({ capsule }) {
    return {
      executionGraphHash: capsule.program.executionGraphHash,
      async rerank() {
        executions += 1;
        return { schema: 'doppler_rerank_evidence/v1', inputHash: sha('a'), outputHash: sha('b'), backendIdentityHash: sha('c') };
      },
      async close() { closed += 1; },
    };
  },
});
const request = { application: first.capsule.release.application, query: 'query', documents: ['document'] };
await assert.rejects(renderer.rerank(request), /no active Capsule/);
const firstDecision = await decision(0);
await coordinator.installCandidate(references[0], firstDecision.digest);
await assert.rejects(renderer.rerank(request), /no active Capsule/);
await bridge.activate(firstDecision, sha('d'));
await assert.rejects(renderer.rerank(request), /no verified revocation snapshot/);
await coordinator.applyRevocationSnapshot(await snapshot(1, []));
assert.equal((await renderer.rerank(request)).capsule.semanticRoot, first.capsule.semanticRoot);
const untrustedRestart = registerDocumentSearchReleaseMain({ ...mainOptions,
  stateStore: createDocumentSearchReleaseStore(path.join(directory, 'release.json')),
  verifyRevocationSnapshot: () => false,
  ipcMain: { handle() {} },
});
await assert.rejects(untrustedRestart.resolveCurrent(), /verified.*signature/,
  'restoring a file must not bypass current signature trust');
await assert.rejects(coordinator.applyRevocationSnapshot(await snapshot(2, [], {
  issuedAtUtc: '2026-09-04T01:00:00.000Z', expiresAtUtc: '2026-09-05T01:00:00.000Z',
})), /future/);
const invalidClock = registerDocumentSearchReleaseMain({ ...mainOptions, now: () => 'invalid', ipcMain: { handle() {} } });
await assert.rejects(invalidClock.resolveCurrent(), /ISO instant/);

const secondDecision = await decision(1);
await coordinator.installCandidate(references[1], secondDecision.digest);
assert.equal((await renderer.rerank(request)).capsule.semanticRoot, first.capsule.semanticRoot, 'install is not activation');
await coordinator.rejectCandidate(sha('e'));
assert.equal((await bridge.status()).failures.length, 1);
await coordinator.installCandidate(references[1], secondDecision.digest);
await bridge.activate(secondDecision, sha('f'));
assert.equal((await renderer.rerank(request)).lifecycle.releaseVersion, '2.0.0');

coordinator = registerDocumentSearchReleaseMain(mainOptions);
assert.equal((await bridge.resolveCurrent()).semanticRoot, second.capsule.semanticRoot, 'restart preserves activation');
await coordinator.applyRevocationSnapshot(await snapshot(2, [second.capsule.semanticRoot]));
const beforeRevoked = executions;
await assert.rejects(renderer.rerank(request), /current Capsule is revoked/);
assert.equal(executions, beforeRevoked);
coordinator = registerDocumentSearchReleaseMain({ ...mainOptions,
  stateStore: createDocumentSearchReleaseStore(path.join(directory, 'release.json')) });
await assert.rejects(renderer.rerank(request), /current Capsule is revoked/, 'revocation survives reopening the durable store');
await assert.rejects(coordinator.applyRevocationSnapshot(await snapshot(1, [])), /monotonically/);
await assert.rejects(coordinator.applyRevocationSnapshot(await snapshot(3, [])), /retain all previously revoked/);
assert.equal(executions, beforeRevoked);
await bridge.rollback(sha('d'));
assert.equal((await renderer.rerank(request)).capsule.semanticRoot, first.capsule.semanticRoot);
assert.equal((await bridge.status()).failures.length, 1, 'rejected evidence survives rollback');
now = '2026-09-06T00:00:00.000Z';
await assert.rejects(renderer.rerank(request), /revocation state is expired/);
await assert.rejects(handler({}, { action: 'resolve-current', path: references[0].path }), /unsupported/);
assert.equal(closed, executions);
console.log('electron-release-episode.test: ok (durable local files and signatures; synthetic execution and IPC)');
} finally { await fs.rm(directory, { recursive: true, force: true }); }
