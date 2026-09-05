import assert from 'node:assert/strict';
import { registerDocumentSearchReleaseMain } from '../../examples/electron-document-search/main.js';
import { exposeDocumentSearchReleaseBridge } from '../../examples/electron-document-search/preload.js';
import { createDocumentSearchRenderer } from '../../examples/electron-document-search/renderer.js';
import { createSignedPackFixture, TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY } from '../helpers/pack-v2-fixture.js';
import {
  RELEASE_DECISION_SCHEMA, signProductionReleaseEvidence, verifyProductionReleaseEvidenceSignature,
} from '../../src/config/production-release-evidence.js';
import { ELECTRON_REVOCATION_SNAPSHOT_SCHEMA } from 'doppler-gpu/electron';

const sha = (character) => `sha256:${character.repeat(64)}`;
const signer = {
  authority: 'electron-episode-fixture',
  publicKeyJwk: TEST_PACK_PUBLIC_KEY,
  privateKeyJwk: { ...TEST_PACK_PUBLIC_KEY, d: 'WQi2FHRfw0jZxl_IXiMp5TAuehMfssojWd2Oj3WaUKU' },
};
const first = await createSignedPackFixture();
const nextRelease = structuredClone(first.pack.release);
nextRelease.lifecycle.releaseVersion = '2.0.0';
nextRelease.lifecycle.supersedes = { packId: first.pack.packId, semanticRoot: first.pack.semanticRoot };
nextRelease.lifecycle.failedUpgrade.previousPackId = first.pack.packId;
nextRelease.lifecycle.failedUpgrade.previousSemanticRoot = first.pack.semanticRoot;
const second = await createSignedPackFixture({ release: nextRelease });
assert.notEqual(first.pack.semanticRoot, second.pack.semanticRoot);
const fixtures = [first, second];
const references = fixtures.map(({ pack }, index) => ({
  packId: pack.packId, semanticRoot: pack.semanticRoot, path: `packs/revision-${index}.json`,
}));
let stored = null;
let handler;
let bridge;
let now = '2026-09-04T00:00:00.000Z';
const stateStore = {
  async load() { return structuredClone(stored); },
  async compareAndSwap(sequence, state) {
    if ((stored?.sequence ?? 0) !== sequence) return false;
    stored = structuredClone(state);
    return true;
  },
};
const mainOptions = {
  stateStore,
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
const revocation = { ...first.pack.release.revocation, authorityId: signer.authority };
async function decision(index) {
  const pack = references[index];
  return signProductionReleaseEvidence({
    schema: RELEASE_DECISION_SCHEMA, releaseId: `electron-episode-${index}`, productionReleaseDigest: sha('1'),
    pack: { ...pack, envelopeDigest: sha('2') },
    eligibility: 'eligible', reasons: [], applicationGateReceipts: [], fleetReceipts: [], knownExclusions: [],
    previousRelease: { releaseId: 'previous', packSemanticRoot: first.pack.semanticRoot },
    rollback: { releaseId: 'previous', packSemanticRoot: first.pack.semanticRoot, authority: 'customer' },
    revocation, activationAuthority: 'customer', selfPromotionAllowed: false,
    createdAtUtc: now, digest: '', signature: null,
  }, signer);
}
async function snapshot(sequence, revokedSemanticRoots) {
  return signProductionReleaseEvidence({
    schema: ELECTRON_REVOCATION_SNAPSHOT_SCHEMA, authorityId: signer.authority,
    policyDigest: revocation.policyDigest, sequence, revokedSemanticRoots,
    issuedAtUtc: '2026-09-04T00:00:00.000Z', expiresAtUtc: '2026-09-05T00:00:00.000Z',
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
  trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
  packSource: { async fetchPack(path) { return fixtures[references.findIndex((ref) => ref.path === path)].pack; } },
  artifactStore: first.artifactStore,
  async programFactory({ pack }) {
    return {
      executionGraphHash: pack.program.executionGraphHash,
      async rerank() {
        executions += 1;
        return { schema: 'doppler_rerank_evidence/v1', inputHash: sha('a'), outputHash: sha('b'), backendIdentityHash: sha('c') };
      },
      async close() { closed += 1; },
    };
  },
});
const request = { application: first.pack.release.application, query: 'query', documents: ['document'] };
await assert.rejects(renderer.rerank(request), /no active Pack/);
const firstDecision = await decision(0);
await coordinator.installCandidate(references[0], firstDecision.digest);
await assert.rejects(renderer.rerank(request), /no active Pack/);
await bridge.activate(firstDecision, sha('d'));
await assert.rejects(renderer.rerank(request), /no verified revocation snapshot/);
await coordinator.applyRevocationSnapshot(await snapshot(1, []));
assert.equal((await renderer.rerank(request)).pack.semanticRoot, first.pack.semanticRoot);

const secondDecision = await decision(1);
await coordinator.installCandidate(references[1], secondDecision.digest);
assert.equal((await renderer.rerank(request)).pack.semanticRoot, first.pack.semanticRoot, 'install is not activation');
await coordinator.rejectCandidate(sha('e'));
assert.equal((await bridge.status()).failures.length, 1);
await coordinator.installCandidate(references[1], secondDecision.digest);
await bridge.activate(secondDecision, sha('f'));
assert.equal((await renderer.rerank(request)).lifecycle.releaseVersion, '2.0.0');

coordinator = registerDocumentSearchReleaseMain(mainOptions);
assert.equal((await bridge.resolveCurrent()).semanticRoot, second.pack.semanticRoot, 'restart preserves activation');
await coordinator.applyRevocationSnapshot(await snapshot(2, [second.pack.semanticRoot]));
const beforeRevoked = executions;
await assert.rejects(renderer.rerank(request), /current Pack is revoked/);
assert.equal(executions, beforeRevoked);
await bridge.rollback(sha('d'));
assert.equal((await renderer.rerank(request)).pack.semanticRoot, first.pack.semanticRoot);
assert.equal((await bridge.status()).failures.length, 1, 'rejected evidence survives rollback');
now = '2026-09-06T00:00:00.000Z';
await assert.rejects(renderer.rerank(request), /revocation state is expired/);
await assert.rejects(handler({}, { action: 'resolve-current', path: references[0].path }), /unsupported/);
assert.equal(closed, executions);
console.log('electron-release-episode.test: ok (signed fixture releases; synthetic execution and IPC)');
