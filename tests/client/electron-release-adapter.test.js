import assert from 'node:assert/strict';

import {
  ELECTRON_REVOCATION_SNAPSHOT_SCHEMA,
  createElectronReleaseStateCoordinator,
} from '../../src/client/electron/release-state.js';
import { createElectronRendererRuntime } from '../../src/client/electron/renderer-runtime.js';
import {
  RELEASE_DECISION_SCHEMA,
  signProductionReleaseEvidence,
  validateReleaseDecision,
  verifyProductionReleaseEvidenceSignature,
} from '../../src/config/production-release-evidence.js';

const privateKeyJwk = {
  crv: 'Ed25519',
  d: 'WQi2FHRfw0jZxl_IXiMp5TAuehMfssojWd2Oj3WaUKU',
  x: 'FLU5-eSyW8ORkAf8HupzJn8juiJ2TrGSw2rgMNqGPfc',
  kty: 'OKP',
};
const publicKeyJwk = { crv: privateKeyJwk.crv, x: privateKeyJwk.x, kty: privateKeyJwk.kty };
const sha = (character) => `sha256:${character.repeat(64)}`;
let now = '2026-08-24T00:00:00.000Z';
let stored = null;
const stateStore = {
  async load() {
    return stored === null ? null : structuredClone(stored);
  },
  async compareAndSwap(expectedSequence, nextState) {
    const sequence = stored?.sequence ?? 0;
    if (sequence !== expectedSequence) return false;
    stored = structuredClone(nextState);
    return true;
  },
};

async function decision(capsule, marker) {
  return signProductionReleaseEvidence({
    schema: RELEASE_DECISION_SCHEMA,
    releaseId: `electron-fixture-release-${marker.repeat(16)}`,
    productionReleaseDigest: sha('1'),
    capsule: {
      ...capsule,
      envelopeDigest: sha('2'),
    },
    eligibility: 'eligible',
    reasons: [],
    applicationGateReceipts: [],
    fleetReceipts: [],
    knownExclusions: [],
    previousRelease: { releaseId: 'previous', capsuleSemanticRoot: sha('3') },
    rollback: { releaseId: 'previous', capsuleSemanticRoot: sha('3'), authority: 'customer' },
    revocation: {
      authorityId: 'fixture-authority',
      policyDigest: sha('4'),
      offlineExpirySeconds: 86400,
      failClosedAfterExpiry: true,
    },
    activationAuthority: 'customer',
    selfPromotionAllowed: false,
    createdAtUtc: now,
    digest: '',
    signature: null,
  }, {
    authority: 'fixture-release-authority',
    privateKeyJwk,
    publicKeyJwk,
  });
}

const coordinator = createElectronReleaseStateCoordinator({
  stateStore,
  verifyReleaseDecision: async () => true,
  verifyRevocationSnapshot: (snapshot) => verifyProductionReleaseEvidenceSignature(
    snapshot,
    { 'fixture-authority': publicKeyJwk }
  ),
  now: () => now,
});
const capsuleA = { capsuleId: 'capsule-a', semanticRoot: sha('a'), path: 'capsules/a.json' };
const decisionA = await decision(capsuleA, 'a');
assert.equal(validateReleaseDecision(decisionA).ok, true);
const malformedDecision = structuredClone(decisionA);
malformedDecision.eligibility = 'blocked';
malformedDecision.reasons = [{
  code: 'invented-rejection',
  scope: 'fixture',
  detail: 'Malformed rejection code for validator coverage.',
  evidenceDigests: [],
}];
assert.ok(validateReleaseDecision(malformedDecision).errors.includes(
  'release decision.reasons[0].code is unsupported.'
));
await coordinator.installCandidate(capsuleA, decisionA.digest);
await coordinator.activateCandidate(decisionA, sha('5'));
await assert.rejects(coordinator.applyRevocationSnapshot({
  schema: ELECTRON_REVOCATION_SNAPSHOT_SCHEMA,
  authorityId: 'fixture-authority',
  sequence: 1,
  expiresAtUtc: '2026-08-25T00:00:00.000Z',
  revokedSemanticRoots: [],
  digest: sha('6'),
  signatureVerified: true,
}), /signatureVerified is not supported/u);
const revocationSnapshot = await signProductionReleaseEvidence({
  schema: ELECTRON_REVOCATION_SNAPSHOT_SCHEMA,
  authorityId: 'fixture-authority',
  policyDigest: sha('4'),
  sequence: 1,
  issuedAtUtc: '2026-08-24T00:00:00.000Z',
  expiresAtUtc: '2026-08-25T00:00:00.000Z',
  revokedSemanticRoots: [],
  digest: '',
  signature: null,
}, {
  authority: 'fixture-authority',
  privateKeyJwk,
  publicKeyJwk,
});
await coordinator.applyRevocationSnapshot(revocationSnapshot);
assert.equal((await coordinator.applyRevocationSnapshot(revocationSnapshot)).sequence, 3);
assert.deepEqual(await coordinator.resolveCurrent(), capsuleA);

const restarted = createElectronReleaseStateCoordinator({
  stateStore,
  verifyReleaseDecision: async () => true,
  verifyRevocationSnapshot: (snapshot) => verifyProductionReleaseEvidenceSignature(
    snapshot,
    { 'fixture-authority': publicKeyJwk }
  ),
  now: () => now,
});
assert.deepEqual(await restarted.resolveCurrent(), capsuleA);

const rejectedCapsule = { capsuleId: 'capsule-rejected', semanticRoot: sha('b'), path: 'capsules/rejected.json' };
await restarted.installCandidate(rejectedCapsule, sha('7'));
await restarted.rejectCandidate(sha('8'));
const rejectedState = await restarted.load();
assert.equal(rejectedState.current.capsule.semanticRoot, capsuleA.semanticRoot);
assert.equal(rejectedState.failures.at(-1).candidateSemanticRoot, rejectedCapsule.semanticRoot);

now = '2026-08-24T01:00:00.000Z';
const capsuleB = { capsuleId: 'capsule-b', semanticRoot: sha('c'), path: 'capsules/b.json' };
const decisionB = await decision(capsuleB, 'b');
await restarted.installCandidate(capsuleB, decisionB.digest);
await restarted.activateCandidate(decisionB, sha('9'));
assert.equal((await restarted.load()).previous.capsule.semanticRoot, capsuleA.semanticRoot);
await restarted.rollback(sha('d'));
assert.equal((await restarted.resolveCurrent()).semanticRoot, capsuleA.semanticRoot);

const controller = new AbortController();
controller.abort();
const request = { application: {}, query: 'query', documents: ['document'] };
let opened = false;
const cancelledRuntime = createElectronRendererRuntime({
  releaseState: restarted,
  openCapsule: async () => {
    opened = true;
    return {};
  },
});
await assert.rejects(
  cancelledRuntime.rerank(request, { signal: controller.signal }),
  (error) => error.code === 'DOPPLER_ELECTRON_CANCELLED'
);
assert.equal(opened, false);

let closed = false;
const deviceLossRuntime = createElectronRendererRuntime({
  releaseState: restarted,
  openCapsule: async () => ({
    ...capsuleA,
    async rerank() {
      const error = new Error('adapter removed');
      error.code = 'GPU_DEVICE_LOST';
      throw error;
    },
    async close() {
      closed = true;
    },
  }),
});
await assert.rejects(
  deviceLossRuntime.rerank(request),
  (error) => error.code === 'DOPPLER_ELECTRON_DEVICE_LOST'
);
assert.equal(closed, true);

now = '2026-08-26T00:00:00.000Z';
await assert.rejects(restarted.resolveCurrent(), /revocation state is expired/u);

const validState = stored;
stored = { schema: 'corrupt' };
await assert.rejects(restarted.load(), /sequence is required/u);
stored = validState;

console.log('electron-release-adapter.test: ok');
