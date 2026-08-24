import assert from 'node:assert/strict';

import {
  ELECTRON_REVOCATION_SNAPSHOT_SCHEMA,
  createElectronReleaseStateCoordinator,
} from '../../src/client/electron/release-state.js';
import { createElectronRendererRuntime } from '../../src/client/electron/renderer-runtime.js';
import {
  RELEASE_DECISION_SCHEMA,
  signProductionReleaseEvidence,
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

async function decision(pack, marker) {
  return signProductionReleaseEvidence({
    schema: RELEASE_DECISION_SCHEMA,
    releaseId: `electron-fixture-release-${marker.repeat(16)}`,
    productionReleaseDigest: sha('1'),
    pack: {
      ...pack,
      envelopeDigest: sha('2'),
    },
    eligibility: 'eligible',
    reasons: [],
    applicationGateReceipts: [],
    fleetReceipts: [],
    knownExclusions: [],
    previousRelease: { releaseId: 'previous', packSemanticRoot: sha('3') },
    rollback: { releaseId: 'previous', packSemanticRoot: sha('3'), authority: 'customer' },
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
  now: () => now,
});
const packA = { packId: 'pack-a', semanticRoot: sha('a'), path: 'packs/a.json' };
const decisionA = await decision(packA, 'a');
await coordinator.installCandidate(packA, decisionA.digest);
await coordinator.activateCandidate(decisionA, sha('5'));
await coordinator.applyRevocationSnapshot({
  schema: ELECTRON_REVOCATION_SNAPSHOT_SCHEMA,
  authorityId: 'fixture-authority',
  sequence: 1,
  expiresAtUtc: '2026-08-25T00:00:00.000Z',
  revokedSemanticRoots: [],
  digest: sha('6'),
  signatureVerified: true,
});
assert.deepEqual(await coordinator.resolveCurrent(), packA);

const restarted = createElectronReleaseStateCoordinator({
  stateStore,
  verifyReleaseDecision: async () => true,
  now: () => now,
});
assert.deepEqual(await restarted.resolveCurrent(), packA);

const rejectedPack = { packId: 'pack-rejected', semanticRoot: sha('b'), path: 'packs/rejected.json' };
await restarted.installCandidate(rejectedPack, sha('7'));
await restarted.rejectCandidate(sha('8'));
const rejectedState = await restarted.load();
assert.equal(rejectedState.current.pack.semanticRoot, packA.semanticRoot);
assert.equal(rejectedState.failures.at(-1).candidateSemanticRoot, rejectedPack.semanticRoot);

now = '2026-08-24T01:00:00.000Z';
const packB = { packId: 'pack-b', semanticRoot: sha('c'), path: 'packs/b.json' };
const decisionB = await decision(packB, 'b');
await restarted.installCandidate(packB, decisionB.digest);
await restarted.activateCandidate(decisionB, sha('9'));
assert.equal((await restarted.load()).previous.pack.semanticRoot, packA.semanticRoot);
await restarted.rollback(sha('d'));
assert.equal((await restarted.resolveCurrent()).semanticRoot, packA.semanticRoot);

const controller = new AbortController();
controller.abort();
let opened = false;
const cancelledRuntime = createElectronRendererRuntime({
  releaseState: restarted,
  openPack: async () => {
    opened = true;
    return {};
  },
});
await assert.rejects(
  cancelledRuntime.rerank('query', ['document'], { signal: controller.signal }),
  (error) => error.code === 'DOPPLER_ELECTRON_CANCELLED'
);
assert.equal(opened, false);

let closed = false;
const deviceLossRuntime = createElectronRendererRuntime({
  releaseState: restarted,
  openPack: async () => ({
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
  deviceLossRuntime.rerank('query', ['document']),
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
