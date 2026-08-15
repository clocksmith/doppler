import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';

import {
  assertBundledAdapterAuthorized,
  assertKnownResolutionNotRevoked,
  authorizeBundledAdapter,
} from '../../src/config/revocation-policy.js';
import {
  configureSignedRevocationAuthority,
  getSignedRevocationStatus,
  refreshSignedRevocations,
  serializeSignedRevocationEnvelope,
} from '../../src/config/revocation-updates.js';
import { buildLayerContext } from '../../src/inference/pipelines/text/generator-helpers.js';
import { doppler } from '../../src/client/doppler-api.js';

globalThis.crypto ??= webcrypto;

const ADAPTER_DIGEST = `sha256:${'b'.repeat(64)}`;
const EMPTY_TARGETS = {
  logicalModelIds: [],
  modelIds: [],
  sourceCheckpointIds: [],
  weightPackIds: [],
  manifestVariantIds: [],
  artifactVariantIds: [],
  adapterIds: [],
  adapterDigests: [],
};
let now = Date.parse('2026-08-15T12:00:00.000Z');
let persisted = null;
let failSave = false;
const responses = [];

async function createSigner(id) {
  const pair = await crypto.subtle.generateKey(
    { name: 'ECDSA', namedCurve: 'P-256' },
    true,
    ['sign', 'verify']
  );
  return {
    id,
    privateKey: pair.privateKey,
    publicKeyJwk: await crypto.subtle.exportKey('jwk', pair.publicKey),
  };
}

function registry(issuedAtUtc, revocations = []) {
  return {
    $schema: 'schema/revocation-registry.schema.json',
    schemaVersion: 1,
    source: 'doppler',
    updatedAtUtc: issuedAtUtc,
    trust: { distribution: 'signed-live', signatureVerification: 'verified' },
    revocations,
  };
}

async function envelope({ signer, epoch, sequence, revocations = [], keyring = null, issuedOffset = 0, lifetime = 60_000 }) {
  const issuedAtUtc = new Date(now + issuedOffset).toISOString();
  const value = {
    schema: 'doppler.signed-revocation-envelope/v1',
    authorityId: 'clocksmith-revocations-v1',
    epoch,
    sequence,
    issuedAtUtc,
    expiresAtUtc: new Date(now + issuedOffset + lifetime).toISOString(),
    registry: registry(issuedAtUtc, revocations),
    keyring,
    signerId: signer.id,
    signature: '',
  };
  const signature = await crypto.subtle.sign(
    { name: 'ECDSA', hash: 'SHA-256' },
    signer.privateKey,
    new TextEncoder().encode(serializeSignedRevocationEnvelope(value))
  );
  value.signature = Buffer.from(signature).toString('base64url');
  return value;
}

async function fetchFn(url, init) {
  assert.equal(url, 'https://revocations.clocksmith.dev/doppler/v1.json');
  assert.equal(init.cache, 'no-store');
  assert.equal(init.redirect, 'error');
  assert.equal(init.headers.accept, 'application/json');
  assert.ok(init.signal instanceof AbortSignal);
  const next = responses.shift();
  if (next instanceof Error) throw next;
  return new Response(JSON.stringify(next), { status: 200 });
}

const onlineV1 = await createSigner('online-v1');
const onlineV2 = await createSigner('online-v2');
const recovery = await createSigner('recovery-v1');
const initial = await envelope({ signer: onlineV1, epoch: 1, sequence: 1 });
responses.push(initial);

const options = {
  authorityId: 'clocksmith-revocations-v1',
  url: 'https://revocations.clocksmith.dev/doppler/v1.json',
  initialEpoch: 1,
  onlineKeys: [{ id: onlineV1.id, publicKeyJwk: onlineV1.publicKeyJwk }],
  recoveryKeys: [{ id: recovery.id, publicKeyJwk: recovery.publicKeyJwk }],
  refreshIntervalMs: 1000,
  requestTimeoutMs: 1000,
  maxBytes: 10_000,
  maxClockSkewMs: 1000,
  maxEnvelopeLifetimeMs: 120_000,
  stateStore: {
    async load() { return structuredClone(persisted); },
    async save(value) {
      if (failSave) throw new Error('state store failed');
      persisted = structuredClone(value);
    },
  },
  fetchFn,
  now: () => now,
};

assert.equal(typeof doppler.revocations.configure, 'function');
assert.equal(typeof doppler.revocations.refresh, 'function');
assert.equal(typeof doppler.revocations.status, 'function');
await assert.rejects(
  () => configureSignedRevocationAuthority({ ...options, unknown: true }),
  /unknown is not supported/
);
const configured = await configureSignedRevocationAuthority(options);
assert.equal(configured.signatureVerification, 'verified');
assert.equal(configured.current, true);
assert.equal(configured.epoch, 1);
assert.equal(configured.sequence, 1);
assert.ok(persisted);

responses.push(initial);
assert.equal((await refreshSignedRevocations({ force: true })).sequence, 1);

const replayRewrite = await envelope({ signer: onlineV1, epoch: 1, sequence: 1, issuedOffset: 1 });
responses.push(replayRewrite);
await assert.rejects(
  () => refreshSignedRevocations({ force: true }),
  /rolled back or replayed/
);

const adapter = {
  id: 'loaded-adapter',
  identity: {
    schema: 'doppler.lora-execution-identity/v1',
    id: 'loaded-adapter',
    digest: ADAPTER_DIGEST,
  },
};
await authorizeBundledAdapter(adapter);
assert.doesNotThrow(() => assertKnownResolutionNotRevoked({ modelId: 'loaded-model' }));
assert.doesNotThrow(() => assertBundledAdapterAuthorized(adapter));

const denyRecord = {
  id: 'loaded-identity-revoked',
  state: 'revoked',
  issuedAtUtc: new Date(now).toISOString(),
  severity: 'security',
  reason: 'Deterministic live invalidation fixture.',
  targets: {
    ...EMPTY_TARGETS,
    modelIds: ['loaded-model'],
    adapterIds: ['loaded-adapter'],
    adapterDigests: [ADAPTER_DIGEST],
  },
  replacements: { ...EMPTY_TARGETS },
  evidencePaths: ['docs/revocation.md'],
};
const recoveryEnvelope = await envelope({
  signer: recovery,
  epoch: 2,
  sequence: 1,
  revocations: [denyRecord],
  keyring: {
    onlineKeys: [{ id: onlineV2.id, publicKeyJwk: onlineV2.publicKeyJwk }],
    revokedKeys: [{ id: onlineV1.id, publicKeyJwk: onlineV1.publicKeyJwk }],
  },
});
responses.push(recoveryEnvelope);
const recovered = await refreshSignedRevocations({ force: true });
assert.equal(recovered.epoch, 2);
assert.equal(recovered.sequence, 1);
assert.throws(() => assertKnownResolutionNotRevoked({ modelId: 'loaded-model' }), /loaded-identity-revoked/);
assert.throws(() => assertBundledAdapterAuthorized(adapter), /loaded-identity-revoked/);
assert.throws(
  () => buildLayerContext({ revocationIdentity: { modelId: 'loaded-model' }, lora: null }),
  /loaded-identity-revoked/
);
assert.throws(
  () => buildLayerContext({ revocationIdentity: { modelId: 'allowed-model' }, lora: adapter }),
  /loaded-identity-revoked/
);

const compromised = await envelope({ signer: onlineV1, epoch: 2, sequence: 2, revocations: [denyRecord] });
responses.push(compromised);
await assert.rejects(
  () => refreshSignedRevocations({ force: true }),
  /signer is not trusted/
);

const retained = await envelope({ signer: onlineV2, epoch: 2, sequence: 2, revocations: [denyRecord] });
responses.push(retained);
assert.equal((await refreshSignedRevocations({ force: true })).sequence, 2);

const excessiveLifetime = await envelope({
  signer: onlineV2,
  epoch: 2,
  sequence: 3,
  revocations: [denyRecord],
  lifetime: options.maxEnvelopeLifetimeMs + 1,
});
responses.push(excessiveLifetime);
await assert.rejects(
  () => refreshSignedRevocations({ force: true }),
  /lifetime exceeds authority policy/
);

const unrevokedKey = await envelope({
  signer: recovery,
  epoch: 3,
  sequence: 1,
  revocations: [denyRecord],
  keyring: {
    onlineKeys: [{ id: onlineV2.id, publicKeyJwk: onlineV2.publicKeyJwk }],
    revokedKeys: [],
  },
});
responses.push(unrevokedKey);
await assert.rejects(
  () => refreshSignedRevocations({ force: true }),
  /omitted or rewrote revoked key online-v1/
);

const unsaved = await envelope({ signer: onlineV2, epoch: 2, sequence: 3, revocations: [denyRecord] });
failSave = true;
responses.push(unsaved);
await assert.rejects(
  () => refreshSignedRevocations({ force: true }),
  /state store failed/
);
assert.equal(getSignedRevocationStatus().sequence, 2);
failSave = false;
responses.push(unsaved);
assert.equal((await refreshSignedRevocations({ force: true })).sequence, 3);

const dropped = await envelope({ signer: onlineV2, epoch: 2, sequence: 4 });
responses.push(dropped);
await assert.rejects(
  () => refreshSignedRevocations({ force: true }),
  /removed or rewrote deny record/
);

responses.push('x'.repeat(options.maxBytes + 1));
await assert.rejects(
  () => refreshSignedRevocations({ force: true }),
  /exceeds maxBytes/
);

responses.push(new Error('offline'));
const offline = await refreshSignedRevocations({ force: true });
assert.equal(offline.offline, true);
assert.equal(offline.sequence, 3);

responses.push(unsaved);
assert.equal((await configureSignedRevocationAuthority(options)).sequence, 3);

now += 61_000;
responses.push(new Error('offline-expired'));
await assert.rejects(
  () => refreshSignedRevocations({ force: true }),
  /without current verified state/
);
assert.throws(
  () => assertKnownResolutionNotRevoked({ modelId: 'allowed-model' }),
  /state is not current/
);
assert.equal(getSignedRevocationStatus().current, false);

console.log('signed-revocation-updates.test: ok');
