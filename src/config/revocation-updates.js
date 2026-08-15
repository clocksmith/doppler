import {
  installLiveRevocationRegistry,
  validateRevocationRegistry,
} from './revocation-policy.js';

const ENVELOPE_SCHEMA = 'doppler.signed-revocation-envelope/v1';
const STATE_SCHEMA = 'doppler.revocation-update-state/v1';
let authority = null;
let state = null;
let status = Object.freeze({ configured: false, signatureVerification: 'unavailable' });

function object(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return value;
}

function exact(value, fields, label) {
  object(value, label);
  const expected = new Set(fields);
  for (const field of Object.keys(value)) if (!expected.has(field)) throw new Error(`${label}.${field} is not supported.`);
  for (const field of fields) {
    if (!Object.prototype.hasOwnProperty.call(value, field)) throw new Error(`${label}.${field} is required.`);
  }
}

function exactOptions(value, required, optional, label) {
  object(value, label);
  const supported = new Set([...required, ...optional]);
  for (const field of Object.keys(value)) if (!supported.has(field)) throw new Error(`${label}.${field} is not supported.`);
  for (const field of required) {
    if (!Object.prototype.hasOwnProperty.call(value, field)) throw new Error(`${label}.${field} is required.`);
  }
}

function text(value, label) {
  const normalized = typeof value === 'string' ? value.trim() : '';
  if (!normalized) throw new Error(`${label} must be non-empty.`);
  return normalized;
}

function instant(value, label) {
  const normalized = text(value, label);
  const time = new Date(normalized).getTime();
  if (!Number.isFinite(time) || new Date(time).toISOString() !== normalized) throw new Error(`${label} must be an ISO instant.`);
  return { value: normalized, time };
}

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${label} must be a positive safe integer.`);
  return value;
}

function nonNegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} must be a non-negative safe integer.`);
  return value;
}

function keyFingerprint(key) {
  return `${key.publicKeyJwk.crv}:${key.publicKeyJwk.x}:${key.publicKeyJwk.y}`;
}

function normalizeKeys(value, label, allowEmpty) {
  if (!Array.isArray(value) || (!allowEmpty && value.length === 0)) {
    throw new Error(`${label} must be ${allowEmpty ? 'an' : 'a non-empty'} array.`);
  }
  const keys = value.map((entry, index) => {
    const keyLabel = `${label}[${index}]`;
    exact(entry, ['id', 'publicKeyJwk'], keyLabel);
    const jwk = object(entry.publicKeyJwk, `${keyLabel}.publicKeyJwk`);
    if (jwk.kty !== 'EC' || jwk.crv !== 'P-256'
      || typeof jwk.x !== 'string' || !jwk.x
      || typeof jwk.y !== 'string' || !jwk.y
      || 'd' in jwk) {
      throw new Error(`${keyLabel}.publicKeyJwk must be an EC P-256 public key.`);
    }
    return { id: text(entry.id, `${keyLabel}.id`), publicKeyJwk: structuredClone(jwk) };
  });
  if (new Set(keys.map((entry) => entry.id)).size !== keys.length) throw new Error(`${label} IDs must be unique.`);
  if (new Set(keys.map(keyFingerprint)).size !== keys.length) throw new Error(`${label} public keys must be unique.`);
  return keys;
}

function keyringsOverlap(left, right) {
  return left.some((leftKey) => right.some((rightKey) => (
    leftKey.id === rightKey.id || keyFingerprint(leftKey) === keyFingerprint(rightKey)
  )));
}

function normalizeAuthority(options) {
  const value = object(options, 'revocation authority');
  exactOptions(value, [
    'authorityId', 'url', 'initialEpoch', 'onlineKeys', 'recoveryKeys',
    'refreshIntervalMs', 'requestTimeoutMs', 'maxBytes', 'maxClockSkewMs',
    'maxEnvelopeLifetimeMs', 'stateStore',
  ], ['fetchFn', 'now'], 'revocation authority');
  const url = new URL(text(value.url, 'revocation authority.url'));
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash) {
    throw new Error('revocation authority.url must be exact credential-free HTTPS without query or fragment.');
  }
  for (const field of ['refreshIntervalMs', 'requestTimeoutMs', 'maxBytes', 'maxEnvelopeLifetimeMs']) {
    positiveInteger(value[field], `revocation authority.${field}`);
  }
  nonNegativeInteger(value.maxClockSkewMs, 'revocation authority.maxClockSkewMs');
  const stateStore = object(value.stateStore, 'revocation authority.stateStore');
  if (typeof stateStore.load !== 'function' || typeof stateStore.save !== 'function') {
    throw new Error('revocation authority.stateStore requires load() and save().');
  }
  const fetchFn = value.fetchFn ?? globalThis.fetch;
  if (typeof fetchFn !== 'function') throw new Error('revocation authority requires fetch.');
  if (value.now !== undefined && typeof value.now !== 'function') throw new Error('revocation authority.now must be a function.');
  const onlineKeys = normalizeKeys(value.onlineKeys, 'revocation authority.onlineKeys', false);
  const recoveryKeys = normalizeKeys(value.recoveryKeys, 'revocation authority.recoveryKeys', false);
  if (keyringsOverlap(onlineKeys, recoveryKeys)) {
    throw new Error('Revocation online and recovery keys must be disjoint.');
  }
  return {
    authorityId: text(value.authorityId, 'revocation authority.authorityId'),
    url: url.href,
    initialEpoch: positiveInteger(value.initialEpoch, 'revocation authority.initialEpoch'),
    onlineKeys,
    recoveryKeys,
    refreshIntervalMs: value.refreshIntervalMs,
    requestTimeoutMs: value.requestTimeoutMs,
    maxBytes: value.maxBytes,
    maxClockSkewMs: value.maxClockSkewMs,
    maxEnvelopeLifetimeMs: value.maxEnvelopeLifetimeMs,
    stateStore,
    fetchFn,
    now: value.now ?? Date.now,
  };
}

function canonical(value) {
  if (value === null) return 'null';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('Signed revocation payload numbers must be finite.');
    return JSON.stringify(value);
  }
  if (typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value !== 'object') throw new Error('Signed revocation payload must contain only JSON values.');
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
}

export function serializeSignedRevocationEnvelope(envelope) {
  const { signature, ...payload } = envelope;
  return canonical(payload);
}

function decodeBase64Url(value) {
  const encoded = text(value, 'signed revocation signature');
  if (!/^[A-Za-z0-9_-]+$/u.test(encoded)) throw new Error('Signed revocation signature must use unpadded base64url.');
  const normalized = encoded.replace(/-/g, '+').replace(/_/g, '/');
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, '=');
  let bytes;
  if (typeof atob === 'function') bytes = Uint8Array.from(atob(padded), (character) => character.charCodeAt(0));
  else if (typeof Buffer !== 'undefined') bytes = new Uint8Array(Buffer.from(padded, 'base64'));
  else throw new Error('Signed revocation verification requires a base64 decoder.');
  if (bytes.byteLength !== 64) throw new Error('Signed revocation P-256 signature must be 64 bytes.');
  return bytes;
}

function normalizeEnvelope(value) {
  exact(value, ['schema', 'authorityId', 'epoch', 'sequence', 'issuedAtUtc', 'expiresAtUtc', 'registry', 'keyring', 'signerId', 'signature'], 'signed revocation envelope');
  if (value.schema !== ENVELOPE_SCHEMA) throw new Error(`signed revocation envelope.schema must be ${ENVELOPE_SCHEMA}.`);
  if (text(value.authorityId, 'signed revocation envelope.authorityId') !== authority.authorityId) throw new Error('Signed revocation authority mismatch.');
  const issued = instant(value.issuedAtUtc, 'signed revocation envelope.issuedAtUtc');
  const expires = instant(value.expiresAtUtc, 'signed revocation envelope.expiresAtUtc');
  if (expires.time <= issued.time) throw new Error('Signed revocation expiry must follow issuance.');
  if (expires.time - issued.time > authority.maxEnvelopeLifetimeMs) {
    throw new Error('Signed revocation envelope lifetime exceeds authority policy.');
  }
  const registry = validateRevocationRegistry(value.registry);
  if (registry.trust.distribution !== 'signed-live' || registry.updatedAtUtc !== issued.value) {
    throw new Error('Signed revocation registry trust or update instant is invalid.');
  }
  let keyring = null;
  if (value.keyring === undefined) throw new Error('signed revocation envelope.keyring must be an object or null.');
  if (value.keyring !== null) {
    exact(value.keyring, ['onlineKeys', 'revokedKeys'], 'signed revocation keyring');
    const onlineKeys = normalizeKeys(value.keyring.onlineKeys, 'signed revocation keyring.onlineKeys', false);
    const revokedKeys = normalizeKeys(value.keyring.revokedKeys, 'signed revocation keyring.revokedKeys', true);
    if (keyringsOverlap(onlineKeys, revokedKeys)) throw new Error('Signed revocation keyring cannot activate a revoked key.');
    if (keyringsOverlap(onlineKeys, authority.recoveryKeys)) throw new Error('Signed revocation online and recovery keys must be disjoint.');
    keyring = { onlineKeys, revokedKeys };
  }
  return {
    raw: value,
    signature: value.signature,
    epoch: positiveInteger(value.epoch, 'signed revocation envelope.epoch'),
    sequence: positiveInteger(value.sequence, 'signed revocation envelope.sequence'),
    issued,
    expires,
    registry,
    keyring,
    signerId: text(value.signerId, 'signed revocation envelope.signerId'),
  };
}

async function verifySignature(envelope, keys) {
  const signer = keys.find((entry) => entry.id === envelope.signerId);
  if (!signer) throw new Error(`Signed revocation signer is not trusted: ${envelope.signerId}.`);
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('Signed revocation verification requires WebCrypto.');
  const key = await subtle.importKey('jwk', signer.publicKeyJwk, { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']);
  const valid = await subtle.verify(
    { name: 'ECDSA', hash: 'SHA-256' },
    key,
    decodeBase64Url(envelope.signature),
    new TextEncoder().encode(serializeSignedRevocationEnvelope(envelope.raw))
  );
  if (!valid) throw new Error('Signed revocation signature mismatch.');
}

async function verifyCandidate(value, { restore = false } = {}) {
  const envelope = normalizeEnvelope(value);
  const recovery = envelope.keyring != null;
  await verifySignature(envelope, recovery ? authority.recoveryKeys : state.onlineKeys);
  if (!restore) {
    const now = authority.now();
    if (envelope.issued.time > now + authority.maxClockSkewMs) throw new Error('Signed revocation envelope is from the future.');
    if (envelope.expires.time <= now) throw new Error('Signed revocation envelope is expired.');
    const replay = state.dataEnvelope
      && envelope.epoch === state.epoch
      && envelope.sequence === state.sequence
      && canonical(envelope.raw) === canonical(state.dataEnvelope.raw);
    if (replay) return { ...envelope, replay: true };
    if (recovery) {
      if (envelope.epoch <= state.epoch) throw new Error('Signed revocation recovery epoch must advance.');
      for (const revokedKey of state.revokedKeys) {
        const retained = envelope.keyring.revokedKeys.find((entry) => entry.id === revokedKey.id);
        if (!retained || keyFingerprint(retained) !== keyFingerprint(revokedKey)) {
          throw new Error(`Signed revocation recovery omitted or rewrote revoked key ${revokedKey.id}.`);
        }
      }
      for (const onlineKey of state.onlineKeys) {
        const retained = envelope.keyring.onlineKeys.find((entry) => (
          entry.id === onlineKey.id && keyFingerprint(entry) === keyFingerprint(onlineKey)
        ));
        const retired = envelope.keyring.revokedKeys.find((entry) => (
          entry.id === onlineKey.id && keyFingerprint(entry) === keyFingerprint(onlineKey)
        ));
        if (!retained && !retired) throw new Error(`Signed revocation recovery omitted retired key ${onlineKey.id}.`);
      }
      if (envelope.keyring.onlineKeys.some((entry) => state.revokedKeys.some((revoked) => (
        entry.id === revoked.id || keyFingerprint(entry) === keyFingerprint(revoked)
      )))) {
        throw new Error('Signed revocation recovery reactivated revoked key material.');
      }
    } else if (envelope.epoch !== state.epoch || envelope.sequence <= state.sequence) {
      throw new Error('Signed revocation update was rolled back or replayed.');
    }
    for (const previous of state.dataEnvelope?.registry.revocations ?? []) {
      const retained = envelope.registry.revocations.find((entry) => entry.id === previous.id);
      if (!retained || canonical(retained) !== canonical(previous)) {
        throw new Error(`Signed revocation update removed or rewrote deny record ${previous.id}.`);
      }
    }
  }
  return envelope;
}

function applyEnvelope(envelope, keyProof) {
  if (envelope.keyring) {
    state.onlineKeys = envelope.keyring.onlineKeys;
    state.revokedKeys = envelope.keyring.revokedKeys;
  }
  state.epoch = envelope.epoch;
  state.sequence = envelope.sequence;
  state.keyProof = keyProof;
  state.dataEnvelope = envelope;
  installLiveRevocationRegistry(envelope.registry, () => {
    const currentTime = authority?.now();
    if (!state?.dataEnvelope || !Number.isFinite(currentTime)
      || state.dataEnvelope.expires.time <= currentTime) {
      throw new Error('Signed live revocation state is not current.');
    }
  });
}

async function restoreState(value) {
  if (value == null) return;
  exact(value, ['schema', 'authorityId', 'keyProof', 'dataEnvelope'], 'revocation update state');
  if (value.schema !== STATE_SCHEMA || value.authorityId !== authority.authorityId) throw new Error('Persisted revocation authority state is invalid.');
  let keyProof = null;
  if (value.keyProof) {
    keyProof = await verifyCandidate(value.keyProof, { restore: true });
    if (!keyProof.keyring || keyProof.epoch < authority.initialEpoch) throw new Error('Persisted revocation key proof is invalid.');
    state.onlineKeys = keyProof.keyring.onlineKeys;
    state.revokedKeys = keyProof.keyring.revokedKeys;
    state.epoch = keyProof.epoch;
    state.sequence = keyProof.sequence - 1;
  }
  if (value.dataEnvelope) {
    const data = await verifyCandidate(value.dataEnvelope, { restore: true });
    if (data.epoch !== state.epoch || data.sequence < state.sequence) throw new Error('Persisted revocation data state was rolled back.');
    applyEnvelope(data, keyProof);
  }
}

function snapshotStatus(overrides = {}) {
  const currentTime = authority.now();
  status = Object.freeze({
    configured: true,
    authorityId: authority.authorityId,
    epoch: state.epoch,
    sequence: state.sequence,
    expiresAtUtc: state.dataEnvelope?.expires.value ?? null,
    signatureVerification: state.dataEnvelope ? 'verified' : 'pending',
    current: Boolean(state.dataEnvelope && Number.isFinite(currentTime)
      && state.dataEnvelope.expires.time > currentTime),
    offline: false,
    lastError: null,
    ...overrides,
  });
  return status;
}

function offlineStatus(reason) {
  if (!state.dataEnvelope || state.dataEnvelope.expires.time <= authority.now()) {
    throw new Error(`Signed revocation refresh failed without current verified state: ${reason}`);
  }
  return snapshotStatus({ offline: true, lastError: reason });
}

async function readBoundedResponse(response) {
  const contentLength = response.headers?.get?.('content-length');
  if (contentLength != null && /^\d+$/u.test(contentLength)
    && Number(contentLength) > authority.maxBytes) {
    throw new Error('Signed revocation response exceeds maxBytes.');
  }
  if (!response.body?.getReader) {
    const raw = await response.text();
    if (new TextEncoder().encode(raw).byteLength > authority.maxBytes) throw new Error('Signed revocation response exceeds maxBytes.');
    return raw;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let size = 0;
  let raw = '';
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > authority.maxBytes) {
        await reader.cancel().catch(() => {});
        throw new Error('Signed revocation response exceeds maxBytes.');
      }
      raw += decoder.decode(value, { stream: true });
    }
    return raw + decoder.decode();
  } finally {
    reader.releaseLock();
  }
}

export async function configureSignedRevocationAuthority(options) {
  const configured = normalizeAuthority(options);
  if (authority && authority.authorityId !== configured.authorityId) throw new Error('A different revocation authority is already configured.');
  authority = configured;
  state = {
    epoch: authority.initialEpoch,
    sequence: 0,
    onlineKeys: authority.onlineKeys,
    revokedKeys: [],
    keyProof: null,
    dataEnvelope: null,
    refreshedAt: 0,
  };
  await restoreState(await authority.stateStore.load());
  snapshotStatus();
  return refreshSignedRevocations({ force: true });
}

export async function refreshSignedRevocations(options = {}) {
  if (!authority) throw new Error('Signed revocation authority is not configured.');
  exactOptions(options, [], ['force'], 'signed revocation refresh options');
  const force = options.force ?? false;
  if (typeof force !== 'boolean') throw new Error('signed revocation refresh options.force must be boolean.');
  const now = authority.now();
  if (!Number.isFinite(now)) throw new Error('Signed revocation authority clock returned a non-finite value.');
  if (!force && state.dataEnvelope && now < state.refreshedAt + authority.refreshIntervalMs && now < state.dataEnvelope.expires.time) return status;
  let response;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), authority.requestTimeoutMs);
  try {
    response = await authority.fetchFn(authority.url, {
      cache: 'no-store',
      redirect: 'error',
      headers: { accept: 'application/json' },
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timeout);
    return offlineStatus(error?.message || String(error));
  }
  if (!response?.ok) {
    clearTimeout(timeout);
    return offlineStatus(`HTTP ${response?.status ?? 'failure'}`);
  }
  let raw;
  try {
    raw = await readBoundedResponse(response);
  } catch (error) {
    if (controller.signal.aborted) return offlineStatus('request timed out');
    throw error;
  } finally {
    clearTimeout(timeout);
  }
  let value;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error('Signed revocation response is not valid JSON.');
  }
  const envelope = await verifyCandidate(value);
  if (envelope.replay) {
    state.refreshedAt = now;
    return snapshotStatus();
  }
  const keyProof = envelope.keyring ? value : state.keyProof;
  const persisted = { schema: STATE_SCHEMA, authorityId: authority.authorityId, keyProof, dataEnvelope: value };
  await authority.stateStore.save(persisted);
  applyEnvelope(envelope, keyProof);
  state.refreshedAt = now;
  return snapshotStatus();
}

export function getSignedRevocationStatus() {
  if (!authority) return status;
  return snapshotStatus({ offline: status.offline, lastError: status.lastError });
}
