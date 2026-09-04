import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { freezePackV2, hashPackV2Envelope, hashPackV2PublicKey } from './pack-v2.js';
import { validatePackReleaseContract } from './pack-release-contract.js';
import { signPackDigest, validatePackSignature, verifyPackDigest } from './pack-signature.js';

export const PACK_RELEASE_EVENT_SCHEMA = 'doppler.pack-release-event/v1';
const ACTIONS = ['eligible', 'blocked', 'promoted', 'quarantined', 'revoked', 'superseded', 'rollback-authorized'];
const EXECUTABLE_ACTIONS = new Set(['eligible', 'promoted', 'rollback-authorized']);
const DIGEST = /^sha256:[0-9a-f]{64}$/;
const FIELDS = ['schema', 'pack', 'sequence', 'previousEventDigest', 'issuedAtUtc', 'expiresAtUtc', 'action', 'release', 'migratedFrom', 'nextSigner', 'digest', 'signature'];

function eventPayload(event) {
  return Object.fromEntries(FIELDS.filter((key) => key !== 'digest' && key !== 'signature').map((key) => [key, event[key]]));
}

export function hashPackReleaseEvent(event) {
  return computeCanonicalSha256(eventPayload(event));
}

function validatePackReference(reference, label, errors) {
  if (!reference || typeof reference !== 'object' || Array.isArray(reference)) {
    errors.push(`${label} must be a Pack reference.`);
    return;
  }
  if (Object.keys(reference).some((key) => !['schema', 'semanticRoot', 'envelopeDigest'].includes(key))) errors.push(`Unknown ${label} field.`);
  if (!['doppler.pack/v2', 'doppler.pack/v3'].includes(reference.schema)) errors.push(`Invalid ${label}.schema.`);
  for (const field of ['semanticRoot', 'envelopeDigest']) if (!DIGEST.test(reference[field])) errors.push(`Invalid ${label}.${field}.`);
}

export function validatePackReleaseEvent(event, { requireSignature = true } = {}) {
  if (!event || typeof event !== 'object' || Array.isArray(event)) return { ok: false, errors: ['Release event must be an object.'] };
  const errors = [];
  for (const key of Object.keys(event)) if (!FIELDS.includes(key)) errors.push(`Unknown release event field: ${key}.`);
  for (const key of FIELDS) if (event[key] === undefined) errors.push(`Release event requires ${key}.`);
  if (event.schema !== PACK_RELEASE_EVENT_SCHEMA) errors.push('Invalid release event schema.');
  validatePackReference(event.pack, 'pack', errors);
  if (!Number.isSafeInteger(event.sequence) || event.sequence < 1) errors.push('Release sequence must be a positive safe integer.');
  if (event.sequence === 1 ? event.previousEventDigest !== null : !DIGEST.test(event.previousEventDigest)) errors.push('Invalid previous release event digest.');
  if (!ACTIONS.includes(event.action)) errors.push('Unsupported release action.');
  for (const field of ['issuedAtUtc', 'expiresAtUtc']) {
    const time = Date.parse(event[field]);
    if (!Number.isFinite(time) || new Date(time).toISOString() !== event[field]) errors.push(`Invalid release ${field}.`);
  }
  if (!(Date.parse(event.expiresAtUtc) > Date.parse(event.issuedAtUtc))) errors.push('Release expiry must follow issuance.');
  errors.push(...validatePackReleaseContract(event.release, {
    targetIds: event.release?.stateSnapshot?.portableAcrossTargetIds ?? [],
  }).errors);
  if (event.migratedFrom !== null) validatePackReference(event.migratedFrom, 'migratedFrom', errors);
  if (event.nextSigner !== null) {
    const key = event.nextSigner;
    if (!key || key.kty !== 'OKP' || key.crv !== 'Ed25519' || typeof key.x !== 'string'
      || Object.keys(key).some((field) => !['kty', 'crv', 'x'].includes(field))) errors.push('Key rotation requires a public Ed25519 JWK.');
  }
  const digest = hashPackReleaseEvent(event);
  if (event.digest !== digest) errors.push('Release event digest mismatch.');
  if (requireSignature || event.signature !== null) errors.push(...validatePackSignature(event.signature, digest));
  return { ok: errors.length === 0, errors };
}

export async function signPackReleaseEvent(params, signer) {
  const draft = structuredClone({ ...params, schema: PACK_RELEASE_EVENT_SCHEMA, digest: null, signature: null });
  draft.digest = hashPackReleaseEvent(draft);
  const validation = validatePackReleaseEvent(draft, { requireSignature: false });
  if (!validation.ok) throw new Error(`Invalid release event: ${validation.errors.join('; ')}`);
  const snapshot = freezePackV2(draft);
  return freezePackV2({ ...snapshot, signature: await signPackDigest(snapshot.digest, signer) });
}

export async function verifyPackReleaseEvents(events, { pack, trustedSigners, policy }) {
  if (!Array.isArray(events) || events.length === 0) throw new Error('Pack v3 execution requires its signed release event history.');
  const checkpoint = policy?.checkpoint;
  if (!checkpoint || !Number.isSafeInteger(checkpoint.sequence) || checkpoint.sequence < 0
    || (checkpoint.sequence === 0 ? checkpoint.digest !== null : !DIGEST.test(checkpoint.digest))) {
    throw new Error('Release policy requires an explicit persisted sequence/digest checkpoint.');
  }
  if (!Number.isSafeInteger(policy.minimumSequence) || policy.minimumSequence < checkpoint.sequence) throw new Error('Invalid minimum release sequence.');
  const now = Date.parse(policy.now);
  if (!Number.isFinite(now) || new Date(now).toISOString() !== policy.now) throw new Error('Release policy requires an explicit ISO verification time.');
  const history = freezePackV2(structuredClone(events));
  const authority = history[0]?.signature?.authority;
  let key = trustedSigners instanceof Map ? trustedSigners.get(authority) : trustedSigners?.[authority];
  let previousDigest = null;
  let issuedAt = -Infinity;
  const revokedRoots = new Set();
  for (const [index, event] of history.entries()) {
    const validation = validatePackReleaseEvent(event);
    if (!validation.ok) throw new Error(`Invalid release event: ${validation.errors.join('; ')}`);
    if (event.sequence !== index + 1 || event.previousEventDigest !== previousDigest) throw new Error('Release event history has a gap, replay, or fork.');
    if (event.signature.authority !== authority) throw new Error('Release authority changed without authorization.');
    if (Date.parse(event.issuedAtUtc) < issuedAt || Date.parse(event.issuedAtUtc) > now) throw new Error('Release issuance chronology is invalid.');
    await verifyPackDigest(event.signature, event.digest, key);
    if (event.sequence === checkpoint.sequence && event.digest !== checkpoint.digest) throw new Error('Release history conflicts with the persisted checkpoint.');
    if (event.action === 'revoked') revokedRoots.add(event.pack.semanticRoot);
    if (event.nextSigner !== null) key = event.nextSigner;
    previousDigest = event.digest;
    issuedAt = Date.parse(event.issuedAtUtc);
  }
  const event = history.at(-1);
  if (event.sequence < policy.minimumSequence || event.sequence < checkpoint.sequence) throw new Error('Release history rolled back below the required sequence.');
  if (Date.parse(event.expiresAtUtc) <= now) throw new Error('Release eligibility has expired.');
  if (event.pack.schema !== pack.schema || event.pack.semanticRoot !== pack.semanticRoot
    || event.pack.envelopeDigest !== hashPackV2Envelope(pack)) throw new Error('Release event does not bind this exact Pack envelope.');
  if (!EXECUTABLE_ACTIONS.has(event.action) || revokedRoots.has(pack.semanticRoot)) throw new Error(`Pack execution is blocked by release state: ${event.action}.`);
  const releaseValidation = validatePackReleaseContract(event.release, { targetIds: pack.targetPlans.map((plan) => plan.targetId) });
  if (!releaseValidation.ok) throw new Error(releaseValidation.errors.join('; '));
  return freezePackV2({
    release: event.release,
    event,
    checkpoint: { sequence: event.sequence, digest: event.digest },
    nextPublicKeyDigest: hashPackV2PublicKey(key),
  });
}
