import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { freezeCapsuleV2, hashCapsuleV2Envelope, hashCapsuleV2PublicKey } from './capsule-v2.js';
import { validateCapsuleReleaseContract } from './capsule-release-contract.js';
import { signCapsuleDigest, validateCapsuleSignature, verifyCapsuleDigest } from './capsule-signature.js';

export const CAPSULE_RELEASE_EVENT_SCHEMA = 'doppler.capsule-release-event/v1';
const ACTIONS = ['eligible', 'blocked', 'promoted', 'quarantined', 'revoked', 'superseded', 'rollback-authorized'];
const EXECUTABLE_ACTIONS = new Set(['eligible', 'promoted', 'rollback-authorized']);
const DIGEST = /^sha256:[0-9a-f]{64}$/;
const FIELDS = ['schema', 'capsule', 'sequence', 'previousEventDigest', 'issuedAtUtc', 'expiresAtUtc', 'action', 'release', 'migratedFrom', 'nextSigner', 'digest', 'signature'];

// Only a fully authenticated, contiguous history may advance durable state on denial.
export class CapsuleReleaseStateError extends Error {
  constructor(cause, checkpoint) {
    super(cause.message, { cause });
    this.name = 'CapsuleReleaseStateError';
    this.checkpoint = freezeCapsuleV2({ ...checkpoint });
  }
}

function resolveReleaseAuthorization(policy, event, now) {
  const expired = Date.parse(event.expiresAtUtc) <= now;
  const decision = policy.retainedLocalUse;
  if (decision === undefined) {
    if (expired) throw new Error('Release eligibility has expired.');
    return { mode: 'managed', verifiedAtUtc: policy.now, eventExpired: false, unseenRevocations: 'unknown', retainedLocalUse: null };
  }
  const fields = ['schema', 'capsule', 'releaseEventDigest', 'applicationDigest', 'acceptedAtUtc', 'acknowledgeUnseenRevocations'];
  if (!decision || typeof decision !== 'object' || Array.isArray(decision)
    || Object.keys(decision).some(key => !fields.includes(key))
    || fields.some(key => decision[key] === undefined)
    || decision.schema !== 'doppler.capsule-retained-local-use/v1') {
    throw new Error('Invalid retainedLocalUse decision schema or fields.');
  }
  const errors = [];
  validateCapsuleReference(decision.capsule, 'retainedLocalUse.capsule', errors);
  if (errors.length || computeCanonicalSha256(decision.capsule) !== computeCanonicalSha256(event.capsule)) {
    throw new Error('Retained local use must bind this exact Capsule envelope.');
  }
  if (decision.releaseEventDigest !== event.digest || policy.checkpoint.sequence < event.sequence) {
    throw new Error('Retained local use requires the exact previously persisted release event.');
  }
  if (decision.applicationDigest !== computeCanonicalSha256(event.release.application)) {
    throw new Error('Retained local use application identity differs from the accepted release.');
  }
  const acceptedAt = Date.parse(decision.acceptedAtUtc);
  if (!Number.isFinite(acceptedAt) || new Date(acceptedAt).toISOString() !== decision.acceptedAtUtc
    || acceptedAt < Date.parse(event.issuedAtUtc) || acceptedAt >= Date.parse(event.expiresAtUtc) || acceptedAt > now) {
    throw new Error('Retained local use requires an application acceptance time within the release eligibility window.');
  }
  if (decision.acknowledgeUnseenRevocations !== true) {
    throw new Error('Retained local use must acknowledge that unseen revocations are unknown.');
  }
  return { mode: 'retained-local', verifiedAtUtc: policy.now, eventExpired: expired,
    unseenRevocations: 'unknown', retainedLocalUse: decision };
}

function eventPayload(event) {
  return Object.fromEntries(FIELDS.filter((key) => key !== 'digest' && key !== 'signature').map((key) => [key, event[key]]));
}

export function hashCapsuleReleaseEvent(event) {
  return computeCanonicalSha256(eventPayload(event));
}

function validateCapsuleReference(reference, label, errors) {
  if (!reference || typeof reference !== 'object' || Array.isArray(reference)) {
    errors.push(`${label} must be a Capsule reference.`);
    return;
  }
  if (Object.keys(reference).some((key) => !['schema', 'semanticRoot', 'envelopeDigest'].includes(key))) errors.push(`Unknown ${label} field.`);
  if (!['doppler.capsule/v2', 'doppler.capsule/v3'].includes(reference.schema)) errors.push(`Invalid ${label}.schema.`);
  for (const field of ['semanticRoot', 'envelopeDigest']) if (!DIGEST.test(reference[field])) errors.push(`Invalid ${label}.${field}.`);
}

export function validateCapsuleReleaseEvent(event, { requireSignature = true } = {}) {
  if (!event || typeof event !== 'object' || Array.isArray(event)) return { ok: false, errors: ['Release event must be an object.'] };
  const errors = [];
  for (const key of Object.keys(event)) if (!FIELDS.includes(key)) errors.push(`Unknown release event field: ${key}.`);
  for (const key of FIELDS) if (event[key] === undefined) errors.push(`Release event requires ${key}.`);
  if (event.schema !== CAPSULE_RELEASE_EVENT_SCHEMA) errors.push('Invalid release event schema.');
  validateCapsuleReference(event.capsule, 'capsule', errors);
  if (!Number.isSafeInteger(event.sequence) || event.sequence < 1) errors.push('Release sequence must be a positive safe integer.');
  if (event.sequence === 1 ? event.previousEventDigest !== null : !DIGEST.test(event.previousEventDigest)) errors.push('Invalid previous release event digest.');
  if (!ACTIONS.includes(event.action)) errors.push('Unsupported release action.');
  for (const field of ['issuedAtUtc', 'expiresAtUtc']) {
    const time = Date.parse(event[field]);
    if (!Number.isFinite(time) || new Date(time).toISOString() !== event[field]) errors.push(`Invalid release ${field}.`);
  }
  if (!(Date.parse(event.expiresAtUtc) > Date.parse(event.issuedAtUtc))) errors.push('Release expiry must follow issuance.');
  errors.push(...validateCapsuleReleaseContract(event.release, {
    targetIds: event.release?.stateSnapshot?.portableAcrossTargetIds ?? [],
  }).errors);
  if (event.migratedFrom !== null) validateCapsuleReference(event.migratedFrom, 'migratedFrom', errors);
  if (event.nextSigner !== null) {
    const key = event.nextSigner;
    if (!key || key.kty !== 'OKP' || key.crv !== 'Ed25519' || typeof key.x !== 'string'
      || Object.keys(key).some((field) => !['kty', 'crv', 'x'].includes(field))) errors.push('Key rotation requires a public Ed25519 JWK.');
  }
  const digest = hashCapsuleReleaseEvent(event);
  if (event.digest !== digest) errors.push('Release event digest mismatch.');
  if (requireSignature || event.signature !== null) errors.push(...validateCapsuleSignature(event.signature, digest));
  return { ok: errors.length === 0, errors };
}

export async function signCapsuleReleaseEvent(params, signer) {
  const draft = structuredClone({ ...params, schema: CAPSULE_RELEASE_EVENT_SCHEMA, digest: null, signature: null });
  draft.digest = hashCapsuleReleaseEvent(draft);
  const validation = validateCapsuleReleaseEvent(draft, { requireSignature: false });
  if (!validation.ok) throw new Error(`Invalid release event: ${validation.errors.join('; ')}`);
  const snapshot = freezeCapsuleV2(draft);
  return freezeCapsuleV2({ ...snapshot, signature: await signCapsuleDigest(snapshot.digest, signer) });
}

export async function verifyCapsuleReleaseEvents(events, { capsule, trustedSigners, policy }) {
  if (!Array.isArray(events) || events.length === 0) throw new Error('Capsule v3 execution requires its signed release event history.');
  if (!policy || typeof policy !== 'object' || Array.isArray(policy)
    || Object.keys(policy).some(key => !['now', 'minimumSequence', 'checkpoint', 'retainedLocalUse'].includes(key))) {
    throw new Error('Invalid release policy or unknown policy field.');
  }
  policy = freezeCapsuleV2(structuredClone(policy));
  const checkpoint = policy?.checkpoint;
  if (!checkpoint || Object.keys(checkpoint).some(key => !['sequence', 'digest'].includes(key))
    || !Number.isSafeInteger(checkpoint.sequence) || checkpoint.sequence < 0
    || (checkpoint.sequence === 0 ? checkpoint.digest !== null : !DIGEST.test(checkpoint.digest))) {
    throw new Error('Release policy requires an explicit persisted sequence/digest checkpoint.');
  }
  if (!Number.isSafeInteger(policy.minimumSequence) || policy.minimumSequence < checkpoint.sequence) throw new Error('Invalid minimum release sequence.');
  const now = Date.parse(policy.now);
  if (!Number.isFinite(now) || new Date(now).toISOString() !== policy.now) throw new Error('Release policy requires an explicit ISO verification time.');
  const history = freezeCapsuleV2(structuredClone(events));
  const authority = history[0]?.signature?.authority;
  let key = trustedSigners instanceof Map ? trustedSigners.get(authority) : trustedSigners?.[authority];
  let previousDigest = null;
  let issuedAt = -Infinity;
  const revokedRoots = new Set();
  for (const [index, event] of history.entries()) {
    const validation = validateCapsuleReleaseEvent(event);
    if (!validation.ok) throw new Error(`Invalid release event: ${validation.errors.join('; ')}`);
    if (event.sequence !== index + 1 || event.previousEventDigest !== previousDigest) throw new Error('Release event history has a gap, replay, or fork.');
    if (event.signature.authority !== authority) throw new Error('Release authority changed without authorization.');
    if (Date.parse(event.issuedAtUtc) < issuedAt || Date.parse(event.issuedAtUtc) > now) throw new Error('Release issuance chronology is invalid.');
    await verifyCapsuleDigest(event.signature, event.digest, key);
    if (event.sequence === checkpoint.sequence && event.digest !== checkpoint.digest) throw new Error('Release history conflicts with the persisted checkpoint.');
    if (event.action === 'revoked') revokedRoots.add(event.capsule.semanticRoot);
    if (event.nextSigner !== null) key = event.nextSigner;
    previousDigest = event.digest;
    issuedAt = Date.parse(event.issuedAtUtc);
  }
  const event = history.at(-1);
  if (event.sequence < checkpoint.sequence) throw new Error('Release history rolled back below the required sequence.');
  const verifiedCheckpoint = { sequence: event.sequence, digest: event.digest };
  let authorization;
  try {
    if (event.sequence < policy.minimumSequence) throw new Error('Release history rolled back below the required sequence.');
    if (event.capsule.schema !== capsule.schema || event.capsule.semanticRoot !== capsule.semanticRoot
      || event.capsule.envelopeDigest !== hashCapsuleV2Envelope(capsule)) throw new Error('Release event does not bind this exact Capsule envelope.');
    if (!EXECUTABLE_ACTIONS.has(event.action) || revokedRoots.has(capsule.semanticRoot)) throw new Error(`Capsule execution is blocked by release state: ${event.action}.`);
    const releaseValidation = validateCapsuleReleaseContract(event.release, { targetIds: capsule.targetPlans.map((plan) => plan.targetId) });
    if (!releaseValidation.ok) throw new Error(releaseValidation.errors.join('; '));
    authorization = resolveReleaseAuthorization(policy, event, now);
  } catch (error) {
    throw new CapsuleReleaseStateError(error, verifiedCheckpoint);
  }
  return freezeCapsuleV2({
    release: event.release,
    event,
    checkpoint: verifiedCheckpoint,
    authorization,
    nextPublicKeyDigest: hashCapsuleV2PublicKey(key),
  });
}
