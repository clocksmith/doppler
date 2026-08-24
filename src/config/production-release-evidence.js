import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';
import { hashPackV2PublicKey } from './pack-v2.js';

export const APPLICATION_GATE_RECEIPT_SCHEMA = 'doppler.application-gate-receipt/v1';
export const ELECTRON_FLEET_RECEIPT_SCHEMA = 'doppler.electron-fleet-receipt/v1';
export const RELEASE_DECISION_SCHEMA = 'doppler.release-decision/v1';
export const RELEASE_FAILURE_BUNDLE_SCHEMA = 'doppler.release-failure-bundle/v1';

const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;
const ID_PATTERN = /^[a-z][a-z0-9-]*$/;
const STATUS_SET = new Set(['passed', 'failed']);
const ELIGIBILITY_SET = new Set(['eligible', 'blocked']);
const REJECTION_CODE_SET = new Set([
  'acceptance-failed',
  'application-gate-failed',
  'artifact-invalid',
  'evidence-expired',
  'evidence-invalid',
  'migration-required',
  'revoked',
  'unsupported-device',
]);

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function canonicalJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function requireObject(value, label, errors) {
  if (!isObject(value)) {
    errors.push(`${label} must be an object.`);
    return false;
  }
  return true;
}

function requireExactKeys(value, allowed, label, errors) {
  if (!isObject(value)) return;
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) errors.push(`${label}.${key} is not allowed.`);
  }
}

function requireString(value, label, errors) {
  if (typeof value !== 'string' || !value.trim()) errors.push(`${label} must be a non-empty string.`);
}

function requireId(value, label, errors) {
  if (!ID_PATTERN.test(value || '')) errors.push(`${label} must be a kebab-case identifier.`);
}

function requireDigest(value, label, errors) {
  if (!DIGEST_PATTERN.test(value || '')) errors.push(`${label} must be a SHA-256 digest.`);
}

function requireInstant(value, label, errors) {
  requireString(value, label, errors);
  if (typeof value !== 'string') return;
  const time = new Date(value).getTime();
  if (!Number.isFinite(time) || new Date(time).toISOString() !== value) {
    errors.push(`${label} must be an ISO instant.`);
  }
}

function validateIdentity(value, label, errors) {
  if (!requireObject(value, label, errors)) return;
  requireExactKeys(value, new Set(['id', 'digest']), label, errors);
  requireId(value.id, `${label}.id`, errors);
  requireDigest(value.digest, `${label}.digest`, errors);
}

function validateSignature(value, label, expectedDigest, errors) {
  if (!requireObject(value, label, errors)) return;
  requireExactKeys(
    value,
    new Set(['authority', 'algorithm', 'publicKeyDigest', 'signedDigest', 'signatureHex']),
    label,
    errors
  );
  requireId(value.authority, `${label}.authority`, errors);
  if (value.algorithm !== 'Ed25519') errors.push(`${label}.algorithm must be "Ed25519".`);
  requireDigest(value.publicKeyDigest, `${label}.publicKeyDigest`, errors);
  requireDigest(value.signedDigest, `${label}.signedDigest`, errors);
  if (value.signedDigest !== expectedDigest) errors.push(`${label}.signedDigest must equal digest.`);
  if (typeof value.signatureHex !== 'string' || !/^[0-9a-f]{128}$/.test(value.signatureHex)) {
    errors.push(`${label}.signatureHex must be a 64-byte hexadecimal Ed25519 signature.`);
  }
}

function validateReceiptReference(value, label, errors) {
  if (!requireObject(value, label, errors)) return;
  requireExactKeys(value, new Set(['targetId', 'digest', 'status']), label, errors);
  requireId(value.targetId, `${label}.targetId`, errors);
  requireDigest(value.digest, `${label}.digest`, errors);
  if (!STATUS_SET.has(value.status)) errors.push(`${label}.status must be passed or failed.`);
}

function validateReleaseTarget(value, label, requireAuthority, errors) {
  if (!requireObject(value, label, errors)) return;
  const keys = new Set(['releaseId', 'packSemanticRoot']);
  if (requireAuthority) keys.add('authority');
  requireExactKeys(value, keys, label, errors);
  requireString(value.releaseId, `${label}.releaseId`, errors);
  requireDigest(value.packSemanticRoot, `${label}.packSemanticRoot`, errors);
  if (requireAuthority && value.authority !== 'customer') {
    errors.push(`${label}.authority must be customer.`);
  }
}

function validateRevocation(value, label, errors) {
  if (!requireObject(value, label, errors)) return;
  requireExactKeys(
    value,
    new Set(['authorityId', 'policyDigest', 'offlineExpirySeconds', 'failClosedAfterExpiry']),
    label,
    errors
  );
  requireId(value.authorityId, `${label}.authorityId`, errors);
  requireDigest(value.policyDigest, `${label}.policyDigest`, errors);
  if (!Number.isSafeInteger(value.offlineExpirySeconds) || value.offlineExpirySeconds < 1) {
    errors.push(`${label}.offlineExpirySeconds must be a positive integer.`);
  }
  if (value.failClosedAfterExpiry !== true) errors.push(`${label}.failClosedAfterExpiry must be true.`);
}

function validateDecisionReason(value, index, errors) {
  const label = `release decision.reasons[${index}]`;
  if (!requireObject(value, label, errors)) return;
  requireExactKeys(value, new Set(['code', 'scope', 'detail', 'evidenceDigests']), label, errors);
  if (!REJECTION_CODE_SET.has(value.code)) errors.push(`${label}.code is unsupported.`);
  requireString(value.scope, `${label}.scope`, errors);
  requireString(value.detail, `${label}.detail`, errors);
  if (!Array.isArray(value.evidenceDigests)) {
    errors.push(`${label}.evidenceDigests must be an array.`);
  } else {
    value.evidenceDigests.forEach((entry, digestIndex) => {
      requireDigest(entry, `${label}.evidenceDigests[${digestIndex}]`, errors);
    });
  }
}

function validateKnownExclusion(value, index, errors) {
  const label = `release decision.knownExclusions[${index}]`;
  if (!requireObject(value, label, errors)) return;
  requireExactKeys(value, new Set(['code', 'scope', 'reason', 'evidenceDigest']), label, errors);
  if (!REJECTION_CODE_SET.has(value.code)) errors.push(`${label}.code is unsupported.`);
  requireString(value.scope, `${label}.scope`, errors);
  requireString(value.reason, `${label}.reason`, errors);
  requireDigest(value.evidenceDigest, `${label}.evidenceDigest`, errors);
}

function evidenceSemanticPayload(value) {
  const { digest: _digest, signature: _signature, ...payload } = value;
  return payload;
}

export function hashProductionReleaseEvidence(value) {
  return `sha256:${sha256Hex(canonicalJson(evidenceSemanticPayload(value)))}`;
}

export function validateApplicationGateReceipt(receipt) {
  const errors = [];
  if (!requireObject(receipt, 'application gate receipt', errors)) return { ok: false, errors };
  requireExactKeys(receipt, new Set([
    'schema', 'receiptId', 'releaseId', 'applicationRevisionDigest', 'workload', 'oracle',
    'packSemanticRoot', 'targetPlanId', 'resolvedExecutionId', 'providerId',
    'deviceTargetId', 'evaluator', 'status', 'observations', 'failedSamples',
    'createdAtUtc', 'digest',
  ]), 'application gate receipt', errors);
  if (receipt.schema !== APPLICATION_GATE_RECEIPT_SCHEMA) {
    errors.push(`application gate receipt.schema must be "${APPLICATION_GATE_RECEIPT_SCHEMA}".`);
  }
  requireId(receipt.receiptId, 'application gate receipt.receiptId', errors);
  requireString(receipt.releaseId, 'application gate receipt.releaseId', errors);
  requireDigest(receipt.applicationRevisionDigest, 'application gate receipt.applicationRevisionDigest', errors);
  validateIdentity(receipt.workload, 'application gate receipt.workload', errors);
  validateIdentity(receipt.oracle, 'application gate receipt.oracle', errors);
  requireDigest(receipt.packSemanticRoot, 'application gate receipt.packSemanticRoot', errors);
  requireId(receipt.targetPlanId, 'application gate receipt.targetPlanId', errors);
  requireDigest(receipt.resolvedExecutionId, 'application gate receipt.resolvedExecutionId', errors);
  requireId(receipt.providerId, 'application gate receipt.providerId', errors);
  requireId(receipt.deviceTargetId, 'application gate receipt.deviceTargetId', errors);
  if (requireObject(receipt.evaluator, 'application gate receipt.evaluator', errors)) {
    requireExactKeys(receipt.evaluator, new Set(['id', 'revisionDigest']), 'application gate receipt.evaluator', errors);
    requireId(receipt.evaluator.id, 'application gate receipt.evaluator.id', errors);
    requireDigest(receipt.evaluator.revisionDigest, 'application gate receipt.evaluator.revisionDigest', errors);
  }
  if (!STATUS_SET.has(receipt.status)) errors.push('application gate receipt.status must be passed or failed.');
  if (requireObject(receipt.observations, 'application gate receipt.observations', errors)) {
    requireExactKeys(receipt.observations, new Set([
      'quality', 'coldLatencyMs', 'warmLatencyMs', 'peakMemoryBytes', 'failureRate',
      'startupPassed', 'recoveryPassed',
    ]), 'application gate receipt.observations', errors);
    for (const field of ['quality', 'coldLatencyMs', 'warmLatencyMs', 'peakMemoryBytes', 'failureRate']) {
      if (typeof receipt.observations[field] !== 'number' || !Number.isFinite(receipt.observations[field])) {
        errors.push(`application gate receipt.observations.${field} must be finite.`);
      }
    }
    for (const field of ['coldLatencyMs', 'warmLatencyMs', 'peakMemoryBytes', 'failureRate']) {
      if (typeof receipt.observations[field] === 'number' && receipt.observations[field] < 0) {
        errors.push(`application gate receipt.observations.${field} must be non-negative.`);
      }
    }
    if (typeof receipt.observations.failureRate === 'number' && receipt.observations.failureRate > 1) {
      errors.push('application gate receipt.observations.failureRate must not exceed 1.');
    }
    for (const field of ['startupPassed', 'recoveryPassed']) {
      if (typeof receipt.observations[field] !== 'boolean') {
        errors.push(`application gate receipt.observations.${field} must be boolean.`);
      }
    }
  }
  if (!Array.isArray(receipt.failedSamples)) errors.push('application gate receipt.failedSamples must be an array.');
  else receipt.failedSamples.forEach((sample, index) => {
    const label = `application gate receipt.failedSamples[${index}]`;
    if (!requireObject(sample, label, errors)) return;
    requireExactKeys(sample, new Set(['sampleId', 'reason', 'evidenceDigest']), label, errors);
    requireString(sample.sampleId, `${label}.sampleId`, errors);
    requireString(sample.reason, `${label}.reason`, errors);
    requireDigest(sample.evidenceDigest, `${label}.evidenceDigest`, errors);
  });
  requireInstant(receipt.createdAtUtc, 'application gate receipt.createdAtUtc', errors);
  requireDigest(receipt.digest, 'application gate receipt.digest', errors);
  if (DIGEST_PATTERN.test(receipt.digest || '') && receipt.digest !== hashProductionReleaseEvidence(receipt)) {
    errors.push('application gate receipt.digest does not match its semantic payload.');
  }
  return { ok: errors.length === 0, errors };
}

function validateDeviceIdentity(device, errors) {
  if (!requireObject(device, 'fleet receipt.device', errors)) return;
  requireExactKeys(device, new Set([
    'os', 'osVersion', 'architecture', 'electronVersion', 'gpuVendor', 'gpuDevice',
    'driverVersion', 'surface', 'hasF16', 'hasSubgroups', 'maxBufferSize',
  ]), 'fleet receipt.device', errors);
  if (device.os !== 'windows' && device.os !== 'macos') errors.push('fleet receipt.device.os is unsupported.');
  if (device.architecture !== 'x64' && device.architecture !== 'arm64') {
    errors.push('fleet receipt.device.architecture is unsupported.');
  }
  for (const field of [
    'osVersion', 'electronVersion', 'gpuVendor', 'gpuDevice', 'driverVersion', 'surface',
  ]) {
    requireString(device[field], `fleet receipt.device.${field}`, errors);
  }
  for (const field of ['hasF16', 'hasSubgroups']) {
    if (typeof device[field] !== 'boolean') {
      errors.push(`fleet receipt.device.${field} must be boolean.`);
    }
  }
  if (!Number.isSafeInteger(device.maxBufferSize) || device.maxBufferSize < 1) {
    errors.push('fleet receipt.device.maxBufferSize must be a positive safe integer.');
  }
}

export function validateElectronFleetReceipt(receipt) {
  const errors = [];
  if (!requireObject(receipt, 'fleet receipt', errors)) return { ok: false, errors };
  requireExactKeys(receipt, new Set([
    'schema', 'receiptId', 'releaseId', 'targetId', 'packSemanticRoot',
    'applicationRevisionDigest', 'workload', 'oracle', 'targetPlanId',
    'resolvedExecutionId', 'providerId', 'device', 'applicationGateDigest', 'status',
    'createdAtUtc', 'digest', 'signature',
  ]), 'fleet receipt', errors);
  if (receipt.schema !== ELECTRON_FLEET_RECEIPT_SCHEMA) {
    errors.push(`fleet receipt.schema must be "${ELECTRON_FLEET_RECEIPT_SCHEMA}".`);
  }
  requireId(receipt.receiptId, 'fleet receipt.receiptId', errors);
  requireString(receipt.releaseId, 'fleet receipt.releaseId', errors);
  requireId(receipt.targetId, 'fleet receipt.targetId', errors);
  requireDigest(receipt.packSemanticRoot, 'fleet receipt.packSemanticRoot', errors);
  requireDigest(receipt.applicationRevisionDigest, 'fleet receipt.applicationRevisionDigest', errors);
  validateIdentity(receipt.workload, 'fleet receipt.workload', errors);
  validateIdentity(receipt.oracle, 'fleet receipt.oracle', errors);
  if (receipt.status === 'passed') {
    requireId(receipt.targetPlanId, 'fleet receipt.targetPlanId', errors);
    requireDigest(receipt.resolvedExecutionId, 'fleet receipt.resolvedExecutionId', errors);
    requireId(receipt.providerId, 'fleet receipt.providerId', errors);
  } else {
    if (receipt.targetPlanId !== null) requireId(receipt.targetPlanId, 'fleet receipt.targetPlanId', errors);
    if (receipt.resolvedExecutionId !== null) requireDigest(receipt.resolvedExecutionId, 'fleet receipt.resolvedExecutionId', errors);
    if (receipt.providerId !== null) requireId(receipt.providerId, 'fleet receipt.providerId', errors);
  }
  validateDeviceIdentity(receipt.device, errors);
  requireDigest(receipt.applicationGateDigest, 'fleet receipt.applicationGateDigest', errors);
  if (!STATUS_SET.has(receipt.status)) errors.push('fleet receipt.status must be passed or failed.');
  requireInstant(receipt.createdAtUtc, 'fleet receipt.createdAtUtc', errors);
  requireDigest(receipt.digest, 'fleet receipt.digest', errors);
  if (DIGEST_PATTERN.test(receipt.digest || '') && receipt.digest !== hashProductionReleaseEvidence(receipt)) {
    errors.push('fleet receipt.digest does not match its semantic payload.');
  }
  validateSignature(receipt.signature, 'fleet receipt.signature', receipt.digest, errors);
  return { ok: errors.length === 0, errors };
}

export function validateReleaseDecision(decision) {
  const errors = [];
  if (!requireObject(decision, 'release decision', errors)) return { ok: false, errors };
  requireExactKeys(decision, new Set([
    'schema', 'releaseId', 'productionReleaseDigest', 'pack', 'eligibility', 'reasons',
    'applicationGateReceipts', 'fleetReceipts', 'knownExclusions', 'previousRelease',
    'rollback', 'revocation', 'activationAuthority', 'selfPromotionAllowed', 'createdAtUtc',
    'digest', 'signature',
  ]), 'release decision', errors);
  if (decision.schema !== RELEASE_DECISION_SCHEMA) {
    errors.push(`release decision.schema must be "${RELEASE_DECISION_SCHEMA}".`);
  }
  requireString(decision.releaseId, 'release decision.releaseId', errors);
  requireDigest(decision.productionReleaseDigest, 'release decision.productionReleaseDigest', errors);
  if (requireObject(decision.pack, 'release decision.pack', errors)) {
    requireExactKeys(decision.pack, new Set(['packId', 'semanticRoot', 'envelopeDigest', 'path']), 'release decision.pack', errors);
    requireString(decision.pack.packId, 'release decision.pack.packId', errors);
    requireDigest(decision.pack.semanticRoot, 'release decision.pack.semanticRoot', errors);
    requireDigest(decision.pack.envelopeDigest, 'release decision.pack.envelopeDigest', errors);
    requireString(decision.pack.path, 'release decision.pack.path', errors);
  }
  if (!ELIGIBILITY_SET.has(decision.eligibility)) errors.push('release decision.eligibility must be eligible or blocked.');
  if (!Array.isArray(decision.reasons)) errors.push('release decision.reasons must be an array.');
  else decision.reasons.forEach((reason, index) => validateDecisionReason(reason, index, errors));
  if (!Array.isArray(decision.applicationGateReceipts)) {
    errors.push('release decision.applicationGateReceipts must be an array.');
  } else {
    decision.applicationGateReceipts.forEach((receipt, index) => (
      validateReceiptReference(receipt, `release decision.applicationGateReceipts[${index}]`, errors)
    ));
  }
  if (!Array.isArray(decision.fleetReceipts)) {
    errors.push('release decision.fleetReceipts must be an array.');
  } else {
    decision.fleetReceipts.forEach((receipt, index) => (
      validateReceiptReference(receipt, `release decision.fleetReceipts[${index}]`, errors)
    ));
  }
  if (!Array.isArray(decision.knownExclusions)) errors.push('release decision.knownExclusions must be an array.');
  else decision.knownExclusions.forEach((exclusion, index) => validateKnownExclusion(exclusion, index, errors));
  validateReleaseTarget(decision.previousRelease, 'release decision.previousRelease', false, errors);
  validateReleaseTarget(decision.rollback, 'release decision.rollback', true, errors);
  validateRevocation(decision.revocation, 'release decision.revocation', errors);
  if (isObject(decision.previousRelease) && isObject(decision.rollback)
    && (decision.previousRelease.releaseId !== decision.rollback.releaseId
      || decision.previousRelease.packSemanticRoot !== decision.rollback.packSemanticRoot)) {
    errors.push('release decision.rollback must bind previousRelease.');
  }
  if (decision.eligibility === 'eligible' && Array.isArray(decision.reasons) && decision.reasons.length > 0) {
    errors.push('eligible release decisions must not contain rejection reasons.');
  }
  if (decision.eligibility === 'blocked' && Array.isArray(decision.reasons) && decision.reasons.length === 0) {
    errors.push('blocked release decisions must contain at least one rejection reason.');
  }
  if (decision.activationAuthority !== 'customer') errors.push('release decision.activationAuthority must be customer.');
  if (decision.selfPromotionAllowed !== false) errors.push('release decision.selfPromotionAllowed must be false.');
  requireInstant(decision.createdAtUtc, 'release decision.createdAtUtc', errors);
  requireDigest(decision.digest, 'release decision.digest', errors);
  if (DIGEST_PATTERN.test(decision.digest || '') && decision.digest !== hashProductionReleaseEvidence(decision)) {
    errors.push('release decision.digest does not match its semantic payload.');
  }
  validateSignature(decision.signature, 'release decision.signature', decision.digest, errors);
  return { ok: errors.length === 0, errors };
}

function hexToBytes(value) {
  return Uint8Array.from(value.match(/.{2}/g) || [], (entry) => Number.parseInt(entry, 16));
}

function bytesToHex(value) {
  return Array.from(value, (entry) => entry.toString(16).padStart(2, '0')).join('');
}

export async function signProductionReleaseEvidence(value, signer) {
  if (!isObject(signer?.privateKeyJwk) || !isObject(signer?.publicKeyJwk)) {
    throw new Error('Production release evidence signing requires privateKeyJwk and publicKeyJwk.');
  }
  if (!ID_PATTERN.test(signer.authority || '')) {
    throw new Error('Production release evidence signing requires a kebab-case authority.');
  }
  const unsigned = { ...value, digest: '', signature: null };
  const digest = hashProductionReleaseEvidence(unsigned);
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('Production release evidence signing requires WebCrypto.');
  const key = await subtle.importKey('jwk', signer.privateKeyJwk, { name: 'Ed25519' }, false, ['sign']);
  const signatureBytes = new Uint8Array(
    await subtle.sign('Ed25519', key, new TextEncoder().encode(digest))
  );
  return {
    ...value,
    digest,
    signature: {
      authority: signer.authority,
      algorithm: 'Ed25519',
      publicKeyDigest: hashPackV2PublicKey(signer.publicKeyJwk),
      signedDigest: digest,
      signatureHex: bytesToHex(signatureBytes),
    },
  };
}

export async function verifyProductionReleaseEvidenceSignature(value, trustedSigners) {
  const signature = value?.signature;
  const publicKeyJwk = trustedSigners instanceof Map
    ? trustedSigners.get(signature?.authority)
    : trustedSigners?.[signature?.authority];
  if (!publicKeyJwk) throw new Error(`Untrusted release evidence authority "${signature?.authority}".`);
  if (hashPackV2PublicKey(publicKeyJwk) !== signature.publicKeyDigest) {
    throw new Error('Release evidence public key digest mismatch.');
  }
  if (hashProductionReleaseEvidence(value) !== value.digest || signature.signedDigest !== value.digest) {
    throw new Error('Release evidence digest mismatch.');
  }
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('Production release evidence verification requires WebCrypto.');
  const key = await subtle.importKey('jwk', publicKeyJwk, { name: 'Ed25519' }, false, ['verify']);
  const valid = await subtle.verify(
    'Ed25519',
    key,
    hexToBytes(signature.signatureHex),
    new TextEncoder().encode(value.digest)
  );
  if (!valid) throw new Error('Release evidence signature mismatch.');
  return true;
}
