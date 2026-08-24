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
  if (typeof value.signatureHex !== 'string' || !/^[0-9a-f]+$/.test(value.signatureHex)) {
    errors.push(`${label}.signatureHex must be hexadecimal.`);
  }
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
    'evaluator', 'status', 'observations', 'failedSamples', 'createdAtUtc', 'digest',
  ]), 'application gate receipt', errors);
  if (receipt.schema !== APPLICATION_GATE_RECEIPT_SCHEMA) {
    errors.push(`application gate receipt.schema must be "${APPLICATION_GATE_RECEIPT_SCHEMA}".`);
  }
  requireId(receipt.receiptId, 'application gate receipt.receiptId', errors);
  requireString(receipt.releaseId, 'application gate receipt.releaseId', errors);
  requireDigest(receipt.applicationRevisionDigest, 'application gate receipt.applicationRevisionDigest', errors);
  validateIdentity(receipt.workload, 'application gate receipt.workload', errors);
  validateIdentity(receipt.oracle, 'application gate receipt.oracle', errors);
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
    for (const field of ['startupPassed', 'recoveryPassed']) {
      if (typeof receipt.observations[field] !== 'boolean') {
        errors.push(`application gate receipt.observations.${field} must be boolean.`);
      }
    }
  }
  if (!Array.isArray(receipt.failedSamples)) errors.push('application gate receipt.failedSamples must be an array.');
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
    'os', 'osVersion', 'architecture', 'electronVersion', 'gpuVendor', 'gpuDevice', 'driverVersion',
  ]), 'fleet receipt.device', errors);
  if (device.os !== 'windows' && device.os !== 'macos') errors.push('fleet receipt.device.os is unsupported.');
  if (device.architecture !== 'x64' && device.architecture !== 'arm64') {
    errors.push('fleet receipt.device.architecture is unsupported.');
  }
  for (const field of ['osVersion', 'electronVersion', 'gpuVendor', 'gpuDevice', 'driverVersion']) {
    requireString(device[field], `fleet receipt.device.${field}`, errors);
  }
}

export function validateElectronFleetReceipt(receipt) {
  const errors = [];
  if (!requireObject(receipt, 'fleet receipt', errors)) return { ok: false, errors };
  requireExactKeys(receipt, new Set([
    'schema', 'receiptId', 'releaseId', 'targetId', 'packSemanticRoot',
    'applicationRevisionDigest', 'workload', 'oracle', 'device', 'applicationGateDigest',
    'status', 'createdAtUtc', 'digest', 'signature',
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
  if (!Array.isArray(decision.applicationGateReceipts)) errors.push('release decision.applicationGateReceipts must be an array.');
  if (!Array.isArray(decision.fleetReceipts)) errors.push('release decision.fleetReceipts must be an array.');
  if (!Array.isArray(decision.knownExclusions)) errors.push('release decision.knownExclusions must be an array.');
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
