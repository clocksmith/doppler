#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { computeCanonicalJsonSha256 } from './lib/canonical-json.js';
import {
  REVOCATION_AUTHORITY_EVIDENCE_CLASSES,
  validateRevocationAuthorityEvidence,
  validateRevocationAuthorityOwnerConfirmation,
} from './lib/signed-revocation-authority-evidence.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(
  REPO_ROOT,
  'tools',
  'policies',
  'signed-revocation-authority-qualification.json'
);
const REQUIRED_HOSTS = Object.freeze(['browser', 'node']);
const REQUIRED_DRILLS = Object.freeze([
  'refresh-current',
  'online-key-rotation',
  'exact-replay',
  'rewritten-replay-rejection',
  'sequence-rollback-rejection',
  'epoch-rollback-rejection',
  'offline-expiry',
  'compromise-recovery',
  'durable-store-restart',
  'loaded-identity-invalidation',
  'application-fail-closed',
]);
const EVIDENCE_FIELDS = Object.freeze([
  'ownerConfirmation',
  'endpointDeployment',
  'packageTrustBinding',
  'onlineKeyCustody',
  'recoveryKeyCustody',
  'custodySeparation',
  'browserDurableState',
  'nodeDurableState',
  'refreshCurrent',
  'onlineKeyRotation',
  'exactReplay',
  'rewrittenReplayRejection',
  'sequenceRollbackRejection',
  'epochRollbackRejection',
  'offlineExpiry',
  'compromiseRecovery',
  'durableStoreRestart',
  'loadedIdentityInvalidation',
  'applicationFailClosed',
  'requalification',
]);
const ROOT_FIELDS = Object.freeze([
  '$schema',
  'schemaVersion',
  'source',
  'goalId',
  'minimumQualifiedAuthorities',
  'ownerConfirmationMaxAgeDays',
  'requiredHosts',
  'requiredDrills',
  'authorities',
]);
const AUTHORITY_FIELDS = Object.freeze([
  'id',
  'lifecycle',
  'owner',
  'ownerConfirmedAtUtc',
  'claimAllowed',
  'deployment',
  'qualifiedAtUtc',
  'expiresAtUtc',
  'evidence',
  'blockers',
]);
const DEPLOYMENT_FIELDS = Object.freeze([
  'endpointUrl',
  'authorityId',
  'transportPolicy',
  'onlineKeyIds',
  'recoveryKeyIds',
  'durableStateStoreIds',
]);
const LIFECYCLES = new Set(['candidate', 'active', 'quarantined', 'revoked', 'retired']);
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function validateExactKeys(value, fields, label, errors) {
  if (!isPlainObject(value)) {
    errors.push(`${label} must be an object`);
    return false;
  }
  const expected = new Set(fields);
  for (const field of Object.keys(value)) {
    if (!expected.has(field)) errors.push(`${label}.${field} is not supported`);
  }
  for (const field of fields) {
    if (!Object.prototype.hasOwnProperty.call(value, field)) {
      errors.push(`${label}.${field} is required`);
    }
  }
  return true;
}

function validateNullableText(value, label, errors) {
  if (value === null) return null;
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string or null`);
  return normalized || null;
}

function validateStringArray(value, label, errors) {
  if (!Array.isArray(value)) {
    errors.push(`${label} must be an array`);
    return [];
  }
  const values = [];
  const seen = new Set();
  for (const entry of value) {
    const normalized = normalizeText(entry);
    if (!normalized) {
      errors.push(`${label} entries must be non-empty strings`);
    } else if (seen.has(normalized)) {
      errors.push(`${label} contains duplicate entry ${normalized}`);
    } else {
      values.push(normalized);
      seen.add(normalized);
    }
  }
  return values;
}

function sameSequence(left, right) {
  return Array.isArray(left)
    && left.length === right.length
    && left.every((entry, index) => entry === right[index]);
}

function parseIsoInstant(value, label, nullable, errors) {
  if (value === null && nullable) return null;
  const normalized = normalizeText(value);
  const instant = normalized ? new Date(normalized) : null;
  if (!instant || !Number.isFinite(instant.getTime()) || instant.toISOString() !== normalized) {
    errors.push(`${label} must be an ISO instant${nullable ? ' or null' : ''}`);
    return null;
  }
  return instant;
}

function confirmationIsCurrent(confirmedAt, now, maxAgeDays) {
  if (!confirmedAt) return false;
  const age = now.getTime() - confirmedAt.getTime();
  return age >= 0 && age <= maxAgeDays * 24 * 60 * 60 * 1000;
}

function isQualifiedEndpoint(value) {
  if (!value) return false;
  try {
    const url = new URL(value);
    return url.protocol === 'https:'
      && Boolean(url.hostname)
      && !url.username
      && !url.password
      && !url.hash;
  } catch {
    return false;
  }
}

function isRepoRelativePath(value) {
  const normalized = normalizeText(value);
  return Boolean(
    normalized
    && !path.isAbsolute(normalized)
    && !normalized.includes('\\')
    && !normalized.split('/').includes('..')
  );
}

async function validateEvidenceReference(value, label, repoRoot, errors) {
  if (value === null) return null;
  if (!validateExactKeys(value, ['path', 'digest'], label, errors)) return null;
  if (!isRepoRelativePath(value.path)) {
    errors.push(`${label}.path must be a repo-relative path`);
    return null;
  }
  const normalized = normalizeText(value.path);
  const digest = normalizeText(value.digest).toLowerCase();
  if (!/^sha256:[0-9a-f]{64}$/.test(digest)) {
    errors.push(`${label}.digest must be a SHA-256 identity`);
  }
  let receipt = null;
  try {
    receipt = JSON.parse(await fs.readFile(path.join(repoRoot, normalized), 'utf8'));
  } catch (error) {
    errors.push(`${label}.path is not readable JSON: ${normalized}: ${error.message}`);
  }
  if (receipt && /^sha256:[0-9a-f]{64}$/.test(digest)) {
    if (computeCanonicalJsonSha256(receipt) !== digest) {
      errors.push(`${label}.digest does not match canonical JSON evidence`);
    }
  }
  return { path: normalized, digest, receipt };
}

async function validateAuthority(authority, context) {
  const { errors, repoRoot, now, maxAgeDays, seenIds, seenAuthorityIds } = context;
  const id = normalizeText(authority?.id) || '<missing-id>';
  if (!validateExactKeys(authority, AUTHORITY_FIELDS, id, errors)) return null;
  if (!ID_PATTERN.test(id)) errors.push(`${id}: id must be lowercase kebab-case`);
  if (seenIds.has(id)) errors.push(`${id}: duplicate qualification id`);
  seenIds.add(id);
  if (!LIFECYCLES.has(authority.lifecycle)) errors.push(`${id}: lifecycle is not recognized`);
  if (typeof authority.claimAllowed !== 'boolean') errors.push(`${id}: claimAllowed must be boolean`);
  const owner = validateNullableText(authority.owner, `${id}: owner`, errors);
  const ownerConfirmedAt = parseIsoInstant(
    authority.ownerConfirmedAtUtc,
    `${id}: ownerConfirmedAtUtc`,
    true,
    errors
  );
  const qualifiedAt = parseIsoInstant(
    authority.qualifiedAtUtc,
    `${id}: qualifiedAtUtc`,
    true,
    errors
  );
  const expiresAt = parseIsoInstant(
    authority.expiresAtUtc,
    `${id}: expiresAtUtc`,
    true,
    errors
  );
  const blockers = validateStringArray(authority.blockers, `${id}: blockers`, errors);

  const deployment = authority.deployment;
  let endpointUrl = null;
  let authorityId = null;
  let onlineKeyIds = [];
  let recoveryKeyIds = [];
  const durableStateStoreIds = {};
  if (validateExactKeys(deployment, DEPLOYMENT_FIELDS, `${id}: deployment`, errors)) {
    endpointUrl = validateNullableText(deployment.endpointUrl, `${id}: deployment.endpointUrl`, errors);
    authorityId = validateNullableText(deployment.authorityId, `${id}: deployment.authorityId`, errors);
    if (authorityId) {
      if (seenAuthorityIds.has(authorityId)) errors.push(`${id}: duplicate deployment authorityId ${authorityId}`);
      seenAuthorityIds.add(authorityId);
    }
    if (deployment.transportPolicy !== 'https-no-redirect') {
      errors.push(`${id}: deployment.transportPolicy must be https-no-redirect`);
    }
    onlineKeyIds = validateStringArray(deployment.onlineKeyIds, `${id}: deployment.onlineKeyIds`, errors);
    recoveryKeyIds = validateStringArray(deployment.recoveryKeyIds, `${id}: deployment.recoveryKeyIds`, errors);
    for (const keyId of onlineKeyIds) {
      if (recoveryKeyIds.includes(keyId)) errors.push(`${id}: online and recovery key IDs must be disjoint`);
    }
    if (validateExactKeys(
      deployment.durableStateStoreIds,
      REQUIRED_HOSTS,
      `${id}: deployment.durableStateStoreIds`,
      errors
    )) {
      for (const host of REQUIRED_HOSTS) {
        durableStateStoreIds[host] = validateNullableText(
          deployment.durableStateStoreIds[host],
          `${id}: deployment.durableStateStoreIds.${host}`,
          errors
        );
      }
    }
  }

  const evidence = {};
  if (validateExactKeys(authority.evidence, EVIDENCE_FIELDS, `${id}: evidence`, errors)) {
    await Promise.all(EVIDENCE_FIELDS.map(async (field) => {
      evidence[field] = await validateEvidenceReference(
        authority.evidence[field],
        `${id}: evidence.${field}`,
        repoRoot,
        errors
      );
    }));
  }
  const evidencePaths = EVIDENCE_FIELDS
    .map((field) => evidence[field]?.path)
    .filter(Boolean);
  if (new Set(evidencePaths).size !== evidencePaths.length) {
    errors.push(`${id}: evidence references must use distinct paths`);
  }
  const missingEvidence = EVIDENCE_FIELDS.filter((field) => !evidence[field]?.receipt);
  const qualificationReasons = [];
  if (authority.lifecycle !== 'active') qualificationReasons.push('lifecycle-not-active');
  if (!owner) qualificationReasons.push('owner-missing');
  if (!confirmationIsCurrent(ownerConfirmedAt, now, maxAgeDays)) {
    qualificationReasons.push('owner-confirmation-stale-or-missing');
  }
  if (!isQualifiedEndpoint(endpointUrl)) qualificationReasons.push('https-endpoint-missing-or-invalid');
  if (!authorityId) qualificationReasons.push('authority-id-missing');
  if (onlineKeyIds.length === 0) qualificationReasons.push('online-keys-missing');
  if (recoveryKeyIds.length === 0) qualificationReasons.push('recovery-keys-missing');
  if (REQUIRED_HOSTS.some((host) => !durableStateStoreIds[host])) {
    qualificationReasons.push('durable-state-store-identity-incomplete');
  }
  if (!qualifiedAt) qualificationReasons.push('qualification-date-missing');
  if (qualifiedAt && qualifiedAt.getTime() > now.getTime()) {
    qualificationReasons.push('qualification-date-in-future');
  }
  if (!expiresAt || expiresAt.getTime() <= now.getTime()) {
    qualificationReasons.push('qualification-expired-or-missing');
  }
  if (qualifiedAt && expiresAt && expiresAt.getTime() <= qualifiedAt.getTime()) {
    qualificationReasons.push('qualification-expiry-not-after-qualification');
  }
  if (missingEvidence.length > 0) qualificationReasons.push('evidence-incomplete');
  const evidenceContext = {
    qualificationId: id,
    owner,
    ownerConfirmedAtUtc: ownerConfirmedAt?.toISOString() ?? null,
    authorityId,
    endpointUrl,
    onlineKeyIds,
    recoveryKeyIds,
    durableStateStoreIds,
    requiredDrillCount: REQUIRED_DRILLS.length,
  };
  if (evidence.ownerConfirmation?.receipt) {
    const result = validateRevocationAuthorityOwnerConfirmation(
      evidence.ownerConfirmation.receipt,
      evidenceContext
    );
    for (const error of result.errors) errors.push(`${id}: evidence.ownerConfirmation: ${error}`);
    if (result.errors.length > 0) qualificationReasons.push('owner-confirmation-invalid');
    for (const reason of result.reasons) {
      if (!qualificationReasons.includes(reason)) qualificationReasons.push(reason);
    }
    if (qualifiedAt && result.confirmedAt && result.confirmedAt.getTime() > qualifiedAt.getTime()) {
      qualificationReasons.push('owner-confirmation-postdates-qualification');
    }
  }
  const evidenceResults = {};
  for (const field of REVOCATION_AUTHORITY_EVIDENCE_CLASSES) {
    if (!evidence[field]?.receipt) continue;
    const result = validateRevocationAuthorityEvidence(evidence[field].receipt, {
      ...evidenceContext,
      evidenceClass: field,
    });
    evidenceResults[field] = result;
    for (const error of result.errors) errors.push(`${id}: evidence.${field}: ${error}`);
    if (result.errors.length > 0) qualificationReasons.push(`${field}-invalid`);
    for (const reason of result.reasons) {
      if (!qualificationReasons.includes(reason)) qualificationReasons.push(reason);
    }
    if (qualifiedAt && result.capturedAt && result.capturedAt.getTime() > qualifiedAt.getTime()) {
      qualificationReasons.push(`${field}-postdates-qualification`);
    }
  }
  const onlineCustodyDomain = evidenceResults.onlineKeyCustody
    ?.observations?.custodyDomainId;
  const recoveryCustodyDomain = evidenceResults.recoveryKeyCustody
    ?.observations?.custodyDomainId;
  const separation = evidenceResults.custodySeparation?.observations;
  if (
    separation
    && (
      separation.onlineCustodyDomainId !== onlineCustodyDomain
      || separation.recoveryCustodyDomainId !== recoveryCustodyDomain
    )
  ) {
    errors.push(`${id}: custody separation domains do not match custody receipts`);
    qualificationReasons.push('custody-separation-identity-mismatch');
  }
  if (blockers.length > 0) qualificationReasons.push('blockers-present');
  const satisfiesQualification = qualificationReasons.length === 0;
  if (authority.claimAllowed === true && !satisfiesQualification) {
    errors.push(`${id}: claimAllowed authority does not satisfy production qualification`);
  }
  if (authority.claimAllowed === false && blockers.length === 0) {
    errors.push(`${id}: non-claimable authority must list blockers`);
  }
  return {
    id,
    lifecycle: authority.lifecycle,
    owner,
    claimAllowed: authority.claimAllowed,
    qualified: satisfiesQualification && authority.claimAllowed === true,
    endpointConfigured: isQualifiedEndpoint(endpointUrl),
    onlineKeys: onlineKeyIds.length,
    recoveryKeys: recoveryKeyIds.length,
    qualifiedHosts: REQUIRED_HOSTS.filter((host) => Boolean(durableStateStoreIds[host])),
    missingEvidence,
    qualificationReasons,
    blockers,
  };
}

export async function validateSignedRevocationAuthorityQualification(policy, options = {}) {
  const repoRoot = options.repoRoot || REPO_ROOT;
  const now = options.now || new Date();
  const errors = [];
  if (!validateExactKeys(policy, ROOT_FIELDS, 'policy', errors)) {
    return {
      ok: false,
      errors,
      authorities: [],
      qualifiedAuthorities: 0,
      candidateAuthorities: 0,
      gateSatisfied: false,
    };
  }
  if (policy.$schema !== '../../src/config/schema/signed-revocation-authority-qualification.schema.json') {
    errors.push('policy.$schema is not supported');
  }
  if (policy.schemaVersion !== 2) errors.push('policy.schemaVersion must be 2');
  if (policy.source !== 'doppler') errors.push('policy.source must be doppler');
  if (policy.goalId !== 'evidence-backed-correctness-performance') {
    errors.push('policy.goalId is not supported');
  }
  if (policy.minimumQualifiedAuthorities !== 1) {
    errors.push('policy.minimumQualifiedAuthorities must be 1');
  }
  if (!Number.isInteger(policy.ownerConfirmationMaxAgeDays)
    || policy.ownerConfirmationMaxAgeDays < 1
    || policy.ownerConfirmationMaxAgeDays > 365) {
    errors.push('policy.ownerConfirmationMaxAgeDays must be an integer from 1 through 365');
  }
  if (!sameSequence(policy.requiredHosts, REQUIRED_HOSTS)) {
    errors.push(`policy.requiredHosts must be ${REQUIRED_HOSTS.join(', ')}`);
  }
  if (!sameSequence(policy.requiredDrills, REQUIRED_DRILLS)) {
    errors.push('policy.requiredDrills must match the production drill contract');
  }
  if (!Array.isArray(policy.authorities)) {
    errors.push('policy.authorities must be an array');
  }
  const authorities = [];
  const context = {
    errors,
    repoRoot,
    now,
    maxAgeDays: policy.ownerConfirmationMaxAgeDays,
    seenIds: new Set(),
    seenAuthorityIds: new Set(),
  };
  for (const authority of policy.authorities || []) {
    const result = await validateAuthority(authority, context);
    if (result) authorities.push(result);
  }
  const qualifiedAuthorities = authorities.filter((authority) => authority.qualified).length;
  const candidateAuthorities = authorities.filter((authority) => (
    authority.lifecycle === 'candidate' && authority.claimAllowed === false
  )).length;
  return {
    ok: errors.length === 0,
    errors,
    authorities,
    qualifiedAuthorities,
    candidateAuthorities,
    gateSatisfied: errors.length === 0
      && qualifiedAuthorities >= policy.minimumQualifiedAuthorities,
    requiredHosts: [...REQUIRED_HOSTS],
    requiredDrills: [...REQUIRED_DRILLS],
  };
}

export async function buildSignedRevocationAuthorityQualificationReport(options = {}) {
  const policyPath = options.policyPath || DEFAULT_POLICY_PATH;
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  return validateSignedRevocationAuthorityQualification(policy, options);
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((token) => token !== '--json');
  if (unsupported.length > 0) throw new Error(`Unknown argument: ${unsupported[0]}`);
  const report = await buildSignedRevocationAuthorityQualificationReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    console.log(
      `signed-revocation-authority: contract ok (${report.qualifiedAuthorities} qualified, ${report.candidateAuthorities} candidates)`
    );
  } else {
    for (const error of report.errors) console.error(`signed-revocation-authority: ${error}`);
  }
  if (!report.ok) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
