#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { isPlainObject } from '../src/formats/plain-object.js';

import {
  validateSignedRevocationAuthorityQualification,
} from './check-signed-revocation-authority-qualification.js';
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
const CAPTURE_SCHEMA = 'doppler.signed-revocation-authority-evaluation-capture/v1';
const CAPTURE_EVIDENCE_FIELDS = Object.freeze([
  'ownerConfirmation',
  ...REVOCATION_AUTHORITY_EVIDENCE_CLASSES,
]);
const REVIEW_REASONS = new Set([
  'lifecycle-not-active',
  'blockers-present',
  'evidence-incomplete',
]);
const DAY_MS = 24 * 60 * 60 * 1000;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function assertExactKeys(value, fields, label) {
  if (!isPlainObject(value)) throw new Error(`${label} must be an object.`);
  const expected = new Set(fields);
  const unsupported = Object.keys(value).filter((field) => !expected.has(field));
  if (unsupported.length > 0) throw new Error(`${label}.${unsupported[0]} is not supported.`);
  const missing = fields.filter((field) => !Object.hasOwn(value, field));
  if (missing.length > 0) throw new Error(`${label}.${missing[0]} is required.`);
}

function requireText(value, label) {
  const normalized = normalizeText(value);
  if (!normalized) throw new Error(`${label} must be a non-empty string.`);
  return normalized;
}

function parseInstant(value, label) {
  const normalized = requireText(value, label);
  const instant = new Date(normalized);
  if (!Number.isFinite(instant.getTime()) || instant.toISOString() !== normalized) {
    throw new Error(`${label} must be an ISO instant.`);
  }
  return instant;
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

async function readJson(filePath, label) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (error) {
    throw new Error(`${label} is not readable JSON: ${error.message}`);
  }
}

async function readEvidence(value, label, repoRoot) {
  if (!isRepoRelativePath(value)) throw new Error(`${label} must be a repo-relative path.`);
  const evidencePath = normalizeText(value);
  const receipt = await readJson(path.join(repoRoot, evidencePath), label);
  return {
    path: evidencePath,
    digest: computeCanonicalJsonSha256(receipt),
    receipt,
  };
}

function hasPriorEvaluation(authority) {
  return Boolean(
    authority.ownerConfirmedAtUtc
    || authority.deployment.endpointUrl
    || authority.deployment.authorityId
    || authority.deployment.onlineKeyIds.length > 0
    || authority.deployment.recoveryKeyIds.length > 0
    || Object.values(authority.deployment.durableStateStoreIds).some(Boolean)
    || authority.qualifiedAtUtc
    || authority.expiresAtUtc
    || Object.values(authority.evidence).some((value) => value !== null)
  );
}

async function validateCapture(capture, context) {
  const { authority, repoRoot, now, ownerMaxAgeDays, evidenceMaxAgeDays } = context;
  assertExactKeys(capture, [
    'schema', 'qualificationId', 'evaluatedAtUtc', 'expiresAtUtc', 'evidencePaths',
  ], 'authority evaluation capture');
  if (capture.schema !== CAPTURE_SCHEMA) throw new Error(`capture.schema must be ${CAPTURE_SCHEMA}.`);
  if (capture.qualificationId !== authority.id) {
    throw new Error(`capture.qualificationId must be ${authority.id}.`);
  }
  const evaluatedAt = parseInstant(capture.evaluatedAtUtc, 'capture.evaluatedAtUtc');
  const expiresAt = parseInstant(capture.expiresAtUtc, 'capture.expiresAtUtc');
  if (evaluatedAt.getTime() > now.getTime()) {
    throw new Error('capture.evaluatedAtUtc must not be in the future.');
  }
  if (expiresAt.getTime() <= evaluatedAt.getTime()) {
    throw new Error('capture.expiresAtUtc must be later than capture.evaluatedAtUtc.');
  }
  if (expiresAt.getTime() > evaluatedAt.getTime() + evidenceMaxAgeDays * DAY_MS) {
    throw new Error(`capture.expiresAtUtc exceeds the ${evidenceMaxAgeDays}-day evidence limit.`);
  }
  assertExactKeys(
    capture.evidencePaths,
    CAPTURE_EVIDENCE_FIELDS,
    'capture.evidencePaths'
  );
  const evidence = {};
  for (const field of CAPTURE_EVIDENCE_FIELDS) {
    evidence[field] = await readEvidence(
      capture.evidencePaths[field],
      `capture.evidencePaths.${field}`,
      repoRoot
    );
  }
  if (
    new Set(Object.values(evidence).map((entry) => entry.path)).size
      !== CAPTURE_EVIDENCE_FIELDS.length
  ) {
    throw new Error('capture.evidencePaths must use distinct paths.');
  }
  const owner = validateRevocationAuthorityOwnerConfirmation(
    evidence.ownerConfirmation.receipt,
    { qualificationId: authority.id, owner: authority.owner }
  );
  if (owner.errors.length > 0) {
    throw new Error(`Owner confirmation is invalid: ${owner.errors.join('; ')}`);
  }
  if (
    owner.confirmedAt
    && expiresAt.getTime() > owner.confirmedAt.getTime() + ownerMaxAgeDays * DAY_MS
  ) {
    throw new Error(`capture.expiresAtUtc exceeds the ${ownerMaxAgeDays}-day owner limit.`);
  }
  const endpoint = evidence.endpointDeployment.receipt?.observations ?? {};
  const trust = evidence.packageTrustBinding.receipt?.observations ?? {};
  const browserStore = evidence.browserDurableState.receipt?.observations ?? {};
  const nodeStore = evidence.nodeDurableState.receipt?.observations ?? {};
  const evidenceContext = {
    qualificationId: authority.id,
    owner: authority.owner,
    authorityId: endpoint.authorityId,
    endpointUrl: endpoint.endpointUrl,
    onlineKeyIds: trust.onlineKeyIds,
    recoveryKeyIds: trust.recoveryKeyIds,
    durableStateStoreIds: {
      browser: browserStore.storeId,
      node: nodeStore.storeId,
    },
    requiredDrillCount: 11,
  };
  const results = {};
  let authorityEvidenceIdentity = null;
  for (const evidenceClass of REVOCATION_AUTHORITY_EVIDENCE_CLASSES) {
    const result = validateRevocationAuthorityEvidence(evidence[evidenceClass].receipt, {
      ...evidenceContext,
      ...authorityEvidenceIdentity,
      evidenceClass,
    });
    if (result.errors.length > 0) {
      throw new Error(`${evidenceClass} evidence is invalid: ${result.errors.join('; ')}`);
    }
    results[evidenceClass] = result;
    if (!authorityEvidenceIdentity) {
      authorityEvidenceIdentity = {
        harnessRevision: result.harnessRevision,
        environmentFingerprint: result.environmentFingerprint,
      };
    }
  }
  const separation = results.custodySeparation.observations;
  if (
    separation.onlineCustodyDomainId
      !== results.onlineKeyCustody.observations.custodyDomainId
    || separation.recoveryCustodyDomainId
      !== results.recoveryKeyCustody.observations.custodyDomainId
  ) {
    throw new Error('Custody separation domains do not match custody receipts.');
  }
  const evidenceInstants = [owner.confirmedAt]
    .concat(Object.values(results).map((result) => result.capturedAt))
    .filter(Boolean);
  if (evidenceInstants.some((instant) => instant.getTime() > evaluatedAt.getTime())) {
    throw new Error('capture.evaluatedAtUtc must not predate retained evidence.');
  }
  if (
    Object.values(results).some((result) => (
      result.capturedAt
      && evaluatedAt.getTime() - result.capturedAt.getTime() > evidenceMaxAgeDays * DAY_MS
    ))
  ) {
    throw new Error(`Retained authority evidence exceeds the ${evidenceMaxAgeDays}-day age limit.`);
  }
  return {
    evaluatedAtUtc: evaluatedAt.toISOString(),
    expiresAtUtc: expiresAt.toISOString(),
    evidence,
    evidenceContext,
    authorityEvidenceIdentity,
    owner,
  };
}

async function writeJsonAtomically(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  try {
    await fs.writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
    await fs.rename(temporaryPath, filePath);
  } finally {
    await fs.rm(temporaryPath, { force: true });
  }
}

export async function recordSignedRevocationAuthorityEvaluation(options) {
  const repoRoot = path.resolve(options.repoRoot || REPO_ROOT);
  const policyPath = path.resolve(options.policyPath || DEFAULT_POLICY_PATH);
  const capturePath = path.resolve(requireText(options.capturePath, 'capturePath'));
  const outputPolicyPath = path.resolve(requireText(options.outputPolicyPath, 'outputPolicyPath'));
  const now = options.now instanceof Date ? options.now : new Date();
  const policy = await readJson(policyPath, 'signed revocation authority policy');
  const current = await validateSignedRevocationAuthorityQualification(policy, { repoRoot, now });
  if (!current.ok) throw new Error(`Authority policy is invalid: ${current.errors[0]}`);
  const capture = await readJson(capturePath, 'authority evaluation capture');
  const authority = policy.authorities.find((entry) => entry.id === capture.qualificationId);
  if (!authority) throw new Error(`Unknown authority qualification: ${capture.qualificationId}.`);
  if (authority.claimAllowed) throw new Error(`Recorder cannot replace claimable authority ${authority.id}.`);
  if (authority.evidence.promotion !== null) {
    throw new Error(`Recorder cannot replace promoted authority ${authority.id}.`);
  }
  if (hasPriorEvaluation(authority) && options.replace !== true) {
    throw new Error(`Authority ${authority.id} already contains evaluation state; use --replace.`);
  }
  const validated = await validateCapture(capture, {
    authority,
    repoRoot,
    now,
    ownerMaxAgeDays: policy.ownerConfirmationMaxAgeDays,
    evidenceMaxAgeDays: policy.evidenceMaxAgeDays,
  });
  const outputPolicy = structuredClone(policy);
  const outputAuthority = outputPolicy.authorities.find((entry) => entry.id === authority.id);
  outputAuthority.ownerConfirmedAtUtc = validated.owner.confirmedAt?.toISOString() ?? null;
  outputAuthority.deployment = {
    endpointUrl: validated.evidenceContext.endpointUrl,
    authorityId: validated.evidenceContext.authorityId,
    transportPolicy: 'https-no-redirect',
    onlineKeyIds: validated.evidenceContext.onlineKeyIds,
    recoveryKeyIds: validated.evidenceContext.recoveryKeyIds,
    durableStateStoreIds: validated.evidenceContext.durableStateStoreIds,
  };
  outputAuthority.harnessRevision = validated.authorityEvidenceIdentity?.harnessRevision ?? null;
  outputAuthority.environmentFingerprint = validated.authorityEvidenceIdentity
    ?.environmentFingerprint ?? null;
  outputAuthority.qualifiedAtUtc = validated.evaluatedAtUtc;
  outputAuthority.expiresAtUtc = validated.expiresAtUtc;
  outputAuthority.lifecycle = 'candidate';
  outputAuthority.claimAllowed = false;
  outputAuthority.evidence = Object.fromEntries(CAPTURE_EVIDENCE_FIELDS.map((field) => [
    field,
    {
      path: validated.evidence[field].path,
      digest: validated.evidence[field].digest,
    },
  ]));
  outputAuthority.evidence.promotion = null;
  outputAuthority.blockers = [
    'production-authority-evaluation-awaiting-explicit-promotion',
    'production-authority-promotion-evidence-missing',
  ];
  let outputReport = await validateSignedRevocationAuthorityQualification(
    outputPolicy,
    { repoRoot, now }
  );
  if (!outputReport.ok) throw new Error(`Recorded authority policy is invalid: ${outputReport.errors[0]}`);
  const authorityReport = outputReport.authorities.find((entry) => entry.id === authority.id);
  outputAuthority.blockers = Array.from(new Set([
    'production-authority-evaluation-awaiting-explicit-promotion',
    'production-authority-promotion-evidence-missing',
    ...authorityReport.qualificationReasons.filter((reason) => !REVIEW_REASONS.has(reason)),
  ]));
  outputReport = await validateSignedRevocationAuthorityQualification(outputPolicy, { repoRoot, now });
  if (!outputReport.ok) throw new Error(`Recorded authority policy is invalid: ${outputReport.errors[0]}`);
  await writeJsonAtomically(outputPolicyPath, outputPolicy);
  return {
    qualificationId: authority.id,
    outputPolicyPath,
    lifecycle: outputAuthority.lifecycle,
    authorityId: outputAuthority.deployment.authorityId,
    endpointUrl: outputAuthority.deployment.endpointUrl,
    harnessRevision: outputAuthority.harnessRevision,
    environmentFingerprint: outputAuthority.environmentFingerprint,
    claimAllowed: false,
    blockers: outputAuthority.blockers,
  };
}

export function parseArgs(argv) {
  const options = {
    policyPath: DEFAULT_POLICY_PATH,
    capturePath: '',
    outputPolicyPath: '',
    replace: false,
    apply: false,
    json: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--policy') options.policyPath = argv[++index] || '';
    else if (token === '--capture') options.capturePath = argv[++index] || '';
    else if (token === '--out') options.outputPolicyPath = argv[++index] || '';
    else if (token === '--replace') options.replace = true;
    else if (token === '--apply') options.apply = true;
    else if (token === '--json') options.json = true;
    else throw new Error(`Unknown argument: ${token}`);
  }
  if (!options.capturePath) throw new Error('--capture is required.');
  if (options.apply && options.outputPolicyPath) throw new Error('--apply and --out are mutually exclusive.');
  if (options.apply) options.outputPolicyPath = options.policyPath;
  if (!options.outputPolicyPath) throw new Error('--out is required unless --apply is used.');
  if (!options.apply && path.resolve(options.outputPolicyPath) === path.resolve(options.policyPath)) {
    throw new Error('Writing the source policy requires explicit --apply.');
  }
  return options;
}

export async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  const result = await recordSignedRevocationAuthorityEvaluation(options);
  if (options.json) console.log(JSON.stringify(result, null, 2));
  else {
    console.log(
      `revocation-authority-record: captured ${result.qualificationId}; `
      + `lifecycle=${result.lifecycle}; claimAllowed=false; `
      + `output=${path.relative(REPO_ROOT, result.outputPolicyPath)}`
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
