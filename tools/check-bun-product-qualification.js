#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { isPlainObject } from '../src/formats/plain-object.js';

import { computeCanonicalJsonSha256 } from './lib/canonical-json.js';
import {
  BUN_QUALIFICATION_EVIDENCE_CLASSES,
  computeBunQualificationEvidenceSetDigest,
  validateBunProductPromotionEvidence,
  validateBunProductQualificationEvidence,
} from './lib/bun-product-qualification-evidence.js';
import {
  validateDopplerRuntimeOwnershipReceipt,
} from './lib/runtime-ownership-execution-evidence.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools/policies/bun-product-qualification.json');
const DEFAULT_SUBSYSTEMS_PATH = path.join(REPO_ROOT, 'src/config/support-tiers/subsystems.json');
const DEFAULT_RELEASE_REGISTRY_PATH = path.join(REPO_ROOT, 'benchmarks/vendors/registry.json');
const DEFAULT_RELEASE_MATRIX_PATH = path.join(REPO_ROOT, 'benchmarks/vendors/release-matrix.json');
const REQUIRED_WORKLOADS = Object.freeze(['generation', 'embedding', 'reranking']);
const EVIDENCE_FIELDS = Object.freeze([
  'execution',
  ...BUN_QUALIFICATION_EVIDENCE_CLASSES,
  'promotion',
]);
const CORRECTNESS_CLASSES = new Set([
  'exact-token',
  'tolerance-bounded-numerical',
  'semantic',
  'held-out-task-metric',
]);
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const DAY_MS = 24 * 60 * 60 * 1000;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function exactKeys(value, fields, label, errors) {
  if (!isPlainObject(value)) {
    errors.push(`${label} must be an object`);
    return false;
  }
  const expected = new Set(fields);
  for (const field of Object.keys(value)) {
    if (!expected.has(field)) errors.push(`${label}.${field} is not supported`);
  }
  for (const field of fields) {
    if (!Object.hasOwn(value, field)) errors.push(`${label}.${field} is required`);
  }
  return true;
}

function text(value, label, errors, nullable = false) {
  if (nullable && value === null) return null;
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string${nullable ? ' or null' : ''}`);
  return normalized || null;
}

function sha256(value, label, errors, nullable = false) {
  const normalized = text(value, label, errors, nullable)?.toLowerCase() ?? null;
  if (normalized && !SHA256_PATTERN.test(normalized)) {
    errors.push(`${label} must be a SHA-256 identity${nullable ? ' or null' : ''}`);
  }
  return normalized;
}

function instant(value, label, errors, nullable = false) {
  if (nullable && value === null) return null;
  const normalized = text(value, label, errors, nullable);
  const parsed = normalized ? new Date(normalized) : null;
  if (!parsed || !Number.isFinite(parsed.getTime()) || parsed.toISOString() !== normalized) {
    errors.push(`${label} must be an ISO instant${nullable ? ' or null' : ''}`);
    return null;
  }
  return parsed;
}

function stringArray(value, label, errors) {
  if (!Array.isArray(value)) {
    errors.push(`${label} must be an array`);
    return [];
  }
  const values = value.map((entry, index) => text(entry, `${label}[${index}]`, errors)).filter(Boolean);
  if (new Set(values).size !== values.length) errors.push(`${label} contains duplicate entries`);
  return values;
}

function sameMembers(left, right) {
  const a = [...left].sort();
  const b = [...right].sort();
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function repoPath(value) {
  const normalized = normalizeText(value);
  return Boolean(
    normalized
    && !path.isAbsolute(normalized)
    && !normalized.includes('\\')
    && !normalized.split('/').includes('..')
  );
}

async function evidenceReference(value, label, repoRoot, errors) {
  if (value === null) return null;
  if (!exactKeys(value, ['path', 'digest'], label, errors)) return null;
  if (!repoPath(value.path)) {
    errors.push(`${label}.path must be repo-relative`);
    return null;
  }
  const evidencePath = normalizeText(value.path);
  const digest = sha256(value.digest, `${label}.digest`, errors);
  let receipt;
  try {
    receipt = JSON.parse(await fs.readFile(path.join(repoRoot, evidencePath), 'utf8'));
  } catch (error) {
    errors.push(`${label}.path is not readable JSON: ${error.message}`);
    return null;
  }
  if (digest && computeCanonicalJsonSha256(receipt) !== digest) {
    errors.push(`${label}.digest does not match canonical JSON evidence`);
  }
  return { path: evidencePath, digest, receipt };
}

function pushUnique(target, values) {
  for (const value of values) {
    if (!target.includes(value)) target.push(value);
  }
}

async function validateQualification(qualification, context) {
  const { errors, repoRoot, now, maxAgeDays, seenIds } = context;
  const initialErrorCount = errors.length;
  const fields = [
    'id',
    'workload',
    'logicalModelId',
    'manifestVariantId',
    'correctnessClass',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'bunVersion',
    'webgpuImplementationId',
    'providerId',
    'qualifiedAtUtc',
    'expiresAtUtc',
    'evidence',
    'claimAllowed',
    'blockers',
  ];
  const id = normalizeText(qualification?.id) || '<missing-qualification>';
  if (!exactKeys(qualification, fields, id, errors)) return null;
  const reasons = [];
  if (!ID_PATTERN.test(id)) errors.push(`${id}: id must be lowercase kebab-case`);
  if (seenIds.has(id)) errors.push(`${id}: duplicate qualification id`);
  seenIds.add(id);
  if (!REQUIRED_WORKLOADS.includes(qualification.workload)) {
    errors.push(`${id}: workload is not recognized`);
  }
  const identity = {
    qualificationId: id,
    workload: qualification.workload,
    logicalModelId: text(qualification.logicalModelId, `${id}.logicalModelId`, errors),
    manifestVariantId: text(qualification.manifestVariantId, `${id}.manifestVariantId`, errors),
    resolvedArtifactVariantId: sha256(
      qualification.resolvedArtifactVariantId,
      `${id}.resolvedArtifactVariantId`,
      errors,
      true
    ),
    resolvedExecutionId: sha256(
      qualification.resolvedExecutionId,
      `${id}.resolvedExecutionId`,
      errors,
      true
    ),
    bunVersion: text(qualification.bunVersion, `${id}.bunVersion`, errors, true),
    webgpuImplementationId: text(
      qualification.webgpuImplementationId,
      `${id}.webgpuImplementationId`,
      errors,
      true
    ),
    providerId: text(qualification.providerId, `${id}.providerId`, errors, true),
  };
  if (!CORRECTNESS_CLASSES.has(qualification.correctnessClass)) {
    errors.push(`${id}.correctnessClass is not recognized`);
  }
  const qualifiedAt = instant(qualification.qualifiedAtUtc, `${id}.qualifiedAtUtc`, errors, true);
  const expiresAt = instant(qualification.expiresAtUtc, `${id}.expiresAtUtc`, errors, true);
  const blockers = stringArray(qualification.blockers, `${id}.blockers`, errors);
  if (typeof qualification.claimAllowed !== 'boolean') errors.push(`${id}.claimAllowed must be boolean`);
  if (qualification.claimAllowed && blockers.length > 0) {
    errors.push(`${id}: claimable qualification must not list blockers`);
  }
  if (!qualification.claimAllowed && blockers.length === 0) {
    errors.push(`${id}: non-claimable qualification must list blockers`);
  }
  const evidence = {};
  const retainedPaths = [];
  if (exactKeys(qualification.evidence, EVIDENCE_FIELDS, `${id}.evidence`, errors)) {
    await Promise.all(EVIDENCE_FIELDS.map(async (field) => {
      evidence[field] = await evidenceReference(
        qualification.evidence[field],
        `${id}.evidence.${field}`,
        repoRoot,
        errors
      );
      if (evidence[field]) retainedPaths.push(evidence[field].path);
    }));
  }
  const missingEvidence = EVIDENCE_FIELDS.filter((field) => !evidence[field]);
  if (missingEvidence.length > 0) reasons.push('bun-evidence-incomplete');
  if (!identity.resolvedArtifactVariantId || !identity.resolvedExecutionId) {
    reasons.push('bun-resolution-identity-incomplete');
  }
  if (!identity.bunVersion || !identity.webgpuImplementationId || !identity.providerId) {
    reasons.push('bun-host-identity-incomplete');
  }
  if (!qualifiedAt) {
    reasons.push('bun-qualification-date-missing');
  } else {
    const age = now.getTime() - qualifiedAt.getTime();
    if (age < 0 || age > maxAgeDays * DAY_MS) reasons.push('bun-qualification-stale-or-future');
  }
  if (!expiresAt || expiresAt.getTime() <= now.getTime()) {
    reasons.push('bun-qualification-expired-or-missing');
  }
  if (qualifiedAt && expiresAt && expiresAt.getTime() > qualifiedAt.getTime() + maxAgeDays * DAY_MS) {
    errors.push(`${id}: expiresAtUtc exceeds the qualification age limit`);
  }
  let execution = null;
  if (evidence.execution?.receipt) {
    execution = validateDopplerRuntimeOwnershipReceipt(evidence.execution.receipt, {
      logicalModelId: identity.logicalModelId,
      resolvedArtifactVariantId: identity.resolvedArtifactVariantId,
      resolvedExecutionId: identity.resolvedExecutionId,
    });
    for (const error of execution.errors) errors.push(`${id}.evidence.execution: ${error}`);
    pushUnique(reasons, execution.reasons);
    if (qualifiedAt && execution.timestamp?.getTime() > qualifiedAt.getTime()) {
      reasons.push('bun-qualification-predates-execution');
    }
  }
  const firstSemanticReceipt = BUN_QUALIFICATION_EVIDENCE_CLASSES
    .map((field) => evidence[field]?.receipt)
    .find(Boolean);
  const semanticContext = {
    ...identity,
    correctnessClass: qualification.correctnessClass,
    harnessRevision: firstSemanticReceipt?.harnessRevision ?? null,
    environmentFingerprint: firstSemanticReceipt?.environmentFingerprint ?? null,
  };
  for (const evidenceClass of BUN_QUALIFICATION_EVIDENCE_CLASSES) {
    const reference = evidence[evidenceClass];
    if (!reference?.receipt) continue;
    const result = validateBunProductQualificationEvidence(reference.receipt, {
      ...semanticContext,
      evidenceClass,
    });
    for (const error of result.errors) errors.push(`${id}.evidence.${evidenceClass}: ${error}`);
    pushUnique(reasons, result.reasons);
    if (qualifiedAt && result.capturedAt?.getTime() > qualifiedAt.getTime()) {
      pushUnique(reasons, ['bun-qualification-predates-semantic-evidence']);
    }
  }
  const qualificationReferences = Object.fromEntries(
    EVIDENCE_FIELDS.filter((field) => field !== 'promotion').map((field) => [
      field,
      qualification.evidence[field],
    ])
  );
  const evidenceSetDigest = computeBunQualificationEvidenceSetDigest(qualificationReferences);
  if (evidence.promotion?.receipt) {
    const result = validateBunProductPromotionEvidence(evidence.promotion.receipt, {
      ...identity,
      qualifiedAtUtc: qualification.qualifiedAtUtc,
      expiresAtUtc: qualification.expiresAtUtc,
      evidenceSetDigest,
    });
    for (const error of result.errors) errors.push(`${id}.evidence.promotion: ${error}`);
    if (qualifiedAt && result.promotedAt?.getTime() < qualifiedAt.getTime()) {
      reasons.push('bun-promotion-predates-qualification');
    }
  } else {
    reasons.push('bun-promotion-evidence-missing');
  }
  if (new Set(retainedPaths).size !== retainedPaths.length) {
    errors.push(`${id}: retained evidence paths must be distinct`);
  }
  if (qualification.claimAllowed && reasons.length > 0) {
    errors.push(`${id}: claimAllowed qualification does not satisfy Bun product support`);
  }
  return {
    id,
    workload: qualification.workload,
    claimAllowed: qualification.claimAllowed,
    qualified: qualification.claimAllowed === true
      && reasons.length === 0
      && errors.length === initialErrorCount,
    missingEvidence,
    reasons,
    blockers,
  };
}

function findById(collection, id) {
  return Array.isArray(collection) ? collection.find((entry) => entry?.id === id) : null;
}

export async function validateBunProductQualificationPolicy(policy, options = {}) {
  const errors = [];
  const repoRoot = options.repoRoot || REPO_ROOT;
  const now = options.now instanceof Date ? options.now : new Date();
  const fields = [
    '$schema',
    'schemaVersion',
    'source',
    'supportSubsystemId',
    'releaseEngineId',
    'requiredWorkloads',
    'minimumQualifiedWorkloads',
    'qualificationMaxAgeDays',
    'requiredEvidenceFields',
    'qualifications',
  ];
  if (!exactKeys(policy, fields, 'Bun qualification policy', errors)) {
    return { errors, qualifications: [], qualifiedWorkloads: 0, gateSatisfied: false };
  }
  if (policy.$schema !== '../../src/config/schema/bun-product-qualification.schema.json') {
    errors.push('Bun qualification policy $schema is invalid');
  }
  if (policy.schemaVersion !== 1) errors.push('Bun qualification policy schemaVersion must be 1');
  if (policy.source !== 'doppler') errors.push('Bun qualification policy source must be doppler');
  if (policy.supportSubsystemId !== 'runtime.bun-webgpu') {
    errors.push('Bun qualification policy supportSubsystemId is invalid');
  }
  if (policy.releaseEngineId !== 'doppler-bun') {
    errors.push('Bun qualification policy releaseEngineId is invalid');
  }
  if (!sameMembers(stringArray(policy.requiredWorkloads, 'requiredWorkloads', errors), REQUIRED_WORKLOADS)) {
    errors.push('Bun qualification policy requiredWorkloads is invalid');
  }
  if (!sameMembers(stringArray(policy.requiredEvidenceFields, 'requiredEvidenceFields', errors), EVIDENCE_FIELDS)) {
    errors.push('Bun qualification policy requiredEvidenceFields is invalid');
  }
  if (policy.minimumQualifiedWorkloads !== 3) {
    errors.push('Bun qualification policy minimumQualifiedWorkloads must be 3');
  }
  if (
    !Number.isInteger(policy.qualificationMaxAgeDays)
    || policy.qualificationMaxAgeDays < 1
    || policy.qualificationMaxAgeDays > 365
  ) {
    errors.push('Bun qualification policy qualificationMaxAgeDays must be in [1, 365]');
  }
  if (!Array.isArray(policy.qualifications)) errors.push('Bun qualifications must be an array');
  const seenIds = new Set();
  const qualifications = [];
  for (const qualification of Array.isArray(policy.qualifications) ? policy.qualifications : []) {
    const result = await validateQualification(qualification, {
      errors,
      repoRoot,
      now,
      maxAgeDays: policy.qualificationMaxAgeDays,
      seenIds,
    });
    if (result) qualifications.push(result);
  }
  const qualified = qualifications.filter((entry) => entry.qualified);
  const missingWorkloads = REQUIRED_WORKLOADS.filter(
    (workload) => !qualified.some((entry) => entry.workload === workload)
  );
  const portfolioQualified = qualified.length >= policy.minimumQualifiedWorkloads
    && missingWorkloads.length === 0;
  const [subsystems, releaseRegistry, releaseMatrix] = await Promise.all([
    options.subsystems || JSON.parse(await fs.readFile(
      options.subsystemsPath || path.join(repoRoot, path.relative(REPO_ROOT, DEFAULT_SUBSYSTEMS_PATH)),
      'utf8'
    )),
    options.releaseRegistry || JSON.parse(await fs.readFile(
      options.releaseRegistryPath || path.join(repoRoot, path.relative(REPO_ROOT, DEFAULT_RELEASE_REGISTRY_PATH)),
      'utf8'
    )),
    options.releaseMatrix || JSON.parse(await fs.readFile(
      options.releaseMatrixPath || path.join(repoRoot, path.relative(REPO_ROOT, DEFAULT_RELEASE_MATRIX_PATH)),
      'utf8'
    )),
  ]);
  const subsystem = findById(subsystems?.subsystems, policy.supportSubsystemId);
  const releaseEngine = findById(releaseRegistry?.products, policy.releaseEngineId);
  const releaseTarget = findById(releaseMatrix?.targets, policy.releaseEngineId);
  if (!subsystem) errors.push('Bun support subsystem is missing');
  if (!releaseEngine) errors.push('Bun release engine is missing');
  if (!releaseTarget) errors.push('Bun release-matrix target is missing');
  const expectedTier = portfolioQualified ? 'tier1' : 'experimental';
  const expectedStatus = portfolioQualified ? 'active' : 'experimental';
  if (subsystem?.tier !== expectedTier) {
    errors.push(`Bun support subsystem tier must be ${expectedTier}`);
  }
  if (releaseEngine?.status !== expectedStatus) {
    errors.push(`Bun release engine status must be ${expectedStatus}`);
  }
  if (releaseTarget?.status !== expectedStatus) {
    errors.push(`Bun release-matrix target status must be ${expectedStatus}`);
  }
  if (!portfolioQualified && qualified.length > 0) {
    errors.push('Partial Bun product promotion is forbidden; all required workloads must promote together');
  }
  return {
    errors,
    qualifications,
    qualifiedWorkloads: qualified.length,
    candidateWorkloads: qualifications.filter((entry) => !entry.claimAllowed).length,
    missingWorkloads,
    portfolioQualified,
    subsystemTier: subsystem?.tier ?? null,
    releaseEngineStatus: releaseEngine?.status ?? null,
    releaseTargetStatus: releaseTarget?.status ?? null,
    gateSatisfied: errors.length === 0 && portfolioQualified,
  };
}

export async function buildBunProductQualificationReport(options = {}) {
  const policyPath = options.policyPath || DEFAULT_POLICY_PATH;
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  const result = await validateBunProductQualificationPolicy(policy, options);
  return {
    ok: result.errors.length === 0,
    policyPath: path.relative(options.repoRoot || REPO_ROOT, policyPath),
    ...result,
  };
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((token) => token !== '--json');
  if (unsupported.length > 0) throw new Error(`Unknown argument: ${unsupported[0]}`);
  const report = await buildBunProductQualificationReport();
  if (json) console.log(JSON.stringify(report, null, 2));
  else if (report.ok) {
    console.log(
      `bun-product-qualification: contract ok, gate ${report.gateSatisfied ? 'satisfied' : 'incomplete'} `
      + `(${report.qualifiedWorkloads}/3 qualified; ${report.candidateWorkloads} candidates)`
    );
  } else {
    for (const error of report.errors) console.error(`bun-product-qualification: ${error}`);
  }
  if (!report.ok) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
