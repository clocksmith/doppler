#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(
  REPO_ROOT,
  'tools',
  'policies',
  'product-integration-qualification.json'
);
const REQUIRED_WORKLOADS = Object.freeze(['generation', 'embedding-retrieval', 'reranking']);
const QUALIFICATION_LEVELS = new Set([
  'contract-ready',
  'runtime-verified',
  'task-qualified',
  'performance-qualified',
  'product-supported',
]);
const LIFECYCLES = new Set([
  'candidate',
  'active',
  'deprecated',
  'quarantined',
  'revoked',
  'retired',
]);
const EVIDENCE_FIELDS = Object.freeze([
  'installToFirstVerifiedOutput',
  'identity',
  'sourceTaskQualityRetention',
  'reliability',
  'memory',
  'coldWarmResponse',
  'browserHardwareQualification',
  'incumbentControl',
  'upgradeRequalification',
  'rollbackRevocation',
]);
const INTEGRATION_FIELDS = Object.freeze([
  'id',
  'applicationName',
  'workload',
  'owner',
  'ownerConfirmedAtUtc',
  'qualificationLevel',
  'lifecycle',
  'claimAllowed',
  'logicalModelId',
  'resolvedArtifactVariantId',
  'resolvedExecutionId',
  'qualifiedAtUtc',
  'expiresAtUtc',
  'evidence',
  'blockers',
]);
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
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

function isRepoRelativePath(value) {
  const normalized = normalizeText(value);
  return Boolean(
    normalized
    && !path.isAbsolute(normalized)
    && !normalized.includes('\\')
    && !normalized.split('/').includes('..')
  );
}

async function validateEvidencePath(value, label, repoRoot, errors) {
  if (value === null) return null;
  if (!isRepoRelativePath(value)) {
    errors.push(`${label} must be a repo-relative path or null`);
    return null;
  }
  const normalized = normalizeText(value);
  try {
    await fs.stat(path.join(repoRoot, normalized));
  } catch {
    errors.push(`${label} does not exist: ${normalized}`);
  }
  return normalized;
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

function validateStringArray(value, label, errors) {
  if (!Array.isArray(value)) {
    errors.push(`${label} must be an array`);
    return [];
  }
  const seen = new Set();
  const values = [];
  for (const item of value) {
    const normalized = normalizeText(item);
    if (!normalized) {
      errors.push(`${label} entries must be non-empty strings`);
    } else if (seen.has(normalized)) {
      errors.push(`${label} contains duplicate entry ${normalized}`);
    } else {
      seen.add(normalized);
      values.push(normalized);
    }
  }
  return values;
}

function validateNullableString(value, label, errors) {
  if (value === null) return null;
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string or null`);
  return normalized || null;
}

function validateNullableSha256(value, label, errors) {
  const normalized = validateNullableString(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !SHA256_PATTERN.test(normalized)) {
    errors.push(`${label} must be a SHA-256 identity or null`);
  }
  return normalized;
}

function ownerConfirmationIsCurrent(confirmedAt, now, maxAgeDays) {
  if (!confirmedAt) return false;
  const age = now.getTime() - confirmedAt.getTime();
  return age >= 0 && age <= maxAgeDays * 24 * 60 * 60 * 1000;
}

async function validateIntegration(integration, context) {
  const { errors, repoRoot, now, maxAgeDays, seenIds } = context;
  const id = normalizeText(integration?.id) || '<missing-id>';
  if (!validateExactKeys(integration, INTEGRATION_FIELDS, id, errors)) return null;
  if (!ID_PATTERN.test(id)) errors.push(`${id}: id must be lowercase kebab-case`);
  if (seenIds.has(id)) errors.push(`${id}: duplicate integration id`);
  seenIds.add(id);
  const applicationName = normalizeText(integration.applicationName);
  if (!applicationName) errors.push(`${id}: applicationName is required`);
  if (!REQUIRED_WORKLOADS.includes(integration.workload)) {
    errors.push(`${id}: workload is not recognized`);
  }
  const owner = validateNullableString(integration.owner, `${id}: owner`, errors);
  const ownerConfirmedAt = parseIsoInstant(
    integration.ownerConfirmedAtUtc,
    `${id}: ownerConfirmedAtUtc`,
    true,
    errors
  );
  if (!QUALIFICATION_LEVELS.has(integration.qualificationLevel)) {
    errors.push(`${id}: qualificationLevel is not recognized`);
  }
  if (!LIFECYCLES.has(integration.lifecycle)) errors.push(`${id}: lifecycle is not recognized`);
  if (typeof integration.claimAllowed !== 'boolean') errors.push(`${id}: claimAllowed must be boolean`);
  const logicalModelId = validateNullableString(
    integration.logicalModelId,
    `${id}: logicalModelId`,
    errors
  );
  const resolvedArtifactVariantId = validateNullableSha256(
    integration.resolvedArtifactVariantId,
    `${id}: resolvedArtifactVariantId`,
    errors
  );
  const resolvedExecutionId = validateNullableSha256(
    integration.resolvedExecutionId,
    `${id}: resolvedExecutionId`,
    errors
  );
  const qualifiedAt = parseIsoInstant(
    integration.qualifiedAtUtc,
    `${id}: qualifiedAtUtc`,
    true,
    errors
  );
  const expiresAt = parseIsoInstant(
    integration.expiresAtUtc,
    `${id}: expiresAtUtc`,
    true,
    errors
  );
  const blockers = validateStringArray(integration.blockers, `${id}: blockers`, errors);
  const evidence = {};
  if (validateExactKeys(integration.evidence, EVIDENCE_FIELDS, `${id}: evidence`, errors)) {
    await Promise.all(EVIDENCE_FIELDS.map(async (field) => {
      evidence[field] = await validateEvidencePath(
        integration.evidence[field],
        `${id}: evidence.${field}`,
        repoRoot,
        errors
      );
    }));
  }
  const missingEvidence = EVIDENCE_FIELDS.filter((field) => !evidence[field]);
  const qualificationReasons = [];
  if (integration.qualificationLevel !== 'product-supported') {
    qualificationReasons.push('qualification-level-not-product-supported');
  }
  if (integration.lifecycle !== 'active') qualificationReasons.push('lifecycle-not-active');
  if (!owner) qualificationReasons.push('owner-missing');
  if (!ownerConfirmationIsCurrent(ownerConfirmedAt, now, maxAgeDays)) {
    qualificationReasons.push('owner-confirmation-stale-or-missing');
  }
  if (!qualifiedAt) qualificationReasons.push('qualification-date-missing');
  if (!expiresAt || expiresAt.getTime() <= now.getTime()) {
    qualificationReasons.push('qualification-expired-or-missing');
  }
  if (!logicalModelId || !resolvedArtifactVariantId || !resolvedExecutionId) {
    qualificationReasons.push('resolution-identity-incomplete');
  }
  if (missingEvidence.length > 0) qualificationReasons.push('evidence-incomplete');
  if (blockers.length > 0) qualificationReasons.push('blockers-present');
  const qualified = qualificationReasons.length === 0;
  if (integration.claimAllowed === true && !qualified) {
    errors.push(`${id}: claimAllowed integration does not satisfy product qualification`);
  }
  if (integration.claimAllowed === true && blockers.length > 0) {
    errors.push(`${id}: claimAllowed integration must not list blockers`);
  }
  if (integration.claimAllowed === false && blockers.length === 0) {
    errors.push(`${id}: non-claimable integration must list blockers`);
  }
  return {
    id,
    applicationName,
    workload: integration.workload,
    owner,
    qualificationLevel: integration.qualificationLevel,
    lifecycle: integration.lifecycle,
    claimAllowed: integration.claimAllowed,
    qualified: qualified && integration.claimAllowed === true,
    qualificationReasons,
    missingEvidence,
  };
}

export async function validateProductIntegrationQualification(policy, options = {}) {
  const errors = [];
  const repoRoot = options.repoRoot || REPO_ROOT;
  const now = options.now instanceof Date ? options.now : new Date();
  const fields = [
    '$schema',
    'schemaVersion',
    'source',
    'goalId',
    'minimumQualifiedIntegrations',
    'requiredWorkloads',
    'ownerConfirmationMaxAgeDays',
    'integrations',
  ];
  if (!validateExactKeys(policy, fields, 'product integration policy', errors)) {
    return {
      errors,
      integrations: [],
      qualifiedIntegrations: 0,
      candidateIntegrations: 0,
      candidateWorkloads: [],
      gateSatisfied: false,
      missingWorkloads: [...REQUIRED_WORKLOADS],
    };
  }
  if (policy.schemaVersion !== 2) errors.push('product integration policy schemaVersion must be 2');
  if (policy.source !== 'doppler') errors.push('product integration policy source must be "doppler"');
  if (policy.goalId !== 'local-webgpu-product-surface') {
    errors.push('product integration policy goalId must be "local-webgpu-product-surface"');
  }
  if (policy.minimumQualifiedIntegrations !== 3) {
    errors.push('product integration policy minimumQualifiedIntegrations must be 3');
  }
  if (JSON.stringify(policy.requiredWorkloads) !== JSON.stringify(REQUIRED_WORKLOADS)) {
    errors.push(`product integration policy requiredWorkloads must be ${REQUIRED_WORKLOADS.join(', ')}`);
  }
  if (
    !Number.isInteger(policy.ownerConfirmationMaxAgeDays)
    || policy.ownerConfirmationMaxAgeDays < 1
    || policy.ownerConfirmationMaxAgeDays > 365
  ) {
    errors.push('product integration policy ownerConfirmationMaxAgeDays must be in [1, 365]');
  }
  if (!Array.isArray(policy.integrations)) {
    errors.push('product integration policy integrations must be an array');
  }
  const seenIds = new Set();
  const integrations = [];
  for (const integration of Array.isArray(policy.integrations) ? policy.integrations : []) {
    const result = await validateIntegration(integration, {
      errors,
      repoRoot,
      now,
      maxAgeDays: policy.ownerConfirmationMaxAgeDays,
      seenIds,
    });
    if (result) integrations.push(result);
  }
  const qualified = integrations.filter((integration) => integration.qualified);
  const candidates = integrations.filter((integration) => (
    integration.lifecycle === 'candidate' && integration.claimAllowed === false
  ));
  const qualifiedApplicationNames = new Set(
    qualified.map((integration) => integration.applicationName.toLowerCase())
  );
  const missingWorkloads = REQUIRED_WORKLOADS.filter(
    (workload) => !qualified.some((integration) => integration.workload === workload)
  );
  return {
    errors,
    integrations,
    qualifiedIntegrations: qualified.length,
    candidateIntegrations: candidates.length,
    candidateWorkloads: Array.from(new Set(candidates.map((integration) => integration.workload))),
    distinctQualifiedApplications: qualifiedApplicationNames.size,
    missingWorkloads,
    gateSatisfied: errors.length === 0
      && qualified.length >= policy.minimumQualifiedIntegrations
      && qualifiedApplicationNames.size >= policy.minimumQualifiedIntegrations
      && missingWorkloads.length === 0,
  };
}

export async function buildProductIntegrationQualificationReport(options = {}) {
  const policyPath = options.policyPath || DEFAULT_POLICY_PATH;
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  const result = await validateProductIntegrationQualification(policy, options);
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
  const report = await buildProductIntegrationQualificationReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    const status = report.gateSatisfied ? 'satisfied' : 'incomplete';
    console.log(
      `product-integration-qualification: contract ok, gate ${status} ` +
      `(${report.qualifiedIntegrations}/3 qualified; ${report.candidateIntegrations} candidates; ` +
      `missing ${report.missingWorkloads.join(', ') || 'none'})`
    );
  } else {
    for (const error of report.errors) console.error(`product-integration-qualification: ${error}`);
  }
  if (!report.ok) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
