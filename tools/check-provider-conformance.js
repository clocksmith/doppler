#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'provider-conformance.json');
const GOAL_IDS = Object.freeze([
  'model-artifact-runtime-contract',
  'correctness-performance-claims',
]);
const REQUIRED_WORKLOADS = Object.freeze(['generation', 'embedding', 'reranking']);
const CORE_PROVIDER_LANE_IDS = Object.freeze(['browser-webgpu', 'node-webgpu']);
const LIFECYCLE_STAGES = Object.freeze(['load', 'execute', 'unload']);
const EVIDENCE_FIELDS = Object.freeze([
  'modelContract',
  'resolutionIdentity',
  'operations',
  'lifecycle',
  'correctness',
  'providerReceipt',
]);
const CORRECTNESS_CLASSES = new Set([
  'exact-token',
  'tolerance-bounded-numerical',
  'semantic',
  'held-out-task-metric',
]);
const PROVIDER_KINDS = new Set(['browser-webgpu', 'node-webgpu', 'doe', 'other-webgpu']);
const PROVIDER_ROLES = new Set(['core', 'optional-named']);
const LIFECYCLE_RESULTS = new Set(['passed', 'failed', 'not-run']);
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const DAY_MS = 24 * 60 * 60 * 1000;

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

function validateNullableString(value, label, errors) {
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

function sameMembers(left, right) {
  const sortedLeft = [...left].sort();
  const sortedRight = [...right].sort();
  return sortedLeft.length === sortedRight.length
    && sortedLeft.every((value, index) => value === sortedRight[index]);
}

function parseIsoInstant(value, label, errors) {
  if (value === null) return null;
  const normalized = normalizeText(value);
  const instant = normalized ? new Date(normalized) : null;
  if (!instant || !Number.isFinite(instant.getTime()) || instant.toISOString() !== normalized) {
    errors.push(`${label} must be an ISO instant or null`);
    return null;
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

async function validateRepoPath(value, label, repoRoot, errors) {
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

function validateClaimState(record, label, reasons, errors) {
  const blockers = validateStringArray(record.blockers, `${label}.blockers`, errors);
  if (typeof record.claimAllowed !== 'boolean') {
    errors.push(`${label}.claimAllowed must be boolean`);
  } else if (record.claimAllowed && blockers.length > 0) {
    errors.push(`${label}: claimable records must not list blockers`);
  } else if (!record.claimAllowed && blockers.length === 0) {
    errors.push(`${label}: non-claimable records must list blockers`);
  }
  if (record.claimAllowed === true && reasons.length > 0) {
    errors.push(`${label}: claimAllowed record does not satisfy provider conformance`);
  }
  return blockers;
}

async function validateProviderResult(provider, context) {
  const {
    errors,
    repoRoot,
    now,
    maxAgeDays,
    laneById,
    suite,
    seenLaneIds,
  } = context;
  const fields = [
    'laneId',
    'implementationId',
    'logicalModelId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'environmentFingerprint',
    'operations',
    'lifecycle',
    'correctness',
    'qualifiedAtUtc',
    'expiresAtUtc',
    'evidence',
    'claimAllowed',
    'blockers',
  ];
  const laneId = normalizeText(provider?.laneId) || '<missing-lane>';
  const label = `${suite.id}: provider ${laneId}`;
  if (!validateExactKeys(provider, fields, label, errors)) return null;
  const reasons = [];
  if (!ID_PATTERN.test(laneId)) errors.push(`${label}: laneId must be lowercase kebab-case`);
  if (seenLaneIds.has(laneId)) errors.push(`${suite.id}: duplicate provider lane ${laneId}`);
  seenLaneIds.add(laneId);
  if (!laneById.has(laneId)) errors.push(`${label}: lane is not declared by providerLanes`);
  const identityFields = [
    'implementationId',
    'logicalModelId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'environmentFingerprint',
  ];
  const identity = Object.fromEntries(identityFields.map((field) => [
    field,
    validateNullableString(provider[field], `${label}.${field}`, errors),
  ]));
  for (const field of identityFields) {
    if (!identity[field]) reasons.push(`${field}-missing`);
  }
  if (identity.logicalModelId && identity.logicalModelId !== suite.logicalModelId) {
    reasons.push('logical-model-mismatch');
  }
  if (
    identity.resolvedArtifactVariantId
    && identity.resolvedArtifactVariantId !== suite.resolvedArtifactVariantId
  ) {
    reasons.push('artifact-variant-mismatch');
  }
  const operations = validateStringArray(provider.operations, `${label}.operations`, errors);
  if (!sameMembers(operations, suite.declaredOperations)) reasons.push('operations-mismatch');
  if (!validateExactKeys(provider.lifecycle, LIFECYCLE_STAGES, `${label}.lifecycle`, errors)) {
    reasons.push('lifecycle-incomplete');
  } else {
    for (const stage of LIFECYCLE_STAGES) {
      if (!LIFECYCLE_RESULTS.has(provider.lifecycle[stage])) {
        errors.push(`${label}.lifecycle.${stage} is not recognized`);
      }
      if (provider.lifecycle[stage] !== 'passed') reasons.push(`lifecycle-${stage}-not-passed`);
    }
  }
  if (!validateExactKeys(provider.correctness, ['class', 'passed'], `${label}.correctness`, errors)) {
    reasons.push('correctness-incomplete');
  } else {
    if (!CORRECTNESS_CLASSES.has(provider.correctness.class)) {
      errors.push(`${label}.correctness.class is not recognized`);
    }
    if (provider.correctness.class !== suite.correctnessClass) {
      reasons.push('correctness-class-mismatch');
    }
    if (typeof provider.correctness.passed !== 'boolean') {
      errors.push(`${label}.correctness.passed must be boolean`);
    }
    if (provider.correctness.passed !== true) reasons.push('correctness-not-passed');
  }
  const qualifiedAt = parseIsoInstant(provider.qualifiedAtUtc, `${label}.qualifiedAtUtc`, errors);
  const expiresAt = parseIsoInstant(provider.expiresAtUtc, `${label}.expiresAtUtc`, errors);
  if (!qualifiedAt) {
    reasons.push('qualification-date-missing');
  } else {
    const age = now.getTime() - qualifiedAt.getTime();
    if (age < 0 || age > maxAgeDays * DAY_MS) reasons.push('qualification-stale-or-future');
  }
  if (!expiresAt || expiresAt.getTime() <= now.getTime()) reasons.push('qualification-expired-or-missing');
  const evidence = {};
  if (validateExactKeys(provider.evidence, EVIDENCE_FIELDS, `${label}.evidence`, errors)) {
    await Promise.all(EVIDENCE_FIELDS.map(async (field) => {
      evidence[field] = await validateRepoPath(
        provider.evidence[field],
        `${label}.evidence.${field}`,
        repoRoot,
        errors
      );
    }));
  }
  const missingEvidence = EVIDENCE_FIELDS.filter((field) => !evidence[field]);
  if (missingEvidence.length > 0) reasons.push('evidence-incomplete');
  validateClaimState(provider, label, reasons, errors);
  return {
    laneId,
    claimAllowed: provider.claimAllowed,
    qualified: provider.claimAllowed === true && reasons.length === 0,
    reasons,
    missingEvidence,
  };
}

async function validateSuite(suite, context) {
  const { errors, repoRoot, now, maxAgeDays, laneById, seenSuiteIds } = context;
  const fields = [
    'id',
    'workload',
    'logicalModelId',
    'resolvedArtifactVariantId',
    'workloadContractPath',
    'declaredOperations',
    'correctnessClass',
    'requiredProviderLaneIds',
    'claimAllowed',
    'providers',
    'blockers',
  ];
  const id = normalizeText(suite?.id) || '<missing-suite>';
  if (!validateExactKeys(suite, fields, id, errors)) return null;
  const reasons = [];
  if (!ID_PATTERN.test(id)) errors.push(`${id}: id must be lowercase kebab-case`);
  if (seenSuiteIds.has(id)) errors.push(`${id}: duplicate suite id`);
  seenSuiteIds.add(id);
  if (!REQUIRED_WORKLOADS.includes(suite.workload)) errors.push(`${id}: workload is not recognized`);
  const logicalModelId = validateNullableString(suite.logicalModelId, `${id}.logicalModelId`, errors);
  const resolvedArtifactVariantId = validateNullableString(
    suite.resolvedArtifactVariantId,
    `${id}.resolvedArtifactVariantId`,
    errors
  );
  if (!logicalModelId) reasons.push('logical-model-missing');
  if (!resolvedArtifactVariantId) reasons.push('artifact-variant-missing');
  const workloadContractPath = await validateRepoPath(
    suite.workloadContractPath,
    `${id}.workloadContractPath`,
    repoRoot,
    errors
  );
  if (!workloadContractPath) reasons.push('workload-contract-missing');
  const declaredOperations = validateStringArray(
    suite.declaredOperations,
    `${id}.declaredOperations`,
    errors
  );
  if (declaredOperations.length === 0) reasons.push('declared-operations-missing');
  if (!CORRECTNESS_CLASSES.has(suite.correctnessClass)) {
    errors.push(`${id}: correctnessClass is not recognized`);
  }
  const requiredProviderLaneIds = validateStringArray(
    suite.requiredProviderLaneIds,
    `${id}.requiredProviderLaneIds`,
    errors
  );
  for (const coreLaneId of CORE_PROVIDER_LANE_IDS) {
    if (!requiredProviderLaneIds.includes(coreLaneId)) reasons.push(`core-provider-${coreLaneId}-missing`);
  }
  for (const laneId of requiredProviderLaneIds) {
    if (!laneById.has(laneId)) errors.push(`${id}: required provider lane ${laneId} is not declared`);
  }
  if (!Array.isArray(suite.providers)) errors.push(`${id}.providers must be an array`);
  const seenLaneIds = new Set();
  const providers = [];
  const suiteContract = {
    id,
    logicalModelId,
    resolvedArtifactVariantId,
    declaredOperations,
    correctnessClass: suite.correctnessClass,
  };
  for (const provider of Array.isArray(suite.providers) ? suite.providers : []) {
    const result = await validateProviderResult(provider, {
      errors,
      repoRoot,
      now,
      maxAgeDays,
      laneById,
      suite: suiteContract,
      seenLaneIds,
    });
    if (result) providers.push(result);
  }
  for (const laneId of requiredProviderLaneIds) {
    const provider = providers.find((entry) => entry.laneId === laneId);
    if (!provider) {
      reasons.push(`required-provider-${laneId}-missing`);
    } else if (!provider.qualified) {
      reasons.push(`required-provider-${laneId}-unqualified`);
    }
  }
  validateClaimState(suite, id, reasons, errors);
  return {
    id,
    workload: suite.workload,
    logicalModelId,
    resolvedArtifactVariantId,
    workloadContractPath,
    declaredOperations,
    correctnessClass: suite.correctnessClass,
    claimAllowed: suite.claimAllowed,
    qualified: suite.claimAllowed === true && reasons.length === 0,
    reasons,
    providers,
  };
}

async function validateProviderLane(lane, context) {
  const { errors, repoRoot, seenLaneIds } = context;
  const fields = ['id', 'kind', 'role', 'contractPath'];
  const id = normalizeText(lane?.id) || '<missing-lane>';
  if (!validateExactKeys(lane, fields, `provider lane ${id}`, errors)) return null;
  if (!ID_PATTERN.test(id)) errors.push(`${id}: provider lane id must be lowercase kebab-case`);
  if (seenLaneIds.has(id)) errors.push(`${id}: duplicate provider lane id`);
  seenLaneIds.add(id);
  if (!PROVIDER_KINDS.has(lane.kind)) errors.push(`${id}: provider kind is not recognized`);
  if (!PROVIDER_ROLES.has(lane.role)) errors.push(`${id}: provider role is not recognized`);
  if (lane.kind === 'doe' && lane.role !== 'optional-named') {
    errors.push(`${id}: Doe must remain an optional-named provider lane`);
  }
  const contractPath = await validateRepoPath(
    lane.contractPath,
    `${id}.contractPath`,
    repoRoot,
    errors
  );
  if (!contractPath) errors.push(`${id}: contractPath is required`);
  return { id, kind: lane.kind, role: lane.role, contractPath };
}

export async function validateProviderConformancePolicy(policy, options = {}) {
  const errors = [];
  const repoRoot = options.repoRoot || REPO_ROOT;
  const now = options.now instanceof Date ? options.now : new Date();
  const fields = [
    '$schema',
    'schemaVersion',
    'source',
    'goalIds',
    'minimumQualifiedSuites',
    'requiredWorkloads',
    'coreProviderLaneIds',
    'qualificationMaxAgeDays',
    'requiredLifecycleStages',
    'requiredEvidenceFields',
    'providerLanes',
    'suites',
  ];
  if (!validateExactKeys(policy, fields, 'provider conformance policy', errors)) {
    return {
      errors,
      suites: [],
      qualifiedSuites: 0,
      candidateSuites: 0,
      candidateWorkloads: [],
      missingWorkloads: [...REQUIRED_WORKLOADS],
      gateSatisfied: false,
    };
  }
  if (policy.schemaVersion !== 1) errors.push('provider conformance policy schemaVersion must be 1');
  if (policy.source !== 'doppler') errors.push('provider conformance policy source must be "doppler"');
  if (policy.$schema !== '../../src/config/schema/provider-conformance-policy.schema.json') {
    errors.push('provider conformance policy $schema is invalid');
  }
  const goalIds = validateStringArray(policy.goalIds, 'provider conformance policy goalIds', errors);
  if (!sameMembers(goalIds, GOAL_IDS)) errors.push('provider conformance policy goalIds are invalid');
  if (policy.minimumQualifiedSuites !== 3) errors.push('provider conformance policy minimumQualifiedSuites must be 3');
  const requiredWorkloads = validateStringArray(
    policy.requiredWorkloads,
    'provider conformance policy requiredWorkloads',
    errors
  );
  if (!sameMembers(requiredWorkloads, REQUIRED_WORKLOADS)) {
    errors.push(`provider conformance policy requiredWorkloads must be ${REQUIRED_WORKLOADS.join(', ')}`);
  }
  const coreProviderLaneIds = validateStringArray(
    policy.coreProviderLaneIds,
    'provider conformance policy coreProviderLaneIds',
    errors
  );
  if (JSON.stringify(coreProviderLaneIds) !== JSON.stringify(CORE_PROVIDER_LANE_IDS)) {
    errors.push(`provider conformance policy coreProviderLaneIds must be ${CORE_PROVIDER_LANE_IDS.join(', ')}`);
  }
  const lifecycleStages = validateStringArray(
    policy.requiredLifecycleStages,
    'provider conformance policy requiredLifecycleStages',
    errors
  );
  if (JSON.stringify(lifecycleStages) !== JSON.stringify(LIFECYCLE_STAGES)) {
    errors.push(`provider conformance policy requiredLifecycleStages must be ${LIFECYCLE_STAGES.join(', ')}`);
  }
  const evidenceFields = validateStringArray(
    policy.requiredEvidenceFields,
    'provider conformance policy requiredEvidenceFields',
    errors
  );
  if (JSON.stringify(evidenceFields) !== JSON.stringify(EVIDENCE_FIELDS)) {
    errors.push('provider conformance policy requiredEvidenceFields are invalid');
  }
  if (
    !Number.isInteger(policy.qualificationMaxAgeDays)
    || policy.qualificationMaxAgeDays < 1
    || policy.qualificationMaxAgeDays > 365
  ) {
    errors.push('provider conformance policy qualificationMaxAgeDays must be in [1, 365]');
  }
  if (!Array.isArray(policy.providerLanes)) errors.push('provider conformance policy providerLanes must be an array');
  const seenLaneIds = new Set();
  const providerLanes = [];
  for (const lane of Array.isArray(policy.providerLanes) ? policy.providerLanes : []) {
    const result = await validateProviderLane(lane, { errors, repoRoot, seenLaneIds });
    if (result) providerLanes.push(result);
  }
  const laneById = new Map(providerLanes.map((lane) => [lane.id, lane]));
  for (const laneId of CORE_PROVIDER_LANE_IDS) {
    const lane = laneById.get(laneId);
    if (!lane) {
      errors.push(`provider conformance policy is missing core lane ${laneId}`);
    } else if (lane.role !== 'core' || lane.kind !== laneId) {
      errors.push(`${laneId}: core lane must use matching kind and role core`);
    }
  }
  for (const lane of providerLanes) {
    if (lane.role === 'core' && !CORE_PROVIDER_LANE_IDS.includes(lane.id)) {
      errors.push(`${lane.id}: undeclared core provider lane`);
    }
  }
  if (!Array.isArray(policy.suites)) errors.push('provider conformance policy suites must be an array');
  const seenSuiteIds = new Set();
  const suites = [];
  for (const suite of Array.isArray(policy.suites) ? policy.suites : []) {
    const result = await validateSuite(suite, {
      errors,
      repoRoot,
      now,
      maxAgeDays: policy.qualificationMaxAgeDays,
      laneById,
      seenSuiteIds,
    });
    if (result) suites.push(result);
  }
  const qualified = suites.filter((suite) => suite.qualified);
  const candidates = suites.filter((suite) => suite.claimAllowed === false);
  const missingWorkloads = REQUIRED_WORKLOADS.filter(
    (workload) => !qualified.some((suite) => suite.workload === workload)
  );
  return {
    errors,
    providerLanes,
    suites,
    qualifiedSuites: qualified.length,
    candidateSuites: candidates.length,
    candidateWorkloads: Array.from(new Set(candidates.map((suite) => suite.workload))),
    missingWorkloads,
    gateSatisfied: errors.length === 0
      && qualified.length >= policy.minimumQualifiedSuites
      && missingWorkloads.length === 0,
  };
}

export async function buildProviderConformanceReport(options = {}) {
  const policyPath = options.policyPath || DEFAULT_POLICY_PATH;
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  const result = await validateProviderConformancePolicy(policy, options);
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
  const report = await buildProviderConformanceReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    const status = report.gateSatisfied ? 'satisfied' : 'incomplete';
    console.log(
      `provider-conformance: contract ok, gate ${status} `
      + `(${report.qualifiedSuites}/3 qualified; ${report.candidateSuites} candidates; `
      + `missing ${report.missingWorkloads.join(', ') || 'none'})`
    );
  } else {
    for (const error of report.errors) console.error(`provider-conformance: ${error}`);
  }
  if (!report.ok) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
