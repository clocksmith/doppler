#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { isPlainObject } from '../src/formats/plain-object.js';

import {
  validateDopplerRuntimeOwnershipReceipt,
  validateRuntimeOwnershipExecutionEvidence,
} from './lib/runtime-ownership-execution-evidence.js';
import {
  RUNTIME_OWNERSHIP_DIMENSION_CLASSES,
  computeRuntimeOwnershipDecisionEvidenceDigest,
  computeRuntimeOwnershipEvidenceSetDigest,
  computeRuntimeOwnershipHypothesisSetDigest,
  validateRuntimeOwnershipDimensionEvidence,
  validateRuntimeOwnershipHypothesisEvidence,
  validateRuntimeOwnershipPromotionEvidence,
} from './lib/runtime-ownership-decision-evidence.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(
  REPO_ROOT,
  'benchmarks',
  'vendors',
  'runtime-ownership-decisions.json'
);
const GOAL_IDS = Object.freeze([
  'model-artifact-runtime-contract',
  'correctness-performance-claims',
]);
const REQUIRED_WORKLOADS = Object.freeze(['generation', 'embedding', 'reranking']);
const DISPOSITIONS = Object.freeze(['incumbent', 'doppler', 'dual']);
const ADVANTAGE_AXES = Object.freeze([
  'unsupported-operation',
  'end-to-end-performance',
  'memory',
  'diagnostic-depth',
  'offline-artifact-control',
  'verified-correction-path',
]);
const EVIDENCE_FIELDS = Object.freeze([
  'sourceExecution',
  'incumbentExecution',
  'dopplerExecution',
  ...RUNTIME_OWNERSHIP_DIMENSION_CLASSES,
  'promotion',
]);
const NON_PROMOTION_EVIDENCE_FIELDS = Object.freeze(
  EVIDENCE_FIELDS.filter((field) => field !== 'promotion')
);
const CORRECTNESS_CLASSES = new Set([
  'exact-token',
  'tolerance-bounded-numerical',
  'semantic',
  'held-out-task-metric',
]);
const THRESHOLD_OPERATORS = new Set([
  'greater-than-or-equal',
  'less-than-or-equal',
  'pass',
]);
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
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
    if (!Object.prototype.hasOwnProperty.call(value, field)) {
      errors.push(`${label}.${field} is required`);
    }
  }
  return true;
}

function stringArray(value, label, errors) {
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

function nullableString(value, label, errors) {
  if (value === null) return null;
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string or null`);
  return normalized || null;
}

function nullableSha256(value, label, errors) {
  const normalized = nullableString(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !SHA256_PATTERN.test(normalized)) {
    errors.push(`${label} must be a SHA-256 identity or null`);
  }
  return normalized;
}

function nullableRevision(value, label, errors) {
  const normalized = nullableString(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !/^[0-9a-f]{40}$/.test(normalized)) {
    errors.push(`${label} must be a lowercase 40-hex revision or null`);
  }
  return normalized;
}

function isoInstant(value, label, errors) {
  if (value === null) return null;
  const normalized = normalizeText(value);
  const instant = normalized ? new Date(normalized) : null;
  if (!instant || !Number.isFinite(instant.getTime()) || instant.toISOString() !== normalized) {
    errors.push(`${label} must be an ISO instant or null`);
    return null;
  }
  return instant;
}

function isRepoPath(value) {
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
  if (!exactKeys(value, ['path', 'digest'], label, errors)) return null;
  if (!isRepoPath(value.path)) {
    errors.push(`${label}.path must be a repo-relative path`);
    return null;
  }
  const evidencePath = normalizeText(value.path);
  const evidenceDigest = normalizeText(value.digest).toLowerCase();
  if (!SHA256_PATTERN.test(evidenceDigest)) {
    errors.push(`${label}.digest must be a SHA-256 identity`);
  }
  let receipt;
  try {
    receipt = JSON.parse(await fs.readFile(path.join(repoRoot, evidencePath), 'utf8'));
  } catch (error) {
    errors.push(`${label}.path is not readable JSON: ${error.message}`);
    return null;
  }
  if (
    SHA256_PATTERN.test(evidenceDigest)
    && computeRuntimeOwnershipDecisionEvidenceDigest(receipt) !== evidenceDigest
  ) {
    errors.push(`${label}.digest does not match canonical JSON evidence`);
  }
  return { path: evidencePath, digest: evidenceDigest, receipt };
}

function validateClaimState(record, label, reasons, errors) {
  const blockers = stringArray(record.blockers, `${label}.blockers`, errors);
  if (typeof record.claimAllowed !== 'boolean') {
    errors.push(`${label}.claimAllowed must be boolean`);
  } else if (record.claimAllowed && blockers.length > 0) {
    errors.push(`${label}: claimable decisions must not list blockers`);
  } else if (!record.claimAllowed && blockers.length === 0) {
    errors.push(`${label}: non-claimable decisions must list blockers`);
  }
  if (record.claimAllowed === true && reasons.length > 0) {
    errors.push(`${label}: claimAllowed decision does not satisfy runtime ownership qualification`);
  }
  return blockers;
}

function pushUnique(target, values) {
  for (const value of values) {
    if (!target.includes(value)) target.push(value);
  }
}

async function validateHypothesis(hypothesis, context) {
  const {
    errors,
    repoRoot,
    decisionId,
    seenAxes,
    expectedIdentity,
    retainedEvidencePaths,
  } = context;
  const initialErrorCount = errors.length;
  const fields = [
    'axis',
    'statement',
    'metric',
    'controlMetric',
    'controlRequirement',
    'threshold',
    'declaredAtUtc',
    'result',
  ];
  const axis = normalizeText(hypothesis?.axis) || '<missing-axis>';
  const label = `${decisionId}: hypothesis ${axis}`;
  if (!exactKeys(hypothesis, fields, label, errors)) {
    return {
      axis,
      complete: false,
      passed: null,
      declaredAt: null,
      evaluatedAt: null,
      reasons: ['hypothesis-invalid'],
    };
  }
  const reasons = [];
  if (!ADVANTAGE_AXES.includes(axis)) errors.push(`${label}: axis is not recognized`);
  if (seenAxes.has(axis)) errors.push(`${decisionId}: duplicate hypothesis axis ${axis}`);
  seenAxes.add(axis);
  for (const field of ['statement', 'metric', 'controlMetric', 'controlRequirement']) {
    if (!normalizeText(hypothesis[field])) errors.push(`${label}.${field} is required`);
  }
  const thresholdFields = ['operator', 'value', 'unit'];
  let operator = null;
  let thresholdValue = null;
  if (!exactKeys(hypothesis.threshold, thresholdFields, `${label}.threshold`, errors)) {
    reasons.push('threshold-invalid');
  } else {
    operator = hypothesis.threshold.operator;
    thresholdValue = hypothesis.threshold.value;
    const unit = nullableString(hypothesis.threshold.unit, `${label}.threshold.unit`, errors);
    if (!THRESHOLD_OPERATORS.has(operator)) errors.push(`${label}.threshold.operator is not recognized`);
    if (operator === 'pass') {
      if (thresholdValue !== null || unit !== null) {
        errors.push(`${label}: pass threshold must use null value and unit`);
      }
    } else if (!Number.isFinite(thresholdValue) || !unit) {
      errors.push(`${label}: numeric threshold requires finite value and unit`);
    }
  }
  const declaredAt = isoInstant(hypothesis.declaredAtUtc, `${label}.declaredAtUtc`, errors);
  if (!declaredAt) reasons.push('declaration-date-missing');
  const resultFields = ['passed', 'observedValue', 'evaluatedAtUtc', 'evidence'];
  let passed = null;
  let evaluatedAt = null;
  let evidenceReference = null;
  if (!exactKeys(hypothesis.result, resultFields, `${label}.result`, errors)) {
    reasons.push('result-invalid');
  } else {
    passed = hypothesis.result.passed;
    if (passed !== null && typeof passed !== 'boolean') {
      errors.push(`${label}.result.passed must be boolean or null`);
    }
    evaluatedAt = isoInstant(
      hypothesis.result.evaluatedAtUtc,
      `${label}.result.evaluatedAtUtc`,
      errors
    );
    evidenceReference = await validateEvidenceReference(
      hypothesis.result.evidence,
      `${label}.result.evidence`,
      repoRoot,
      errors
    );
    const observedValue = hypothesis.result.observedValue;
    const entirelyNull = passed === null
      && observedValue === null
      && hypothesis.result.evaluatedAtUtc === null
      && hypothesis.result.evidence === null;
    if (entirelyNull) {
      reasons.push('result-incomplete');
    } else if (passed === null || observedValue === null || !evaluatedAt || !evidenceReference) {
      errors.push(`${label}.result must be complete or entirely null`);
      reasons.push('result-invalid');
    }
    if (declaredAt && evaluatedAt && evaluatedAt.getTime() < declaredAt.getTime()) {
      reasons.push('evaluated-before-declaration');
    }
    if (evidenceReference) {
      retainedEvidencePaths.push(evidenceReference.path);
      const validation = validateRuntimeOwnershipHypothesisEvidence(
        evidenceReference.receipt,
        {
          ...expectedIdentity,
          decisionId,
          axis,
          metric: hypothesis.metric,
          controlMetric: hypothesis.controlMetric,
          operator,
          thresholdValue,
        }
      );
      for (const error of validation.errors) errors.push(`${label}.result.evidence: ${error}`);
      if (validation.passed !== null && passed !== validation.passed) {
        errors.push(`${label}.result.passed does not match semantic hypothesis evidence`);
      }
      if (
        validation.observedValue !== null
        && !Object.is(observedValue, validation.observedValue)
      ) {
        errors.push(`${label}.result.observedValue does not match semantic hypothesis evidence`);
      }
      if (
        validation.evaluatedAt
        && evaluatedAt
        && validation.evaluatedAt.getTime() !== evaluatedAt.getTime()
      ) {
        errors.push(`${label}.result.evaluatedAtUtc does not match semantic hypothesis evidence`);
      }
    }
  }
  return {
    axis,
    complete: reasons.length === 0 && errors.length === initialErrorCount,
    passed,
    declaredAt,
    evaluatedAt,
    reasons,
  };
}

async function validateDecision(decision, context) {
  const { errors, repoRoot, now, maxAgeDays, seenIds } = context;
  const initialErrorCount = errors.length;
  const fields = [
    'id',
    'workload',
    'logicalModelId',
    'manifestVariantId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'sourceProviderId',
    'sourceArtifactId',
    'sourceExecutionId',
    'incumbentProviderId',
    'incumbentArtifactId',
    'incumbentExecutionId',
    'correctnessClass',
    'harnessRevision',
    'environmentFingerprint',
    'hypotheses',
    'disposition',
    'decisionRationale',
    'qualifiedAtUtc',
    'expiresAtUtc',
    'evidence',
    'claimAllowed',
    'blockers',
  ];
  const id = normalizeText(decision?.id) || '<missing-decision>';
  if (!exactKeys(decision, fields, id, errors)) return null;
  const reasons = [];
  if (!ID_PATTERN.test(id)) errors.push(`${id}: id must be lowercase kebab-case`);
  if (seenIds.has(id)) errors.push(`${id}: duplicate decision id`);
  seenIds.add(id);
  if (!REQUIRED_WORKLOADS.includes(decision.workload)) errors.push(`${id}: workload is not recognized`);
  const identityFields = [
    'logicalModelId',
    'sourceProviderId',
    'sourceArtifactId',
    'incumbentProviderId',
    'incumbentArtifactId',
  ];
  const identity = Object.fromEntries(identityFields.map((field) => [
    field,
    nullableString(decision[field], `${id}.${field}`, errors),
  ]));
  for (const field of identityFields) {
    if (!identity[field]) reasons.push(`${field}-missing`);
  }
  identity.manifestVariantId = nullableString(
    decision.manifestVariantId,
    `${id}.manifestVariantId`,
    errors
  );
  identity.resolvedArtifactVariantId = nullableSha256(
    decision.resolvedArtifactVariantId,
    `${id}.resolvedArtifactVariantId`,
    errors
  );
  identity.resolvedExecutionId = nullableSha256(
    decision.resolvedExecutionId,
    `${id}.resolvedExecutionId`,
    errors
  );
  identity.sourceExecutionId = nullableSha256(
    decision.sourceExecutionId,
    `${id}.sourceExecutionId`,
    errors
  );
  identity.incumbentExecutionId = nullableSha256(
    decision.incumbentExecutionId,
    `${id}.incumbentExecutionId`,
    errors
  );
  for (const field of [
    'manifestVariantId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'sourceExecutionId',
    'incumbentExecutionId',
  ]) {
    if (!identity[field]) reasons.push(`${field}-missing`);
  }
  if (identity.incumbentProviderId === 'doppler') errors.push(`${id}: incumbentProviderId must not be doppler`);
  if (identity.sourceProviderId === 'doppler') errors.push(`${id}: sourceProviderId must not be doppler`);
  if (
    identity.sourceProviderId
    && identity.incumbentProviderId
    && identity.sourceProviderId === identity.incumbentProviderId
  ) {
    errors.push(`${id}: sourceProviderId and incumbentProviderId must identify distinct controls`);
  }
  if (!CORRECTNESS_CLASSES.has(decision.correctnessClass)) {
    errors.push(`${id}: correctnessClass is not recognized`);
  }
  const harnessRevision = nullableRevision(
    decision.harnessRevision,
    `${id}.harnessRevision`,
    errors
  );
  const environmentFingerprint = nullableSha256(
    decision.environmentFingerprint,
    `${id}.environmentFingerprint`,
    errors
  );
  if (!harnessRevision || !environmentFingerprint) {
    reasons.push('decision-evidence-identity-incomplete');
  }
  if (!Array.isArray(decision.hypotheses)) errors.push(`${id}.hypotheses must be an array`);
  const seenAxes = new Set();
  const hypotheses = [];
  const retainedEvidencePaths = [];
  const expectedIdentity = {
    workload: decision.workload,
    logicalModelId: identity.logicalModelId,
    sourceExecutionId: identity.sourceExecutionId,
    incumbentExecutionId: identity.incumbentExecutionId,
    resolvedArtifactVariantId: identity.resolvedArtifactVariantId,
    resolvedExecutionId: identity.resolvedExecutionId,
    harnessRevision,
    environmentFingerprint,
  };
  for (const hypothesis of Array.isArray(decision.hypotheses) ? decision.hypotheses : []) {
    hypotheses.push(await validateHypothesis(hypothesis, {
      errors,
      repoRoot,
      decisionId: id,
      seenAxes,
      expectedIdentity,
      retainedEvidencePaths,
    }));
  }
  if (hypotheses.length === 0) reasons.push('material-advantage-hypothesis-missing');
  if (hypotheses.some((hypothesis) => !hypothesis.complete)) reasons.push('hypothesis-results-incomplete');
  const passedAdvantages = hypotheses.filter((hypothesis) => hypothesis.passed === true).length;
  if (decision.disposition !== null && !DISPOSITIONS.includes(decision.disposition)) {
    errors.push(`${id}: disposition is not recognized`);
  }
  if (decision.disposition === null) {
    reasons.push('disposition-missing');
  } else if (decision.disposition === 'doppler' && passedAdvantages === 0) {
    reasons.push('doppler-disposition-without-material-advantage');
  } else if (decision.disposition === 'incumbent' && passedAdvantages > 0) {
    reasons.push('incumbent-disposition-conflicts-with-material-advantage');
  } else if (decision.disposition === 'dual' && passedAdvantages === 0) {
    reasons.push('dual-disposition-without-material-advantage');
  }
  if (!nullableString(decision.decisionRationale, `${id}.decisionRationale`, errors)) {
    reasons.push('decision-rationale-missing');
  }
  const qualifiedAt = isoInstant(decision.qualifiedAtUtc, `${id}.qualifiedAtUtc`, errors);
  const expiresAt = isoInstant(decision.expiresAtUtc, `${id}.expiresAtUtc`, errors);
  if (!qualifiedAt) {
    reasons.push('qualification-date-missing');
  } else {
    const age = now.getTime() - qualifiedAt.getTime();
    if (age < 0 || age > maxAgeDays * DAY_MS) reasons.push('qualification-stale-or-future');
    if (hypotheses.some(
      (hypothesis) => hypothesis.evaluatedAt?.getTime() > qualifiedAt.getTime()
    )) {
      reasons.push('qualification-predates-hypothesis-evidence');
    }
  }
  if (!expiresAt || expiresAt.getTime() <= now.getTime()) reasons.push('qualification-expired-or-missing');
  const evidence = {};
  if (exactKeys(decision.evidence, EVIDENCE_FIELDS, `${id}.evidence`, errors)) {
    await Promise.all(EVIDENCE_FIELDS.map(async (field) => {
      evidence[field] = await validateEvidenceReference(
        decision.evidence[field],
        `${id}.evidence.${field}`,
        repoRoot,
        errors
      );
      if (evidence[field]) retainedEvidencePaths.push(evidence[field].path);
    }));
  }
  const missingEvidence = EVIDENCE_FIELDS.filter((field) => !evidence[field]);
  if (missingEvidence.length > 0) reasons.push('evidence-incomplete');
  if (!evidence.promotion) reasons.push('disposition-promotion-evidence-missing');
  const externalStatuses = {};
  const externalExecutions = [
    {
      role: 'source',
      evidenceField: 'sourceExecution',
      providerId: identity.sourceProviderId,
      artifactId: identity.sourceArtifactId,
      executionId: identity.sourceExecutionId,
    },
    {
      role: 'incumbent',
      evidenceField: 'incumbentExecution',
      providerId: identity.incumbentProviderId,
      artifactId: identity.incumbentArtifactId,
      executionId: identity.incumbentExecutionId,
    },
  ];
  for (const execution of externalExecutions) {
    if (!evidence[execution.evidenceField]) continue;
    const receipt = evidence[execution.evidenceField].receipt;
    const result = validateRuntimeOwnershipExecutionEvidence(receipt, {
      role: execution.role,
      providerId: execution.providerId,
      artifactId: execution.artifactId,
      workload: decision.workload,
      logicalModelId: identity.logicalModelId,
    });
    externalStatuses[execution.role] = result.status;
    for (const error of result.errors) {
      errors.push(`${id}.evidence.${execution.evidenceField}: ${error}`);
    }
    pushUnique(reasons, result.reasons);
    if (qualifiedAt && result.completedAt?.getTime() > qualifiedAt.getTime()) {
      pushUnique(reasons, ['qualification-predates-execution-evidence']);
    }
    if (
      execution.executionId
      && result.evidenceId
      && execution.executionId !== result.evidenceId
    ) {
      errors.push(
        `${id}.${execution.role}ExecutionId does not match canonical `
        + `${execution.evidenceField} evidence identity`
      );
    }
  }
  if (evidence.dopplerExecution) {
    const receipt = evidence.dopplerExecution.receipt;
    if (receipt) {
      const result = validateDopplerRuntimeOwnershipReceipt(receipt, {
        logicalModelId: identity.logicalModelId,
        resolvedArtifactVariantId: identity.resolvedArtifactVariantId,
        resolvedExecutionId: identity.resolvedExecutionId,
      });
      for (const error of result.errors) {
        errors.push(`${id}.evidence.dopplerExecution: ${error}`);
      }
      pushUnique(reasons, result.reasons);
      if (qualifiedAt && result.timestamp?.getTime() > qualifiedAt.getTime()) {
        pushUnique(reasons, ['qualification-predates-execution-evidence']);
      }
    }
  }
  const dimensionResults = {};
  for (const evidenceClass of RUNTIME_OWNERSHIP_DIMENSION_CLASSES) {
    const reference = evidence[evidenceClass];
    if (!reference) continue;
    const result = validateRuntimeOwnershipDimensionEvidence(reference.receipt, {
      ...expectedIdentity,
      decisionId: id,
      evidenceClass,
    });
    dimensionResults[evidenceClass] = result;
    for (const error of result.errors) {
      errors.push(`${id}.evidence.${evidenceClass}: ${error}`);
    }
    pushUnique(reasons, result.reasons);
    if (qualifiedAt && result.capturedAt?.getTime() > qualifiedAt.getTime()) {
      pushUnique(reasons, ['qualification-predates-dimension-evidence']);
    }
  }
  const unsupportedOperationPassed = hypotheses.some(
    (hypothesis) => hypothesis.axis === 'unsupported-operation' && hypothesis.passed === true
  );
  if (unsupportedOperationPassed) {
    const correctness = dimensionResults.correctness;
    const unsupportedOperationConfirmed = externalStatuses.incumbent === 'failed'
      && correctness?.passed === true
      && correctness.observations?.incumbentAcceptable === false
      && correctness.observations?.dopplerAcceptable === true;
    if (externalStatuses.incumbent !== 'failed') {
      errors.push(`${id}: unsupported-operation advantage requires a failed incumbent execution`);
    }
    if (unsupportedOperationConfirmed) {
      const reasonIndex = reasons.indexOf('incumbent-execution-not-passed');
      if (reasonIndex >= 0) reasons.splice(reasonIndex, 1);
    } else {
      pushUnique(reasons, ['unsupported-operation-not-confirmed-by-correctness']);
    }
  }
  if (evidence.promotion?.receipt) {
    const evidenceSetReferences = Object.fromEntries(
      NON_PROMOTION_EVIDENCE_FIELDS.map((field) => [
        field,
        evidence[field]
          ? { path: evidence[field].path, digest: evidence[field].digest }
          : null,
      ])
    );
    const result = validateRuntimeOwnershipPromotionEvidence(
      evidence.promotion.receipt,
      {
        decisionId: id,
        workload: decision.workload,
        logicalModelId: identity.logicalModelId,
        manifestVariantId: identity.manifestVariantId,
        resolvedArtifactVariantId: identity.resolvedArtifactVariantId,
        resolvedExecutionId: identity.resolvedExecutionId,
        sourceProviderId: identity.sourceProviderId,
        sourceArtifactId: identity.sourceArtifactId,
        sourceExecutionId: identity.sourceExecutionId,
        incumbentProviderId: identity.incumbentProviderId,
        incumbentArtifactId: identity.incumbentArtifactId,
        incumbentExecutionId: identity.incumbentExecutionId,
        correctnessClass: decision.correctnessClass,
        harnessRevision,
        environmentFingerprint,
        disposition: decision.disposition,
        decisionRationale: decision.decisionRationale,
        evidenceSetDigest: computeRuntimeOwnershipEvidenceSetDigest(evidenceSetReferences),
        hypothesisSetDigest: computeRuntimeOwnershipHypothesisSetDigest(decision.hypotheses),
        qualifiedAtUtc: qualifiedAt?.toISOString() ?? null,
        expiresAtUtc: expiresAt?.toISOString() ?? null,
      }
    );
    for (const error of result.errors) errors.push(`${id}.evidence.promotion: ${error}`);
    if (result.errors.length > 0) reasons.push('disposition-promotion-evidence-invalid');
  }
  if (new Set(retainedEvidencePaths).size !== retainedEvidencePaths.length) {
    errors.push(`${id}: retained evidence paths must be distinct`);
  }
  const blockers = validateClaimState(decision, id, reasons, errors);
  return {
    id,
    workload: decision.workload,
    logicalModelId: identity.logicalModelId,
    manifestVariantId: identity.manifestVariantId,
    resolvedArtifactVariantId: identity.resolvedArtifactVariantId,
    resolvedExecutionId: identity.resolvedExecutionId,
    sourceProviderId: identity.sourceProviderId,
    sourceArtifactId: identity.sourceArtifactId,
    sourceExecutionId: identity.sourceExecutionId,
    incumbentProviderId: identity.incumbentProviderId,
    incumbentArtifactId: identity.incumbentArtifactId,
    incumbentExecutionId: identity.incumbentExecutionId,
    correctnessClass: decision.correctnessClass,
    harnessRevision,
    environmentFingerprint,
    hypothesisAxes: Array.from(seenAxes),
    disposition: decision.disposition,
    claimAllowed: decision.claimAllowed,
    qualified: decision.claimAllowed === true
      && reasons.length === 0
      && errors.length === initialErrorCount,
    passedAdvantages,
    missingEvidence,
    blockers,
    reasons,
  };
}

export async function validateRuntimeOwnershipDecisions(policy, options = {}) {
  const errors = [];
  const repoRoot = options.repoRoot || REPO_ROOT;
  const now = options.now instanceof Date ? options.now : new Date();
  const fields = [
    '$schema',
    'schemaVersion',
    'source',
    'goalIds',
    'minimumQualifiedDecisions',
    'requiredWorkloads',
    'qualificationMaxAgeDays',
    'allowedDispositions',
    'materialAdvantageAxes',
    'requiredEvidenceFields',
    'decisions',
  ];
  if (!exactKeys(policy, fields, 'runtime ownership policy', errors)) {
    return {
      errors,
      decisions: [],
      qualifiedDecisions: 0,
      candidateDecisions: 0,
      candidateWorkloads: [],
      missingWorkloads: [...REQUIRED_WORKLOADS],
      gateSatisfied: false,
    };
  }
  if (policy.$schema !== 'schema/runtime-ownership-decisions.schema.json') {
    errors.push('runtime ownership policy $schema is invalid');
  }
  if (policy.schemaVersion !== 5) errors.push('runtime ownership policy schemaVersion must be 5');
  if (policy.source !== 'doppler') errors.push('runtime ownership policy source must be "doppler"');
  const fixedArrays = [
    ['goalIds', GOAL_IDS],
    ['requiredWorkloads', REQUIRED_WORKLOADS],
    ['allowedDispositions', DISPOSITIONS],
    ['materialAdvantageAxes', ADVANTAGE_AXES],
    ['requiredEvidenceFields', EVIDENCE_FIELDS],
  ];
  for (const [field, expected] of fixedArrays) {
    const actual = stringArray(policy[field], `runtime ownership policy ${field}`, errors);
    if (!sameMembers(actual, expected)) errors.push(`runtime ownership policy ${field} is invalid`);
  }
  if (policy.minimumQualifiedDecisions !== 3) {
    errors.push('runtime ownership policy minimumQualifiedDecisions must be 3');
  }
  if (
    !Number.isInteger(policy.qualificationMaxAgeDays)
    || policy.qualificationMaxAgeDays < 1
    || policy.qualificationMaxAgeDays > 365
  ) {
    errors.push('runtime ownership policy qualificationMaxAgeDays must be in [1, 365]');
  }
  if (!Array.isArray(policy.decisions)) errors.push('runtime ownership policy decisions must be an array');
  const seenIds = new Set();
  const decisions = [];
  for (const decision of Array.isArray(policy.decisions) ? policy.decisions : []) {
    const result = await validateDecision(decision, {
      errors,
      repoRoot,
      now,
      maxAgeDays: policy.qualificationMaxAgeDays,
      seenIds,
    });
    if (result) decisions.push(result);
  }
  const qualified = decisions.filter((decision) => decision.qualified);
  const candidates = decisions.filter((decision) => decision.claimAllowed === false);
  const missingWorkloads = REQUIRED_WORKLOADS.filter(
    (workload) => !qualified.some((decision) => decision.workload === workload)
  );
  return {
    errors,
    decisions,
    qualifiedDecisions: qualified.length,
    candidateDecisions: candidates.length,
    candidateWorkloads: Array.from(new Set(candidates.map((decision) => decision.workload))),
    missingWorkloads,
    gateSatisfied: errors.length === 0
      && qualified.length >= policy.minimumQualifiedDecisions
      && missingWorkloads.length === 0,
  };
}

export async function buildRuntimeOwnershipDecisionReport(options = {}) {
  const policyPath = options.policyPath || DEFAULT_POLICY_PATH;
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  const result = await validateRuntimeOwnershipDecisions(policy, options);
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
  const report = await buildRuntimeOwnershipDecisionReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    const status = report.gateSatisfied ? 'satisfied' : 'incomplete';
    console.log(
      `runtime-ownership: contract ok, gate ${status} `
      + `(${report.qualifiedDecisions}/3 qualified; ${report.candidateDecisions} candidates; `
      + `missing ${report.missingWorkloads.join(', ') || 'none'})`
    );
  } else {
    for (const error of report.errors) console.error(`runtime-ownership: ${error}`);
  }
  if (!report.ok) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
