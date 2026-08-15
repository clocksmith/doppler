#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

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
  'correctness',
  'taskQuality',
  'usability',
  'memory',
  'endToEndPerformance',
  'diagnosticDepth',
  'distributionCost',
  'integrationBurden',
  'providerRisk',
]);
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
const DAY_MS = 24 * 60 * 60 * 1000;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
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

async function evidencePath(value, label, repoRoot, errors) {
  if (value === null) return null;
  if (!isRepoPath(value)) {
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

function thresholdMatches(operator, thresholdValue, observedValue) {
  if (operator === 'greater-than-or-equal') return observedValue >= thresholdValue;
  if (operator === 'less-than-or-equal') return observedValue <= thresholdValue;
  return null;
}

async function validateHypothesis(hypothesis, context) {
  const { errors, repoRoot, decisionId, seenAxes } = context;
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
  const resultFields = ['passed', 'observedValue', 'evaluatedAtUtc', 'evidencePath'];
  let passed = null;
  let evaluatedAt = null;
  let resultPath = null;
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
    resultPath = await evidencePath(
      hypothesis.result.evidencePath,
      `${label}.result.evidencePath`,
      repoRoot,
      errors
    );
    if (passed === null || !evaluatedAt || !resultPath) reasons.push('result-incomplete');
    if (declaredAt && evaluatedAt && evaluatedAt.getTime() < declaredAt.getTime()) {
      reasons.push('evaluated-before-declaration');
    }
    if (operator !== 'pass' && passed !== null) {
      const observed = hypothesis.result.observedValue;
      if (!Number.isFinite(observed)) {
        errors.push(`${label}: numeric threshold requires a finite observedValue`);
      } else if (thresholdMatches(operator, thresholdValue, observed) !== passed) {
        errors.push(`${label}: passed does not match the declared threshold and observedValue`);
      }
    }
    if (operator === 'pass' && passed !== null && hypothesis.result.observedValue === null) {
      errors.push(`${label}: qualitative pass result requires observedValue`);
    }
  }
  return {
    complete: reasons.length === 0,
    passed,
    declaredAt,
    evaluatedAt,
    reasons,
  };
}

async function validateDecision(decision, context) {
  const { errors, repoRoot, now, maxAgeDays, seenIds } = context;
  const fields = [
    'id',
    'workload',
    'logicalModelId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'sourceProviderId',
    'sourceArtifactId',
    'sourceExecutionId',
    'incumbentProviderId',
    'incumbentArtifactId',
    'incumbentExecutionId',
    'correctnessClass',
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
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'sourceProviderId',
    'sourceArtifactId',
    'sourceExecutionId',
    'incumbentProviderId',
    'incumbentArtifactId',
    'incumbentExecutionId',
  ];
  const identity = Object.fromEntries(identityFields.map((field) => [
    field,
    nullableString(decision[field], `${id}.${field}`, errors),
  ]));
  for (const field of identityFields) {
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
  if (!Array.isArray(decision.hypotheses)) errors.push(`${id}.hypotheses must be an array`);
  const seenAxes = new Set();
  const hypotheses = [];
  for (const hypothesis of Array.isArray(decision.hypotheses) ? decision.hypotheses : []) {
    hypotheses.push(await validateHypothesis(hypothesis, { errors, repoRoot, decisionId: id, seenAxes }));
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
      evidence[field] = await evidencePath(
        decision.evidence[field],
        `${id}.evidence.${field}`,
        repoRoot,
        errors
      );
    }));
  }
  const missingEvidence = EVIDENCE_FIELDS.filter((field) => !evidence[field]);
  if (missingEvidence.length > 0) reasons.push('evidence-incomplete');
  const blockers = validateClaimState(decision, id, reasons, errors);
  return {
    id,
    workload: decision.workload,
    logicalModelId: identity.logicalModelId,
    resolvedArtifactVariantId: identity.resolvedArtifactVariantId,
    resolvedExecutionId: identity.resolvedExecutionId,
    sourceProviderId: identity.sourceProviderId,
    sourceArtifactId: identity.sourceArtifactId,
    sourceExecutionId: identity.sourceExecutionId,
    incumbentProviderId: identity.incumbentProviderId,
    incumbentArtifactId: identity.incumbentArtifactId,
    incumbentExecutionId: identity.incumbentExecutionId,
    correctnessClass: decision.correctnessClass,
    hypothesisAxes: Array.from(seenAxes),
    disposition: decision.disposition,
    claimAllowed: decision.claimAllowed,
    qualified: decision.claimAllowed === true && reasons.length === 0,
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
  if (policy.schemaVersion !== 1) errors.push('runtime ownership policy schemaVersion must be 1');
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
