#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { validateRevocationRegistry, findResolutionRevocation } from '../src/config/revocation-policy.js';
import { computeCanonicalSha256 } from '../src/utils/canonical-hash.js';
import {
  validateRuntimePromotionActivationEvidence,
  validateRuntimePromotionDecisionEvidence,
} from './lib/runtime-promotion-monitoring-evidence.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'runtime-promotion-monitoring.json');
const DEFAULT_REVOCATION_PATH = path.join(REPO_ROOT, 'src', 'config', 'revocation-registry.json');
const CHANGE_CLASSES = Object.freeze([
  'scheduling-allocation-cache',
  'numerical-kernel',
  'precision-quantization',
  'model-artifact',
  'adapter',
  'provider-integration',
]);
const SCOPE_FIELDS = Object.freeze([
  'modelId',
  'artifactVariantId',
  'executionId',
  'providerId',
  'environmentFingerprint',
  'workloadId',
]);
const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function exactObject(value, fields, label, errors) {
  if (!isPlainObject(value)) {
    errors.push(`${label} must be an object`);
    return false;
  }
  const expected = new Set(fields);
  for (const field of Object.keys(value)) {
    if (!expected.has(field)) errors.push(`${label}.${field} is not supported`);
  }
  for (const field of fields) {
    if (!Object.prototype.hasOwnProperty.call(value, field)) errors.push(`${label}.${field} is required`);
  }
  return true;
}

function text(value, label, errors) {
  const normalized = typeof value === 'string' ? value.trim() : '';
  if (!normalized) errors.push(`${label} must be a non-empty string`);
  return normalized || null;
}

function digest(value, label, errors) {
  const normalized = text(value, label, errors);
  if (normalized && !DIGEST_PATTERN.test(normalized)) errors.push(`${label} must be a SHA-256 identity`);
  return normalized;
}

function instant(value, label, errors, nullable = false) {
  if (nullable && value === null) return null;
  const normalized = text(value, label, errors);
  if (!normalized) return null;
  const parsed = new Date(normalized);
  if (!Number.isFinite(parsed.getTime()) || parsed.toISOString() !== normalized) {
    errors.push(`${label} must be an ISO instant${nullable ? ' or null' : ''}`);
    return null;
  }
  return parsed;
}

function uniqueStrings(value, label, errors, minimum = 1) {
  if (!Array.isArray(value)) {
    errors.push(`${label} must be an array`);
    return [];
  }
  const normalized = value.map((entry, index) => text(entry, `${label}[${index}]`, errors)).filter(Boolean);
  if (normalized.length < minimum) errors.push(`${label} requires at least ${minimum} entries`);
  if (new Set(normalized).size !== normalized.length) errors.push(`${label} must not contain duplicates`);
  return normalized;
}

function sameMembers(left, right) {
  const a = [...left].sort();
  const b = [...right].sort();
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function sameScope(left, right) {
  return SCOPE_FIELDS.every((field) => left[field] === right[field]);
}

function validateScope(value, label, errors) {
  exactObject(value, SCOPE_FIELDS, label, errors);
  const scope = Object.fromEntries(SCOPE_FIELDS.map((field) => [field, text(value?.[field], `${label}.${field}`, errors)]));
  digest(scope.artifactVariantId, `${label}.artifactVariantId`, errors);
  digest(scope.executionId, `${label}.executionId`, errors);
  return scope;
}

function validateChecks(value, expectedIds, label, errors) {
  if (!Array.isArray(value)) {
    errors.push(`${label} must be an array`);
    return [];
  }
  const ids = [];
  for (let index = 0; index < value.length; index += 1) {
    const entry = value[index];
    const entryLabel = `${label}[${index}]`;
    exactObject(entry, ['id', 'passed'], entryLabel, errors);
    const id = text(entry?.id, `${entryLabel}.id`, errors);
    if (id) ids.push(id);
    if (typeof entry?.passed !== 'boolean') errors.push(`${entryLabel}.passed must be boolean`);
  }
  if (new Set(ids).size !== ids.length) errors.push(`${label} contains duplicate ids`);
  if (!sameMembers(ids, expectedIds)) errors.push(`${label} ids must match the frozen plan`);
  return value;
}

async function repoPath(value, label, repoRoot, errors, pathExists) {
  const normalized = text(value, label, errors);
  if (!normalized) return null;
  if (path.isAbsolute(normalized) || normalized.includes('\\') || normalized.split('/').includes('..')) {
    errors.push(`${label} must be repo-relative`);
    return null;
  }
  if (!await pathExists(path.join(repoRoot, normalized))) errors.push(`${label} does not exist: ${normalized}`);
  return normalized;
}

async function validateEvidenceReference(value, label, context) {
  const { errors, repoRoot, loadJson, pathExists } = context;
  if (!exactObject(value, ['path', 'digest'], label, errors)) return null;
  const evidencePath = await repoPath(value.path, `${label}.path`, repoRoot, errors, pathExists);
  const evidenceDigest = digest(value.digest, `${label}.digest`, errors);
  if (!evidencePath) return null;
  let receipt;
  try {
    receipt = await loadJson(path.join(repoRoot, evidencePath));
  } catch (error) {
    errors.push(`${label}.path could not be read as JSON: ${error.message}`);
    return null;
  }
  if (evidenceDigest && computeCanonicalSha256(receipt) !== evidenceDigest) {
    errors.push(`${label}.digest does not match canonical JSON evidence`);
  }
  return { path: evidencePath, receipt };
}

function degradationPercent(metric, observed) {
  const scale = Math.abs(metric.baseline);
  if (metric.direction === 'maximize') return ((metric.baseline - observed) / scale) * 100;
  return ((observed - metric.baseline) / scale) * 100;
}

function receiptCoreHash(receipt) {
  const { receiptHash, ...core } = receipt;
  return computeCanonicalSha256(core);
}

async function validateOptimizationReceipt(promotion, label, context) {
  const { errors, repoRoot, loadJson, pathExists } = context;
  const receiptPath = await repoPath(
    promotion.optimizationReceiptPath,
    `${label}.optimizationReceiptPath`,
    repoRoot,
    errors,
    pathExists
  );
  if (!receiptPath) return null;
  let receipt;
  try {
    receipt = await loadJson(path.join(repoRoot, receiptPath));
  } catch (error) {
    errors.push(`${label}.optimizationReceiptPath could not be read: ${error.message}`);
    return null;
  }
  if (receipt?.schema !== 'doppler.runtime-optimization-receipt/v1') {
    errors.push(`${label}: optimization receipt schema is not supported`);
    return receipt;
  }
  const declaredHash = digest(promotion.optimizationReceiptHash, `${label}.optimizationReceiptHash`, errors);
  if (receipt.receiptHash !== declaredHash || receiptCoreHash(receipt) !== declaredHash) {
    errors.push(`${label}: optimization receipt hash mismatch`);
  }
  if (receipt.candidateId !== promotion.candidateId) errors.push(`${label}: candidateId does not match receipt`);
  if (receipt.candidateHash !== promotion.candidateHash) errors.push(`${label}: candidateHash does not match receipt`);
  if (receipt.campaign?.changeClass !== promotion.changeClass) errors.push(`${label}: changeClass does not match receipt`);
  if (receipt.decision?.accepted !== true || receipt.promotion?.recommended !== true) {
    errors.push(`${label}: optimization receipt must recommend an accepted candidate`);
  }
  if (receipt.promotion?.runtimeMutationApplied !== false) {
    errors.push(`${label}: evaluator must not have applied a runtime mutation`);
  }
  if (!sameMembers(receipt.promotion?.requiredStages ?? [], ['shadow', 'canary'])) {
    errors.push(`${label}: optimization receipt must require shadow and canary`);
  }
  return receipt;
}

async function validatePromotion(promotion, index, context) {
  const { errors, revocations } = context;
  const label = `promotion[${index}]`;
  exactObject(promotion, [
    'id',
    'optimizationReceiptPath',
    'optimizationReceiptHash',
    'activationEvidence',
    'candidateId',
    'candidateHash',
    'changeClass',
    'activatedAtUtc',
    'scope',
    'rollbackTarget',
    'plan',
    'observations',
    'decision',
  ], label, errors);
  const id = text(promotion?.id, `${label}.id`, errors);
  const retainedEvidencePaths = [promotion?.optimizationReceiptPath].filter(Boolean);
  if (id && !ID_PATTERN.test(id)) errors.push(`${label}.id must be lowercase kebab-case`);
  text(promotion?.candidateId, `${label}.candidateId`, errors);
  digest(promotion?.candidateHash, `${label}.candidateHash`, errors);
  if (!CHANGE_CLASSES.includes(promotion?.changeClass)) errors.push(`${label}.changeClass is not supported`);
  const activatedAt = instant(promotion?.activatedAtUtc, `${label}.activatedAtUtc`, errors);
  const scope = validateScope(promotion?.scope, `${label}.scope`, errors);
  const activationReference = await validateEvidenceReference(
    promotion?.activationEvidence,
    `${label}.activationEvidence`,
    context
  );
  if (activationReference?.path) retainedEvidencePaths.push(activationReference.path);
  if (activationReference?.receipt) {
    const activation = validateRuntimePromotionActivationEvidence(
      activationReference.receipt,
      {
        promotionId: id,
        candidateId: promotion?.candidateId,
        candidateHash: promotion?.candidateHash,
        activatedAtUtc: activatedAt?.toISOString() ?? null,
        scope,
      }
    );
    for (const error of activation.errors) {
      errors.push(`${label}.activationEvidence: ${error}`);
    }
  }

  exactObject(
    promotion?.rollbackTarget,
    ['kind', 'id', 'digest', 'knownSafe', 'evidence'],
    `${label}.rollbackTarget`,
    errors
  );
  if (!['runtime-profile', 'kernel-variant', 'execution-graph'].includes(promotion?.rollbackTarget?.kind)) {
    errors.push(`${label}.rollbackTarget.kind is not supported`);
  }
  text(promotion?.rollbackTarget?.id, `${label}.rollbackTarget.id`, errors);
  digest(promotion?.rollbackTarget?.digest, `${label}.rollbackTarget.digest`, errors);
  if (promotion?.rollbackTarget?.knownSafe !== true) errors.push(`${label}.rollbackTarget.knownSafe must be true`);
  const rollbackEvidenceReference = await validateEvidenceReference(
    promotion?.rollbackTarget?.evidence,
    `${label}.rollbackTarget.evidence`,
    context
  );
  if (rollbackEvidenceReference?.path) {
    retainedEvidencePaths.push(rollbackEvidenceReference.path);
  }

  const plan = promotion?.plan;
  exactObject(plan, [
    'owner',
    'declaredAtUtc',
    'primaryMetric',
    'controlMetricIds',
    'neighborWorkloadIds',
    'minimumObservations',
    'revocationConditions',
  ], `${label}.plan`, errors);
  text(plan?.owner, `${label}.plan.owner`, errors);
  const declaredAt = instant(plan?.declaredAtUtc, `${label}.plan.declaredAtUtc`, errors);
  if (declaredAt && activatedAt && declaredAt.getTime() > activatedAt.getTime()) {
    errors.push(`${label}: monitor plan must be declared before activation`);
  }
  exactObject(plan?.primaryMetric, ['id', 'direction', 'baseline', 'maxDegradationPercent'], `${label}.plan.primaryMetric`, errors);
  const metric = plan?.primaryMetric ?? {};
  text(metric.id, `${label}.plan.primaryMetric.id`, errors);
  if (!['maximize', 'minimize'].includes(metric.direction)) errors.push(`${label}.plan.primaryMetric.direction is not supported`);
  if (!Number.isFinite(metric.baseline) || metric.baseline === 0) errors.push(`${label}.plan.primaryMetric.baseline must be finite and non-zero`);
  if (!Number.isFinite(metric.maxDegradationPercent) || metric.maxDegradationPercent < 0) {
    errors.push(`${label}.plan.primaryMetric.maxDegradationPercent must be non-negative`);
  }
  const controlIds = uniqueStrings(plan?.controlMetricIds, `${label}.plan.controlMetricIds`, errors);
  const neighborIds = uniqueStrings(plan?.neighborWorkloadIds, `${label}.plan.neighborWorkloadIds`, errors);
  if (!Number.isInteger(plan?.minimumObservations) || plan.minimumObservations < 2) {
    errors.push(`${label}.plan.minimumObservations must be an integer of at least 2`);
  }
  const revocationConditions = uniqueStrings(plan?.revocationConditions, `${label}.plan.revocationConditions`, errors);
  const receipt = await validateOptimizationReceipt(promotion, label, context);
  if (receipt && !sameMembers(revocationConditions, receipt.promotion?.revocationConditions ?? [])) {
    errors.push(`${label}: revocationConditions must match the frozen optimization receipt`);
  }

  if (!Array.isArray(promotion?.observations)) {
    errors.push(`${label}.observations must be an array`);
    return { id, status: null };
  }
  const observationIds = new Set();
  let latestObservation = null;
  let breach = false;
  for (let observationIndex = 0; observationIndex < promotion.observations.length; observationIndex += 1) {
    const evidenceReference = await validateEvidenceReference(
      promotion.observations[observationIndex],
      `${label}.observations[${observationIndex}]`,
      context
    );
    const observation = evidenceReference?.receipt;
    if (evidenceReference?.path) retainedEvidencePaths.push(evidenceReference.path);
    const observationLabel = `${label}.observations[${observationIndex}]`;
    exactObject(observation, [
      'schema', 'id', 'observedAtUtc', 'scope', 'primaryMetric', 'controls', 'neighbors',
    ], observationLabel, errors);
    if (observation?.schema !== 'doppler.runtime-promotion-observation-evidence/v1') {
      errors.push(`${observationLabel}.schema is not supported`);
    }
    const observationId = text(observation?.id, `${observationLabel}.id`, errors);
    if (observationId && observationIds.has(observationId)) errors.push(`${label}.observations contains duplicate id ${observationId}`);
    if (observationId) observationIds.add(observationId);
    const observedAt = instant(observation?.observedAtUtc, `${observationLabel}.observedAtUtc`, errors);
    if (observedAt && activatedAt && observedAt.getTime() < activatedAt.getTime()) {
      errors.push(`${observationLabel} predates activation`);
    }
    if (observedAt && latestObservation && observedAt.getTime() <= latestObservation.getTime()) {
      errors.push(`${label}.observations must be strictly chronological`);
    }
    if (observedAt) latestObservation = observedAt;
    const observedScope = validateScope(observation?.scope, `${observationLabel}.scope`, errors);
    if (!sameScope(observedScope, scope)) errors.push(`${observationLabel}.scope does not match promotion scope`);
    exactObject(observation?.primaryMetric, ['id', 'value'], `${observationLabel}.primaryMetric`, errors);
    if (observation?.primaryMetric?.id !== metric.id) errors.push(`${observationLabel}.primaryMetric.id does not match plan`);
    const observedValue = observation?.primaryMetric?.value;
    if (!Number.isFinite(observedValue)) {
      errors.push(`${observationLabel}.primaryMetric.value must be finite`);
    } else if (Number.isFinite(metric.baseline) && metric.baseline !== 0
      && degradationPercent(metric, observedValue) > metric.maxDegradationPercent) {
      breach = true;
    }
    const controls = validateChecks(observation?.controls, controlIds, `${observationLabel}.controls`, errors);
    const neighbors = validateChecks(observation?.neighbors, neighborIds, `${observationLabel}.neighbors`, errors);
    if ([...controls, ...neighbors].some((entry) => entry?.passed === false)) breach = true;
  }

  const expectedStatus = breach
    ? 'revoke'
    : promotion.observations.length >= plan.minimumObservations
      ? 'retain'
      : 'monitoring';
  const decision = promotion?.decision;
  exactObject(decision, [
    'status', 'decidedAtUtc', 'reason', 'revocationRecordId', 'authority',
    'runtimeMutationApplied', 'evidence',
  ], `${label}.decision`, errors);
  if (decision?.status !== expectedStatus) errors.push(`${label}.decision.status must be ${expectedStatus}`);
  if (decision?.authority !== 'human') errors.push(`${label}.decision.authority must be human`);
  if (decision?.runtimeMutationApplied !== false) errors.push(`${label}.decision.runtimeMutationApplied must be false`);
  const decisionAt = instant(decision?.decidedAtUtc, `${label}.decision.decidedAtUtc`, errors, true);
  const decisionEvidence = decision?.evidence === null
    ? null
    : await validateEvidenceReference(
      decision?.evidence,
      `${label}.decision.evidence`,
      context
    );
  if (decisionEvidence?.path) retainedEvidencePaths.push(decisionEvidence.path);
  if (expectedStatus === 'monitoring') {
    if (
      decisionAt !== null
      || decision?.reason !== null
      || decision?.revocationRecordId !== null
      || decisionEvidence !== null
    ) {
      errors.push(`${label}: monitoring decision must not claim a terminal outcome`);
    }
  } else {
    text(decision?.reason, `${label}.decision.reason`, errors);
    if (!decisionAt || (latestObservation && decisionAt.getTime() < latestObservation.getTime())) {
      errors.push(`${label}.decision.decidedAtUtc must follow the latest observation`);
    }
    if (expectedStatus === 'retain' && decision?.revocationRecordId !== null) {
      errors.push(`${label}: retain decision must not name a revocation record`);
    }
    if (expectedStatus === 'revoke') {
      const revocationId = text(decision?.revocationRecordId, `${label}.decision.revocationRecordId`, errors);
      const match = findResolutionRevocation({
        modelId: scope.modelId,
        artifactVariantId: scope.artifactVariantId,
      }, revocations);
      if (!match || match.revocation.id !== revocationId) {
        errors.push(`${label}: revoke decision requires a matching active revocation record`);
      }
    }
    if (!decisionEvidence?.receipt) {
      errors.push(`${label}: terminal decision requires semantic decision evidence`);
    } else {
      const result = validateRuntimePromotionDecisionEvidence(decisionEvidence.receipt, {
        promotionId: id,
        candidateId: promotion?.candidateId,
        candidateHash: promotion?.candidateHash,
        scope,
        status: expectedStatus,
        decidedAtUtc: decisionAt?.toISOString() ?? null,
        reason: decision?.reason,
        revocationRecordId: decision?.revocationRecordId,
      });
      for (const error of result.errors) errors.push(`${label}.decision.evidence: ${error}`);
    }
  }
  if (new Set(retainedEvidencePaths).size !== retainedEvidencePaths.length) {
    errors.push(`${label}: retained evidence paths must be distinct`);
  }
  return { id, status: expectedStatus };
}

export async function validateRuntimePromotionMonitoringPolicy(policy, options = {}) {
  const errors = [];
  const repoRoot = options.repoRoot ?? REPO_ROOT;
  const loadJson = options.loadJson ?? (async (filePath) => JSON.parse(await fs.readFile(filePath, 'utf8')));
  const pathExists = options.pathExists ?? (async (filePath) => fs.stat(filePath).then(() => true, () => false));
  let revocations;
  try {
    revocations = validateRevocationRegistry(options.revocationRegistry);
  } catch (error) {
    return { ok: false, promotions: 0, monitoring: 0, retained: 0, revoked: 0, coverageSatisfied: false, errors: [error.message] };
  }
  exactObject(policy, ['$schema', 'schemaVersion', 'source', 'requiredChangeClasses', 'promotions'], 'monitoring policy', errors);
  if (policy?.$schema !== '../../src/config/schema/runtime-promotion-monitoring.schema.json') errors.push('monitoring policy $schema is not recognized');
  if (policy?.schemaVersion !== 3) errors.push('monitoring policy schemaVersion must be 3');
  if (policy?.source !== 'doppler') errors.push('monitoring policy source must be doppler');
  const requiredClasses = uniqueStrings(policy?.requiredChangeClasses, 'monitoring policy requiredChangeClasses', errors);
  if (!sameMembers(requiredClasses, CHANGE_CLASSES)) errors.push('monitoring policy requiredChangeClasses is incomplete');
  if (!Array.isArray(policy?.promotions)) errors.push('monitoring policy promotions must be an array');
  const results = [];
  if (Array.isArray(policy?.promotions)) {
    for (let index = 0; index < policy.promotions.length; index += 1) {
      results.push(await validatePromotion(policy.promotions[index], index, {
        errors,
        repoRoot,
        revocations,
        loadJson,
        pathExists,
      }));
    }
  }
  const ids = results.map((result) => result.id).filter(Boolean);
  if (new Set(ids).size !== ids.length) errors.push('monitoring policy promotion ids must be unique');
  const counts = Object.fromEntries(['monitoring', 'retain', 'revoke'].map((status) => [
    status,
    results.filter((result) => result.status === status).length,
  ]));
  return {
    ok: errors.length === 0,
    promotions: results.length,
    monitoring: counts.monitoring,
    retained: counts.retain,
    revoked: counts.revoke,
    coverageSatisfied: results.length > 0 && counts.monitoring === 0 && errors.length === 0,
    errors,
  };
}

export async function buildRuntimePromotionMonitoringReport({
  repoRoot = REPO_ROOT,
  policyPath = DEFAULT_POLICY_PATH,
  revocationPath = DEFAULT_REVOCATION_PATH,
} = {}) {
  const [policy, revocationRegistry] = await Promise.all([
    fs.readFile(policyPath, 'utf8').then(JSON.parse),
    fs.readFile(revocationPath, 'utf8').then(JSON.parse),
  ]);
  return validateRuntimePromotionMonitoringPolicy(policy, { repoRoot, revocationRegistry });
}

export async function main() {
  const report = await buildRuntimePromotionMonitoringReport();
  if (!report.ok) {
    for (const error of report.errors) console.error(`[promotion-monitoring] ${error}`);
    process.exitCode = 1;
    return;
  }
  console.log(
    `[promotion-monitoring] contract ok, coverage ${report.coverageSatisfied ? 'satisfied' : 'incomplete'} ` +
    `(${report.promotions} promotions; ${report.monitoring} monitoring, ${report.retained} retained, ${report.revoked} revoked)`
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`[promotion-monitoring] ${error.message}`);
    process.exitCode = 1;
  });
}
