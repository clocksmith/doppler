#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { computeCanonicalSha256 } from '../src/utils/canonical-hash.js';
import {
  validateRuntimePromotionMonitoringPolicy,
} from './check-runtime-promotion-monitoring.js';
import {
  validateRuntimePromotionActivationEvidence,
  validateRuntimePromotionDecisionEvidence,
} from './lib/runtime-promotion-monitoring-evidence.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(
  REPO_ROOT,
  'tools',
  'policies',
  'runtime-promotion-monitoring.json'
);
const DEFAULT_REVOCATION_PATH = path.join(
  REPO_ROOT,
  'src',
  'config',
  'revocation-registry.json'
);
const CAPTURE_SCHEMA = 'doppler.runtime-promotion-monitoring-capture/v1';

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
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

async function readEvidencePath(value, label, repoRoot) {
  if (!isRepoRelativePath(value)) throw new Error(`${label} must be a repo-relative path.`);
  const evidencePath = normalizeText(value);
  const receipt = await readJson(path.join(repoRoot, evidencePath), label);
  return { path: evidencePath, digest: computeCanonicalSha256(receipt), receipt };
}

function receiptCoreHash(receipt) {
  const { receiptHash, ...core } = receipt;
  return computeCanonicalSha256(core);
}

function sameJson(left, right) {
  return computeCanonicalSha256(left) === computeCanonicalSha256(right);
}

function assertMonotonicReplacement(existing, replacement) {
  if (existing.decision.status !== 'monitoring') {
    throw new Error(`Terminal promotion ${existing.id} cannot be replaced.`);
  }
  const frozenFields = [
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
  ];
  for (const field of frozenFields) {
    if (!sameJson(existing[field], replacement[field])) {
      throw new Error(`Replacement cannot change frozen promotion field ${field}.`);
    }
  }
  if (replacement.observations.length < existing.observations.length) {
    throw new Error('Replacement must retain all prior monitoring observations.');
  }
  for (let index = 0; index < existing.observations.length; index += 1) {
    if (!sameJson(existing.observations[index], replacement.observations[index])) {
      throw new Error(`Replacement cannot rewrite monitoring observation ${index}.`);
    }
  }
}

function validateOptimizationReceipt(receipt) {
  if (receipt?.schema !== 'doppler.runtime-optimization-receipt/v1') {
    throw new Error('Optimization receipt schema is not supported.');
  }
  if (receipt.receiptHash !== receiptCoreHash(receipt)) {
    throw new Error('Optimization receipt hash does not match canonical receipt content.');
  }
  if (receipt.decision?.accepted !== true || receipt.promotion?.recommended !== true) {
    throw new Error('Optimization receipt must recommend an accepted candidate.');
  }
  if (receipt.promotion?.runtimeMutationApplied !== false) {
    throw new Error('Optimization evaluator must not have applied a runtime mutation.');
  }
  const stages = [...(receipt.promotion?.requiredStages ?? [])].sort();
  if (stages.join(',') !== 'canary,shadow') {
    throw new Error('Optimization receipt must require shadow and canary stages.');
  }
  return receipt;
}

async function validateCapture(capture, context) {
  const { repoRoot, now } = context;
  assertExactKeys(capture, [
    'schema',
    'promotionId',
    'optimizationReceiptPath',
    'activationEvidencePath',
    'rollbackTarget',
    'plan',
    'observationPaths',
    'decisionEvidencePath',
  ], 'monitoring capture');
  if (capture.schema !== CAPTURE_SCHEMA) throw new Error(`capture.schema must be ${CAPTURE_SCHEMA}.`);
  const promotionId = requireText(capture.promotionId, 'capture.promotionId');
  const optimization = await readEvidencePath(
    capture.optimizationReceiptPath,
    'capture.optimizationReceiptPath',
    repoRoot
  );
  validateOptimizationReceipt(optimization.receipt);
  const activation = await readEvidencePath(
    capture.activationEvidencePath,
    'capture.activationEvidencePath',
    repoRoot
  );
  const activationResult = validateRuntimePromotionActivationEvidence(activation.receipt, {
    promotionId,
    candidateId: optimization.receipt.candidateId,
    candidateHash: optimization.receipt.candidateHash,
  });
  if (activationResult.errors.length > 0) {
    throw new Error(`Activation evidence is invalid: ${activationResult.errors.join('; ')}`);
  }
  if (activationResult.activatedAt.getTime() > now.getTime()) {
    throw new Error('Activation evidence must not be in the future.');
  }
  assertExactKeys(capture.rollbackTarget, [
    'kind', 'id', 'digest', 'evidencePath',
  ], 'capture.rollbackTarget');
  const rollbackEvidence = await readEvidencePath(
    capture.rollbackTarget.evidencePath,
    'capture.rollbackTarget.evidencePath',
    repoRoot
  );
  assertExactKeys(capture.plan, [
    'owner', 'declaredAtUtc', 'primaryMetric', 'controlMetricIds',
    'neighborWorkloadIds', 'minimumObservations',
  ], 'capture.plan');
  const planDeclaredAt = parseInstant(capture.plan.declaredAtUtc, 'capture.plan.declaredAtUtc');
  if (planDeclaredAt.getTime() > activationResult.activatedAt.getTime()) {
    throw new Error('Monitoring plan must be declared before activation.');
  }
  if (!Array.isArray(capture.observationPaths)) {
    throw new Error('capture.observationPaths must be an array.');
  }
  const observations = [];
  for (let index = 0; index < capture.observationPaths.length; index += 1) {
    const evidence = await readEvidencePath(
      capture.observationPaths[index],
      `capture.observationPaths[${index}]`,
      repoRoot
    );
    if (evidence.receipt?.schema !== 'doppler.runtime-promotion-observation-evidence/v1') {
      throw new Error(`capture.observationPaths[${index}] is not promotion observation evidence.`);
    }
    const observedAt = parseInstant(
      evidence.receipt.observedAtUtc,
      `capture.observationPaths[${index}].observedAtUtc`
    );
    if (observedAt.getTime() > now.getTime()) {
      throw new Error(`capture.observationPaths[${index}] must not be in the future.`);
    }
    observations.push(evidence);
  }
  let decision = null;
  let decisionResult = null;
  if (capture.decisionEvidencePath !== null) {
    decision = await readEvidencePath(
      capture.decisionEvidencePath,
      'capture.decisionEvidencePath',
      repoRoot
    );
    decisionResult = validateRuntimePromotionDecisionEvidence(decision.receipt, {
      promotionId,
      candidateId: optimization.receipt.candidateId,
      candidateHash: optimization.receipt.candidateHash,
      scope: activationResult.scope,
    });
    if (decisionResult.errors.length > 0) {
      throw new Error(`Decision evidence is invalid: ${decisionResult.errors.join('; ')}`);
    }
    if (decisionResult.decidedAt.getTime() > now.getTime()) {
      throw new Error('Decision evidence must not be in the future.');
    }
  }
  const allPaths = [
    optimization.path,
    activation.path,
    rollbackEvidence.path,
    ...observations.map((entry) => entry.path),
    ...(decision ? [decision.path] : []),
  ];
  if (new Set(allPaths).size !== allPaths.length) {
    throw new Error('Monitoring capture evidence paths must be distinct.');
  }
  return {
    promotionId,
    optimization,
    activation,
    activationResult,
    rollbackEvidence,
    observations,
    decision,
    decisionResult,
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

export async function recordRuntimePromotionMonitoring(options) {
  const repoRoot = path.resolve(options.repoRoot || REPO_ROOT);
  const policyPath = path.resolve(options.policyPath || DEFAULT_POLICY_PATH);
  const revocationPath = path.resolve(options.revocationPath || DEFAULT_REVOCATION_PATH);
  const capturePath = path.resolve(requireText(options.capturePath, 'capturePath'));
  const outputPolicyPath = path.resolve(requireText(options.outputPolicyPath, 'outputPolicyPath'));
  const now = options.now instanceof Date ? options.now : new Date();
  const [policy, revocationRegistry, capture] = await Promise.all([
    readJson(policyPath, 'promotion monitoring policy'),
    readJson(revocationPath, 'revocation registry'),
    readJson(capturePath, 'promotion monitoring capture'),
  ]);
  const current = await validateRuntimePromotionMonitoringPolicy(policy, {
    repoRoot,
    revocationRegistry,
  });
  if (!current.ok) throw new Error(`Promotion monitoring policy is invalid: ${current.errors[0]}`);
  const existingIndex = policy.promotions.findIndex((entry) => entry.id === capture.promotionId);
  if (existingIndex >= 0 && options.replace !== true) {
    throw new Error(`Promotion ${capture.promotionId} already exists; use --replace.`);
  }
  const validated = await validateCapture(capture, { repoRoot, now });
  const optimizationReceipt = validated.optimization.receipt;
  const activation = validated.activationResult;
  const decision = validated.decisionResult;
  const promotion = {
    id: validated.promotionId,
    optimizationReceiptPath: validated.optimization.path,
    optimizationReceiptHash: optimizationReceipt.receiptHash,
    activationEvidence: {
      path: validated.activation.path,
      digest: validated.activation.digest,
    },
    candidateId: optimizationReceipt.candidateId,
    candidateHash: optimizationReceipt.candidateHash,
    changeClass: optimizationReceipt.campaign?.changeClass,
    activatedAtUtc: activation.activatedAt.toISOString(),
    scope: activation.scope,
    rollbackTarget: {
      kind: capture.rollbackTarget.kind,
      id: capture.rollbackTarget.id,
      digest: capture.rollbackTarget.digest,
      knownSafe: true,
      evidence: {
        path: validated.rollbackEvidence.path,
        digest: validated.rollbackEvidence.digest,
      },
    },
    plan: {
      ...capture.plan,
      revocationConditions: optimizationReceipt.promotion.revocationConditions,
    },
    observations: validated.observations.map((entry) => ({
      path: entry.path,
      digest: entry.digest,
    })),
    decision: decision
      ? {
        status: decision.status,
        decidedAtUtc: decision.decidedAt.toISOString(),
        reason: decision.reason,
        revocationRecordId: decision.revocationRecordId,
        authority: 'human',
        runtimeMutationApplied: false,
        evidence: {
          path: validated.decision.path,
          digest: validated.decision.digest,
        },
      }
      : {
        status: 'monitoring',
        decidedAtUtc: null,
        reason: null,
        revocationRecordId: null,
        authority: 'human',
        runtimeMutationApplied: false,
        evidence: null,
      },
  };
  const outputPolicy = structuredClone(policy);
  if (existingIndex >= 0) {
    assertMonotonicReplacement(policy.promotions[existingIndex], promotion);
    outputPolicy.promotions[existingIndex] = promotion;
  } else {
    outputPolicy.promotions.push(promotion);
  }
  const outputReport = await validateRuntimePromotionMonitoringPolicy(outputPolicy, {
    repoRoot,
    revocationRegistry,
  });
  if (!outputReport.ok) {
    throw new Error(`Recorded promotion monitoring policy is invalid: ${outputReport.errors[0]}`);
  }
  await writeJsonAtomically(outputPolicyPath, outputPolicy);
  return {
    promotionId: promotion.id,
    candidateId: promotion.candidateId,
    status: promotion.decision.status,
    coverageSatisfied: outputReport.coverageSatisfied,
    outputPolicyPath,
    runtimeMutationApplied: false,
  };
}

export function parseArgs(argv) {
  const options = {
    policyPath: DEFAULT_POLICY_PATH,
    revocationPath: DEFAULT_REVOCATION_PATH,
    capturePath: '',
    outputPolicyPath: '',
    replace: false,
    apply: false,
    json: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--policy') options.policyPath = argv[++index] || '';
    else if (token === '--revocations') options.revocationPath = argv[++index] || '';
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
  const result = await recordRuntimePromotionMonitoring(options);
  if (options.json) console.log(JSON.stringify(result, null, 2));
  else {
    console.log(
      `promotion-monitoring-record: captured ${result.promotionId}; `
      + `status=${result.status}; runtimeMutationApplied=false; `
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
