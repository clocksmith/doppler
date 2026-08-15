#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { validateRuntimeOwnershipDecisions } from './check-runtime-ownership-decisions.js';
import {
  validateDopplerRuntimeOwnershipReceipt,
  validateRuntimeOwnershipExecutionEvidence,
} from './lib/runtime-ownership-execution-evidence.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(
  REPO_ROOT,
  'benchmarks',
  'vendors',
  'runtime-ownership-decisions.json'
);
const CAPTURE_SCHEMA = 'doppler.runtime-ownership-evaluation-capture/v1';
const DISPOSITIONS = new Set(['incumbent', 'doppler', 'dual']);
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
const RESULT_FIELDS = Object.freeze([
  'axis',
  'passed',
  'observedValue',
  'evaluatedAtUtc',
  'evidencePath',
]);
const DAY_MS = 24 * 60 * 60 * 1000;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function assertExactKeys(value, expectedFields, label) {
  if (!isPlainObject(value)) throw new Error(`${label} must be an object.`);
  const expected = new Set(expectedFields);
  const unsupported = Object.keys(value).filter((field) => !expected.has(field));
  if (unsupported.length > 0) {
    throw new Error(`${label} contains unsupported field ${unsupported[0]}.`);
  }
  const missing = expectedFields.filter((field) => !Object.hasOwn(value, field));
  if (missing.length > 0) throw new Error(`${label}.${missing[0]} is required.`);
}

function requireText(value, label) {
  const normalized = normalizeText(value);
  if (!normalized) throw new Error(`${label} must be a non-empty string.`);
  return normalized;
}

function parseIsoInstant(value, label) {
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

async function requireRepoPath(value, label, repoRoot) {
  if (!isRepoRelativePath(value)) throw new Error(`${label} must be a repo-relative path.`);
  const normalized = normalizeText(value);
  try {
    await fs.stat(path.join(repoRoot, normalized));
  } catch {
    throw new Error(`${label} does not exist: ${normalized}.`);
  }
  return normalized;
}

async function readJson(filePath, label) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (error) {
    throw new Error(`${label} is not readable JSON: ${error.message}`);
  }
}

function hasPriorEvaluation(decision) {
  return Boolean(
    decision.sourceExecutionId
    || decision.incumbentExecutionId
    || decision.resolvedArtifactVariantId
    || decision.resolvedExecutionId
    || decision.disposition
    || decision.qualifiedAtUtc
    || decision.expiresAtUtc
    || decision.hypotheses.some((hypothesis) => hypothesis.result.passed !== null)
    || Object.values(decision.evidence).some((value) => value !== null)
  );
}

function validateDisposition(disposition, hypothesisResults) {
  if (!DISPOSITIONS.has(disposition)) {
    throw new Error('capture.recommendedDisposition is not recognized.');
  }
  const passedAdvantages = hypothesisResults.filter((result) => result.passed).length;
  if (disposition === 'incumbent' && passedAdvantages > 0) {
    throw new Error('An incumbent recommendation conflicts with a passing material advantage.');
  }
  if (disposition !== 'incumbent' && passedAdvantages === 0) {
    throw new Error(`${disposition} recommendation requires a passing material advantage.`);
  }
}

async function validateHypothesisResults(capture, decision, repoRoot) {
  if (!Array.isArray(capture.hypothesisResults)) {
    throw new Error('capture.hypothesisResults must be an array.');
  }
  const expectedAxes = decision.hypotheses.map((hypothesis) => hypothesis.axis);
  const seenAxes = new Set();
  const results = [];
  for (const result of capture.hypothesisResults) {
    assertExactKeys(result, RESULT_FIELDS, 'capture hypothesis result');
    const axis = requireText(result.axis, 'capture hypothesis result.axis');
    if (!expectedAxes.includes(axis)) throw new Error(`Unexpected hypothesis result axis ${axis}.`);
    if (seenAxes.has(axis)) throw new Error(`Duplicate hypothesis result axis ${axis}.`);
    seenAxes.add(axis);
    if (typeof result.passed !== 'boolean') {
      throw new Error(`${axis}.passed must be boolean.`);
    }
    if (!Number.isFinite(result.observedValue) && !normalizeText(result.observedValue)) {
      throw new Error(`${axis}.observedValue must be a finite number or non-empty string.`);
    }
    const evaluatedAt = parseIsoInstant(result.evaluatedAtUtc, `${axis}.evaluatedAtUtc`);
    const evidencePath = await requireRepoPath(
      result.evidencePath,
      `${axis}.evidencePath`,
      repoRoot
    );
    results.push({
      axis,
      passed: result.passed,
      observedValue: result.observedValue,
      evaluatedAtUtc: evaluatedAt.toISOString(),
      evidencePath,
      evaluatedAt,
    });
  }
  if (seenAxes.size !== expectedAxes.length) {
    const missing = expectedAxes.find((axis) => !seenAxes.has(axis));
    throw new Error(`Missing hypothesis result axis ${missing}.`);
  }
  return results;
}

async function validateEvidencePaths(capture, repoRoot) {
  assertExactKeys(capture.evidence, EVIDENCE_FIELDS, 'capture.evidence');
  const evidence = {};
  for (const field of EVIDENCE_FIELDS) {
    evidence[field] = await requireRepoPath(
      capture.evidence[field],
      `capture.evidence.${field}`,
      repoRoot
    );
  }
  return evidence;
}

async function validateExternalReceipt(evidence, decision, role, repoRoot) {
  const evidenceField = `${role}Execution`;
  const receipt = await readJson(
    path.join(repoRoot, evidence[evidenceField]),
    `${role} execution evidence`
  );
  const source = role === 'source';
  const result = validateRuntimeOwnershipExecutionEvidence(receipt, {
    role,
    providerId: source ? decision.sourceProviderId : decision.incumbentProviderId,
    artifactId: source ? decision.sourceArtifactId : decision.incumbentArtifactId,
    workload: decision.workload,
    logicalModelId: decision.logicalModelId,
  });
  if (result.errors.length > 0) {
    throw new Error(`${role} execution evidence is invalid: ${result.errors.join('; ')}`);
  }
  return result;
}

async function validateDopplerReceipt(evidence, decision, repoRoot) {
  const receipt = await readJson(
    path.join(repoRoot, evidence.dopplerExecution),
    'Doppler execution evidence'
  );
  const result = validateDopplerRuntimeOwnershipReceipt(receipt, {
    logicalModelId: decision.logicalModelId,
    resolvedArtifactVariantId: decision.resolvedArtifactVariantId,
    resolvedExecutionId: decision.resolvedExecutionId,
  });
  if (result.errors.length > 0) {
    throw new Error(`Doppler execution evidence is invalid: ${result.errors.join('; ')}`);
  }
  return result;
}

async function validateCapture(capture, context) {
  const { decision, repoRoot, now, qualificationMaxAgeDays } = context;
  assertExactKeys(capture, [
    'schema',
    'decisionId',
    'recommendedDisposition',
    'decisionRationale',
    'qualifiedAtUtc',
    'expiresAtUtc',
    'hypothesisResults',
    'evidence',
  ], 'runtime ownership evaluation capture');
  if (capture.schema !== CAPTURE_SCHEMA) {
    throw new Error(`capture.schema must be ${CAPTURE_SCHEMA}.`);
  }
  if (capture.decisionId !== decision.id) {
    throw new Error(`capture.decisionId must be ${decision.id}.`);
  }
  const decisionRationale = requireText(capture.decisionRationale, 'capture.decisionRationale');
  const hypothesisResults = await validateHypothesisResults(capture, decision, repoRoot);
  validateDisposition(capture.recommendedDisposition, hypothesisResults);
  const qualifiedAt = parseIsoInstant(capture.qualifiedAtUtc, 'capture.qualifiedAtUtc');
  const expiresAt = parseIsoInstant(capture.expiresAtUtc, 'capture.expiresAtUtc');
  if (qualifiedAt.getTime() > now.getTime()) {
    throw new Error('capture.qualifiedAtUtc must not be in the future.');
  }
  if (hypothesisResults.some((result) => result.evaluatedAt.getTime() > qualifiedAt.getTime())) {
    throw new Error('capture.qualifiedAtUtc must not predate hypothesis evaluation.');
  }
  if (expiresAt.getTime() <= qualifiedAt.getTime()) {
    throw new Error('capture.expiresAtUtc must be later than capture.qualifiedAtUtc.');
  }
  if (expiresAt.getTime() > qualifiedAt.getTime() + qualificationMaxAgeDays * DAY_MS) {
    throw new Error(
      `capture.expiresAtUtc exceeds the ${qualificationMaxAgeDays}-day policy limit.`
    );
  }
  const evidence = await validateEvidencePaths(capture, repoRoot);
  const source = await validateExternalReceipt(evidence, decision, 'source', repoRoot);
  const incumbent = await validateExternalReceipt(evidence, decision, 'incumbent', repoRoot);
  const doppler = await validateDopplerReceipt(evidence, decision, repoRoot);
  return {
    disposition: capture.recommendedDisposition,
    decisionRationale,
    qualifiedAtUtc: qualifiedAt.toISOString(),
    expiresAtUtc: expiresAt.toISOString(),
    hypothesisResults,
    evidence,
    source,
    incumbent,
    doppler,
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

export async function recordRuntimeOwnershipEvaluation(options) {
  const repoRoot = path.resolve(options.repoRoot || REPO_ROOT);
  const policyPath = path.resolve(options.policyPath || DEFAULT_POLICY_PATH);
  const capturePath = path.resolve(requireText(options.capturePath, 'capturePath'));
  const outputPolicyPath = path.resolve(requireText(options.outputPolicyPath, 'outputPolicyPath'));
  const now = options.now instanceof Date ? options.now : new Date();
  const policy = await readJson(policyPath, 'runtime ownership policy');
  const current = await validateRuntimeOwnershipDecisions(policy, { repoRoot, now });
  if (current.errors.length > 0) {
    throw new Error(`Runtime ownership policy is invalid: ${current.errors[0]}`);
  }
  const capture = await readJson(capturePath, 'runtime ownership evaluation capture');
  const decision = policy.decisions.find((entry) => entry.id === capture.decisionId);
  if (!decision) throw new Error(`Unknown runtime ownership decision: ${capture.decisionId}.`);
  if (decision.claimAllowed) {
    throw new Error(`Recorder cannot replace claimable runtime ownership decision ${decision.id}.`);
  }
  if (hasPriorEvaluation(decision) && options.replace !== true) {
    throw new Error(`Decision ${decision.id} already contains evaluation state; use --replace.`);
  }
  const validated = await validateCapture(capture, {
    decision,
    repoRoot,
    now,
    qualificationMaxAgeDays: policy.qualificationMaxAgeDays,
  });
  const outputPolicy = structuredClone(policy);
  const outputDecision = outputPolicy.decisions.find((entry) => entry.id === decision.id);
  outputDecision.sourceExecutionId = validated.source.evidenceId;
  outputDecision.incumbentExecutionId = validated.incumbent.evidenceId;
  outputDecision.resolvedArtifactVariantId = validated.doppler.resolution
    ?.resolvedArtifactVariantId ?? null;
  outputDecision.resolvedExecutionId = validated.doppler.resolution?.resolvedExecutionId ?? null;
  outputDecision.hypotheses = outputDecision.hypotheses.map((hypothesis) => {
    const result = validated.hypothesisResults.find((entry) => entry.axis === hypothesis.axis);
    return {
      ...hypothesis,
      result: {
        passed: result.passed,
        observedValue: result.observedValue,
        evaluatedAtUtc: result.evaluatedAtUtc,
        evidencePath: result.evidencePath,
      },
    };
  });
  outputDecision.disposition = validated.disposition;
  outputDecision.decisionRationale = validated.decisionRationale;
  outputDecision.qualifiedAtUtc = validated.qualifiedAtUtc;
  outputDecision.expiresAtUtc = validated.expiresAtUtc;
  outputDecision.evidence = validated.evidence;
  outputDecision.claimAllowed = false;
  outputDecision.blockers = ['runtime-ownership-evaluation-awaiting-explicit-promotion'];
  let outputReport = await validateRuntimeOwnershipDecisions(outputPolicy, { repoRoot, now });
  if (outputReport.errors.length > 0) {
    throw new Error(`Recorded runtime ownership policy is invalid: ${outputReport.errors[0]}`);
  }
  const decisionReport = outputReport.decisions.find((entry) => entry.id === decision.id);
  outputDecision.blockers = Array.from(new Set([
    'runtime-ownership-evaluation-awaiting-explicit-promotion',
    ...decisionReport.reasons,
  ]));
  outputReport = await validateRuntimeOwnershipDecisions(outputPolicy, { repoRoot, now });
  if (outputReport.errors.length > 0) {
    throw new Error(`Recorded runtime ownership policy is invalid: ${outputReport.errors[0]}`);
  }
  await writeJsonAtomically(outputPolicyPath, outputPolicy);
  return {
    decisionId: decision.id,
    outputPolicyPath,
    sourceExecutionId: outputDecision.sourceExecutionId,
    incumbentExecutionId: outputDecision.incumbentExecutionId,
    resolvedArtifactVariantId: outputDecision.resolvedArtifactVariantId,
    resolvedExecutionId: outputDecision.resolvedExecutionId,
    disposition: outputDecision.disposition,
    claimAllowed: false,
    blockers: outputDecision.blockers,
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
  const result = await recordRuntimeOwnershipEvaluation(options);
  if (options.json) console.log(JSON.stringify(result, null, 2));
  else {
    console.log(
      `runtime-ownership-record: captured ${result.decisionId}; `
      + `disposition=${result.disposition}; claimAllowed=false; `
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
