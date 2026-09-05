#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { isPlainObject } from '../src/formats/plain-object.js';

import {
  validateProductIntegrationQualification,
} from './check-product-integration-qualification.js';
import { computeCanonicalJsonSha256 } from './lib/canonical-json.js';
import {
  PRODUCT_OUTCOME_EVIDENCE_CLASSES,
  validateProductIntegrationOutcomeEvidence,
  validateProductIntegrationOwnerConfirmation,
} from './lib/product-integration-evidence.js';
import {
  validateDopplerRuntimeOwnershipReceipt,
} from './lib/runtime-ownership-execution-evidence.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(
  REPO_ROOT,
  'tools',
  'policies',
  'product-integration-qualification.json'
);
const CAPTURE_SCHEMA = 'doppler.product-integration-evaluation-capture/v1';
const CAPTURE_EVIDENCE_FIELDS = Object.freeze([
  'ownerConfirmation',
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
const REVIEW_REASONS = new Set([
  'qualification-level-not-product-supported',
  'lifecycle-not-active',
  'blockers-present',
  'evidence-incomplete',
]);
const DAY_MS = 24 * 60 * 60 * 1000;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
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

async function readEvidencePath(value, label, repoRoot) {
  if (!isRepoRelativePath(value)) throw new Error(`${label} must be a repo-relative path.`);
  const normalized = normalizeText(value);
  let receipt;
  try {
    receipt = JSON.parse(await fs.readFile(path.join(repoRoot, normalized), 'utf8'));
  } catch (error) {
    throw new Error(`${label} is not readable JSON: ${error.message}`);
  }
  return {
    path: normalized,
    digest: computeCanonicalJsonSha256(receipt),
    receipt,
  };
}

async function readJson(filePath, label) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (error) {
    throw new Error(`${label} is not readable JSON: ${error.message}`);
  }
}

function hasPriorEvaluation(integration) {
  return Boolean(
    integration.ownerConfirmedAtUtc
    || integration.resolvedArtifactVariantId
    || integration.resolvedExecutionId
    || integration.qualifiedAtUtc
    || integration.expiresAtUtc
    || Object.values(integration.evidence).some((value) => value !== null)
  );
}

async function validateCapture(capture, context) {
  const { integration, repoRoot, now, maxAgeDays } = context;
  assertExactKeys(capture, [
    'schema',
    'integrationId',
    'evaluatedAtUtc',
    'expiresAtUtc',
    'evidencePaths',
  ], 'product integration evaluation capture');
  if (capture.schema !== CAPTURE_SCHEMA) {
    throw new Error(`capture.schema must be ${CAPTURE_SCHEMA}.`);
  }
  if (capture.integrationId !== integration.id) {
    throw new Error(`capture.integrationId must be ${integration.id}.`);
  }
  const evaluatedAt = parseIsoInstant(capture.evaluatedAtUtc, 'capture.evaluatedAtUtc');
  const expiresAt = parseIsoInstant(capture.expiresAtUtc, 'capture.expiresAtUtc');
  if (evaluatedAt.getTime() > now.getTime()) {
    throw new Error('capture.evaluatedAtUtc must not be in the future.');
  }
  if (expiresAt.getTime() <= evaluatedAt.getTime()) {
    throw new Error('capture.expiresAtUtc must be later than capture.evaluatedAtUtc.');
  }
  if (expiresAt.getTime() > evaluatedAt.getTime() + maxAgeDays * DAY_MS) {
    throw new Error(`capture.expiresAtUtc exceeds the ${maxAgeDays}-day policy limit.`);
  }
  assertExactKeys(
    capture.evidencePaths,
    CAPTURE_EVIDENCE_FIELDS,
    'capture.evidencePaths'
  );
  const distinctEvidencePaths = new Set(Object.values(capture.evidencePaths));
  if (distinctEvidencePaths.size !== CAPTURE_EVIDENCE_FIELDS.length) {
    throw new Error('capture.evidencePaths must use a distinct path for every evidence class.');
  }
  const evidence = {};
  for (const field of CAPTURE_EVIDENCE_FIELDS) {
    evidence[field] = await readEvidencePath(
      capture.evidencePaths[field],
      `capture.evidencePaths.${field}`,
      repoRoot
    );
  }
  const baseContext = {
    integrationId: integration.id,
    applicationName: integration.applicationName,
    workload: integration.workload,
    owner: integration.owner,
    logicalModelId: integration.logicalModelId,
  };
  const identity = validateDopplerRuntimeOwnershipReceipt(evidence.identity.receipt, {
    logicalModelId: integration.logicalModelId,
    resolvedArtifactVariantId: integration.resolvedArtifactVariantId,
    resolvedExecutionId: integration.resolvedExecutionId,
  });
  if (identity.errors.length > 0) {
    throw new Error(`Identity receipt is invalid: ${identity.errors.join('; ')}`);
  }
  let outcomeContext = {
    ...baseContext,
    resolvedArtifactVariantId: identity.resolution?.resolvedArtifactVariantId ?? null,
    resolvedExecutionId: identity.resolution?.resolvedExecutionId ?? null,
  };
  const outcomeResults = [];
  let applicationEvidenceIdentity = null;
  for (const evidenceClass of PRODUCT_OUTCOME_EVIDENCE_CLASSES) {
    const result = validateProductIntegrationOutcomeEvidence(
      evidence[evidenceClass].receipt,
      { ...outcomeContext, ...applicationEvidenceIdentity, evidenceClass }
    );
    if (result.errors.length > 0) {
      throw new Error(`${evidenceClass} evidence is invalid: ${result.errors.join('; ')}`);
    }
    outcomeResults.push({ evidenceClass, ...result });
    if (!applicationEvidenceIdentity) {
      applicationEvidenceIdentity = {
        applicationRevision: result.applicationRevision,
        harnessRevision: result.harnessRevision,
        environmentFingerprint: result.environmentFingerprint,
      };
    }
    if (
      !outcomeContext.resolvedArtifactVariantId
      && !outcomeContext.resolvedExecutionId
      && result.resolution
    ) {
      outcomeContext = {
        ...outcomeContext,
        resolvedArtifactVariantId: result.resolution.resolvedArtifactVariantId,
        resolvedExecutionId: result.resolution.resolvedExecutionId,
      };
    }
  }
  const owner = validateProductIntegrationOwnerConfirmation(
    evidence.ownerConfirmation.receipt,
    { ...baseContext, applicationRevision: applicationEvidenceIdentity?.applicationRevision }
  );
  if (owner.errors.length > 0) {
    throw new Error(`Owner confirmation is invalid: ${owner.errors.join('; ')}`);
  }
  const evidenceInstants = [owner.confirmedAt, identity.timestamp]
    .concat(outcomeResults.map((result) => result.capturedAt))
    .filter(Boolean);
  if (evidenceInstants.some((instant) => instant.getTime() > evaluatedAt.getTime())) {
    throw new Error('capture.evaluatedAtUtc must not predate retained evidence.');
  }
  return {
    evaluatedAtUtc: evaluatedAt.toISOString(),
    expiresAtUtc: expiresAt.toISOString(),
    evidence,
    owner,
    identity,
    outcomeResults,
    applicationEvidenceIdentity,
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

export async function recordProductIntegrationEvaluation(options) {
  const repoRoot = path.resolve(options.repoRoot || REPO_ROOT);
  const policyPath = path.resolve(options.policyPath || DEFAULT_POLICY_PATH);
  const capturePath = path.resolve(requireText(options.capturePath, 'capturePath'));
  const outputPolicyPath = path.resolve(requireText(options.outputPolicyPath, 'outputPolicyPath'));
  const now = options.now instanceof Date ? options.now : new Date();
  const policy = await readJson(policyPath, 'product integration policy');
  const current = await validateProductIntegrationQualification(policy, { repoRoot, now });
  if (current.errors.length > 0) {
    throw new Error(`Product integration policy is invalid: ${current.errors[0]}`);
  }
  const capture = await readJson(capturePath, 'product integration evaluation capture');
  const integration = policy.integrations.find((entry) => entry.id === capture.integrationId);
  if (!integration) throw new Error(`Unknown product integration: ${capture.integrationId}.`);
  if (integration.claimAllowed) {
    throw new Error(`Recorder cannot replace claimable product integration ${integration.id}.`);
  }
  if (integration.evidence.promotion !== null) {
    throw new Error(`Recorder cannot replace promoted product integration ${integration.id}.`);
  }
  if (hasPriorEvaluation(integration) && options.replace !== true) {
    throw new Error(`Integration ${integration.id} already contains evaluation state; use --replace.`);
  }
  const validated = await validateCapture(capture, {
    integration,
    repoRoot,
    now,
    maxAgeDays: policy.ownerConfirmationMaxAgeDays,
  });
  const outputPolicy = structuredClone(policy);
  const outputIntegration = outputPolicy.integrations.find((entry) => entry.id === integration.id);
  outputIntegration.ownerConfirmedAtUtc = validated.owner.confirmedAt?.toISOString() ?? null;
  outputIntegration.resolvedArtifactVariantId = validated.identity.resolution
    ?.resolvedArtifactVariantId ?? null;
  outputIntegration.resolvedExecutionId = validated.identity.resolution?.resolvedExecutionId ?? null;
  outputIntegration.applicationRevision = validated.applicationEvidenceIdentity
    ?.applicationRevision ?? null;
  outputIntegration.harnessRevision = validated.applicationEvidenceIdentity?.harnessRevision ?? null;
  outputIntegration.environmentFingerprint = validated.applicationEvidenceIdentity
    ?.environmentFingerprint ?? null;
  outputIntegration.qualifiedAtUtc = validated.evaluatedAtUtc;
  outputIntegration.expiresAtUtc = validated.expiresAtUtc;
  outputIntegration.lifecycle = 'candidate';
  outputIntegration.claimAllowed = false;
  outputIntegration.evidence = Object.fromEntries(CAPTURE_EVIDENCE_FIELDS.map((field) => [
    field,
    {
      path: validated.evidence[field].path,
      digest: validated.evidence[field].digest,
    },
  ]));
  outputIntegration.evidence.promotion = null;
  outputIntegration.blockers = [
    'application-evaluation-awaiting-explicit-promotion',
    'product-support-promotion-evidence-missing',
  ];
  let outputReport = await validateProductIntegrationQualification(outputPolicy, { repoRoot, now });
  if (outputReport.errors.length > 0) {
    throw new Error(`Recorded product integration policy is invalid: ${outputReport.errors[0]}`);
  }
  const integrationReport = outputReport.integrations.find((entry) => entry.id === integration.id);
  outputIntegration.blockers = Array.from(new Set([
    'application-evaluation-awaiting-explicit-promotion',
    'product-support-promotion-evidence-missing',
    ...integrationReport.qualificationReasons.filter((reason) => !REVIEW_REASONS.has(reason)),
  ]));
  outputReport = await validateProductIntegrationQualification(outputPolicy, { repoRoot, now });
  if (outputReport.errors.length > 0) {
    throw new Error(`Recorded product integration policy is invalid: ${outputReport.errors[0]}`);
  }
  await writeJsonAtomically(outputPolicyPath, outputPolicy);
  return {
    integrationId: integration.id,
    outputPolicyPath,
    qualificationLevel: outputIntegration.qualificationLevel,
    lifecycle: outputIntegration.lifecycle,
    ownerConfirmedAtUtc: outputIntegration.ownerConfirmedAtUtc,
    resolvedArtifactVariantId: outputIntegration.resolvedArtifactVariantId,
    resolvedExecutionId: outputIntegration.resolvedExecutionId,
    applicationRevision: outputIntegration.applicationRevision,
    harnessRevision: outputIntegration.harnessRevision,
    environmentFingerprint: outputIntegration.environmentFingerprint,
    claimAllowed: false,
    blockers: outputIntegration.blockers,
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
  const result = await recordProductIntegrationEvaluation(options);
  if (options.json) console.log(JSON.stringify(result, null, 2));
  else {
    console.log(
      `product-integration-record: captured ${result.integrationId}; `
      + `level=${result.qualificationLevel}; lifecycle=${result.lifecycle}; `
      + `claimAllowed=false; output=${path.relative(REPO_ROOT, result.outputPolicyPath)}`
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
