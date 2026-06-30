#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'claim-evidence-contract.json');
const DEFAULT_PACKAGE_PATH = path.join(REPO_ROOT, 'package.json');
const REQUIRED_CLAIM_FIELDS = Object.freeze([
  'modelId',
  'mode',
  'surface',
  'verificationSource',
  'lastVerifiedAt',
  'artifactFormat',
]);

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function resolveRepoPath(value, label, errors) {
  const normalized = normalizeText(value);
  if (
    !normalized
    || path.isAbsolute(normalized)
    || normalized.includes('\\')
    || normalized.split('/').includes('..')
  ) {
    errors.push(`${label} must be a repo-relative path`);
    return null;
  }
  return path.join(REPO_ROOT, normalized);
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function validateStringArray(value, label, errors) {
  if (!Array.isArray(value) || value.length === 0) {
    errors.push(`${label} must be a non-empty array`);
    return [];
  }
  const out = [];
  const seen = new Set();
  for (const item of value) {
    const normalized = normalizeText(item);
    if (!normalized) {
      errors.push(`${label} entries must be non-empty strings`);
      continue;
    }
    if (seen.has(normalized)) {
      errors.push(`${label} contains duplicate entry ${normalized}`);
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function isRepoRelativeJsonPath(value) {
  const normalized = normalizeText(value);
  return Boolean(
    normalized
    && !path.isAbsolute(normalized)
    && !normalized.includes('\\')
    && !normalized.split('/').includes('..')
    && normalized.endsWith('.json')
  );
}

function validatePolicy(policy, errors) {
  if (!isPlainObject(policy)) {
    errors.push('claim evidence policy must be an object');
    return false;
  }
  if (policy.schemaVersion !== 1) {
    errors.push('claim evidence policy schemaVersion must be 1');
  }
  if (policy.source !== 'doppler') {
    errors.push('claim evidence policy source must be "doppler"');
  }
  for (const field of [
    'releaseClaimPolicyPath',
    'benchmarkPolicyPath',
    'localInferenceClaimMatrixPath',
    'releaseMatrixPath',
    'goalMatrixPath',
  ]) {
    if (!isRepoRelativeJsonPath(policy[field])) {
      errors.push(`claim evidence policy ${field} must be a repo-relative JSON path`);
    }
  }
  validateStringArray(policy.requiredScripts, 'claim evidence policy requiredScripts', errors);
  validateStringArray(policy.requiredGoalEvidence, 'claim evidence policy requiredGoalEvidence', errors);
  validateStringArray(policy.requiredReleaseMatrixSources, 'claim evidence policy requiredReleaseMatrixSources', errors);
  return errors.length === 0;
}

function validateScripts(policy, packageJson, errors) {
  const scripts = isPlainObject(packageJson?.scripts) ? packageJson.scripts : {};
  for (const script of policy.requiredScripts) {
    if (!Object.prototype.hasOwnProperty.call(scripts, script)) {
      errors.push(`package.json is missing required claim evidence script ${script}`);
    }
  }
}

function validateReleaseClaims(policy, releaseClaimPolicy, errors) {
  if (!isPlainObject(releaseClaimPolicy)) {
    errors.push(`${policy.releaseClaimPolicyPath}: must be an object`);
    return { claimCount: 0 };
  }
  if (releaseClaimPolicy.schemaVersion !== 1) {
    errors.push(`${policy.releaseClaimPolicyPath}: schemaVersion must be 1`);
  }
  if (!Array.isArray(releaseClaimPolicy.claims) || releaseClaimPolicy.claims.length === 0) {
    errors.push(`${policy.releaseClaimPolicyPath}: claims must be a non-empty array`);
    return { claimCount: 0 };
  }
  const seen = new Set();
  for (const claim of releaseClaimPolicy.claims) {
    const modelId = normalizeText(claim?.modelId) || 'unknown-model';
    for (const field of REQUIRED_CLAIM_FIELDS) {
      if (field === 'surface') {
        if (!Array.isArray(claim?.surface) || claim.surface.length === 0) {
          errors.push(`${modelId}: release claim surface must be a non-empty array`);
        }
      } else if (!normalizeText(claim?.[field])) {
        errors.push(`${modelId}: release claim ${field} is required`);
      }
    }
    const mode = normalizeText(claim?.mode);
    const key = `${modelId}:${mode}`;
    if (seen.has(key)) {
      errors.push(`${modelId}: duplicate release claim mode ${mode}`);
    }
    seen.add(key);
    if (!isRepoRelativeJsonPath(claim?.evidence?.reportPath)) {
      errors.push(`${modelId}: release claim evidence.reportPath must be repo-relative JSON`);
    }
    if (!isRepoRelativeJsonPath(claim?.performanceEvidence?.reportPath)) {
      errors.push(`${modelId}: release claim performanceEvidence.reportPath must be repo-relative JSON`);
    }
    if (!normalizeText(claim?.performanceEvidence?.metricPath)) {
      errors.push(`${modelId}: release claim performanceEvidence.metricPath is required`);
    }
  }
  return { claimCount: releaseClaimPolicy.claims.length };
}

function validateBenchmarkPolicy(policy, benchmarkPolicy, localClaimMatrix, errors) {
  const requiredTimingFields = validateStringArray(
    benchmarkPolicy?.requiredTimingFields,
    `${policy.benchmarkPolicyPath}: requiredTimingFields`,
    errors
  );
  validateStringArray(
    benchmarkPolicy?.requiredCompareMetricIds,
    `${policy.benchmarkPolicyPath}: requiredCompareMetricIds`,
    errors
  );
  const matrixMeasurements = new Set(validateStringArray(
    localClaimMatrix?.requiredMeasurements,
    `${policy.localInferenceClaimMatrixPath}: requiredMeasurements`,
    errors
  ));
  for (const field of requiredTimingFields) {
    if (!matrixMeasurements.has(field)) {
      errors.push(`${policy.localInferenceClaimMatrixPath}: requiredMeasurements must include benchmark timing field ${field}`);
    }
  }
  if (!isPlainObject(localClaimMatrix?.promotionGates)) {
    errors.push(`${policy.localInferenceClaimMatrixPath}: promotionGates must be an object`);
  }
}

function validateReleaseMatrix(policy, releaseMatrix, errors) {
  const sources = isPlainObject(releaseMatrix?.sources) ? releaseMatrix.sources : {};
  for (const sourceId of policy.requiredReleaseMatrixSources) {
    const source = sources[sourceId];
    if (!isPlainObject(source)) {
      errors.push(`${policy.releaseMatrixPath}: sources.${sourceId} is required`);
      continue;
    }
    if (!isRepoRelativeJsonPath(source.path)) {
      errors.push(`${policy.releaseMatrixPath}: sources.${sourceId}.path must be repo-relative JSON`);
    }
    if (!/^[0-9a-f]{64}$/.test(normalizeText(source.sha256))) {
      errors.push(`${policy.releaseMatrixPath}: sources.${sourceId}.sha256 must be sha256 hex`);
    }
  }
}

function validateGoalEvidence(policy, goalMatrix, errors) {
  const goals = Array.isArray(goalMatrix?.goals) ? goalMatrix.goals : [];
  const claimGoal = goals.find((goal) => goal?.id === 'correctness-performance-claims');
  if (!claimGoal) {
    errors.push(`${policy.goalMatrixPath}: correctness-performance-claims goal is required`);
    return;
  }
  const evidence = new Set([
    ...(Array.isArray(claimGoal.evidencePaths) ? claimGoal.evidencePaths : []),
    ...claimGoal.rows.flatMap((row) => Array.isArray(row?.evidencePaths) ? row.evidencePaths : []),
  ]);
  for (const requiredEvidence of policy.requiredGoalEvidence) {
    if (!evidence.has(requiredEvidence)) {
      errors.push(`${policy.goalMatrixPath}: correctness-performance-claims must include evidence ${requiredEvidence}`);
    }
  }
}

export async function buildClaimEvidenceContractReport(options = {}) {
  const policyPath = options.policyPath || DEFAULT_POLICY_PATH;
  const errors = [];
  const policy = await readJson(policyPath);
  if (!validatePolicy(policy, errors)) {
    return {
      ok: false,
      policyPath: path.relative(REPO_ROOT, policyPath),
      errors,
      claimCount: 0,
    };
  }
  const paths = {
    releaseClaimPolicy: resolveRepoPath(policy.releaseClaimPolicyPath, 'releaseClaimPolicyPath', errors),
    benchmarkPolicy: resolveRepoPath(policy.benchmarkPolicyPath, 'benchmarkPolicyPath', errors),
    localInferenceClaimMatrix: resolveRepoPath(policy.localInferenceClaimMatrixPath, 'localInferenceClaimMatrixPath', errors),
    releaseMatrix: resolveRepoPath(policy.releaseMatrixPath, 'releaseMatrixPath', errors),
    goalMatrix: resolveRepoPath(policy.goalMatrixPath, 'goalMatrixPath', errors),
    packageJson: options.packagePath || DEFAULT_PACKAGE_PATH,
  };
  if (errors.length > 0) {
    return {
      ok: false,
      policyPath: path.relative(REPO_ROOT, policyPath),
      errors,
      claimCount: 0,
    };
  }
  const [
    releaseClaimPolicy,
    benchmarkPolicy,
    localClaimMatrix,
    releaseMatrix,
    goalMatrix,
    packageJson,
  ] = await Promise.all([
    readJson(paths.releaseClaimPolicy),
    readJson(paths.benchmarkPolicy),
    readJson(paths.localInferenceClaimMatrix),
    readJson(paths.releaseMatrix),
    readJson(paths.goalMatrix),
    readJson(paths.packageJson),
  ]);
  validateScripts(policy, packageJson, errors);
  const releaseClaimSummary = validateReleaseClaims(policy, releaseClaimPolicy, errors);
  validateBenchmarkPolicy(policy, benchmarkPolicy, localClaimMatrix, errors);
  validateReleaseMatrix(policy, releaseMatrix, errors);
  validateGoalEvidence(policy, goalMatrix, errors);
  return {
    ok: errors.length === 0,
    policyPath: path.relative(REPO_ROOT, policyPath),
    errors,
    claimCount: releaseClaimSummary.claimCount,
    requiredScripts: policy.requiredScripts.length,
    requiredReleaseMatrixSources: policy.requiredReleaseMatrixSources.length,
  };
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((token) => token !== '--json');
  if (unsupported.length > 0) {
    throw new Error(`Unknown argument: ${unsupported[0]}`);
  }
  const report = await buildClaimEvidenceContractReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    console.log(`claim-evidence-contract: evidence ok (${report.claimCount} release claims)`);
  } else {
    for (const error of report.errors) {
      console.error(`claim-evidence-contract: ${error}`);
    }
  }
  if (!report.ok) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
