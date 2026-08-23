#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'model-release-platform.json');
const MATRIX_PATH = path.join(REPO_ROOT, 'src', 'config', 'goal-completion-matrix.json');

const EXPECTED = Object.freeze({
  forgeStages: ['inspect', 'normalize', 'analyze', 'lower', 'specialize', 'search', 'verify', 'qualify', 'package', 'sign'],
  runtimeSteps: ['validate', 'select', 'bind', 'allocate', 'execute', 'observe'],
  providers: ['doppler-webgpu', 'browser-dawn', 'doe-runtime', 'onnx-runtime', 'webnn', 'vendor-native', 'cpu-reference'],
  apiSurfaces: ['open-pack', 'dynamic-load', 'openai-server', 'generation', 'embedding-reranking'],
  applicationClasses: ['generation', 'embedding', 'reranking'],
  recovery: ['content-hash-shard-resume', 'failed-upgrade-preserves-previous-pack', 'portable-state-snapshot-identity'],
  goalRows: [
    ['local-webgpu-product-surface', 'pack-first-runtime-convergence'],
    ['local-webgpu-product-surface', 'model-release-qualification-offer'],
    ['model-artifact-runtime-contract', 'complete-pack-release-closure'],
  ],
});

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function exactArray(actual, expected, label, errors) {
  if (!Array.isArray(actual) || JSON.stringify(actual) !== JSON.stringify(expected)) {
    errors.push(`${label} must equal ${JSON.stringify(expected)}`);
  }
}

function indexById(rows, label, errors) {
  const result = new Map();
  if (!Array.isArray(rows)) {
    errors.push(`${label} must be an array`);
    return result;
  }
  for (const row of rows) {
    const id = normalizeText(row?.id);
    if (!id) {
      errors.push(`${label} contains a row without id`);
      continue;
    }
    if (result.has(id)) errors.push(`${label} contains duplicate id ${id}`);
    result.set(id, row);
  }
  return result;
}

function isRepoRelative(value) {
  const normalized = normalizeText(value);
  return Boolean(normalized && !path.isAbsolute(normalized) && !normalized.split('/').includes('..'));
}

async function validatePaths(rows, repoRoot, errors) {
  for (const [label, values] of rows) {
    if (!Array.isArray(values) || values.length === 0) {
      errors.push(`${label} must contain evidence paths`);
      continue;
    }
    for (const value of values) {
      if (!isRepoRelative(value)) {
        errors.push(`${label} contains invalid repo-relative path ${String(value)}`);
        continue;
      }
      try {
        await fs.stat(path.join(repoRoot, value));
      } catch {
        errors.push(`${label} references missing path ${value}`);
      }
    }
  }
}

function validateTrackedRequirements(rows, blockerCodes, label, errors, pathRows) {
  const index = indexById(rows, label, errors);
  for (const row of index.values()) {
    pathRows.push([`${label}.${row.id}.evidencePaths`, row.evidencePaths]);
    if (row.implementationState === 'implemented' && row.blockerCode !== null) {
      errors.push(`${label}.${row.id}: implemented rows must have blockerCode null`);
    }
    if (row.implementationState === 'partial') {
      if (!normalizeText(row.blockerCode)) {
        errors.push(`${label}.${row.id}: partial rows require a blockerCode`);
      } else if (!blockerCodes.has(row.blockerCode)) {
        errors.push(`${label}.${row.id}: unknown blockerCode ${row.blockerCode}`);
      }
    }
  }
  return index;
}

export async function validateModelReleasePlatform(policy, matrix, options = {}) {
  const repoRoot = options.repoRoot || REPO_ROOT;
  const errors = [];
  const pathRows = [];
  const blockerCodes = new Set((matrix?.blockers || []).map((row) => row.code));

  if (policy?.id !== 'doppler-model-release-platform') errors.push('policy.id must be doppler-model-release-platform');
  if (policy?.positioning?.unitOfValue !== 'supported-model-release') {
    errors.push('positioning.unitOfValue must be supported-model-release');
  }
  exactArray(policy?.architecture?.forgeStages, EXPECTED.forgeStages, 'architecture.forgeStages', errors);
  exactArray(policy?.architecture?.runtimeSteps, EXPECTED.runtimeSteps, 'architecture.runtimeSteps', errors);
  pathRows.push(['architecture.authorityPaths', policy?.architecture?.authorityPaths]);
  pathRows.push(['modelIR.authorityPaths', policy?.modelIR?.authorityPaths]);
  pathRows.push(['pack.authorityPaths', policy?.pack?.authorityPaths]);

  validateTrackedRequirements(
    policy?.pack?.requiredReleaseElements,
    blockerCodes,
    'pack.requiredReleaseElements',
    errors,
    pathRows
  );
  const providerIndex = indexById(policy?.providers?.targets, 'providers.targets', errors);
  exactArray(Array.from(providerIndex.keys()), EXPECTED.providers, 'providers.targets ids', errors);
  if (policy?.providers?.selectionLaw !== 'qualified-only') errors.push('providers.selectionLaw must be qualified-only');
  if (policy?.providers?.runtimeMayInventPlan !== false) errors.push('providers.runtimeMayInventPlan must be false');
  if (policy?.providers?.doeRequired !== false) errors.push('providers.doeRequired must be false');
  for (const provider of providerIndex.values()) {
    pathRows.push([`providers.targets.${provider.id}.qualificationAuthority`, [provider.qualificationAuthority]]);
  }

  const apiIndex = indexById(policy?.apiConvergence, 'apiConvergence', errors);
  exactArray(Array.from(apiIndex.keys()), EXPECTED.apiSurfaces, 'apiConvergence ids', errors);
  for (const row of apiIndex.values()) {
    pathRows.push([`apiConvergence.${row.id}.evidencePaths`, row.evidencePaths]);
    if (row.mode === 'migration-required' && !blockerCodes.has(row.blockerCode)) {
      errors.push(`apiConvergence.${row.id}: migration-required row needs a defined blockerCode`);
    }
    if (row.mode !== 'migration-required' && row.blockerCode !== null) {
      errors.push(`apiConvergence.${row.id}: non-migration row must have blockerCode null`);
    }
  }

  const recoveryIndex = validateTrackedRequirements(
    policy?.recovery,
    blockerCodes,
    'recovery',
    errors,
    pathRows
  );
  exactArray(Array.from(recoveryIndex.keys()), EXPECTED.recovery, 'recovery ids', errors);
  exactArray(policy?.commercialOffer?.applicationClasses, EXPECTED.applicationClasses, 'commercialOffer.applicationClasses', errors);
  if (policy?.commercialOffer?.currentAssessment !== 'unestablished') {
    errors.push('commercialOffer.currentAssessment must remain unestablished until external evidence is promoted');
  }
  if (!blockerCodes.has(policy?.commercialOffer?.blockerCode)) {
    errors.push(`commercialOffer.blockerCode is not defined: ${String(policy?.commercialOffer?.blockerCode)}`);
  }
  exactArray(
    (policy?.promotionSequence || []).map((row) => row.order),
    [1, 2, 3, 4, 5, 6, 7],
    'promotionSequence order',
    errors
  );

  const goals = new Map((matrix?.goals || []).map((goal) => [goal.id, goal]));
  for (const [goalId, rowId] of EXPECTED.goalRows) {
    const row = goals.get(goalId)?.rows?.find((candidate) => candidate.id === rowId);
    if (!row) errors.push(`goal matrix is missing ${goalId}/${rowId}`);
    else pathRows.push([`goal matrix ${goalId}/${rowId}.evidencePaths`, row.evidencePaths]);
  }

  await validatePaths(pathRows, repoRoot, errors);
  return {
    ok: errors.length === 0,
    errors,
    partialRequirements: [
      ...(policy?.pack?.requiredReleaseElements || []),
      ...(policy?.recovery || []),
    ].filter((row) => row.implementationState === 'partial').map((row) => row.id),
    migrationSurfaces: (policy?.apiConvergence || [])
      .filter((row) => row.mode === 'migration-required')
      .map((row) => row.id),
    commercialAssessment: policy?.commercialOffer?.currentAssessment || null,
  };
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

export async function buildModelReleasePlatformReport(options = {}) {
  const repoRoot = options.repoRoot || REPO_ROOT;
  const [policy, matrix] = await Promise.all([
    readJson(options.policyPath || POLICY_PATH),
    readJson(options.matrixPath || MATRIX_PATH),
  ]);
  return validateModelReleasePlatform(policy, matrix, { repoRoot });
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((token) => token !== '--json');
  if (unsupported.length > 0) throw new Error(`Unknown argument: ${unsupported[0]}`);
  const report = await buildModelReleasePlatformReport();
  if (json) console.log(JSON.stringify(report, null, 2));
  else if (report.ok) {
    console.log(
      `model-release-platform: contract ok; ${report.partialRequirements.length} partial requirements, `
      + `${report.migrationSurfaces.length} migration surfaces, commercial=${report.commercialAssessment}`
    );
  } else {
    for (const error of report.errors) console.error(`model-release-platform: ${error}`);
  }
  if (!report.ok) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
