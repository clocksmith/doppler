#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_MATRIX_PATH = path.join(REPO_ROOT, 'src', 'config', 'goal-completion-matrix.json');
const DEFAULT_PACKAGE_PATH = path.join(REPO_ROOT, 'package.json');
const DEFAULT_SUBSYSTEMS_PATH = path.join(REPO_ROOT, 'src', 'config', 'support-tiers', 'subsystems.json');

const GOAL_STATUSES = new Set(['complete', 'partial']);
const ROW_STATUSES = new Set(['covered', 'complete', 'partial', 'experimental', 'blocked', 'diagnostic']);
const CLAIMABLE_ROW_STATUSES = new Set(['covered', 'complete']);
const TIERS = new Set(['tier1', 'experimental', 'internal-only', 'not-applicable']);
const NULLABLE_ROW_FIELDS = ['supportSubsystemId', 'packageBin', 'packageExport', 'smokeCommand'];
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

const REQUIRED_GOAL_LABELS = new Map([
  ['local-webgpu-product-surface', 'Make local WebGPU inference a real product surface'],
  ['model-artifact-runtime-contract', 'Own the model artifact and runtime contract'],
  ['correctness-performance-claims', 'Make correctness and performance evidence-backed'],
]);

const REQUIRED_GOAL_ROWS = new Map([
  [
    'local-webgpu-product-surface',
    [
      'hosted-browser-demo',
      'npx-doppler-gpu',
      'root-api',
      'cli',
      'node-runtime',
      'bun-runtime',
      'openai-compatible-server',
    ],
  ],
  [
    'model-artifact-runtime-contract',
    [
      'rdrr-manifest-runtime',
      'hosted-registry-ids',
      'sharded-weights',
      'tokenizer-metadata',
      'quantization-contract',
      'execution-graph-kernel-refs',
      'dtype-session-policy',
      'model-support-matrix',
      'model-coverage-breadth',
    ],
  ],
  [
    'correctness-performance-claims',
    [
      'release-receipts',
      'benchmark-artifacts',
      'command-parity',
      'support-matrices',
      'explicit-kernel-paths',
      'apples-to-apples-compare-rules',
      'fail-closed-unsupported-paths',
      'claim-promotion-coverage',
    ],
  ],
]);

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function resolveInputPath(value) {
  const normalized = normalizeText(value);
  if (!normalized) {
    throw new Error('Path argument must not be empty');
  }
  return path.isAbsolute(normalized) ? normalized : path.resolve(REPO_ROOT, normalized);
}

function parseArgs(argv) {
  const args = {
    matrixPath: DEFAULT_MATRIX_PATH,
    packagePath: DEFAULT_PACKAGE_PATH,
    subsystemsPath: DEFAULT_SUBSYSTEMS_PATH,
    json: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    const nextValue = () => {
      const candidate = argv[i + 1];
      if (candidate == null || String(candidate).startsWith('--')) {
        throw new Error(`Missing value for ${token}`);
      }
      i += 1;
      return resolveInputPath(candidate);
    };
    if (token === '--matrix') {
      args.matrixPath = nextValue();
      continue;
    }
    if (token === '--package') {
      args.packagePath = nextValue();
      continue;
    }
    if (token === '--subsystems') {
      args.subsystemsPath = nextValue();
      continue;
    }
    if (token === '--json') {
      args.json = true;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  return args;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function isIsoInstant(value) {
  const normalized = normalizeText(value);
  if (!normalized) return false;
  const date = new Date(normalized);
  return Number.isFinite(date.getTime()) && date.toISOString() === normalized;
}

function isRepoRelativePath(value) {
  const candidate = normalizeText(value);
  return Boolean(
    candidate
    && !path.isAbsolute(candidate)
    && !candidate.includes('\\')
    && !candidate.split('/').includes('..')
  );
}

async function validateRepoPath(repoRoot, relativePath, label, errors) {
  const normalized = normalizeText(relativePath);
  if (!isRepoRelativePath(normalized)) {
    errors.push(`${label}: path must be repo-relative and must not traverse upward (${String(relativePath)})`);
    return;
  }
  try {
    await fs.stat(path.join(repoRoot, normalized));
  } catch {
    errors.push(`${label}: path does not exist (${normalized})`);
  }
}

async function validateEvidencePaths(repoRoot, paths, label, errors) {
  if (!Array.isArray(paths) || paths.length === 0) {
    errors.push(`${label}: evidencePaths must be a non-empty array`);
    return;
  }
  const seen = new Set();
  for (const evidencePath of paths) {
    const normalized = normalizeText(evidencePath);
    if (!normalized) {
      errors.push(`${label}: evidencePaths entries must be non-empty strings`);
      continue;
    }
    if (seen.has(normalized)) {
      errors.push(`${label}: duplicate evidence path ${normalized}`);
      continue;
    }
    seen.add(normalized);
    await validateRepoPath(repoRoot, normalized, `${label}: evidencePath`, errors);
  }
}

function validateStringArray(value, label, errors) {
  if (!Array.isArray(value)) {
    errors.push(`${label} must be an array`);
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

function validateNullableString(value, label, errors) {
  if (value === null) return null;
  const normalized = normalizeText(value);
  if (!normalized) {
    errors.push(`${label} must be a non-empty string or null`);
    return null;
  }
  return normalized;
}

function validateRequiredString(value, label, errors) {
  const normalized = normalizeText(value);
  if (!normalized) {
    errors.push(`${label} is required`);
  }
  return normalized;
}

function buildSubsystemMap(subsystemRegistry) {
  const subsystems = Array.isArray(subsystemRegistry?.subsystems) ? subsystemRegistry.subsystems : [];
  return new Map(
    subsystems
      .map((subsystem) => [normalizeText(subsystem?.id), subsystem])
      .filter(([id]) => Boolean(id))
  );
}

function packageBins(packageJson) {
  return isPlainObject(packageJson?.bin) ? packageJson.bin : {};
}

function packageExports(packageJson) {
  return isPlainObject(packageJson?.exports) ? packageJson.exports : {};
}

function packageScripts(packageJson) {
  return isPlainObject(packageJson?.scripts) ? packageJson.scripts : {};
}

function validateIdentifier(value, label, errors) {
  const normalized = validateRequiredString(value, label, errors);
  if (normalized && !ID_PATTERN.test(normalized)) {
    errors.push(`${label} must be a lowercase kebab-case id (${normalized})`);
  }
  return normalized;
}

function parseNpmRunScript(command) {
  const parts = normalizeText(command).split(/\s+/);
  if (parts[0] !== 'npm' || parts[1] !== 'run' || !parts[2]) {
    return null;
  }
  return parts[2];
}

function validateMatrixHeader(matrix, errors) {
  if (!isPlainObject(matrix)) {
    errors.push('goal-completion matrix must be an object');
    return false;
  }
  if (matrix.schemaVersion !== 1) {
    errors.push('goal-completion matrix schemaVersion must be 1');
  }
  if (normalizeText(matrix.source) !== 'doppler') {
    errors.push('goal-completion matrix source must be "doppler"');
  }
  if (!isIsoInstant(matrix.updatedAtUtc)) {
    errors.push('goal-completion matrix updatedAtUtc must be an ISO instant');
  }
  if (!Array.isArray(matrix.goals)) {
    errors.push('goal-completion matrix goals must be an array');
    return false;
  }
  if (!Array.isArray(matrix.blockers)) {
    errors.push('goal-completion matrix blockers must be an array');
    return false;
  }
  return true;
}

async function validateBlockerDefinitions(matrix, repoRoot, errors) {
  const blockerByCode = new Map();
  for (const blocker of matrix.blockers) {
    const code = validateIdentifier(blocker?.code, 'blocker.code', errors);
    if (!code) continue;
    if (blockerByCode.has(code)) {
      errors.push(`duplicate blocker code ${code}`);
    }
    blockerByCode.set(code, blocker);
    validateRequiredString(blocker?.description, `${code}: description`, errors);
    validateRequiredString(blocker?.exitCriteria, `${code}: exitCriteria`, errors);
    await validateEvidencePaths(repoRoot, blocker?.evidencePaths, `${code}: blocker`, errors);
  }
  return blockerByCode;
}

function validateRequiredGoals(goals, errors) {
  const goalById = new Map();
  for (const goal of goals) {
    const goalId = normalizeText(goal?.id);
    if (!goalId) {
      errors.push('goal entry is missing id');
      continue;
    }
    if (goalById.has(goalId)) {
      errors.push(`duplicate goal id ${goalId}`);
    }
    goalById.set(goalId, goal);
    if (!REQUIRED_GOAL_ROWS.has(goalId)) {
      errors.push(`unexpected goal id ${goalId}`);
    }
  }
  for (const requiredGoalId of REQUIRED_GOAL_ROWS.keys()) {
    if (!goalById.has(requiredGoalId)) {
      errors.push(`missing required goal ${requiredGoalId}`);
    }
  }
  return goalById;
}

function validateRequiredRows(goalId, rows, errors) {
  const requiredRows = REQUIRED_GOAL_ROWS.get(goalId) || [];
  const rowById = new Map();
  for (const row of rows) {
    const rowId = normalizeText(row?.id);
    if (!rowId) {
      errors.push(`${goalId}: row entry is missing id`);
      continue;
    }
    if (rowById.has(rowId)) {
      errors.push(`${goalId}: duplicate row id ${rowId}`);
    }
    rowById.set(rowId, row);
    if (!requiredRows.includes(rowId)) {
      errors.push(`${goalId}: unexpected row ${rowId}`);
    }
  }
  for (const requiredRowId of requiredRows) {
    if (!rowById.has(requiredRowId)) {
      errors.push(`${goalId}: missing required row ${requiredRowId}`);
    }
  }
  return rowById;
}

function validateBlockerRefs(blockerCodes, blockerByCode, usedBlockers, label, errors) {
  for (const code of blockerCodes) {
    usedBlockers.add(code);
    if (!blockerByCode.has(code)) {
      errors.push(`${label}: undefined blocker code ${code}`);
    }
  }
}

function countRowsByStatus(rows) {
  const counts = Object.create(null);
  for (const row of rows) {
    const status = normalizeText(row?.status) || 'missing';
    counts[status] = (counts[status] || 0) + 1;
  }
  return counts;
}

function collectGoalBlockers(goal, rows) {
  const blockers = new Set();
  for (const code of Array.isArray(goal?.blockers) ? goal.blockers : []) {
    const normalized = normalizeText(code);
    if (normalized) blockers.add(normalized);
  }
  for (const row of rows) {
    for (const code of Array.isArray(row?.blockers) ? row.blockers : []) {
      const normalized = normalizeText(code);
      if (normalized) blockers.add(normalized);
    }
  }
  return Array.from(blockers).sort();
}

function validateRowReferences(row, rowId, packageJson, subsystemById, errors) {
  for (const field of NULLABLE_ROW_FIELDS) {
    if (!Object.prototype.hasOwnProperty.call(row, field)) {
      errors.push(`${rowId}: ${field} is required; use null when disabled`);
    }
  }

  const supportSubsystemId = validateNullableString(row.supportSubsystemId, `${rowId}: supportSubsystemId`, errors);
  const packageBin = validateNullableString(row.packageBin, `${rowId}: packageBin`, errors);
  const packageExport = validateNullableString(row.packageExport, `${rowId}: packageExport`, errors);
  const smokeCommand = validateNullableString(row.smokeCommand, `${rowId}: smokeCommand`, errors);
  const bins = packageBins(packageJson);
  const exportsMap = packageExports(packageJson);
  const scripts = packageScripts(packageJson);

  if (supportSubsystemId) {
    const subsystem = subsystemById.get(supportSubsystemId);
    if (!subsystem) {
      errors.push(`${rowId}: supportSubsystemId ${supportSubsystemId} is not declared in support tiers`);
    } else if (normalizeText(row.tier) !== normalizeText(subsystem.tier)) {
      errors.push(`${rowId}: tier ${row.tier} must match support subsystem ${supportSubsystemId} tier ${subsystem.tier}`);
    }
  }
  if (packageBin && !Object.prototype.hasOwnProperty.call(bins, packageBin)) {
    errors.push(`${rowId}: packageBin ${packageBin} is not declared in package.json`);
  }
  if (packageExport && !Object.prototype.hasOwnProperty.call(exportsMap, packageExport)) {
    errors.push(`${rowId}: packageExport ${packageExport} is not declared in package.json`);
  }
  if (smokeCommand && !smokeCommand.startsWith('npm run ')) {
    errors.push(`${rowId}: smokeCommand must be an npm script command or null`);
  } else if (smokeCommand) {
    const scriptName = parseNpmRunScript(smokeCommand);
    if (!scriptName) {
      errors.push(`${rowId}: smokeCommand must name an npm script`);
    } else if (!Object.prototype.hasOwnProperty.call(scripts, scriptName)) {
      errors.push(`${rowId}: smokeCommand script ${scriptName} is not declared in package.json`);
    }
  }
  return { smokeCommand };
}

async function validateRow(row, goalId, context) {
  const { repoRoot, packageJson, subsystemById, blockerByCode, usedBlockers, errors } = context;
  const rowId = validateIdentifier(row?.id, `${goalId}: row.id`, errors);
  if (!rowId) return;
  validateRequiredString(row?.label, `${rowId}: label`, errors);
  if (!TIERS.has(normalizeText(row?.tier))) {
    errors.push(`${rowId}: tier must be one of ${Array.from(TIERS).join(', ')}`);
  }
  const status = normalizeText(row?.status);
  if (!ROW_STATUSES.has(status)) {
    errors.push(`${rowId}: status must be one of ${Array.from(ROW_STATUSES).join(', ')}`);
  }
  if (typeof row?.claimAllowed !== 'boolean') {
    errors.push(`${rowId}: claimAllowed must be boolean`);
  }

  const rowBlockers = validateStringArray(row?.blockers, `${rowId}: blockers`, errors);
  validateBlockerRefs(rowBlockers, blockerByCode, usedBlockers, rowId, errors);
  if (row?.claimAllowed === true && !CLAIMABLE_ROW_STATUSES.has(status)) {
    errors.push(`${rowId}: claimAllowed rows must use status covered or complete`);
  }
  if (CLAIMABLE_ROW_STATUSES.has(status) && row?.claimAllowed !== true) {
    errors.push(`${rowId}: status ${status} rows must be claimAllowed`);
  }
  if (row?.claimAllowed === true && rowBlockers.length > 0) {
    errors.push(`${rowId}: claimAllowed rows must not list blockers`);
  }
  if (row?.claimAllowed === false && rowBlockers.length === 0) {
    errors.push(`${rowId}: non-claimable rows must list blocker codes`);
  }
  if (!CLAIMABLE_ROW_STATUSES.has(status) && rowBlockers.length === 0) {
    errors.push(`${rowId}: status ${status || '<missing>'} requires blocker codes`);
  }
  const references = validateRowReferences(row, rowId, packageJson, subsystemById, errors);
  if (row?.claimAllowed === true && !references.smokeCommand) {
    errors.push(`${rowId}: claimAllowed rows must declare a smokeCommand`);
  }
  if (row?.claimAllowed === false && references.smokeCommand !== null) {
    errors.push(`${rowId}: non-claimable rows must set smokeCommand to null`);
  }
  await validateEvidencePaths(repoRoot, row?.evidencePaths, `${rowId}: row`, errors);
}

async function validateGoal(goal, context) {
  const { repoRoot, blockerByCode, usedBlockers, errors } = context;
  const goalId = validateIdentifier(goal?.id, 'goal.id', errors);
  if (!goalId) return;
  validateRequiredString(goal?.label, `${goalId}: label`, errors);
  const expectedGoalLabel = REQUIRED_GOAL_LABELS.get(goalId);
  if (expectedGoalLabel && normalizeText(goal?.label) !== expectedGoalLabel) {
    errors.push(`${goalId}: label must be "${expectedGoalLabel}"`);
  }
  const status = normalizeText(goal?.status);
  if (!GOAL_STATUSES.has(status)) {
    errors.push(`${goalId}: status must be one of ${Array.from(GOAL_STATUSES).join(', ')}`);
  }
  if (typeof goal?.claimAllowed !== 'boolean') {
    errors.push(`${goalId}: claimAllowed must be boolean`);
  }
  if (!Array.isArray(goal?.rows) || goal.rows.length === 0) {
    errors.push(`${goalId}: rows must be a non-empty array`);
    return;
  }

  const goalBlockers = validateStringArray(goal?.blockers, `${goalId}: blockers`, errors);
  validateBlockerRefs(goalBlockers, blockerByCode, usedBlockers, goalId, errors);
  if (status === 'complete' && goal?.claimAllowed !== true) {
    errors.push(`${goalId}: complete goals must be claimAllowed`);
  }
  if (status === 'partial' && goal?.claimAllowed !== false) {
    errors.push(`${goalId}: partial goals must not be claimAllowed`);
  }
  if (goal?.claimAllowed === true && goalBlockers.length > 0) {
    errors.push(`${goalId}: claimAllowed goals must not list blockers`);
  }
  if (goal?.claimAllowed === false && goalBlockers.length === 0) {
    errors.push(`${goalId}: non-claimable goals must list blocker codes`);
  }
  await validateEvidencePaths(repoRoot, goal?.evidencePaths, `${goalId}: goal`, errors);

  validateRequiredRows(goalId, goal.rows, errors);
  const childContext = { ...context };
  for (const row of goal.rows) {
    await validateRow(row, goalId, childContext);
  }
  const rowsClaimable = goal.rows.every((row) => row?.claimAllowed === true);
  const rowsComplete = goal.rows.every((row) => CLAIMABLE_ROW_STATUSES.has(normalizeText(row?.status)));
  const childBlockers = new Set();
  for (const row of goal.rows) {
    for (const code of Array.isArray(row?.blockers) ? row.blockers : []) {
      childBlockers.add(normalizeText(code));
    }
  }
  if (goal?.claimAllowed === true && !rowsClaimable) {
    errors.push(`${goalId}: claimAllowed goal has non-claimable rows`);
  }
  if (goal?.claimAllowed === false && rowsClaimable) {
    errors.push(`${goalId}: non-claimable goal must have at least one non-claimable row`);
  }
  if (status === 'complete' && !rowsComplete) {
    errors.push(`${goalId}: complete goal has non-complete rows`);
  }
  if (status === 'partial' && rowsComplete && rowsClaimable) {
    errors.push(`${goalId}: partial goal must identify at least one incomplete row`);
  }
  for (const code of goalBlockers) {
    if (!childBlockers.has(code)) {
      errors.push(`${goalId}: blocker ${code} must be owned by at least one row`);
    }
  }
}

export async function validateGoalCompletionMatrix(matrix, options = {}) {
  const repoRoot = options.repoRoot || REPO_ROOT;
  const packageJson = options.packageJson || {};
  const subsystemRegistry = options.subsystemRegistry || {};
  const errors = [];
  if (!validateMatrixHeader(matrix, errors)) {
    return errors;
  }

  const blockerByCode = await validateBlockerDefinitions(matrix, repoRoot, errors);
  validateRequiredGoals(matrix.goals, errors);
  const subsystemById = buildSubsystemMap(subsystemRegistry);
  const usedBlockers = new Set();

  for (const goal of matrix.goals) {
    await validateGoal(goal, {
      repoRoot,
      packageJson,
      subsystemById,
      blockerByCode,
      usedBlockers,
      errors,
    });
  }

  for (const code of blockerByCode.keys()) {
    if (!usedBlockers.has(code)) {
      errors.push(`blocker ${code} is defined but unused`);
    }
  }

  return errors;
}

function summarizeGoals(matrix) {
  const goals = Array.isArray(matrix?.goals) ? matrix.goals : [];
  return goals.map((goal) => {
    const rows = Array.isArray(goal.rows) ? goal.rows : [];
    const claimableRows = rows.filter((row) => row?.claimAllowed === true).length;
    const blockedRows = rows.filter((row) => row?.claimAllowed === false).length;
    return {
      id: goal.id,
      label: goal.label,
      status: goal.status,
      claimAllowed: goal.claimAllowed,
      rows: rows.length,
      claimableRows,
      blockedRows,
      completionPercent: rows.length > 0 ? Math.round((claimableRows / rows.length) * 100) : 0,
      statusCounts: countRowsByStatus(rows),
      blockers: collectGoalBlockers(goal, rows),
    };
  });
}

export async function buildGoalCompletionReport(options = {}) {
  const matrixPath = options.matrixPath || DEFAULT_MATRIX_PATH;
  const packagePath = options.packagePath || DEFAULT_PACKAGE_PATH;
  const subsystemsPath = options.subsystemsPath || DEFAULT_SUBSYSTEMS_PATH;
  const [matrix, packageJson, subsystemRegistry] = await Promise.all([
    readJson(matrixPath),
    readJson(packagePath),
    readJson(subsystemsPath),
  ]);
  const errors = await validateGoalCompletionMatrix(matrix, {
    repoRoot: options.repoRoot || REPO_ROOT,
    packageJson,
    subsystemRegistry,
  });
  return {
    ok: errors.length === 0,
    matrixPath: path.relative(options.repoRoot || REPO_ROOT, matrixPath),
    errors,
    goals: summarizeGoals(matrix),
  };
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const report = await buildGoalCompletionReport(args);
  if (args.json) {
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) {
      process.exitCode = 1;
    }
    return;
  }
  if (!report.ok) {
    for (const error of report.errors) {
      console.error(`goal-completion: ${error}`);
    }
    process.exitCode = 1;
    return;
  }
  const rowCount = report.goals.reduce((total, goal) => total + goal.rows, 0);
  console.log(`goal-completion: matrix ok (${report.goals.length} goals, ${rowCount} rows)`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
