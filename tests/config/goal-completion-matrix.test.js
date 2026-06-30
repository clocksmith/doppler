import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  buildGoalCompletionReport,
  validateGoalCompletionMatrix,
} from '../../tools/check-goal-completion.js';

const REPO_ROOT = process.cwd();
const MATRIX_PATH = path.join(REPO_ROOT, 'src', 'config', 'goal-completion-matrix.json');
const PACKAGE_PATH = path.join(REPO_ROOT, 'package.json');
const SUBSYSTEMS_PATH = path.join(REPO_ROOT, 'src', 'config', 'support-tiers', 'subsystems.json');

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function rowById(matrix, goalId, rowId) {
  return matrix.goals
    .find((goal) => goal.id === goalId)
    .rows
    .find((row) => row.id === rowId);
}

async function validateFixture(matrix) {
  const [packageJson, subsystemRegistry] = await Promise.all([
    readJson(PACKAGE_PATH),
    readJson(SUBSYSTEMS_PATH),
  ]);
  return validateGoalCompletionMatrix(matrix, {
    repoRoot: REPO_ROOT,
    packageJson,
    subsystemRegistry,
  });
}

const matrix = await readJson(MATRIX_PATH);

{
  const report = await buildGoalCompletionReport({
    matrixPath: MATRIX_PATH,
    packagePath: PACKAGE_PATH,
    subsystemsPath: SUBSYSTEMS_PATH,
    repoRoot: REPO_ROOT,
  });
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.goals.length, 3);
  assert.deepEqual(report.goals.map((goal) => goal.id), [
    'local-webgpu-product-surface',
    'model-artifact-runtime-contract',
    'correctness-performance-claims',
  ]);
}

{
  const broken = clone(matrix);
  const goal = broken.goals.find((entry) => entry.id === 'local-webgpu-product-surface');
  goal.rows = goal.rows.filter((row) => row.id !== 'bun-runtime');
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes('local-webgpu-product-surface: missing required row bun-runtime'),
    errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  rowById(broken, 'local-webgpu-product-surface', 'bun-runtime').blockers = ['unknown-blocker'];
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes('bun-runtime: undefined blocker code unknown-blocker'),
    errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  rowById(broken, 'local-webgpu-product-surface', 'bun-runtime').claimAllowed = true;
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes('bun-runtime: claimAllowed rows must use status covered or complete'),
    errors.join('\n')
  );
  assert.ok(
    errors.includes('bun-runtime: claimAllowed rows must not list blockers'),
    errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  rowById(broken, 'local-webgpu-product-surface', 'npx-doppler-gpu').packageBin = 'missing-bin';
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes('npx-doppler-gpu: packageBin missing-bin is not declared in package.json'),
    errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  rowById(broken, 'model-artifact-runtime-contract', 'rdrr-manifest-runtime').supportSubsystemId = 'missing.subsystem';
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes('rdrr-manifest-runtime: supportSubsystemId missing.subsystem is not declared in support tiers'),
    errors.join('\n')
  );
}

console.log('goal-completion-matrix.test: ok');
