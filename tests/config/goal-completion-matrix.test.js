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
  assert.equal(report.actions.length, matrix.blockers.length);
  assert.deepEqual(report.actions.map((action) => action.priority), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.equal(report.actions[0].code, 'maintained-application-integrations-missing');
  assert.equal(report.actions[0].completionClass, 'application');
  assert.equal(report.actions[0].statusCommand, 'npm run product:integrations:check');
  assert.deepEqual(report.actions[0].rows, [
    'local-webgpu-product-surface/maintained-application-integrations',
  ]);
}

{
  const broken = clone(matrix);
  broken.blockers[1].priority = broken.blockers[0].priority;
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes(`${broken.blockers[1].code}: duplicate blocker priority ${broken.blockers[0].priority}`),
    errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  broken.blockers[0].completionClass = 'static-guess';
  const errors = await validateFixture(broken);
  assert.ok(
    errors.some((error) => error.startsWith(`${broken.blockers[0].code}: completionClass must be one of`)),
    errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  broken.blockers[0].statusCommand = 'npm run missing:status';
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes(`${broken.blockers[0].code}: statusCommand script missing:status is not declared in package.json`),
    errors.join('\n')
  );
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
  const goal = broken.goals.find((entry) => entry.id === 'correctness-performance-claims');
  goal.rows = goal.rows.filter((row) => row.id !== 'bounded-recursive-improvement');
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes('correctness-performance-claims: missing required row bounded-recursive-improvement'),
    errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  const goal = broken.goals.find((entry) => entry.id === 'correctness-performance-claims');
  goal.rows = goal.rows.filter((row) => row.id !== 'revocation-propagation');
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes('correctness-performance-claims: missing required row revocation-propagation'),
    errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  const goal = broken.goals.find((entry) => entry.id === 'correctness-performance-claims');
  goal.rows = goal.rows.filter((row) => row.id !== 'post-promotion-monitoring');
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes('correctness-performance-claims: missing required row post-promotion-monitoring'),
    errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  const goal = broken.goals.find((entry) => entry.id === 'local-webgpu-product-surface');
  goal.rows = goal.rows.filter((row) => row.id !== 'maintained-application-integrations');
  const errors = await validateFixture(broken);
  assert.ok(
    errors.includes('local-webgpu-product-surface: missing required row maintained-application-integrations'),
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
