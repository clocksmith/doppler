#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { closeSync, existsSync, mkdtempSync, openSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const ROOT_DIR = process.cwd();
const DEFAULT_POLICY_PATH = resolve(ROOT_DIR, 'tools/policies/test-coverage-policy.json');
const NODE_TEST_SETUP_PATH = resolve(ROOT_DIR, 'tools/node-test-runtime-setup.mjs');

const suites = {
  unit: [
    'tests/config',
    'tests/converter',
    'tests/integration',
    'tests/inference',
  ],
  gpu: [
    'tests/kernels',
  ],
  all: [
    'tests/config',
    'tests/converter',
    'tests/integration',
    'tests/inference',
    'tests/kernels',
  ],
};

function parseArgs() {
  const args = process.argv.slice(2);
  const directories = [];
  let suite = null;
  let enforceThreshold = true;
  let policyPath = DEFAULT_POLICY_PATH;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--suite') {
      const value = args[i + 1];
      if (!value || value.startsWith('--')) {
        throw new Error('Missing value for --suite');
      }
      suite = value;
      i += 1;
      continue;
    }
    if (arg.startsWith('--suite=')) {
      suite = arg.split('=', 2)[1];
      continue;
    }
    if (arg === '--policy') {
      const value = args[i + 1];
      if (!value || value.startsWith('--')) {
        throw new Error('Missing value for --policy');
      }
      policyPath = resolve(ROOT_DIR, value);
      i += 1;
      continue;
    }
    if (arg.startsWith('--policy=')) {
      policyPath = resolve(ROOT_DIR, arg.split('=', 2)[1]);
      continue;
    }
    if (arg === '--no-threshold') {
      enforceThreshold = false;
      continue;
    }
    directories.push(arg);
  }

  return { directories, suite, enforceThreshold, policyPath };
}

function collectTestFiles(dir, files) {
  const entries = readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      collectTestFiles(fullPath, files);
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.test.js')) {
      files.push(fullPath);
    }
  }
}

function collectFilesFromRoot(pathValue, files) {
  if (!existsSync(pathValue)) {
    throw new Error(`Test path not found: ${pathValue}`);
  }
  const stats = statSync(pathValue);
  if (stats.isFile()) {
    if (!String(pathValue).endsWith('.test.js')) {
      throw new Error(`Test file must end with .test.js: ${pathValue}`);
    }
    files.push(pathValue);
    return;
  }
  collectTestFiles(pathValue, files);
}

function listRootsFromSuite(suiteName, explicitDirs) {
  if (explicitDirs.length > 0) {
    return explicitDirs.map((dir) => resolve(ROOT_DIR, dir));
  }
  if (!Object.hasOwn(suites, suiteName)) {
    throw new Error(`Unknown --suite "${suiteName}". Valid suites: ${Object.keys(suites).join(', ')}`);
  }
  return suites[suiteName].map((dir) => resolve(ROOT_DIR, dir));
}

function resolveTestFiles(suiteName, directories) {
  const roots = listRootsFromSuite(suiteName, directories);
  const files = [];
  for (const root of roots) {
    collectFilesFromRoot(root, files);
  }
  return files.sort();
}

function loadPolicy(policyPath) {
  if (!existsSync(policyPath)) {
    throw new Error(`coverage policy file not found: ${policyPath}`);
  }
  const parsed = JSON.parse(readFileSync(policyPath, 'utf8'));
  const thresholds = parsed?.thresholds || {};
  if (
    typeof thresholds.line !== 'number'
    || typeof thresholds.branch !== 'number'
    || typeof thresholds.functions !== 'number'
  ) {
    throw new Error('coverage policy thresholds must define numeric line, branch, and functions values');
  }
  return parsed;
}

function resolveExcludedTests(excludeTests) {
  if (!Array.isArray(excludeTests)) {
    return new Set();
  }
  return new Set(excludeTests.map((testPath) => resolve(ROOT_DIR, testPath)));
}

function parseCoverageSummary(outputText) {
  const regex = /^# all[^|]*\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|/gm;
  let match = null;
  while (true) {
    const found = regex.exec(outputText);
    if (!found) break;
    match = found;
  }
  if (!match) return null;
  return {
    line: Number.parseFloat(match[1]),
    branch: Number.parseFloat(match[2]),
    functions: Number.parseFloat(match[3]),
  };
}

function runCoverage(testFiles, concurrency, timeoutMs) {
  const nodeArgs = [
    '--test',
    `--test-concurrency=${concurrency}`,
    '--experimental-test-coverage',
    '--import',
    NODE_TEST_SETUP_PATH,
    ...testFiles,
  ];

  const logDir = mkdtempSync(join(tmpdir(), 'doppler-node-coverage-'));
  const logPath = join(logDir, 'node-test-output.log');
  const fd = openSync(logPath, 'w');

  const result = spawnSync(process.execPath, nodeArgs, {
    cwd: ROOT_DIR,
    stdio: ['ignore', fd, fd],
    timeout: timeoutMs,
  });
  closeSync(fd);

  const outputText = readFileSync(logPath, 'utf8');
  rmSync(logDir, { recursive: true, force: true });

  return {
    ...result,
    outputText,
  };
}

function printSummary(summary, thresholds) {
  console.log(
    `[coverage] all files line=${summary.line.toFixed(2)}% branch=${summary.branch.toFixed(2)}% functions=${summary.functions.toFixed(2)}%`
  );
  if (!thresholds) {
    return;
  }
  console.log(
    `[coverage] thresholds line>=${thresholds.line}% branch>=${thresholds.branch}% functions>=${thresholds.functions}%`
  );
}

function evaluateThresholds(summary, thresholds) {
  const failures = [];
  if (summary.line < thresholds.line) {
    failures.push(`line ${summary.line.toFixed(2)}% < ${thresholds.line}%`);
  }
  if (summary.branch < thresholds.branch) {
    failures.push(`branch ${summary.branch.toFixed(2)}% < ${thresholds.branch}%`);
  }
  if (summary.functions < thresholds.functions) {
    failures.push(`functions ${summary.functions.toFixed(2)}% < ${thresholds.functions}%`);
  }
  return failures;
}

function main() {
  const { directories, suite, enforceThreshold, policyPath } = parseArgs();
  const policy = loadPolicy(policyPath);
  const selectedSuite = suite || policy.suite || 'all';
  const concurrency = policy?.nodeTest?.concurrency ?? 1;
  const timeoutMs = policy?.nodeTest?.timeoutMs ?? 1200000;
  const excludedTests = resolveExcludedTests(policy.excludeTests);
  const discoveredFiles = resolveTestFiles(selectedSuite, directories);
  const testFiles = discoveredFiles.filter((filePath) => !excludedTests.has(filePath));
  const excludedCount = discoveredFiles.length - testFiles.length;

  if (excludedCount > 0) {
    console.log(`[coverage] excluded ${excludedCount} tests via policy`);
  }

  if (testFiles.length === 0) {
    console.error(`[coverage] no test files found for suite="${selectedSuite}"`);
    process.exit(1);
  }

  const result = runCoverage(testFiles, concurrency, timeoutMs);
  process.stdout.write(result.outputText ?? '');

  if (result.error?.code === 'ETIMEDOUT') {
    console.error(`[coverage] timed out after ${timeoutMs}ms`);
    process.exit(1);
  }

  const combinedOutput = result.outputText ?? '';
  const summary = parseCoverageSummary(combinedOutput);
  if (!summary) {
    console.error('[coverage] unable to parse coverage summary from node --test output');
    process.exit(result.status ?? 1);
  }

  printSummary(summary, enforceThreshold ? policy.thresholds : null);

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }

  if (!enforceThreshold) {
    return;
  }

  const failures = evaluateThresholds(summary, policy.thresholds);
  if (failures.length > 0) {
    for (const failure of failures) {
      console.error(`[coverage] threshold failed: ${failure}`);
    }
    process.exit(1);
  }
}

main();
