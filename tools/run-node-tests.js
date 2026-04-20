#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT_DIR = process.cwd();

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
  let suite = 'all';
  let forceExit = false;
  let includePending = false;

  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === '--force-exit') {
      forceExit = true;
      continue;
    }

    if (args[i] === '--include-pending') {
      includePending = true;
      continue;
    }

    if (args[i] === '--suite') {
      const value = args[i + 1];
      if (!value || value.startsWith('--')) {
        throw new Error('Missing value for --suite');
      }
      suite = value;
      i += 1;
      continue;
    }

    if (args[i].startsWith('--suite=')) {
      suite = args[i].split('=', 2)[1];
      continue;
    }

    if (args[i].startsWith('--')) {
      throw new Error(`Unknown argument: ${args[i]}`);
    }
    directories.push(args[i]);
  }

  return { suite, directories, forceExit, includePending };
}

function isPendingTestFile(name) {
  return name.endsWith('.pending.test.js');
}

function collectTestFiles(dir, files, { includePending }) {
  const entries = readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      collectTestFiles(fullPath, files, { includePending });
      continue;
    }
    if (!entry.isFile() || !entry.name.endsWith('.test.js')) continue;
    if (isPendingTestFile(entry.name) && !includePending) continue;
    files.push(fullPath);
  }
}

function collectFilesFromRoot(pathValue, files, { includePending }) {
  if (!existsSync(pathValue)) {
    throw new Error(`Test path not found: ${pathValue}`);
  }
  const stats = statSync(pathValue);
  if (stats.isFile()) {
    const normalized = String(pathValue);
    if (!normalized.endsWith('.test.js')) {
      throw new Error(`Test file must end with .test.js: ${pathValue}`);
    }
    // Explicitly-named files run regardless of pending status: the caller
    // asked for this exact file, so honor that.
    files.push(pathValue);
    return;
  }
  collectTestFiles(pathValue, files, { includePending });
}

function listRootsFromSuite(suiteName, explicitDirs) {
  if (explicitDirs.length > 0) return explicitDirs;
  return suites[suiteName] ? suites[suiteName].map((dir) => resolve(ROOT_DIR, dir)) : suites.all.map((dir) => resolve(ROOT_DIR, dir));
}

const TEST_FILE_RUNNER = resolve(ROOT_DIR, 'tools/run-node-test-file.js');

function runTestFile(file) {
  return spawnSync(
    process.execPath,
    [
      TEST_FILE_RUNNER,
      file,
    ],
    {
      cwd: ROOT_DIR,
      encoding: 'utf8',
    }
  );
}

async function main() {
  const { suite, directories, forceExit, includePending } = parseArgs();
  if (!Object.hasOwn(suites, suite)) {
    throw new Error(`Unknown --suite "${suite}". Valid suites: ${Object.keys(suites).join(', ')}`);
  }
  const selectedRoots = listRootsFromSuite(suite, directories.map((dir) => resolve(ROOT_DIR, dir)));
  const testFiles = [];

  for (const root of selectedRoots) {
    collectFilesFromRoot(root, testFiles, { includePending });
  }

  if (testFiles.length === 0) {
    console.log('[node-tests] no matching tests found');
    return;
  }

  if (includePending) {
    console.log('[node-tests] --include-pending: pending-feature tests will run');
  }

  const failures = [];
  for (const file of testFiles.sort()) {
    const rel = relative(ROOT_DIR, file);
    const result = runTestFile(file);
    if (result.stdout) {
      process.stdout.write(result.stdout);
    }
    if (result.stderr) {
      process.stderr.write(result.stderr);
    }
    if (result.status === 0) {
      console.log(`[node-tests] ok: ${rel}`);
    } else {
      failures.push({
        file,
        error: result.stderr || result.stdout || `exit code ${result.status ?? 1}`,
      });
      console.error(`[node-tests] fail: ${rel}`);
      if (!result.stderr && !result.stdout) {
        console.error(`exit code ${result.status ?? 1}`);
      }
    }
  }

  if (failures.length > 0) {
    console.error(`[node-tests] failed ${failures.length}/${testFiles.length}`);
    process.exit(1);
  }

  console.log(`[node-tests] ok: ${testFiles.length} files`);
  if (forceExit) {
    process.exit(0);
  }
}

await main();
