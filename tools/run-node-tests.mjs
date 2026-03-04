#!/usr/bin/env node

import { existsSync, readdirSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import './node-test-runtime-setup.mjs';

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

  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === '--force-exit') {
      forceExit = true;
      continue;
    }

    if (args[i] === '--suite') {
      const value = args[i + 1];
      if (value) {
        suite = value;
        i += 1;
      }
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

  return { suite, directories, forceExit };
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

function addTestFileIfValid(pathValue, files) {
  if (!existsSync(pathValue)) {
    return false;
  }
  const stats = statSync(pathValue);
  if (!stats.isFile()) {
    return false;
  }
  const normalized = String(pathValue);
  if (normalized.endsWith('.test.js')) {
    files.push(pathValue);
  }
  return true;
}

function listRootsFromSuite(suiteName, explicitDirs) {
  if (explicitDirs.length > 0) return explicitDirs;
  return suites[suiteName] ? suites[suiteName].map((dir) => resolve(ROOT_DIR, dir)) : suites.all.map((dir) => resolve(ROOT_DIR, dir));
}

async function main() {
  const { suite, directories, forceExit } = parseArgs();
  const resolvedSuite = Object.hasOwn(suites, suite) ? suite : 'all';
  const selectedRoots = listRootsFromSuite(resolvedSuite, directories.map((dir) => resolve(ROOT_DIR, dir)));
  const testFiles = [];

  for (const root of selectedRoots) {
    if (!existsSync(root)) {
      // keep behavior permissive: skip missing directories
      continue;
    }
    if (addTestFileIfValid(root, testFiles)) {
      continue;
    }
    collectTestFiles(root, testFiles);
  }

  if (testFiles.length === 0) {
    console.log('[node-tests] no matching tests found');
    return;
  }

  const failures = [];
  for (const file of testFiles.sort()) {
    const rel = relative(ROOT_DIR, file);
    try {
      // eslint-disable-next-line no-await-in-loop
      await import(pathToFileURL(file).href);
      console.log(`[node-tests] ok: ${rel}`);
    } catch (error) {
      failures.push({ file, error });
      console.error(`[node-tests] fail: ${rel}`);
      console.error(error?.stack || String(error));
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
