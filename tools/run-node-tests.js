#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { relative, resolve } from 'node:path';
import { resolveTestFiles } from './lib/node-test-suites.js';
import { runNodeTestScripts } from './lib/node-test-command-chain.js';

const ROOT_DIR = process.cwd();

function parseArgs(args) {
  const directories = [];
  let suite = 'all';
  let includePending = false;
  let list = false;

  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === '--list') {
      list = true;
      continue;
    }
    if (args[i] === '--force-exit') {
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

  return { suite, directories, includePending, list };
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

function runTests(args, seen = new Set()) {
  const { suite, directories, includePending, list } = parseArgs(args);
  const discovered = resolveTestFiles(suite, directories, { includePending });
  if (list) {
    console.log(JSON.stringify(discovered.map((file) => relative(ROOT_DIR, file)), null, 2));
    return;
  }
  const testFiles = discovered.filter((file) => !seen.has(file));
  const omitted = discovered.length - testFiles.length;
  if (omitted) console.log(`[node-tests] already passed in this check invocation: ${omitted} files`);

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
      seen.add(file);
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
    throw new Error(`[node-tests] failed ${failures.length}/${testFiles.length}`);
  }

  console.log(`[node-tests] ok: ${testFiles.length} files`);
}

try {
  const args = process.argv.slice(2);
  if (args[0] === '--scripts') {
    runNodeTestScripts(args.slice(1), runTests);
  } else {
    runTests(args);
  }
  if (args.includes('--force-exit')) process.exit(0);
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
