#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const file = String(process.argv[2] || '').trim();
if (!file) {
  console.error('run-node-test-file: test file path is required.');
  process.exit(1);
}

// Native test completion includes registered async assertions before open-handle cleanup.
const childEnv = { ...process.env };
// A nested invocation must start a harness, not inherit its parent's worker role.
delete childEnv.NODE_TEST_CONTEXT;
const result = spawnSync(process.execPath, [
  '--test',
  '--test-force-exit',
  '--import',
  fileURLToPath(new URL('./node-test-file-bootstrap.js', import.meta.url)),
  resolve(file),
], { stdio: 'inherit', env: childEnv });
if (result.error) console.error(result.error.stack || String(result.error));
process.exitCode = result.status ?? 1;
