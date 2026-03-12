#!/usr/bin/env node

import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

await import('./node-test-runtime-setup.js');

const file = String(process.argv[2] || '').trim();
if (!file) {
  console.error('run-node-test-file: test file path is required.');
  process.exit(1);
}

try {
  await import(pathToFileURL(resolve(file)).href);
  process.exit(0);
} catch (error) {
  console.error(error?.stack || String(error));
  process.exit(1);
}
