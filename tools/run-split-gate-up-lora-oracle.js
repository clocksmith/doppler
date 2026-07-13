#!/usr/bin/env node

import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runBrowserOracle } from './lib/run-browser-oracle.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

runBrowserOracle({
  argv: process.argv.slice(2),
  root: ROOT,
  defaultOutput: 'reports/training/native-parity/split-gate-up-lora-oracle.json',
  modulePath: 'tests/training/browser/split-gate-up-lora-oracle.js',
  exportName: 'runSplitGateUpLoraOracle',
  sourcePaths: {
    studentFixture: 'src/experimental/training/distillation/student-fixture.js',
    autograd: 'src/experimental/training/autograd.js',
    lora: 'src/experimental/training/lora.js',
    oracle: 'tests/training/browser/split-gate-up-lora-oracle.js',
  },
}).catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
