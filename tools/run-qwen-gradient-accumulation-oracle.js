#!/usr/bin/env node

import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runBrowserOracle } from './lib/run-browser-oracle.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

runBrowserOracle({
  argv: process.argv.slice(2),
  root: ROOT,
  defaultOutput: 'reports/training/native-parity/qwen-gradient-accumulation-oracle.json',
  modulePath: 'tests/training/browser/qwen-gradient-accumulation-oracle.js',
  exportName: 'runQwenGradientAccumulationOracle',
  sourcePaths: {
    accumulator: 'src/experimental/training/qwen-gradient-accumulator.js',
    optimizer: 'src/experimental/training/optimizer.js',
    oracle: 'tests/training/browser/qwen-gradient-accumulation-oracle.js',
  },
}).catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
