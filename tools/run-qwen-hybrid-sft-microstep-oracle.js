#!/usr/bin/env node

import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runBrowserOracle } from './lib/run-browser-oracle.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

runBrowserOracle({
  argv: process.argv.slice(2),
  root: ROOT,
  defaultOutput: 'reports/training/native-parity/qwen-hybrid-sft-microstep-oracle.json',
  modulePath: 'tests/training/browser/qwen-hybrid-sft-microstep-oracle.js',
  exportName: 'runQwenHybridSftMicrostepOracle',
  sourcePaths: {
    microstep: 'src/experimental/training/qwen-hybrid-sft-microstep.js',
    hybridModule: 'src/experimental/training/qwen-hybrid-decoder-training-module.js',
    fullReference: 'src/experimental/training/qwen-full-decoder-reference.js',
    oracle: 'tests/training/browser/qwen-hybrid-sft-microstep-oracle.js',
  },
}).catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
