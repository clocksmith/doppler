#!/usr/bin/env node

import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runBrowserOracle } from './lib/run-browser-oracle.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

runBrowserOracle({
  argv: process.argv.slice(2),
  root: ROOT,
  defaultOutput: 'reports/training/native-parity/qwen-hybrid-decoder-backward-oracle.json',
  modulePath: 'tests/training/browser/qwen-hybrid-decoder-backward-oracle.js',
  exportName: 'runQwenHybridDecoderBackwardOracle',
  sourcePaths: {
    fullReference: 'src/experimental/training/qwen-full-decoder-reference.js',
    linearReference: 'src/experimental/training/qwen-linear-decoder-reference.js',
    hybridModule: 'src/experimental/training/qwen-hybrid-decoder-training-module.js',
    checkpointedHybridModule: 'src/experimental/training/qwen-checkpointed-hybrid-decoder-training-module.js',
    oracle: 'tests/training/browser/qwen-hybrid-decoder-backward-oracle.js',
  },
}).catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
