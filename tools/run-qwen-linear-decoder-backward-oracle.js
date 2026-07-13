#!/usr/bin/env node

import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runBrowserOracle } from './lib/run-browser-oracle.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

runBrowserOracle({
  argv: process.argv.slice(2),
  root: ROOT,
  defaultOutput: 'reports/training/native-parity/qwen-linear-decoder-backward-oracle.json',
  modulePath: 'tests/training/browser/qwen-linear-decoder-backward-oracle.js',
  exportName: 'runQwenLinearDecoderBackwardOracle',
  sourcePaths: {
    reference: 'src/experimental/training/qwen-linear-decoder-reference.js',
    attentionModule: 'src/experimental/training/qwen-linear-attention-training-core.js',
    mlpModule: 'src/experimental/training/qwen-decoder-mlp-training-module.js',
    decoderModule: 'src/experimental/training/qwen-linear-decoder-training-module.js',
    oracle: 'tests/training/browser/qwen-linear-decoder-backward-oracle.js',
  },
}).catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
