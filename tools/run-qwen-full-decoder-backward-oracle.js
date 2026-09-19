#!/usr/bin/env node

import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runBrowserOracle } from './lib/run-browser-oracle.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

runBrowserOracle({
  argv: process.argv.slice(2),
  root: ROOT,
  defaultOutput: 'reports/training/native-parity/qwen-full-decoder-backward-oracle.json',
  modulePath: 'tests/training/browser/qwen-full-decoder-backward-oracle.js',
  exportName: 'runQwenFullDecoderBackwardOracle',
  sourcePaths: {
    gatedBackwardShader: 'src/gpu/kernels/backward/sigmoid_gated_backward.wgsl',
    reference: 'src/experimental/training/qwen-full-decoder-reference.js',
    attentionModule: 'src/experimental/training/qwen-full-attention-training-module.js',
    decoderModule: 'src/experimental/training/qwen-full-decoder-training-module.js',
    oracle: 'tests/training/browser/qwen-full-decoder-backward-oracle.js',
  },
}).catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
