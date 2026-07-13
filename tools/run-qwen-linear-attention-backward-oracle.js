#!/usr/bin/env node

import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runBrowserOracle } from './lib/run-browser-oracle.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

runBrowserOracle({
  argv: process.argv.slice(2),
  root: ROOT,
  defaultOutput: 'reports/training/native-parity/qwen-linear-attention-backward-oracle.json',
  modulePath: 'tests/training/browser/qwen-linear-attention-backward-oracle.js',
  exportName: 'runQwenLinearAttentionBackwardOracle',
  sourcePaths: {
    causalConvShader: 'src/gpu/kernels/backward/causal_conv1d_silu_backward.wgsl',
    gatedRmsNormShader: 'src/gpu/kernels/backward/gated_rmsnorm_backward.wgsl',
    reference: 'src/experimental/training/qwen-linear-attention-reference.js',
    oracle: 'tests/training/browser/qwen-linear-attention-backward-oracle.js',
  },
}).catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
