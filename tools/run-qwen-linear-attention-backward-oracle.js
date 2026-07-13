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
    causalConvForwardShader: 'src/gpu/kernels/causal_conv1d_silu.wgsl',
    causalConvShader: 'src/gpu/kernels/backward/causal_conv1d_silu_backward.wgsl',
    gatedRmsNormForwardShader: 'src/gpu/kernels/gated_rmsnorm.wgsl',
    gatedRmsNormShader: 'src/gpu/kernels/backward/gated_rmsnorm_backward.wgsl',
    preparationForwardShader: 'src/gpu/kernels/qwen_linear_attention_prepare.wgsl',
    preparationBackwardShader: 'src/gpu/kernels/backward/qwen_linear_attention_prepare_backward.wgsl',
    gatedDeltaShader: 'src/gpu/kernels/backward/gated_delta_recurrent_backward.wgsl',
    checkpointForwardShader: 'src/gpu/kernels/backward/gated_delta_recurrent_checkpoint_forward.wgsl',
    recurrentReference: 'src/experimental/training/qwen-gated-delta-reference.js',
    reference: 'src/experimental/training/qwen-linear-attention-reference.js',
    trainingCore: 'src/experimental/training/qwen-linear-attention-training-core.js',
    oracle: 'tests/training/browser/qwen-linear-attention-backward-oracle.js',
  },
}).catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
