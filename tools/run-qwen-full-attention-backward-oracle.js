#!/usr/bin/env node

import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runBrowserOracle } from './lib/run-browser-oracle.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

runBrowserOracle({
  argv: process.argv.slice(2),
  root: ROOT,
  defaultOutput: 'reports/training/native-parity/qwen-full-attention-backward-oracle.json',
  modulePath: 'tests/training/browser/qwen-full-attention-backward-oracle.js',
  exportName: 'runQwenFullAttentionBackwardOracle',
  sourcePaths: {
    splitForwardShader: 'src/gpu/kernels/qwen_attention_split_q_gate.wgsl',
    splitBackwardShader: 'src/gpu/kernels/backward/qwen_attention_split_q_gate_backward.wgsl',
    sigmoidBackwardShader: 'src/gpu/kernels/backward/sigmoid_gated_backward.wgsl',
    gqaSoftmaxShader: 'src/gpu/kernels/backward/qwen_gqa_softmax_recompute.wgsl',
    gqaScoresShader: 'src/gpu/kernels/backward/qwen_gqa_softmax_backward_scores.wgsl',
    gqaBackwardShader: 'src/gpu/kernels/backward/qwen_gqa_attention_backward.wgsl',
    ropeForwardShader: 'src/gpu/kernels/rope.wgsl',
    ropeBackwardShader: 'src/gpu/kernels/backward/rope_backward.wgsl',
    reference: 'src/experimental/training/qwen-full-attention-reference.js',
    trainingModule: 'src/experimental/training/qwen-full-attention-training-module.js',
    oracle: 'tests/training/browser/qwen-full-attention-backward-oracle.js',
  },
}).catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
