import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = process.cwd();
const KERNEL_PATH_DIR = path.join(ROOT, 'src', 'config', 'presets', 'kernel-paths');
const WGSL_DIR = path.join(ROOT, 'src', 'gpu', 'kernels');

function loadKernelPath(fileName) {
  const filePath = path.join(KERNEL_PATH_DIR, fileName);
  const raw = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(raw);
}

function collectAttentionKernels(kernelPathConfig) {
  const kernels = new Set();
  const sections = [kernelPathConfig.decode, kernelPathConfig.prefill].filter(Boolean);
  for (const section of sections) {
    const steps = Array.isArray(section.steps) ? section.steps : [];
    for (const step of steps) {
      if (step?.op === 'attention' && typeof step.kernel === 'string') {
        kernels.add(step.kernel);
      }
    }
  }
  return kernels;
}

function hasMaxSubtractedSoftmax(source) {
  const hasExpMinus = /exp\s*\(\s*[^\)]*-\s*[^\)]*\)/.test(source);
  const hasMaxTracker = /(max_score|global_max|running_max|m_i|m_new|shared_max|chunk_max_val)/.test(source);
  return hasExpMinus && hasMaxTracker;
}

const kernelPathFiles = [
  'gemma3-f16-f16a-online.json',
  'gemma3-f16-f32a.json',
  'gemma3-f16-fused-f32a-online.json',
  'gemma3-q4k-dequant-f16a-online.json',
  'gemma3-q4k-dequant-f32a-online.json',
];

const kernels = new Set();
for (const fileName of kernelPathFiles) {
  const kernelPath = loadKernelPath(fileName);
  for (const kernel of collectAttentionKernels(kernelPath)) {
    kernels.add(kernel);
  }
}

// BDPA decode kernel is experimental but tracked as long-context path.
kernels.add('attention_bdpa_decode_f16.wgsl');

for (const kernel of kernels) {
  const kernelPath = path.join(WGSL_DIR, kernel);
  const source = fs.readFileSync(kernelPath, 'utf8');
  assert.equal(
    hasMaxSubtractedSoftmax(source),
    true,
    `Kernel ${kernel} must include max-subtracted softmax logic.`
  );
}

console.log('attention-softmax-stability.test: ok');
