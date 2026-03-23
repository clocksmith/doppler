import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = process.cwd();
const WGSL_DIR = path.join(ROOT, 'src', 'gpu', 'kernels');
const CONVERSION_DIR = path.join(ROOT, 'src', 'config', 'conversion');

function hasMaxSubtractedSoftmax(source) {
  const hasExpMinus = /exp\s*\(\s*[^\)]*-\s*[^\)]*\)/.test(source);
  const hasMaxTracker = /(max_score|global_max|running_max|m_i|m_new|shared_max|chunk_max_val)/.test(source);
  return hasExpMinus && hasMaxTracker;
}

function collectAttentionKernelsFromGraph(execution) {
  const kernels = new Set();
  const phases = ['decode', 'prefill'];
  for (const phase of phases) {
    const steps = execution[phase];
    if (!Array.isArray(steps)) continue;
    for (const step of steps) {
      const op = step[0];
      if (op !== 'attention') continue;
      const kernelKey = step[1];
      const entry = execution.kernels?.[kernelKey];
      if (entry?.kernel) kernels.add(entry.kernel);
    }
  }
  return kernels;
}

function collectJsonFiles(dir) {
  const results = [];
  const stack = [dir];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) stack.push(fullPath);
      else if (entry.isFile() && entry.name.endsWith('.json')) results.push(fullPath);
    }
  }
  return results;
}

const kernels = new Set();

for (const configPath of collectJsonFiles(CONVERSION_DIR)) {
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  const execution = config.execution ?? config.inference?.execution;
  if (!execution?.kernels) continue;
  for (const kernel of collectAttentionKernelsFromGraph(execution)) {
    kernels.add(kernel);
  }
}

// BDPA decode kernel is experimental but tracked as long-context path.
kernels.add('attention_bdpa_decode_f16.wgsl');

for (const kernel of kernels) {
  const kernelPath = path.join(WGSL_DIR, kernel);
  if (!fs.existsSync(kernelPath)) continue;
  const source = fs.readFileSync(kernelPath, 'utf8');
  assert.equal(
    hasMaxSubtractedSoftmax(source),
    true,
    `Kernel ${kernel} must include max-subtracted softmax logic.`
  );
}

console.log('attention-softmax-stability.test: ok');
