import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const kernelsDir = path.join(repoRoot, 'src/gpu/kernels');

const sourceKernel = 'attention_streaming.wgsl';
const generatedVariants = [
  'attention_streaming_f16.wgsl',
  'attention_streaming_f16kv.wgsl',
];

async function readMaxHeadDim(kernelFile) {
  const source = await fs.readFile(path.join(kernelsDir, kernelFile), 'utf8');
  const match = source.match(/const MAX_HEAD_DIM: u32 = (\d+)u;/);
  assert.ok(match, `${kernelFile} must declare MAX_HEAD_DIM`);
  return Number(match[1]);
}

const sourceMaxHeadDim = await readMaxHeadDim(sourceKernel);
assert.ok(
  sourceMaxHeadDim >= 512,
  `${sourceKernel} must support Gemma 4 global-attention headDim=512 on the streaming path`
);

for (const kernelFile of generatedVariants) {
  const variantMaxHeadDim = await readMaxHeadDim(kernelFile);
  assert.equal(
    variantMaxHeadDim,
    sourceMaxHeadDim,
    `${kernelFile} must keep MAX_HEAD_DIM in sync with ${sourceKernel}`
  );
}

console.log('attention-streaming-head-dim-contract.test: ok');
