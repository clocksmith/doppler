import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../../src/inference/pipelines/text/lora-apply.js', import.meta.url),
  'utf8'
);

assert.equal(
  (source.match(/transposeB: false/g) || []).length,
  4,
  'runtime and recorded LoRA A/B matmuls must consume exported [K,r] and [r,N] tensors'
);
assert.doesNotMatch(source, /transposeB: 'auto'/);

console.log('lora-runtime-matmul-layout-contract.test: ok');
