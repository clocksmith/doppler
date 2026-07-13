import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../../src/experimental/training/lora.js', import.meta.url),
  'utf8'
);

assert.equal(
  (source.match(/transposeB: false/g) || []).length,
  4,
  'both LoRA forward matmuls and both autograd records must declare [K,N] weight layout'
);
assert.match(
  source,
  /runMatmul\(a, b, tokens, this\.rank, this\.A\.shape\[0\], \{[\s\S]*?transposeB: false,[\s\S]*?\}\),[\s\S]*?\{ M: tokens, N: this\.rank, K: this\.A\.shape\[0\], transposeB: false \}/
);
assert.match(
  source,
  /runMatmul\(a, b, tokens, this\.B\.shape\[1\], this\.rank, \{[\s\S]*?transposeB: false,[\s\S]*?\}\),[\s\S]*?\{ M: tokens, N: this\.B\.shape\[1\], K: this\.rank, transposeB: false \}/
);

console.log('lora-matmul-layout-contract.test: ok');
