import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const [wrapper, shader] = await Promise.all([
  readFile(new URL('../../src/gpu/kernels/backward/utils.js', import.meta.url), 'utf8'),
  readFile(new URL('../../src/gpu/kernels/backward/matmul_backward.wgsl', import.meta.url), 'utf8'),
]);

assert.equal(
  (wrapper.match(/view\.setUint32\(16, transposeB \? 1 : 0, true\);/g) || []).length,
  2,
  'immediate and recorded matmul backward paths must encode transposeB identically'
);
assert.match(
  shader,
  /if \(u\.transpose_b == 0u\) \{[\s\S]*W\[col \* u\.N \+ wt_row\][\s\S]*\} else \{[\s\S]*W\[wt_row \* u\.K \+ col\]/,
  'transposeB=false must read W[K,N], while transposeB=true must read W[N,K]'
);

console.log('matmul-backward-transpose-contract.test: ok');
