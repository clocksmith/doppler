import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { resolveMatmulBackwardDxVariant } from '../../src/gpu/kernels/backward/utils.js';
import { resolveMatmulBackwardOptions } from '../../src/experimental/training/autograd.js';

const [wrapper, shader, f16Shader] = await Promise.all([
  readFile(new URL('../../src/gpu/kernels/backward/utils.js', import.meta.url), 'utf8'),
  readFile(new URL('../../src/gpu/kernels/backward/matmul_backward.wgsl', import.meta.url), 'utf8'),
  readFile(new URL('../../src/gpu/kernels/backward/matmul_backward_f16w.wgsl', import.meta.url), 'utf8'),
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
assert.equal(resolveMatmulBackwardDxVariant({ dtype: 'f32' }), 'default');
assert.equal(resolveMatmulBackwardDxVariant({ dtype: 'f16' }), 'f16_weight');
assert.throws(
  () => resolveMatmulBackwardDxVariant({ dtype: 'q4k' }),
  /does not support weight dtype/
);
assert.match(f16Shader, /var<storage, read> W: array<u32>/);
assert.match(f16Shader, /unpack2x16float\(W\[index >> 1u\]\)/);
assert.match(
  f16Shader,
  /if \(u\.transpose_b == 0u\) \{[\s\S]*load_f16\(col \* u\.N \+ wt_row\)[\s\S]*\} else \{[\s\S]*load_f16\(wt_row \* u\.K \+ col\)/
);
assert.deepEqual(
  resolveMatmulBackwardOptions({ stopGradInputs: [1] }),
  { stopGradInputs: [1], computeGradInput: true, computeGradWeight: false }
);
assert.deepEqual(
  resolveMatmulBackwardOptions({ stopGradInputs: [0, 1] }),
  { stopGradInputs: [0, 1], computeGradInput: false, computeGradWeight: false }
);

console.log('matmul-backward-transpose-contract.test: ok');
