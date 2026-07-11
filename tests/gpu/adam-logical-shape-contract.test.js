import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../../src/gpu/kernels/backward/adam.js', import.meta.url),
  'utf8'
);

assert.match(
  source,
  /function tensorElementCount\(tensor\)[\s\S]*tensor\.shape\.reduce/
);
assert.equal(
  source.includes('Math.floor(params.buffer.size / bytesPerElement)'),
  false,
  'Adam must not update pooled bucket padding as logical parameter elements'
);
assert.equal(
  (source.match(/count \?\? tensorElementCount\(params\)/g) || []).length,
  2,
  'immediate and recorded Adam paths must use logical tensor element counts'
);

console.log('adam-logical-shape-contract.test: ok');
