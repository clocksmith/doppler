import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(
  new URL('../../src/loader/tensors/tensor-loader.js', import.meta.url),
  'utf8'
);

assert.match(
  source,
  /blocksPerRow\s*=\s*Math\.ceil\(K\s*\/\s*QK_K\);/,
  'rowwise Q4K parity probes must compute packed blocks per row'
);
assert.match(
  source,
  /numBlocks\s*=\s*rowCount\s*\*\s*blocksPerRow;/,
  'rowwise Q4K parity probes must populate numBlocks for packed-block accounting'
);
assert.match(
  source,
  /denseElementCount\s*=\s*rowCount\s*\*\s*K;/,
  'rowwise Q4K parity probes must use rows * K for logical dense output size'
);
assert.match(
  source,
  /requestedOutputBytes\s*=\s*denseElementCount\s*\*\s*bytesPerElem;/,
  'Q4K parity readback size must come from logical dense element count'
);
assert.equal(
  source.includes('requestedOutputBytes = numBlocks * QK_K * bytesPerElem'),
  false,
  'Q4K parity readback must not use padded packed-block length for rowwise tensors'
);

console.log('tensor-loader-q4k-parity-probe-contract.test: ok');
