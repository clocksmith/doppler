import assert from 'node:assert/strict';

import { normBufferMatchesSize } from '../../src/inference/pipelines/text/attention/projections.js';

const paddedGpuBuffer = { size: 4096 };
const projectionNormWeight = { shape: [960] };

assert.equal(
  normBufferMatchesSize(paddedGpuBuffer, 960, projectionNormWeight),
  true,
  'logical weight shape must win over allocator-padded GPU buffer bytes'
);
assert.equal(normBufferMatchesSize(paddedGpuBuffer, 64, projectionNormWeight), false);
assert.equal(
  normBufferMatchesSize(null, 960, projectionNormWeight),
  false,
  'logical shape must not make an absent reused-K norm buffer executable'
);
assert.equal(normBufferMatchesSize({ size: 3840 }, 960), true);

console.log('qk-norm-weight-shape.test: ok');
