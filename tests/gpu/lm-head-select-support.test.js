import assert from 'node:assert/strict';

const { supportsLmHeadSelectLogitsF16 } = await import('../../src/gpu/kernels/lm-head-select-logits.js');
const { createWeightBuffer } = await import('../../src/gpu/weight-buffer.js');

// Plain objects stand in for GPU buffers: the predicate reads metadata only.
const head = (dtype, layout = 'row', materializations = null) =>
  createWeightBuffer({ label: `head-${dtype}` }, dtype, layout, [8, 256], `head-${dtype}`, materializations);

assert.equal(supportsLmHeadSelectLogitsF16(head('f16')), true, 'f16 row heads use the selected-row kernel');
assert.equal(supportsLmHeadSelectLogitsF16(head('f16', 'column')), true, 'f16 column heads use the selected-row kernel');
// q4k heads crashed scoreChoices with "requires f16 weights"; they must take the
// full LM-head matmul and gather the selected logits instead.
assert.equal(supportsLmHeadSelectLogitsF16(head('q4k')), false, 'q4k heads take the full matmul gather');
assert.equal(supportsLmHeadSelectLogitsF16(head('f32')), false, 'f32 heads take the full matmul gather');
assert.equal(
  supportsLmHeadSelectLogitsF16(head('q4k', 'row', { f16: { buffer: { label: 'f16-copy' }, layout: 'row' } })),
  true,
  'a q4k head with a resident f16 materialization uses that materialization'
);
assert.equal(supportsLmHeadSelectLogitsF16({ label: 'raw-gpu-buffer' }), false, 'raw buffers take the full matmul gather');
assert.equal(supportsLmHeadSelectLogitsF16(null), false);

console.log('lm-head-select-support.test: ok');
