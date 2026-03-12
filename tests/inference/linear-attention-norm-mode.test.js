import assert from 'node:assert/strict';

const { inferLinearNormMode, applyLinearNormWeightOffset } = await import(
  '../../src/inference/pipelines/text/linear-attention.js'
);

const layout = { headVDim: 128, valueDim: 2048 };

// --- inferLinearNormMode: Float32Array ---

{
  const shared = inferLinearNormMode(new Float32Array(layout.headVDim), layout);
  assert.equal(shared, 'shared');
}

{
  const perHead = inferLinearNormMode(new Float32Array(layout.valueDim), layout);
  assert.equal(perHead, 'per_head');
}

// --- inferLinearNormMode: Float64Array ---

{
  const shared = inferLinearNormMode(new Float64Array(layout.headVDim), layout);
  assert.equal(shared, 'shared');
}

{
  const perHead = inferLinearNormMode(new Float64Array(layout.valueDim), layout);
  assert.equal(perHead, 'per_head');
}

// --- inferLinearNormMode: Uint16Array (f16-like) ---

{
  const shared = inferLinearNormMode(new Uint16Array(layout.headVDim), layout);
  assert.equal(shared, 'shared');
}

{
  const perHead = inferLinearNormMode(new Uint16Array(layout.valueDim), layout);
  assert.equal(perHead, 'per_head');
}

// --- inferLinearNormMode: Int16Array ---

{
  const shared = inferLinearNormMode(new Int16Array(layout.headVDim), layout);
  assert.equal(shared, 'shared');
}

{
  const perHead = inferLinearNormMode(new Int16Array(layout.valueDim), layout);
  assert.equal(perHead, 'per_head');
}

// --- inferLinearNormMode: generic ArrayBuffer.isView (Uint8Array) ---

{
  const shared = inferLinearNormMode(new Uint8Array(layout.headVDim), layout);
  assert.equal(shared, 'shared');
}

{
  const perHead = inferLinearNormMode(new Uint8Array(layout.valueDim), layout);
  assert.equal(perHead, 'per_head');
}

// --- inferLinearNormMode: raw ArrayBuffer (f32 byte count assumed) ---

{
  const shared = inferLinearNormMode(
    new ArrayBuffer(layout.headVDim * Float32Array.BYTES_PER_ELEMENT),
    layout
  );
  assert.equal(shared, 'shared');
}

{
  const perHead = inferLinearNormMode(
    new ArrayBuffer(layout.valueDim * Float32Array.BYTES_PER_ELEMENT),
    layout
  );
  assert.equal(perHead, 'per_head');
}

// --- inferLinearNormMode: object with .size/.dtype ---

{
  const shared = inferLinearNormMode(
    { size: layout.headVDim * Float32Array.BYTES_PER_ELEMENT, dtype: 'f32' },
    layout
  );
  assert.equal(shared, 'shared');
}

{
  const perHead = inferLinearNormMode(
    { size: layout.valueDim * Uint16Array.BYTES_PER_ELEMENT, dtype: 'f16' },
    layout
  );
  assert.equal(perHead, 'per_head');
}

{
  const bf16Shared = inferLinearNormMode(
    { size: layout.headVDim * 2, dtype: 'bf16' },
    layout
  );
  assert.equal(bf16Shared, 'shared');
}

// --- inferLinearNormMode: non-matching element count → null ---

{
  const noMatch = inferLinearNormMode(new Float32Array(999), layout);
  assert.equal(noMatch, null);
}

{
  const noMatch = inferLinearNormMode(new Uint16Array(7), layout);
  assert.equal(noMatch, null);
}

{
  const noMatch = inferLinearNormMode(
    new ArrayBuffer(333 * Float32Array.BYTES_PER_ELEMENT),
    layout
  );
  assert.equal(noMatch, null);
}

{
  const noMatch = inferLinearNormMode(
    { size: 512 * Float32Array.BYTES_PER_ELEMENT, dtype: 'f32' },
    layout
  );
  assert.equal(noMatch, null);
}

// --- inferLinearNormMode: degenerate inputs → null ---

{
  const empty = inferLinearNormMode(new Float32Array(0), layout);
  assert.equal(empty, null);
}

{
  const nullWeight = inferLinearNormMode(null, layout);
  assert.equal(nullWeight, null);
}

{
  const undefinedWeight = inferLinearNormMode(undefined, layout);
  assert.equal(undefinedWeight, null);
}

{
  const noSize = inferLinearNormMode({}, layout);
  assert.equal(noSize, null);
}

// --- inferLinearNormMode: different projection layouts ---

{
  const smallLayout = { headVDim: 64, valueDim: 512 };
  const shared = inferLinearNormMode(new Float32Array(64), smallLayout);
  assert.equal(shared, 'shared');
  const perHead = inferLinearNormMode(new Float32Array(512), smallLayout);
  assert.equal(perHead, 'per_head');
  const noMatch = inferLinearNormMode(new Float32Array(128), smallLayout);
  assert.equal(noMatch, null);
}

// --- inferLinearNormMode: Qwen 3.5 real dimensions ---

{
  const qwenLayout = { headVDim: 128, valueDim: 2048 };
  const shared = inferLinearNormMode(
    { size: 128 * Uint16Array.BYTES_PER_ELEMENT, dtype: 'f16' },
    qwenLayout
  );
  assert.equal(shared, 'shared');
  const perHead = inferLinearNormMode(
    { size: 2048 * Uint16Array.BYTES_PER_ELEMENT, dtype: 'f16' },
    qwenLayout
  );
  assert.equal(perHead, 'per_head');
}

// --- applyLinearNormWeightOffset: identity regardless of flag ---

{
  const values = new Float32Array([0.25, -0.5, 0.75, 1.5]);
  const result = applyLinearNormWeightOffset(values, false);
  assert.equal(result, values);
  assert.deepEqual(Array.from(result), [0.25, -0.5, 0.75, 1.5]);
}

{
  const values = new Float32Array([0.25, -0.5, 0.75, 1.5]);
  const result = applyLinearNormWeightOffset(values, true);
  assert.equal(result, values);
  assert.deepEqual(Array.from(result), [0.25, -0.5, 0.75, 1.5]);
}

{
  const empty = new Float32Array(0);
  const result = applyLinearNormWeightOffset(empty, true);
  assert.equal(result, empty);
  assert.equal(result.length, 0);
}

// --- applyLinearNormWeightOffset: rejects non-Float32Array ---

assert.throws(
  () => applyLinearNormWeightOffset(new Float64Array([1]), false),
  /Float32Array/
);

assert.throws(
  () => applyLinearNormWeightOffset(new Uint16Array([1]), true),
  /Float32Array/
);

assert.throws(
  () => applyLinearNormWeightOffset([1, 2, 3], false),
  /Float32Array/
);

console.log('linear-attention-norm-mode.test: ok');
