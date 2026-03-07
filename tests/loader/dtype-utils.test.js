import assert from 'node:assert/strict';

globalThis.GPUBuffer = class GPUBuffer {};

const { applyBufferLayout, shouldDequantizeToF16 } = await import('../../src/loader/dtype-utils.js');
const { getBufferDtype } = await import('../../src/gpu/weight-buffer.js');

{
  const upper = new GPUBuffer();
  applyBufferLayout(upper, { dtype: 'F16', role: 'norm' });
  assert.equal(getBufferDtype(upper), 'f16');
}

{
  const lower = new GPUBuffer();
  applyBufferLayout(lower, { dtype: 'f16', role: 'norm' });
  assert.equal(getBufferDtype(lower), 'f16');
}

{
  const explicit = new GPUBuffer();
  applyBufferLayout(explicit, { dtype: 'F32', role: 'norm' }, 'F32');
  assert.equal(getBufferDtype(explicit), 'f32');
}

{
  assert.equal(shouldDequantizeToF16({ role: 'matmul' }), true);
  assert.equal(shouldDequantizeToF16({ role: 'norm' }), false);
  assert.throws(
    () => shouldDequantizeToF16({}),
    /Tensor role is required/
  );
}

console.log('dtype-utils.test: ok');
