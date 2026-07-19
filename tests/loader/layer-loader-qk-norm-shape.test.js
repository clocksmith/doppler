import assert from 'node:assert/strict';

import { isWeightBuffer, tagBufferDtype } from '../../src/gpu/weight-buffer.js';
import { describeQKNormTensor } from '../../src/loader/layer-loader.js';

function makePaddedBuffer(label) {
  const buffer = {
    label,
    size: 4096,
    usage: 0,
    destroy() {},
    mapAsync() {},
    getMappedRange() {},
    unmap() {},
  };
  tagBufferDtype(buffer, 'f32');
  return buffer;
}

const qNorm = makePaddedBuffer('q_norm');
const kNorm = makePaddedBuffer('k_norm');
const location = { shape: [960], dtype: 'F32', layout: 'row' };
const describedQNorm = describeQKNormTensor(qNorm, location, 'model.layers.0.self_attn.q_norm.weight');
const describedKNorm = describeQKNormTensor(kNorm, location, 'model.layers.0.self_attn.k_norm.weight');

assert.equal(isWeightBuffer(describedQNorm), true);
assert.equal(isWeightBuffer(describedKNorm), true);
assert.deepEqual(describedQNorm.shape, [960]);
assert.deepEqual(describedKNorm.shape, [960]);
assert.equal(describedQNorm.buffer, qNorm);
assert.equal(describedKNorm.buffer, kNorm);
assert.equal(describedQNorm.dtype, 'f32');
assert.equal(describedKNorm.dtype, 'f32');

console.log('layer-loader-qk-norm-shape.test: ok');
