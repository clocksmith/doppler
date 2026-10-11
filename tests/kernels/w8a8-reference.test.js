import assert from 'node:assert/strict';
import { quantizeW8A8Reference, matmulW8A8Reference, rescaleW8A8Reference } from './reference/w8a8.js';

const quantize = (values, rows = 1, K = values.length / rows) => quantizeW8A8Reference(new Float32Array(values), rows, K);
const tie = quantize([127, -127, 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5]);
assert.deepEqual([...tie.scales], [1]);
assert.deepEqual([...tie.codes], [127, -127, 0, 2, 2, 4, 0, -2, -2, -4, 0, 0]);
assert.equal(tie.rowStride, 16);
assert.deepEqual([...tie.bytes.subarray(0, 4)], [0, 0, 128, 63]);
assert.deepEqual([...quantize([0, -0, 0]).codes], [0, 0, 0, 0]);
assert.deepEqual([...quantize([0, 0]).scales], [1]);

// F32 division would produce the tie 1.5; exact input/stored-scale is below it.
const exactRatio = quantize([0.001003009034320712, 0.00001184656321129296]);
assert.equal(Math.fround(0.00001184656321129296 / exactRatio.scales[0]), 1.5);
assert.equal(exactRatio.codes[1], 1);

const a = quantize([127, 2, -3, 0, 0, 0], 2, 3);
const w = quantize([127, 4, 5, -127, 4, 5], 2, 3);
const product = matmulW8A8Reference(a.bytes, w.bytes, 2, 2, 3);
assert.deepEqual([...product.accumulators], [16122, -16136, 0, 0]);
assert.deepEqual([...product.output], [16122, -16136, 0, 0]);
assert.equal(rescaleW8A8Reference(16777217, 1, 1).output, 16777216);
assert.equal(rescaleW8A8Reference(3, Math.fround(0.1), Math.fround(0.2)).output,
  Math.fround(Math.fround(3 * Math.fround(0.1)) * Math.fround(0.2)));
assert(Object.is(rescaleW8A8Reference(-1, 2 ** -126, 0.5).output, -0));
assert.throws(() => rescaleW8A8Reference(2147483647, 2 ** 127, 1), /overflow/);
assert.throws(() => rescaleW8A8Reference(1, 2 ** -127, 1), /normal F32/);
for (const invalid of [NaN, Infinity, -Infinity, 2 ** -149, -(2 ** -149)]) {
  assert.throws(() => quantize([invalid]), /source/);
}
assert.throws(() => quantize([2 ** -126]), /scale/);
assert.throws(() => quantizeW8A8Reference(new Float32Array(133145), 1, 133145), /I32/);
const boundary = quantizeW8A8Reference(new Float32Array(133144).fill(127), 1, 133144);
assert.equal(matmulW8A8Reference(boundary.bytes, boundary.bytes, 1, 1, 133144).accumulators[0], 2147479576);
const corruptCode = a.bytes.slice(); corruptCode[4] = 128;
assert.throws(() => matmulW8A8Reference(corruptCode, w.bytes, 2, 2, 3), /code or tail/);
const corruptTail = a.bytes.slice(); corruptTail[7] = 1;
assert.throws(() => matmulW8A8Reference(corruptTail, w.bytes, 2, 2, 3), /code or tail/);
assert.throws(() => matmulW8A8Reference(a.bytes.slice(1), w.bytes, 2, 2, 3), /byte size/);
console.log('Independent W8A8 rational quantization, I32, rescale and invalid-input controls passed.');
