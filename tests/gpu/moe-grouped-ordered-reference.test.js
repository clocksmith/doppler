import assert from 'node:assert/strict';
import { f32Bits, f32FromBits, fmaF32, laneTreeDotF32, packedValueF32,
  orderedPackedFixture, orderedInputFixture, compareF32Bits } from '../../tools/experiments/moe-grouped-ordered-reference.js';

// A fused result survives cancellation where separately rounded arithmetic loses it.
const a = 1 + 2 ** -23, b = 1 - 2 ** -23;
assert.equal(fmaF32(a, b, -1), -(2 ** -46));
assert.equal(Math.fround(Math.fround(a * b) - 1), 0);
// Double rounding loses the tiny addend that decides a midpoint.
assert.equal(f32Bits(fmaF32(a, 1.5, -(2 ** -80))), 0x3fc00001);
assert.equal(f32Bits(Math.fround(a * 1.5 - 2 ** -80)), 0x3fc00002);
assert.equal(f32Bits(fmaF32(2 ** -126, 2 ** -24, 0)), 0);
assert.equal(f32Bits(fmaF32(2 ** -126, 3 * 2 ** -24, 0)), 2);
assert.equal(f32Bits(fmaF32(-0, 1, -0)), 0x80000000);
assert.equal(fmaF32(2 ** 127, 2, 0), Infinity);
assert.throws(() => fmaF32(Infinity, 1, 0), /Finite F32/);

// Literal original-format blocks: Q4_0 nibble halves, Q4_1 signed additive
// minimum (not subtraction/absolute value), and Q8_0 signed -128/+127.
for (const [dtype, size, header] of [['q4_0', 18, 2], ['q4_1', 20, 4], ['q8_0', 34, 2]]) {
  const raw = new Uint8Array(size), view = new DataView(raw.buffer);
  view.setUint16(0, 0xbc00, true); // d=-1, exactly representable.
  if (dtype === 'q4_1') view.setUint16(2, 0xc000, true); // m=-2.
  if (dtype === 'q8_0') {
    raw[2] = 0x80; raw[3] = 0x7f;
    assert.equal(packedValueF32(raw, dtype, 0), 128);
    assert.equal(packedValueF32(raw, dtype, 1), -127);
  } else {
    raw[header] = 0xf0;
    assert.equal(packedValueF32(raw, dtype, 0), dtype === 'q4_0' ? 8 : -2);
    assert.equal(packedValueF32(raw, dtype, 16), dtype === 'q4_0' ? -7 : -17);
  }
}
// Explicit cancellation across 32-lane accumulation and the baseline tree.
const x = new Float32Array(96).fill(1), w = new Float32Array(96);
w[0] = 2 ** 24; w[32] = 1; w[64] = -(2 ** 24); w[1] = 3;
assert.equal(laneTreeDotF32(x, w), 3);
assert.equal(Array.from(w).reduce((sum, value) => sum + value, 0), 4);
assert.deepEqual(compareF32Bits(new Float32Array([0]), new Float32Array([-0])).first,
  { index: 0, actual: 0, expected: -0, actualBits: 0, expectedBits: 0x80000000, finite: true });
assert.equal(compareF32Bits(new Float32Array([NaN]), new Float32Array([NaN])).passed, false);
assert.throws(() => compareF32Bits(new Float32Array(1), new Float32Array(2)), /coverage/);
for (const dtype of ['q4_0', 'q4_1', 'q8_0']) {
  const raw = orderedPackedFixture(dtype, [2, 3, 64], 2);
  const weights = Float32Array.from({ length: 64 }, (_, i) => packedValueF32(raw, dtype, i));
  const input = orderedInputFixture(1, 64, 5);
  assert.ok(Number.isFinite(laneTreeDotF32(input, weights)));
  assert.equal(f32FromBits(f32Bits(weights[0])), weights[0]);
}
console.log('moe-grouped-ordered-reference: exact F32 rounding, signed packed formats, lane/tree and bitwise gates pass');
