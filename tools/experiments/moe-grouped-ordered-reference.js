// Independent CPU fixtures only. This module is never a runtime fallback.
import assert from 'node:assert/strict';

const scratch = new DataView(new ArrayBuffer(4));
export function f32Bits(value) {
  scratch.setFloat32(0, value, true);
  return scratch.getUint32(0, true);
}
export function f32FromBits(bits) {
  scratch.setUint32(0, bits, true);
  return scratch.getFloat32(0, true);
}
function dyadic(value) {
  const bits = f32Bits(value), exponent = (bits >>> 23) & 255;
  assert.notEqual(exponent, 255, 'Finite F32 fixture values required');
  const significand = BigInt((bits & 0x7fffff) | (exponent ? 0x800000 : 0));
  return { integer: bits >>> 31 ? -significand : significand,
    exponent: exponent ? exponent - 150 : -149, negative: Boolean(bits >>> 31) };
}
function roundEven(integer, shift) {
  if (shift <= 0) return integer << BigInt(-shift);
  const count = BigInt(shift), truncated = integer >> count;
  const remainder = integer - (truncated << count), half = 1n << (count - 1n);
  return truncated + BigInt(remainder > half || (remainder === half && Boolean(truncated & 1n)));
}
// Binary64 a*b+c is not a general F32 FMA oracle: a tiny addend can decide
// which side of a F32 midpoint an exact product occupies. Use exact dyadics.
export function fmaF32(a, b, c) {
  const left = dyadic(a), right = dyadic(b), addend = dyadic(c);
  const productExponent = left.exponent + right.exponent;
  const exponent = Math.min(productExponent, addend.exponent);
  const integer = (left.integer * right.integer << BigInt(productExponent - exponent))
    + (addend.integer << BigInt(addend.exponent - exponent));
  if (integer === 0n) {
    return left.integer * right.integer === 0n && addend.integer === 0n
      && left.negative !== right.negative && addend.negative ? -0 : 0;
  }
  const sign = integer < 0n ? 0x80000000 : 0, magnitude = integer < 0n ? -integer : integer;
  const length = magnitude.toString(2).length;
  let binaryExponent = length - 1 + exponent;
  if (binaryExponent < -126) {
    return f32FromBits((sign | Number(roundEven(magnitude, -149 - exponent))) >>> 0);
  }
  let significand = roundEven(magnitude, length - 24);
  if (significand === 0x1000000n) { significand >>= 1n; binaryExponent++; }
  if (binaryExponent > 127) return sign ? -Infinity : Infinity;
  return f32FromBits((sign | ((binaryExponent + 127) << 23) | Number(significand & 0x7fffffn)) >>> 0);
}
export function laneTreeDotF32(input, weights) {
  assert.equal(input.length, weights.length);
  const sums = new Float32Array(32);
  for (let lane = 0; lane < 32; lane++) {
    for (let k = lane; k < input.length; k += 32) sums[lane] = fmaF32(input[k], weights[k], sums[lane]);
  }
  for (let stride = 16; stride > 0; stride >>= 1) {
    for (let lane = 0; lane < stride; lane++) sums[lane] = Math.fround(sums[lane] + sums[lane + stride]);
  }
  return sums[0];
}
function half(bits) {
  const exponent = (bits >>> 10) & 31, fraction = bits & 1023;
  assert.notEqual(exponent, 31, 'Finite binary16 fixture scale required');
  return (bits & 0x8000 ? -1 : 1) * (exponent ? 1024 + fraction : fraction)
    * 2 ** (exponent ? exponent - 25 : -24);
}
const blockBytes = { q4_0: 18, q4_1: 20, q8_0: 34 };
export function packedValueF32(raw, dtype, index) {
  const bytes = blockBytes[dtype];
  assert.ok(bytes, 'Explicit supported packed dtype required');
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const offset = Math.floor(index / 32) * bytes, local = index % 32;
  const scale = half(view.getUint16(offset, true));
  if (dtype === 'q8_0') return Math.fround(scale * view.getInt8(offset + 2 + local));
  const byte = raw[offset + (dtype === 'q4_1' ? 4 : 2) + local % 16];
  const quant = (byte >>> (local < 16 ? 0 : 4)) & 15;
  if (dtype === 'q4_0') return Math.fround(scale * (quant - 8));
  return Math.fround(Math.fround(scale * quant) + half(view.getUint16(offset + 2, true)));
}
export function orderedPackedFixture(dtype, shape, salt) {
  assert.ok(blockBytes[dtype]);
  const count = shape.reduce((a, b) => a * b, 1);
  assert.equal(shape.at(-1) % 32, 0);
  const blocks = count / 32, bytes = blockBytes[dtype];
  const raw = new Uint8Array(Math.ceil(blocks * bytes / 4) * 4), view = new DataView(raw.buffer);
  const scales = [0x0401, 0x1401, 0x2401, 0x3401, 0x3c01];
  const minimums = [0xb801, 0x1001, 0xc001, 0x0000];
  for (let block = 0; block < blocks; block++) {
    const offset = block * bytes;
    view.setUint16(offset, scales[(block + salt) % scales.length] | ((block + salt) % 3 ? 0 : 0x8000), true);
    if (dtype === 'q4_1') view.setUint16(offset + 2, minimums[(block + salt) % minimums.length], true);
    const data = offset + (dtype === 'q4_1' ? 4 : 2);
    if (dtype === 'q8_0') {
      for (let k = 0; k < 32; k++) raw[data + k] = (block * 13 + k * 67 + salt) & 255;
    } else {
      for (let k = 0; k < 16; k++) raw[data + k] = ((block * 3 + k * 7 + salt) & 15)
        | (((block * 11 + k * 5 + salt + 1) & 15) << 4);
    }
  }
  return raw;
}
export function orderedInputFixture(rows, K, salt) {
  const magnitudes = [65536, 2 ** -16, 1, 1 + 2 ** -23, 256, 2 ** -8, 3, 1 - 2 ** -24];
  return Float32Array.from({ length: rows * K }, (_, i) => {
    const k = i % K, row = Math.floor(i / K), block = Math.floor(k / 32);
    const magnitude = magnitudes[(k + row + salt) % magnitudes.length];
    return ((block + row + salt) % 2 ? -1 : 1) * magnitude;
  });
}
export function compareF32Bits(actual, expected) {
  assert.equal(actual.length, expected.length, 'Complete output coverage required');
  let mismatches = 0, first = null, maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    const actualBits = f32Bits(actual[i]), expectedBits = f32Bits(expected[i]);
    const finite = Number.isFinite(actual[i]) && Number.isFinite(expected[i]);
    if (!finite || actualBits !== expectedBits) {
      mismatches++;
      first ??= { index: i, actual: actual[i], expected: expected[i], actualBits, expectedBits, finite };
    }
    maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
  }
  return { passed: mismatches === 0, elements: actual.length, mismatches, first, maxAbs };
}
