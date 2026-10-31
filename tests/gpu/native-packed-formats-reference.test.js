import assert from 'node:assert/strict';
import { decodePacked, packedFixture } from '../helpers/q4_0-fixture.js';

// Independent byte sentinels for the reference used by the physical basis-vector
// fixture. These expectations do not call the shader decoder or its bit formulas.
const q41 = new Uint8Array(20), v41 = new DataView(q41.buffer);
v41.setUint16(0, 0xb400, true); // d = -0.25
v41.setUint16(2, 0x3800, true); // minimum = +0.5
q41.fill(0xf0, 4);
const a = decodePacked('q4_1', q41, 1, 32);
assert(a.subarray(0, 16).every(x => x === 0.5));
assert(a.subarray(16).every(x => x === -3.25));

const q50 = new Uint8Array(44), v50 = new DataView(q50.buffer);
for (const base of [0, 22]) {
  v50.setUint16(base, 0x3800, true);
  v50.setUint32(base + 2, 0x80010001, true);
  q50.fill(0xf0, base + 6, base + 22);
}
const b = decodePacked('q5_0', q50, 2, 32);
for (const base of [0, 32]) {
  assert.equal(b[base], 0);
  assert.equal(b[base + 1], -8);
  assert.equal(b[base + 15], -8);
  assert.equal(b[base + 16], 7.5);
  assert.equal(b[base + 17], -0.5);
  assert.equal(b[base + 31], 7.5);
}

const q6 = new Uint8Array(420), v6 = new DataView(q6.buffer);
const scales = [-128, 127, -1, 1, -64, 63, 0, 17, 127, -128, 1, -1, 63, -64, 17, 0];
for (const base of [0, 210]) {
  q6.fill(0xff, base, base + 64);
  q6.fill(0xff, base + 128, base + 160);
  scales.forEach((scale, index) => v6.setInt8(base + 192 + index, scale));
  v6.setUint16(base + 208, base === 0 ? 0x2400 : 0xa400, true); // +/- 1/64
}
const c = decodePacked('q6k', q6, 2, 256);
for (let row = 0; row < 2; row++) for (let group = 0; group < 16; group++) {
  const quant = group < 8 ? 31 : -32;
  const expected = (row === 0 ? 1 : -1) * scales[group] * quant / 64;
  assert(c.subarray(row * 256 + group * 16, row * 256 + (group + 1) * 16).every(x => x === expected));
}
for (const dtype of ['q4_1', 'q5_0', 'q6k']) {
  const K = dtype === 'q6k' ? 768 : 96;
  const decoded = decodePacked(dtype, packedFixture(dtype, 7, K), 7, K);
  assert(decoded.every(Number.isFinite));
  assert(decoded.some(x => x < 0) && decoded.some(x => x > 0));
}
console.log('Native packed reference byte sentinels pass; physical vector decoder remains a separate gate.');
