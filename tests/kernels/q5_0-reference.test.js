import assert from 'node:assert/strict';
import { dequantQ5_0Ref } from './reference/dequant.js';

// Hand-derived from the GGUF Q5_0 block definition: d = 0.5, qh bits 0 and 31 set,
// qs[j] = j | ((15 - j) << 4). Value j is (j | bit_j << 4) - 16; value 16 + j is ((15 - j) | bit_{16+j} << 4) - 16.
const block = new Uint8Array(22);
const view = new DataView(block.buffer);
view.setUint16(0, 0x3800, true);
view.setUint32(2, 0x80000001, true);
for (let j = 0; j < 16; j++) block[6 + j] = j | ((15 - j) << 4);
const expected = [];
for (let j = 0; j < 16; j++) expected[j] = ((j | (j === 0 ? 16 : 0)) - 16) * 0.5;
for (let j = 0; j < 16; j++) expected[16 + j] = (((15 - j) | (j === 15 ? 16 : 0)) - 16) * 0.5;
assert.deepEqual(Array.from(dequantQ5_0Ref(block, 1)), expected);
assert.equal(expected[0], 0);
assert.equal(expected[1], -7.5);
assert.equal(expected[16], -0.5);
assert.equal(expected[31], 0);
console.log('q5_0-reference.test: ok');
