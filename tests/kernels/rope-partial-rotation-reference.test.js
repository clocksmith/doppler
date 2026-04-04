import assert from 'node:assert/strict';

import { ropeRef } from './reference/rope.js';

const cos = new Float32Array([0.5]);
const sin = new Float32Array([Math.sqrt(3) / 2]);
const input = new Float32Array(512);

input[0] = 1;
input[64] = 100;
input[256] = 2;
input[320] = 200;

const output = ropeRef(input, cos, sin, 1, 1, 512, 0, { rotaryDim: 128 });

assert.ok(Math.abs(output[0] - (1 * 0.5 - 2 * sin[0])) < 1e-6);
assert.ok(Math.abs(output[256] - (1 * sin[0] + 2 * 0.5)) < 1e-6);
assert.equal(output[64], 100);
assert.equal(output[320], 200);

console.log('rope-partial-rotation-reference.test: ok');
