import assert from 'node:assert/strict';

import { ropeRef } from './reference/rope.js';

{
  const cos = new Float32Array([0.5]);
  const sin = new Float32Array([Math.sqrt(3) / 2]);
  const input = new Float32Array(512);

  input[0] = 1;
  input[64] = 2;
  input[128] = 100;
  input[256] = 200;
  input[320] = 300;

  const output = ropeRef(input, cos, sin, 1, 1, 512, 0, { rotaryDim: 128 });

  assert.ok(Math.abs(output[0] - (1 * 0.5 - 2 * sin[0])) < 1e-6);
  assert.ok(Math.abs(output[64] - (1 * sin[0] + 2 * 0.5)) < 1e-6);
  assert.equal(output[128], 100);
  assert.equal(output[256], 200);
  assert.equal(output[320], 300);
}

{
  const cos = new Float32Array([0.5, 0.25]);
  const sin = new Float32Array([Math.sqrt(3) / 2, Math.sqrt(15) / 4]);
  const input = new Float32Array(8);

  input[0] = 1;
  input[1] = 3;
  input[2] = 100;
  input[3] = 200;
  input[4] = 2;
  input[5] = 4;
  input[6] = 300;
  input[7] = 400;

  const output = ropeRef(input, cos, sin, 1, 1, 8, 0, { rotaryDim: 4, pairSpanDim: 8 });

  assert.ok(Math.abs(output[0] - (1 * cos[0] - 2 * sin[0])) < 1e-6);
  assert.ok(Math.abs(output[4] - (1 * sin[0] + 2 * cos[0])) < 1e-6);
  assert.ok(Math.abs(output[1] - (3 * cos[1] - 4 * sin[1])) < 1e-6);
  assert.ok(Math.abs(output[5] - (3 * sin[1] + 4 * cos[1])) < 1e-6);
  assert.equal(output[2], 100);
  assert.equal(output[3], 200);
  assert.equal(output[6], 300);
  assert.equal(output[7], 400);
}

console.log('rope-partial-rotation-reference.test: ok');
