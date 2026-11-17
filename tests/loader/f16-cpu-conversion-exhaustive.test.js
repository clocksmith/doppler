import assert from 'node:assert/strict';

import { float16ToFloat32 } from '../../src/formats/float16.js';
import { convertF16ToF32CPU } from '../../src/loader/tensors/load-executor.js';

// Cover every binary16 encoding, including signed zero, subnormals and NaNs.
const backing = new Uint16Array(65538);
for (let i = 0; i < 65536; i++) backing[i + 1] = i;
const input = backing.subarray(1, 65537);
const output = convertF16ToF32CPU(input);
assert.equal(output.length, input.length);
for (let i = 0; i < input.length; i++) {
  assert.ok(Object.is(output[i], Math.fround(float16ToFloat32(input[i]))), `binary16 encoding ${i}`);
}
assert.equal(convertF16ToF32CPU(new Uint16Array(0)).length, 0);
