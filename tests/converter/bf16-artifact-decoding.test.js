import assert from 'node:assert/strict';
import { bf16ToFloat32, float32ToBFloat16, transformTensorBytes } from '../../src/converter/tensor-transform.js';

for (let bits = 0; bits <= 0xffff; bits++) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 7) & 0xff;
  const fraction = bits & 0x7f;
  const expected = exponent === 255
    ? fraction ? NaN : sign * Infinity
    : exponent === 0 ? sign * fraction * 2 ** -133
      : sign * (1 + fraction / 128) * 2 ** (exponent - 127);
  assert.ok(Object.is(bf16ToFloat32(bits), expected), `BF16 pattern ${bits.toString(16)}`);
  // Encoding another value between reads must not leak scratch state.
  float32ToBFloat16(17.25);
  assert.ok(Object.is(bf16ToFloat32(bits), expected));
}

const source = new Uint16Array([0, 0x8000, 0x3f00, 0x3f80, 0xc000, 0x7f80, 0x7fc0]);
const original = source.slice();
const result = transformTensorBytes({ name: 'model.layers.0.input_layernorm.weight',
  shape: [source.length], dtype: 'BF16' }, new Uint8Array(source.buffer), { targetQuant: 'f16' });
assert.equal(result.outDtype, 'F16');
assert.deepEqual(new Uint16Array(result.tensorData.buffer, result.tensorData.byteOffset, source.length),
  new Uint16Array([0, 0x8000, 0x3800, 0x3c00, 0xc000, 0x7c00, 0x7e00]));
assert.deepEqual(source, original, 'Artifact conversion must not mutate source bytes');
console.log('bf16-artifact-decoding.test: ok');
