import assert from 'node:assert/strict';

import {
  QK4_K_BLOCK_SIZE,
  QK_K,
  calculateQuantizationError,
  dequantizeQ4KM,
  float16ToFloat32,
  float32ToFloat16,
  getQ4KSize,
  getQuantizedSize,
  quantizeF16ToQ4KM,
  quantizeQ4KBlock,
  quantizeToQ4KM,
  quantizeToQ4KMColumnWise,
  quantizeToQ4KMRowWise,
  shouldQuantize,
  transposeF32,
} from '../../src/converter/quantizer.js';

{
  const input = [0, 1, -1, 0.5, -0.25, 12.25, -24.5];
  for (const value of input) {
    const f16 = float32ToFloat16(value);
    const roundTrip = float16ToFloat32(f16);
    assert.ok(Number.isFinite(roundTrip));
    assert.ok(Math.abs(roundTrip - value) < 0.05, `round-trip drift too high for ${value}`);
  }
}

{
  const posInf = float16ToFloat32(float32ToFloat16(Infinity));
  const negInf = float16ToFloat32(float32ToFloat16(-Infinity));
  const nan = float16ToFloat32(float32ToFloat16(NaN));
  assert.equal(posInf, Infinity);
  assert.equal(negInf, -Infinity);
  assert.ok(Number.isNaN(nan));
}

{
  const blockInput = new Float32Array(QK_K);
  for (let i = 0; i < blockInput.length; i += 1) {
    blockInput[i] = Math.sin(i / 16);
  }
  const block = quantizeQ4KBlock(blockInput, 0);
  assert.equal(block.byteLength, QK4_K_BLOCK_SIZE);
}

{
  const shape = [2, 300];
  const data = new Float32Array(shape[0] * shape[1]);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = ((i % 29) - 14) / 7;
  }

  const flat = quantizeToQ4KM(data, shape);
  const row = quantizeToQ4KMRowWise(data, shape);
  const col = quantizeToQ4KMColumnWise(data, shape);

  assert.equal(flat.quantized.length, getQ4KSize(shape, 'flat'));
  assert.equal(row.quantized.length, getQ4KSize(shape, 'row'));
  assert.equal(col.quantized.length, getQ4KSize(shape, 'col'));
  assert.equal(getQ4KSize(shape, 'unknown-layout'), getQ4KSize(shape, 'flat'));
  assert.equal(getQuantizedSize(shape), getQ4KSize(shape, 'flat'));

  assert.equal(col.transposedShape[0], shape[1]);
  assert.equal(col.transposedShape[1], shape[0]);

  const dequantized = dequantizeQ4KM(flat.quantized, flat.numBlocks, shape);
  assert.equal(dequantized.length, data.length);
  const err = calculateQuantizationError(data, dequantized);
  assert.ok(Number.isFinite(err.mse));
  assert.ok(Number.isFinite(err.maxError));
  assert.ok(Number.isFinite(err.snr));
}

{
  await assert.rejects(
    async () => quantizeToQ4KM(new Float32Array(3), [2, 2]),
    /doesn't match shape/
  );
  await assert.rejects(
    async () => calculateQuantizationError(new Float32Array(2), new Float32Array(3)),
    /Length mismatch/
  );
}

{
  const shape = [1, 300];
  const f16 = new Uint16Array(shape[0] * shape[1]);
  for (let i = 0; i < f16.length; i += 1) {
    f16[i] = float32ToFloat16((i % 17) / 4);
  }
  const result = quantizeF16ToQ4KM(f16, shape);
  assert.ok(result.quantized.length > 0);
  assert.ok(result.numBlocks >= 1);
}

{
  const matrix = new Float32Array([
    1, 2, 3,
    4, 5, 6,
  ]);
  const transposed = transposeF32(matrix, [2, 3]);
  assert.deepEqual([...transposed], [1, 4, 2, 5, 3, 6]);
}

{
  assert.throws(
    () => shouldQuantize('tensor.invalid', null),
    /reduce/
  );
  assert.equal(shouldQuantize('tensor.invalid', []), false);

  const shouldBeQuantized = shouldQuantize(
    'model.layers.0.self_attn.q_proj.weight',
    [1024, 1024]
  );
  assert.equal(shouldBeQuantized, true);

  const excluded = shouldQuantize(
    'model.layers.0.self_attn.q_proj.weight',
    [1024, 1024],
    { modulesToNotConvert: ['model.layers.*.self_attn.q_proj.weight'] }
  );
  assert.equal(excluded, false);
}

console.log('quantizer.test: ok');
