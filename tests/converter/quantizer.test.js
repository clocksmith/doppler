import assert from 'node:assert/strict';

import {
  QK4_K_BLOCK_SIZE,
  QK_K,
  calculateQuantizationError,
  dequantizeQ4KM,
  dequantizeQ4KMRowWise,
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
import { decodeQ4KBlockReference } from '../../tools/lib/q4k-projection-reference.js';

// After coding values, neighboring representable scale/minimum pairs must not
// lower squared reconstruction error. Includes partial subblocks, not padding.
for (const length of [17, 256, 300]) {
  const data = Float32Array.from({ length }, (_, i) => Math.sin(i * 1.73) * Math.cos(i * 0.37));
  const { quantized } = quantizeToQ4KMRowWise(data, [1, length]);
  for (let start = 0; start < length; start += QK_K) {
    const decoded = decodeQ4KBlockReference(quantized, start / QK_K * QK4_K_BLOCK_SIZE);
    for (let sb = 0; sb < 8 && start + sb * 32 < length; sb += 1) {
      const end = Math.min(length, start + (sb + 1) * 32);
      const errorFor = (scaleBits, minBits) => {
        const scale = decoded.d * scaleBits;
        const minimum = decoded.dmin * minBits;
        let error = 0;
        for (let i = start + sb * 32; i < end; i += 1) {
          const code = scale > 0 ? Math.max(0, Math.min(15, Math.round((data[i] + minimum) / scale))) : 0;
          const delta = data[i] - Math.fround(scale * code - minimum);
          error += delta * delta;
        }
        return error;
      };
      const actual = errorFor(decoded.scaleBits[sb], decoded.minBits[sb]);
      for (let ds = -1; ds <= 1; ds += 1) {
        for (let dm = -1; dm <= 1; dm += 1) {
          const scale = decoded.scaleBits[sb] + ds;
          const minimum = decoded.minBits[sb] + dm;
          if (scale < 0 || scale > 63 || minimum < 0 || minimum > 63) continue;
          assert.ok(actual <= errorFor(scale, minimum) + 1e-12,
            `length ${length}, subblock ${start / 32 + sb}: neighboring grid reduces error`);
        }
      }
    }
  }
  assert.deepEqual(quantizeToQ4KMRowWise(data, [1, length]).quantized, quantized);
}

// Q4_K minima are non-negative offsets subtracted during decoding. A positive
// subblock must include zero in its representable range, not lose its offset.
for (const value of [1, 0.5, -1, 0]) {
  const decoded = decodeQ4KBlockReference(quantizeQ4KBlock(new Float32Array(QK_K).fill(value), 0));
  assert.ok(decoded.values.every((actual) => Math.abs(actual - value) < 0.002),
    `constant ${value} must survive Q4_K encoding`);
}

// Codes must minimize error against the scales actually stored in the bytes,
// including six-bit subblock parameters and half-precision multipliers.
for (const length of [32, 256, 300]) {
  const data = Float32Array.from({ length }, (_, index) => Math.sin(index * 0.17) * (1 + index / 31));
  const { quantized } = quantizeToQ4KMRowWise(data, [1, length]);
  for (let blockStart = 0; blockStart < length; blockStart += QK_K) {
    const decoded = decodeQ4KBlockReference(quantized, blockStart / QK_K * QK4_K_BLOCK_SIZE);
    for (let i = 0; i < Math.min(QK_K, length - blockStart); i += 1) {
      const subblock = Math.floor(i / 32);
      const error = Math.abs(data[blockStart + i] - decoded.values[i]);
      for (let code = 0; code < 16; code += 1) {
        const candidate = Math.fround(decoded.scales[subblock] * code - decoded.minima[subblock]);
        assert.ok(error <= Math.abs(data[blockStart + i] - candidate) + 1e-6,
          `row length ${length}, element ${blockStart + i}: a stored code has lower error`);
      }
    }
  }
}

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
  const shape = [1, 300];
  const data = new Float32Array(shape[0] * shape[1]);
  for (let i = 0; i < 256; i += 1) {
    data[i] = ((i % 17) - 8) / 4;
  }
  for (let i = 256; i < data.length; i += 1) {
    data[i] = -10 + (i - 256) * 0.01;
  }

  const row = quantizeToQ4KMRowWise(data, shape);
  const dequantized = dequantizeQ4KMRowWise(row.quantized, shape);
  const err = calculateQuantizationError(data, dequantized);

  assert.ok(err.maxError < 0.25, `row-wise partial-block quantization drift too high: ${err.maxError}`);
  assert.ok(
    Math.max(...dequantized.slice(256)) < -9.5,
    'row-wise tail block should not be pulled toward zero by padded zeros'
  );
}

{
  const shape = [2, 2, 300];
  const data = new Float32Array(shape[0] * shape[1] * shape[2]);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = ((i % 37) - 18) / 9;
  }

  const row = quantizeToQ4KMRowWise(data, shape);
  const dequantized = dequantizeQ4KMRowWise(row.quantized, shape);

  assert.equal(row.quantized.length, getQ4KSize(shape, 'row'));
  assert.ok(
    getQ4KSize(shape, 'row') > getQ4KSize(shape, 'flat'),
    'batched row-wise Q4K should preserve per-row padding'
  );
  assert.equal(dequantized.length, data.length);
  const err = calculateQuantizationError(data, dequantized);
  assert.ok(Number.isFinite(err.mse));
  assert.ok(Number.isFinite(err.maxError));
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
