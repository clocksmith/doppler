import assert from 'node:assert/strict';

import {
  dequantizeQ4KMRowWise,
  quantizeToQ4KMRowWise,
} from '../../src/converter/quantizer.js';
import {
  compareFloatArrays,
  decodeQ4KBlockReference,
  projectF16RowWiseReference,
  projectQ4KRowWiseReference,
} from '../../tools/lib/q4k-projection-reference.js';
import { float32ToFloat16 } from '../../src/converter/quantizer.js';

const rows = 5;
const columns = 512;
const weights = new Float32Array(rows * columns);
const activation = new Float32Array(columns);
for (let index = 0; index < weights.length; index += 1) {
  weights[index] = Math.sin(index * 0.017) * 0.25 + Math.cos(index * 0.003) * 0.1;
}
for (let index = 0; index < activation.length; index += 1) {
  activation[index] = Math.sin(index * 0.011) * 0.5;
}

const quantized = quantizeToQ4KMRowWise(weights, [rows, columns]).quantized;
const productionDecoded = dequantizeQ4KMRowWise(quantized, [rows, columns]);
const reference = projectQ4KRowWiseReference(quantized, [rows, columns], activation);

const firstBlock = decodeQ4KBlockReference(quantized);
assert.deepEqual(Array.from(firstBlock.values), Array.from(productionDecoded.slice(0, 256)));
assert.equal(reference.decodedScales.length, rows * 2 * 8);
assert.equal(reference.decodedMinima.length, rows * 2 * 8);

const expectedQ4 = new Float32Array(rows);
for (let row = 0; row < rows; row += 1) {
  let sum = 0;
  for (let column = 0; column < columns; column += 1) {
    sum += activation[column] * productionDecoded[row * columns + column];
  }
  expectedQ4[row] = sum;
}
assert.deepEqual(Array.from(reference.output), Array.from(expectedQ4));

const f16Bytes = new Uint8Array(weights.length * 2);
const f16View = new DataView(f16Bytes.buffer);
for (let index = 0; index < weights.length; index += 1) {
  f16View.setUint16(index * 2, float32ToFloat16(weights[index]), true);
}
const f16Output = projectF16RowWiseReference(f16Bytes, [rows, columns], activation);
assert.equal(f16Output.length, rows);
assert.ok(f16Output.every(Number.isFinite));

const exact = compareFloatArrays(reference.output, reference.output);
assert.equal(exact.maxAbsDiff, 0);
assert.equal(exact.rmse, 0);
assert.equal(exact.cosineSimilarity, 1);

console.log('q4k-projection-reference.test: ok');
