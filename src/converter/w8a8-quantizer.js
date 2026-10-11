import { validateW8A8Policy } from '../config/schema/w8a8.schema.js';
import { createW8A8StorageDescriptor } from '../formats/rdrr/w8a8-storage.js';

// All operands are F32 dyadics. Multiplication by a half-integer <=126.5
// uses at most 32 significand bits, so these F64 threshold comparisons are
// exact. This rounds the exact input/stored-scale ratio, not an F32 quotient.
/** @param {number} value @param {number} scale */
export function roundW8A8Code(value, scale) {
  const magnitude = Math.abs(value);
  let lower = 0;
  let upper = 127;
  while (lower < upper) {
    const midpoint = Math.floor((lower + upper) / 2);
    const boundary = scale * (midpoint + 0.5);
    if (magnitude > boundary || (magnitude === boundary && midpoint % 2 !== 0)) {
      lower = midpoint + 1;
    } else {
      upper = midpoint;
    }
  }
  return value < 0 ? -lower : lower;
}

/** @param {Float32Array} values @param {number[]} shape @param {import('../config/schema/w8a8.schema.js').W8A8Policy} policy */
export function quantizeToW8A8Rows(values, shape, policy) {
  validateW8A8Policy(policy);
  if (!(values instanceof Float32Array) || !Array.isArray(shape) || shape.length !== 2
    || !Number.isSafeInteger(shape[0]) || shape[0] < 1) {
    throw new Error('W8A8 conversion requires F32 values and a positive [N,K] shape.');
  }
  const [rows, columns] = shape;
  const storage = createW8A8StorageDescriptor(columns);
  if (values.length !== rows * columns) throw new Error('W8A8 source element count does not match shape.');
  const quantized = new Uint8Array(rows * storage.rowStrideBytes);
  const view = new DataView(quantized.buffer);
  for (let row = 0; row < rows; row++) {
    let maximum = 0;
    for (let column = 0; column < columns; column++) {
      const magnitude = Math.abs(values[row * columns + column]);
      if (!Number.isFinite(magnitude) || (magnitude !== 0 && magnitude < 2 ** -126)) {
        throw new Error(`W8A8 row ${row} contains a nonfinite or nonzero subnormal input.`);
      }
      maximum = Math.max(maximum, magnitude);
    }
    // Division by odd 127 cannot tie at an F32 midpoint; its nearest distance
    // exceeds the F64 rounding error by over 20 bits. F64 then F32 is RN-even.
    const scale = maximum === 0 ? 1 : Math.fround(maximum / 127);
    if (scale < 2 ** -126) throw new Error(`W8A8 row ${row} scale is subnormal or zero.`);
    const start = row * storage.rowStrideBytes;
    view.setFloat32(start, scale, true);
    for (let column = 0; column < columns; column++) {
      view.setInt8(start + 4 + column, roundW8A8Code(values[row * columns + column], scale));
    }
  }
  return { quantized, storage };
}
