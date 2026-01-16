

import { createHash } from 'crypto';
import { ALIGNMENT } from './types.js';


export async function computeHash(data, algorithm = 'sha256') {
  if (algorithm === 'blake3') {
    try {
      const blake3Module = await import('blake3');
      return blake3Module.blake3(data).toString('hex');
    } catch (err) {
      throw new Error(`blake3 hashing requested but unavailable: ${err.message}`);
    }
  }

  const hash = createHash('sha256');
  hash.update(data);
  return hash.digest('hex');
}


export function alignOffset(offset, alignment = ALIGNMENT) {
  const remainder = offset % alignment;
  return remainder === 0 ? offset : offset + (alignment - remainder);
}


export function createPadding(size) {
  return new Uint8Array(size);
}


export function getBytesPerElement(dtype) {
  const dtypeLower = dtype.toLowerCase();
  if (dtypeLower === 'f32' || dtypeLower === 'float32') return 4;
  if (dtypeLower === 'f16' || dtypeLower === 'float16') return 2;
  if (dtypeLower === 'bf16' || dtypeLower === 'bfloat16') return 2;
  if (dtypeLower === 'i32' || dtypeLower === 'int32') return 4;
  if (dtypeLower === 'i16' || dtypeLower === 'int16') return 2;
  if (dtypeLower === 'i8' || dtypeLower === 'int8') return 1;
  // Q4_K, Q8_0, etc. - these are block-quantized, don't transpose
  return 0;
}


export function transpose2D(data, rows, cols, dtype) {
  const bytesPerElement = getBytesPerElement(dtype);
  const result = new Uint8Array(data.length);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const srcOffset = (r * cols + c) * bytesPerElement;
      const dstOffset = (c * rows + r) * bytesPerElement;
      for (let b = 0; b < bytesPerElement; b++) {
        result[dstOffset + b] = data[srcOffset + b];
      }
    }
  }

  return result;
}
