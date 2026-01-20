

import { createHash } from 'crypto';
import { ALIGNMENT } from './types.js';
import { DTYPE_SIZES } from '../../config/schema/index.js';


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
  const size = DTYPE_SIZES[dtype?.toLowerCase()];
  // Return 0 for quantized types (Q4_K, Q8_0, etc.) - block-quantized, don't transpose
  return size ?? 0;
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
