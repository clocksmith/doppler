import { Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES } from '../../config/schema/quantization.schema.js';

/** @type {typeof import('./q4_0-storage.js').q40StorageBytes} */
export function q40StorageBytes(shape) {
  if (!Array.isArray(shape) || shape.length < 2 || shape.length > 3
    || shape.some(value => !Number.isSafeInteger(value) || value < 1)
    || shape.at(-1) % Q4_0_BLOCK_SIZE !== 0) {
    throw new Error('Q4_0 requires explicit matrix/bank geometry with K divisible by 32.');
  }
  const bytes = shape.reduce((total, value) => total * value, 1) / Q4_0_BLOCK_SIZE * Q4_0_BLOCK_BYTES;
  if (!Number.isSafeInteger(bytes) || bytes > 0xffffffff) throw new Error('Q4_0 storage exceeds addressable byte extents.');
  return bytes;
}

/** @type {typeof import('./q4_0-storage.js').validateQ40Storage} */
export function validateQ40Storage(shape, storage, layout) {
  const bytes = q40StorageBytes(shape);
  if (layout !== 'row' || (storage != null && (storage.packing !== 'q4_0'
    || storage.blockBytes !== Q4_0_BLOCK_BYTES || storage.blockShape?.length !== 1
    || storage.blockShape?.[0] !== Q4_0_BLOCK_SIZE))) {
    throw new Error('Q4_0 requires unchanged row-packed 18-byte blocks of 32 values.');
  }
  return bytes;
}
