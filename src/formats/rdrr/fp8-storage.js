import { float8ToFloat32 } from '../float8.js';
import { validateFP8Policy } from '../../config/schema/fp8.schema.js';

/** @type {typeof import('./fp8-storage.js').createFP8StorageDescriptor} */
export function createFP8StorageDescriptor(policy) {
  validateFP8Policy(policy);
  return { packing: 'fp8-block-v1', encoding: policy.encoding, blockShape: [...policy.blockShape],
    blockBytes: 4 + Math.ceil(policy.blockShape[0] * policy.blockShape[1] / 4) * 4,
    scaleOffsetBytes: 0, dataOffsetBytes: 4, scaleSemantics: 'multiply' };
}

/** @type {typeof import('./fp8-storage.js').validateFP8StorageDescriptor} */
export function validateFP8StorageDescriptor(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('FP8 requires an explicit storage descriptor.');
  const value = /** @type {import('./fp8-storage.js').FP8StorageDescriptor} */ (input);
  const expected = createFP8StorageDescriptor({ schema: 'doppler.fp8/v1', encoding: value.encoding,
    blockShape: value.blockShape, packing: 'fp8-block-v1', scaleMode: 'identity', scaleSuffix: null,
    scaleSemantics: 'multiply', scaleDtype: 'f32', inputDtype: 'f32', accumulatorDtype: 'f32', outputDtype: 'f32', dequantizedWeightPolicy: 'finite-normal-or-zero', nonfinite: 'preserve-storage-reject-gpu' });
  if (Object.keys(value).length !== Object.keys(expected).length
    || Object.entries(expected).some(([key, entry]) => JSON.stringify(/** @type {Record<string, unknown>} */ (input)[key]) !== JSON.stringify(entry))) {
    throw new Error('FP8 storage descriptor does not match its versioned block packing.');
  }
  return expected;
}

/** @type {typeof import('./fp8-storage.js').fp8StorageBytes} */
export function fp8StorageBytes(shape, storage) {
  if (!Array.isArray(shape) || shape.length !== 2
    || !shape.every(value => Number.isSafeInteger(value) && value > 0 && value <= 0xffffff)) {
    throw new Error('FP8 storage requires explicit positive [N,K] geometry.');
  }
  const count = Math.ceil(shape[0] / storage.blockShape[0]) * Math.ceil(shape[1] / storage.blockShape[1]);
  const size = count * storage.blockBytes;
  if (!Number.isSafeInteger(size) || size > 0xffffffff) throw new Error('FP8 storage exceeds u32 addressability.');
  return size;
}

/** @type {typeof import('./fp8-storage.js').validateFP8StorageBytes} */
export function validateFP8StorageBytes(bytes, shape, input) {
  const storage = validateFP8StorageDescriptor(input);
  if (!(bytes instanceof Uint8Array) || bytes.byteLength !== fp8StorageBytes(shape, storage)) {
    throw new Error('FP8 storage byte length differs from its declared block geometry.');
  }
  const [rows, columns] = shape, [blockRows, blockColumns] = storage.blockShape;
  const blocksK = Math.ceil(columns / blockColumns);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let block = 0; block * storage.blockBytes < bytes.length; block++) {
    const start = block * storage.blockBytes;
    const scale = view.getFloat32(start, true);
    if (!Number.isFinite(scale) || scale < 2 ** -126) throw new Error('FP8 block scale must be positive normal F32.');
    for (let offset = 0; offset < storage.blockBytes - 4; offset++) {
      const row = Math.floor(block / blocksK) * blockRows + Math.floor(offset / blockColumns);
      const column = (block % blocksK) * blockColumns + offset % blockColumns;
      if ((offset >= blockRows * blockColumns || row >= rows || column >= columns) && bytes[start + 4 + offset] !== 0) {
        throw new Error('FP8 tail and word padding must contain zero codes.');
      }
    }
  }
  return storage;
}

/** @type {typeof import('./fp8-storage.js').validateFP8GPUStorageBytes} */
export function validateFP8GPUStorageBytes(bytes, shape, input) {
  const storage = validateFP8StorageBytes(bytes, shape, input);
  const [blockN, blockK] = storage.blockShape;
  const blocksK = Math.ceil(shape[1] / blockK);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let block = 0; block * storage.blockBytes < bytes.length; block++) {
    const base = block * storage.blockBytes;
    const scale = view.getFloat32(base, true);
    for (let n = 0; n < blockN; n++) for (let k = 0; k < blockK; k++) {
      if (Math.floor(block / blocksK) * blockN + n >= shape[0]
        || (block % blocksK) * blockK + k >= shape[1]) continue;
      const value = float8ToFloat32(bytes[base + 4 + n * blockK + k], storage.encoding);
      if (!Number.isFinite(value)) throw new Error('FP8 GPU execution does not admit encoding-defined NaN/infinity codes; source bytes are unchanged.');
      const scaled = Math.fround(value * scale);
      if (!Number.isFinite(scaled) || (value !== 0 && Math.abs(scaled) < 2 ** -126)) {
        throw new Error('FP8 GPU dequantization requires finite normal-or-zero scaled weights.');
      }
    }
  }
  return storage;
}
