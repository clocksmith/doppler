import { W8A8_POLICY } from '../../config/schema/w8a8.schema.js';

/** @param {number} columns @returns {import('./w8a8-storage.js').W8A8StorageDescriptor} */
export function createW8A8StorageDescriptor(columns) {
  if (!Number.isSafeInteger(columns) || columns < 1 || columns > W8A8_POLICY.maxReductionWidth) {
    throw new Error('W8A8 reduction width must be a positive integer at most 133144.');
  }
  const paddedK = Math.ceil(columns / 4) * 4;
  return {
    packing: 'w8a8-row-v1', blockShape: [1, columns], blockBytes: 4 + paddedK,
    paddedK, rowStrideBytes: 4 + paddedK, scaleOffsetBytes: 0, dataOffsetBytes: 4,
  };
}

/** @param {unknown} input @param {number} columns */
export function validateW8A8StorageDescriptor(input, columns) {
  const expected = createW8A8StorageDescriptor(columns);
  const storage = /** @type {Record<string, unknown>} */ (input);
  if (!storage || typeof storage !== 'object' || Array.isArray(storage)) {
    throw new Error('W8A8 requires an explicit row storage descriptor.');
  }
  for (const [key, value] of Object.entries(expected)) {
    if (JSON.stringify(storage[key]) !== JSON.stringify(value)) {
      throw new Error(`W8A8 storage.${key} must be ${JSON.stringify(value)}.`);
    }
  }
  if (storage.companions !== undefined && (!Array.isArray(storage.companions) || storage.companions.length)) {
    throw new Error('W8A8 row storage cannot use companion tensors.');
  }
  return expected;
}

/** @param {Uint8Array} bytes @param {number[]} shape @param {unknown} storage */
export function validateW8A8StorageBytes(bytes, shape, storage) {
  if (!(bytes instanceof Uint8Array) || !Array.isArray(shape) || shape.length !== 2
    || !Number.isSafeInteger(shape[0]) || shape[0] < 1) {
    throw new Error('W8A8 requires raw bytes and a positive two-dimensional [N,K] shape.');
  }
  const descriptor = validateW8A8StorageDescriptor(storage, shape[1]);
  const size = shape[0] * descriptor.rowStrideBytes;
  if (!Number.isSafeInteger(size) || bytes.byteLength !== size) {
    throw new Error('W8A8 byte length does not match the declared row storage.');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let row = 0; row < shape[0]; row++) {
    const start = row * descriptor.rowStrideBytes;
    const scale = view.getFloat32(start, true);
    if (!Number.isFinite(scale) || scale < 2 ** -126) {
      throw new Error(`W8A8 row ${row} requires a positive normal F32 scale.`);
    }
    let nonzero = false;
    for (let column = 0; column < descriptor.paddedK; column++) {
      const code = view.getInt8(start + 4 + column);
      if (code === -128 || (column >= shape[1] && code !== 0)) {
        throw new Error(`W8A8 row ${row} contains an invalid code or nonzero padding.`);
      }
      nonzero ||= code !== 0;
    }
    if (!nonzero && scale !== 1) {
      throw new Error(`W8A8 zero row ${row} requires scale 1.`);
    }
  }
  return descriptor;
}
