// One paged, cached binary reader for byte-range sources. Pages are fetched
// once and kept; a page requested while its fetch is in flight reuses that
// request instead of issuing another readRange. Bounds and short reads fail
// with the owning parser's label. Format decoding stays with each format.

/**
 * @typedef {{ name?: string, size: number, readRange(offset: number, length: number): Promise<ArrayBuffer | Uint8Array> | ArrayBuffer | Uint8Array }} ByteRangeSource
 */

/**
 * @param {ArrayBuffer | Uint8Array | unknown} value
 * @param {string} label
 * @returns {Uint8Array}
 */
function toUint8Array(value, label) {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }
  throw new Error(`${label} must return ArrayBuffer or Uint8Array.`);
}

/**
 * The label a source goes by in messages: its trimmed name, or the parser's default.
 * @param {{ name?: unknown } | null | undefined} source
 * @param {string} defaultLabel
 */
export function describeSource(source, defaultLabel) {
  const raw = typeof source?.name === 'string' ? source.name.trim() : '';
  return raw || defaultLabel;
}

/**
 * @param {ByteRangeSource} source
 * @param {{ pageBytes: number, parser: string, defaultLabel: string }} options
 */
export function createPagedByteReader(source, { pageBytes, parser, defaultLabel }) {
  if (!Number.isInteger(pageBytes) || pageBytes <= 0) {
    throw new Error(`${parser}: pageBytes must be a positive integer.`);
  }
  if (!source || typeof source.readRange !== 'function') {
    throw new Error(`${parser}: source.readRange(offset, length) is required.`);
  }
  const size = Number(source.size);
  if (!Number.isFinite(size) || size < 8) {
    throw new Error(`${parser}: source.size must be a finite byte length >= 8.`);
  }
  const label = describeSource(source, defaultLabel);
  /** @type {Map<number, Uint8Array | Promise<Uint8Array>>} */
  const pages = new Map();

  /** @param {number} pageIndex */
  function loadPage(pageIndex) {
    const cached = pages.get(pageIndex);
    if (cached) {
      return cached;
    }
    const pageOffset = pageIndex * pageBytes;
    const pageLength = Math.min(pageBytes, size - pageOffset);
    const pending = Promise.resolve(source.readRange(pageOffset, pageLength)).then(
      (raw) => {
        const bytes = toUint8Array(raw, `${label} page ${pageIndex}`);
        if (bytes.byteLength !== pageLength) {
          throw new Error(
            `${parser}: short read for ${label} page ${pageIndex}. ` +
            `Expected ${pageLength} bytes, got ${bytes.byteLength}.`
          );
        }
        pages.set(pageIndex, bytes);
        return bytes;
      },
      (error) => {
        pages.delete(pageIndex);
        throw error;
      }
    );
    pages.set(pageIndex, pending);
    return pending;
  }

  /**
   * @param {number} offset
   * @param {number} length
   * @returns {Promise<Uint8Array>}
   */
  async function readSlice(offset, length) {
    if (!Number.isFinite(offset) || !Number.isFinite(length)) {
      throw new Error(`${parser}: invalid read request (${offset}, ${length}).`);
    }
    const start = Math.max(0, Math.floor(offset));
    const byteLength = Math.max(0, Math.floor(length));
    if (start + byteLength > size) {
      throw new Error(
        `${parser}: read exceeds ${label} bounds (offset=${start}, length=${byteLength}, size=${size}).`
      );
    }
    if (byteLength === 0) {
      return new Uint8Array(0);
    }
    const pageStart = Math.floor(start / pageBytes);
    const pageEnd = Math.floor((start + byteLength - 1) / pageBytes);
    if (pageStart === pageEnd) {
      const page = await loadPage(pageStart);
      const sliceOffset = start - pageStart * pageBytes;
      return page.subarray(sliceOffset, sliceOffset + byteLength);
    }
    const out = new Uint8Array(byteLength);
    let writeOffset = 0;
    for (let pageIndex = pageStart; pageIndex <= pageEnd; pageIndex++) {
      const page = await loadPage(pageIndex);
      const pageBase = pageIndex * pageBytes;
      const sliceStart = pageIndex === pageStart ? start - pageBase : 0;
      const sliceEnd = pageIndex === pageEnd
        ? (start + byteLength) - pageBase
        : page.byteLength;
      const slice = page.subarray(sliceStart, sliceEnd);
      out.set(slice, writeOffset);
      writeOffset += slice.byteLength;
    }
    return out;
  }

  /**
   * @param {number} offset
   * @param {number} byteLength
   */
  async function view(offset, byteLength) {
    const bytes = await readSlice(offset, byteLength);
    return new DataView(bytes.buffer, bytes.byteOffset, byteLength);
  }

  return Object.freeze({
    label,
    size,
    pageBytes,
    readSlice,
    /** @param {number} offset */
    readUint8: async (offset) => (await readSlice(offset, 1))[0],
    /** @param {number} offset */
    readInt16: async (offset) => (await view(offset, 2)).getInt16(0, true),
    /** @param {number} offset */
    readUint16: async (offset) => (await view(offset, 2)).getUint16(0, true),
    /** @param {number} offset */
    readInt32: async (offset) => (await view(offset, 4)).getInt32(0, true),
    /** @param {number} offset */
    readUint32: async (offset) => (await view(offset, 4)).getUint32(0, true),
    /** @param {number} offset */
    readFloat32: async (offset) => (await view(offset, 4)).getFloat32(0, true),
    /** @param {number} offset */
    readFloat64: async (offset) => (await view(offset, 8)).getFloat64(0, true),
    /** @param {number} offset */
    readBigInt64: async (offset) => (await view(offset, 8)).getBigInt64(0, true),
    /** @param {number} offset */
    readBigUint64: async (offset) => (await view(offset, 8)).getBigUint64(0, true),
  });
}
