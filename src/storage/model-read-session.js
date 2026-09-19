import { getFileSizeInBackend } from './shards/index.js';

// An opened backend handle belongs to one model for its entire lifetime.
/** @type {import('./model-read-session.js').createModelReadSession} */
export async function createModelReadSession(backend, modelId, chunkBytes) {
  if (!modelId) throw new Error('A model ID is required to open a read session.');
  if (!Number.isSafeInteger(chunkBytes) || chunkBytes < 1) {
    throw new Error('Storage readChunkBytes must be a positive integer.');
  }
  const store = await backend.openModelSession(modelId, { create: false });
  let closed = false;
  function requireOpen() {
    if (closed) throw new Error(`Storage session for ${modelId} is closed.`);
  }
  /** @param {number} offset @param {number | null} length */
  function validateRange(offset, length) {
    if (!Number.isSafeInteger(offset) || offset < 0
      || (length !== null && (!Number.isSafeInteger(length) || length < 0))) {
      throw new Error('Storage range requires non-negative integer offsets and lengths.');
    }
  }
  /** @param {string} path */
  async function readFile(path) {
    requireOpen();
    return store.readFile(path);
  }
  /** @param {string} path @param {number} offset @param {number | null} length */
  async function readRange(path, offset = 0, length = null) {
    requireOpen();
    validateRange(offset, length);
    if ('readFileRange' in store) return store.readFileRange(path, offset, length);
    const bytes = new Uint8Array(await store.readFile(path));
    return bytes.slice(offset, length === null ? undefined : offset + length).buffer;
  }
  const session = /** @satisfies {import('./model-read-session.js').ModelReadSession} */ ({
    modelId,
    readFile,
    readRange,
    async readText(path) {
      requireOpen();
      if (store.readText) return store.readText(path);
      return new TextDecoder().decode(await store.readFile(path));
    },
    async readManifest() {
      requireOpen();
      if ('readManifest' in store) return store.readManifest();
      return store.readText('manifest.json');
    },
    async getFileSize(path) { requireOpen(); return getFileSizeInBackend(store, path); },
    async *streamRange(path, offset = 0, length = null, options = {}) {
      requireOpen();
      validateRange(offset, length);
      const size = options.chunkBytes ?? chunkBytes;
      if (!Number.isSafeInteger(size) || size < 1) throw new Error('Invalid storage chunkBytes.');
      if ('readFileRangeStream' in store) {
        for await (const chunk of store.readFileRangeStream(path, offset, length, { chunkBytes: size })) {
          requireOpen();
          yield chunk;
        }
      } else {
        const end = length === null ? await store.getFileSize(path) : offset + length;
        for (let start = offset; start < end; start += size) {
          const bytes = await readRange(path, start, Math.min(size, end - start));
          if (bytes.byteLength === 0) break;
          yield new Uint8Array(bytes);
        }
      }
    },
    async close() {
      if (closed) return;
      closed = true;
      await store.close();
    },
  });
  return Object.freeze(session);
}
