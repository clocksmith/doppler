// A storage context owns one reusable reader per file and closes every opened handle.
/** @param {string | URL} fileRef */
function getNodeFileCacheKey(fileRef) {
  if (typeof fileRef === 'string') {
    return fileRef;
  }
  if (fileRef instanceof URL) {
    return fileRef.href;
  }
  return String(fileRef);
}

/** @type {import('./node-file-access.js').createNodeFileAccess} */
export function createNodeFileAccess() {
  /** @type {Map<string, import('./node-file-access.js').NodeFileReader>} */
  const readers = new Map();

  /** @param {string | URL} fileRef */
  const getReader = (fileRef) => {
    const cacheKey = getNodeFileCacheKey(fileRef);
    let reader = readers.get(cacheKey);
    if (reader) {
      return reader;
    }
    /** @type {Promise<import('node:fs/promises').FileHandle> | null} */
    let handlePromise = null;
    /** @type {Promise<number> | null} */
    let sizePromise = null;
    let closed = false;
    const ensureHandle = async () => {
      if (closed) {
        throw new Error(`artifact storage context: file reader already closed for "${cacheKey}".`);
      }
      if (!handlePromise) {
        handlePromise = import('node:fs/promises')
          .then((fs) => fs.open(fileRef, 'r'))
          .catch((error) => {
            handlePromise = null;
            throw error;
          });
      }
      return handlePromise;
    };
    const getSize = async () => {
      if (!sizePromise) {
        sizePromise = (async () => {
          const handle = await ensureHandle();
          const stats = await handle.stat();
          return Number(stats.size);
        })().catch((error) => {
          sizePromise = null;
          throw error;
        });
      }
      return sizePromise;
    };
    reader = {
      async readRange(offset = 0, length = null) {
        const start = Math.max(0, Math.floor(Number(offset) || 0));
        const fileSize = await getSize();
        const end = length == null
          ? fileSize
          : Math.min(fileSize, start + Math.max(0, Math.floor(Number(length) || 0)));
        if (end <= start) {
          return new ArrayBuffer(0);
        }
        const handle = await ensureHandle();
        const out = Buffer.allocUnsafe(end - start);
        let position = 0;
        while (position < out.length) {
          const { bytesRead } = await handle.read(out, position, out.length - position, start + position);
          if (bytesRead === 0) {
            break;
          }
          position += bytesRead;
        }
        return out.buffer.slice(out.byteOffset, out.byteOffset + position);
      },
      async getSize() {
        return getSize();
      },
      async close() {
        closed = true;
        const handle = await handlePromise;
        handlePromise = null;
        if (handle) {
          await handle.close();
        }
      },
    };
    readers.set(cacheKey, reader);
    return reader;
  };

  return {
    async readRange(fileRef, offset = 0, length = null) {
      return getReader(fileRef).readRange(offset, length);
    },
    async getSize(fileRef) {
      return getReader(fileRef).getSize();
    },
    async close() {
      const pending = Array.from(readers.values(), (reader) => reader.close());
      readers.clear();
      await Promise.all(pending);
    },
  };
}
