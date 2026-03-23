import { isOPFSAvailable } from '../quota.js';

function createLimiter(maxConcurrent) {
  let active = 0;
  const queue = [];

  const acquire = async () => {
    if (active < maxConcurrent) {
      active += 1;
      return;
    }
    await new Promise((resolve) => queue.push(resolve));
    active += 1;
  };

  const release = () => {
    active = Math.max(0, active - 1);
    const next = queue.shift();
    if (next) {
      next();
    }
  };

  return { acquire, release };
}

export function createOpfsStore(config) {
  const {
    opfsRootDir,
    useSyncAccessHandle,
    maxConcurrentHandles,
  } = config;
  let rootDir = null;
  let modelsDir = null;
  let currentModelDir = null;
  let currentModelId = null;
  const syncAccessRequested = useSyncAccessHandle === true;
  if (syncAccessRequested && typeof FileSystemSyncAccessHandle === 'undefined') {
    throw new Error(
      'OPFS sync access handles were explicitly requested but are unavailable in this runtime.'
    );
  }
  const syncAccessEnabled = syncAccessRequested
    && typeof FileSystemSyncAccessHandle !== 'undefined';
  const handleLimiter = syncAccessEnabled ? createLimiter(maxConcurrentHandles) : null;

  if (syncAccessEnabled && (!Number.isInteger(maxConcurrentHandles) || maxConcurrentHandles < 1)) {
    throw new Error('Invalid opfs.maxConcurrentHandles');
  }

  async function init() {
    if (!isOPFSAvailable()) {
      throw new Error('OPFS not available in this browser');
    }
    rootDir = await navigator.storage.getDirectory();
    modelsDir = await rootDir.getDirectoryHandle(opfsRootDir, { create: true });
  }

  async function openModel(modelId, options = {}) {
    if (!modelsDir) {
      await init();
    }
    const create = options.create !== false;
    currentModelDir = await modelsDir.getDirectoryHandle(modelId, { create });
    currentModelId = modelId;
    return currentModelDir;
  }

  function getCurrentModelId() {
    return currentModelId;
  }

  async function openSyncAccessHandle(fileHandle) {
    if (!syncAccessEnabled || !handleLimiter || typeof fileHandle.createSyncAccessHandle !== 'function') {
      return null;
    }
    await handleLimiter.acquire();
    try {
      const handle = await fileHandle.createSyncAccessHandle();
      return {
        handle,
        release: () => {
          handle.close();
          handleLimiter.release();
        }
      };
    } catch (error) {
      handleLimiter.release();
      if (error?.name === 'InvalidStateError' || error?.name === 'NotAllowedError') {
        throw new Error(
          `OPFS sync access handles were explicitly requested but could not be opened: ${error.name}.`
        );
      }
      throw error;
    }
  }

  async function ensureModelDir() {
    if (!currentModelDir) {
      throw new Error('No model directory open. Call openModelStore first.');
    }
  }

  function normalizePathSegments(filename) {
    const normalized = String(filename || '').replace(/\\/g, '/').trim();
    if (!normalized) {
      throw new Error('Filename is required');
    }
    const parts = normalized.split('/').filter(Boolean);
    if (parts.length === 0) {
      throw new Error('Filename is required');
    }
    for (const part of parts) {
      if (part === '.' || part === '..') {
        throw new Error(`Invalid relative storage path: ${filename}`);
      }
    }
    return parts;
  }

  async function resolveDirectoryForPath(filename, options = {}) {
    await ensureModelDir();
    const parts = normalizePathSegments(filename);
    const leafName = parts.pop();
    let dir = currentModelDir;
    for (const part of parts) {
      dir = await dir.getDirectoryHandle(part, { create: options.createDirs === true });
    }
    return {
      dir,
      leafName,
    };
  }

  async function getFileHandle(filename, options = {}) {
    const { dir, leafName } = await resolveDirectoryForPath(filename, {
      createDirs: options.create === true,
    });
    return dir.getFileHandle(leafName, { create: options.create === true });
  }

  function normalizeWriteStreamOptions(options = {}) {
    const append = options?.append === true;
    const expectedOffsetRaw = options?.expectedOffset;
    const expectedOffset = expectedOffsetRaw == null
      ? null
      : Number(expectedOffsetRaw);
    if (
      expectedOffset != null
      && (!Number.isInteger(expectedOffset) || expectedOffset < 0)
    ) {
      throw new Error('createWriteStream expectedOffset must be a non-negative integer');
    }
    return { append, expectedOffset };
  }

  async function getFileSize(filename) {
    const fileHandle = await getFileHandle(filename, { create: false });
    const access = await openSyncAccessHandle(fileHandle);
    if (access) {
      try {
        return access.handle.getSize();
      } finally {
        access.release();
      }
    }
    const file = await fileHandle.getFile();
    return file.size;
  }

  async function readFile(filename) {
    const fileHandle = await getFileHandle(filename, { create: false });
    const access = await openSyncAccessHandle(fileHandle);
    if (access) {
      try {
        const size = access.handle.getSize();
        const buffer = new Uint8Array(size);
        let offset = 0;
        while (offset < size) {
          const view = buffer.subarray(offset);
          const read = access.handle.read(view, { at: offset });
          if (read <= 0) {
            break;
          }
          offset += read;
        }
        return buffer.buffer;
      } finally {
        access.release();
      }
    }

    const file = await fileHandle.getFile();
    return file.arrayBuffer();
  }

  async function readFileRange(filename, offset = 0, length = null) {
    const fileHandle = await getFileHandle(filename, { create: false });
    const access = await openSyncAccessHandle(fileHandle);

    const startRaw = Number(offset);
    const start = Number.isFinite(startRaw) ? Math.max(0, Math.floor(startRaw)) : 0;

    if (access) {
      try {
        const size = access.handle.getSize();
        const end = length == null
          ? size
          : Math.min(size, start + Math.max(0, Number.isFinite(Number(length)) ? Math.floor(Number(length)) : 0));
        const want = Math.max(0, end - start);
        const buffer = new Uint8Array(want);
        let readOffset = 0;
        while (readOffset < want) {
          const view = buffer.subarray(readOffset);
          const read = access.handle.read(view, { at: start + readOffset });
          if (read <= 0) break;
          readOffset += read;
        }
        return buffer.buffer;
      } finally {
        access.release();
      }
    }

    const file = await fileHandle.getFile();
    const end = length == null
      ? file.size
      : Math.min(file.size, start + Math.max(0, Number.isFinite(Number(length)) ? Math.floor(Number(length)) : 0));
    return file.slice(start, end).arrayBuffer();
  }

  async function* readFileRangeStream(filename, offset = 0, length = null, options = {}) {
    const rawChunkBytes = options?.chunkBytes;
    const chunkBytes = Number.isFinite(rawChunkBytes) && rawChunkBytes > 0
      ? Math.floor(rawChunkBytes)
      : (4 * 1024 * 1024);
    const startRaw = Number(offset);
    const start = Number.isFinite(startRaw) ? Math.max(0, Math.floor(startRaw)) : 0;

    const fileHandle = await getFileHandle(filename, { create: false });
    const access = await openSyncAccessHandle(fileHandle);

    if (access) {
      try {
        const size = access.handle.getSize();
        const end = length == null
          ? size
          : Math.min(size, start + Math.max(0, Number.isFinite(Number(length)) ? Math.floor(Number(length)) : 0));
        let at = start;
        const scratch = new Uint8Array(chunkBytes);
        while (at < end) {
          const want = Math.min(chunkBytes, end - at);
          const view = want === scratch.byteLength ? scratch : scratch.subarray(0, want);
          const read = access.handle.read(view, { at });
          if (read <= 0) break;
          // Copy out to avoid consumers seeing a mutated scratch buffer.
          yield view.slice(0, read);
          at += read;
        }
        return;
      } finally {
        access.release();
      }
    }

    // Fallback: repeated slice reads (allocates per-chunk, but avoids full-file materialization).
    const file = await fileHandle.getFile();
    const end = length == null
      ? file.size
      : Math.min(file.size, start + Math.max(0, Number.isFinite(Number(length)) ? Math.floor(Number(length)) : 0));
    for (let at = start; at < end; at += chunkBytes) {
      const ab = await file.slice(at, Math.min(end, at + chunkBytes)).arrayBuffer();
      yield new Uint8Array(ab);
    }
  }

  async function readText(filename) {
    try {
      const fileHandle = await getFileHandle(filename, { create: false });
      const file = await fileHandle.getFile();
      return await file.text();
    } catch (error) {
      if (error.name === 'NotFoundError') {
        return null;
      }
      throw error;
    }
  }

  async function writeFile(filename, data) {
    const fileHandle = await getFileHandle(filename, { create: true });
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
    const access = await openSyncAccessHandle(fileHandle);
    if (access) {
      try {
        access.handle.truncate(0);
        access.handle.write(bytes, { at: 0 });
        access.handle.flush();
      } finally {
        access.release();
      }
      return;
    }

    const writable = await fileHandle.createWritable();
    try {
      await writable.write(bytes);
      await writable.close();
    } catch (error) {
      try { await writable.abort(); } catch { /* best-effort cleanup */ }
      throw error;
    }
  }

  async function createWriteStream(filename, options = {}) {
    const { append, expectedOffset } = normalizeWriteStreamOptions(options);
    const fileHandle = await getFileHandle(filename, { create: true });
    const existingSize = await getFileSize(filename).catch((error) => {
      if (error?.name === 'NotFoundError') {
        return 0;
      }
      throw error;
    });
    if (expectedOffset != null && expectedOffset !== existingSize) {
      throw new Error(
        `createWriteStream expectedOffset mismatch for ${filename}: expected ${expectedOffset}, got ${existingSize}`
      );
    }
    const access = await openSyncAccessHandle(fileHandle);
    if (access) {
      const initialOffset = append ? existingSize : 0;
      let offset = initialOffset;
      let closed = false;
      if (!append) {
        access.handle.truncate(0);
      }
      return {
        write: async (chunk) => {
          if (closed) {
            throw new Error('Write after close');
          }
          const bytes = chunk instanceof ArrayBuffer ? new Uint8Array(chunk) : chunk;
          access.handle.write(bytes, { at: offset });
          offset += bytes.byteLength;
        },
        close: async () => {
          if (closed) return;
          closed = true;
          access.handle.flush();
          access.release();
        },
        abort: async () => {
          if (closed) return;
          closed = true;
          access.handle.truncate(initialOffset);
          access.release();
        },
      };
    }

    const writable = await fileHandle.createWritable({ keepExistingData: append });
    let closed = false;
    let offset = append ? existingSize : 0;
    return {
      write: async (chunk) => {
        if (closed) {
          throw new Error('Write after close');
        }
        const bytes = chunk instanceof ArrayBuffer ? new Uint8Array(chunk) : chunk;
        await writable.write({
          type: 'write',
          position: offset,
          data: bytes,
        });
        offset += bytes.byteLength;
      },
      close: async () => {
        if (closed) return;
        closed = true;
        await writable.close();
      },
      abort: async () => {
        if (closed) return;
        closed = true;
        await writable.abort();
      },
    };
  }

  async function deleteFile(filename) {
    const { dir, leafName } = await resolveDirectoryForPath(filename, { createDirs: false });
    try {
      await dir.removeEntry(leafName);
      return true;
    } catch (error) {
      if (error.name === 'NotFoundError') {
        return false;
      }
      throw error;
    }
  }

  async function listFiles() {
    await ensureModelDir();
    const files = [];
    async function walk(dir, prefix = '') {
      for await (const [name, handle] of dir.entries()) {
        const relativePath = prefix ? `${prefix}/${name}` : name;
        if (handle.kind === 'file') {
          files.push(relativePath);
          continue;
        }
        if (handle.kind === 'directory') {
          await walk(handle, relativePath);
        }
      }
    }
    await walk(currentModelDir);
    return files;
  }

  async function listModels() {
    if (!modelsDir) {
      await init();
    }
    const models = [];
    for await (const [name, handle] of modelsDir.entries()) {
      if (handle.kind === 'directory') {
        models.push(name);
      }
    }
    return models;
  }

  async function deleteModel(modelId) {
    if (!modelsDir) {
      await init();
    }
    try {
      await modelsDir.removeEntry(modelId, { recursive: true });
      if (currentModelId === modelId) {
        currentModelId = null;
        currentModelDir = null;
      }
      return true;
    } catch (error) {
      if (error.name === 'NotFoundError') {
        return false;
      }
      throw error;
    }
  }

  async function cleanup() {
    rootDir = null;
    modelsDir = null;
    currentModelDir = null;
    currentModelId = null;
  }

  return {
    init,
    openModel,
    getCurrentModelId,
    getFileSize,
    readFile,
    readFileRange,
    readFileRangeStream,
    readText,
    writeFile,
    createWriteStream,
    deleteFile,
    listFiles,
    listModels,
    deleteModel,
    cleanup,
  };
}
