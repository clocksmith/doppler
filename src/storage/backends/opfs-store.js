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
  // SyncAccessHandle is typically only available in dedicated workers, but we
  // allow an optimistic attempt anywhere and fall back gracefully if the
  // browser rejects it (NotAllowedError / InvalidStateError).
  const syncAccessEnabled = !!useSyncAccessHandle
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
        return null;
      }
      throw error;
    }
  }

  async function ensureModelDir() {
    if (!currentModelDir) {
      throw new Error('No model directory open. Call openModelStore first.');
    }
  }

  async function readFile(filename) {
    await ensureModelDir();
    const fileHandle = await currentModelDir.getFileHandle(filename);
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
    await ensureModelDir();
    const fileHandle = await currentModelDir.getFileHandle(filename);
    const access = await openSyncAccessHandle(fileHandle);

    const start = Math.max(0, offset | 0);

    if (access) {
      try {
        const size = access.handle.getSize();
        const end = length == null
          ? size
          : Math.min(size, start + Math.max(0, length | 0));
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
      : Math.min(file.size, start + Math.max(0, length | 0));
    return file.slice(start, end).arrayBuffer();
  }

  async function* readFileRangeStream(filename, offset = 0, length = null, options = {}) {
    const chunkBytes = Math.max(1, options.chunkBytes | 0);
    const start = Math.max(0, offset | 0);

    await ensureModelDir();
    const fileHandle = await currentModelDir.getFileHandle(filename);
    const access = await openSyncAccessHandle(fileHandle);

    if (access) {
      try {
        const size = access.handle.getSize();
        const end = length == null
          ? size
          : Math.min(size, start + Math.max(0, length | 0));
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
      : Math.min(file.size, start + Math.max(0, length | 0));
    for (let at = start; at < end; at += chunkBytes) {
      const ab = await file.slice(at, Math.min(end, at + chunkBytes)).arrayBuffer();
      yield new Uint8Array(ab);
    }
  }

  async function readText(filename) {
    await ensureModelDir();
    try {
      const fileHandle = await currentModelDir.getFileHandle(filename);
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
    await ensureModelDir();
    const fileHandle = await currentModelDir.getFileHandle(filename, { create: true });
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
    await writable.write(bytes);
    await writable.close();
  }

  async function createWriteStream(filename) {
    await ensureModelDir();
    const fileHandle = await currentModelDir.getFileHandle(filename, { create: true });
    const access = await openSyncAccessHandle(fileHandle);
    if (access) {
      let offset = 0;
      let closed = false;
      access.handle.truncate(0);
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
          access.handle.truncate(0);
          access.release();
        },
      };
    }

    const writable = await fileHandle.createWritable();
    return {
      write: async (chunk) => {
        const bytes = chunk instanceof ArrayBuffer ? new Uint8Array(chunk) : chunk;
        await writable.write(bytes);
      },
      close: async () => writable.close(),
      abort: async () => writable.abort(),
    };
  }

  async function deleteFile(filename) {
    await ensureModelDir();
    try {
      await currentModelDir.removeEntry(filename);
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
    for await (const [name, handle] of currentModelDir.entries()) {
      if (handle.kind === 'file') {
        files.push(name);
      }
    }
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
