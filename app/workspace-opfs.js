import { log } from '../src/debug/index.js';

const DEFAULT_ROOT_DIR = 'workspaces';
const DEFAULT_WORKSPACE_ID = 'default';

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

function normalizePath(path) {
  if (typeof path !== 'string') {
    throw new Error('Invalid path');
  }
  let clean = path.trim().replace(/\\/g, '/');
  while (clean.startsWith('/')) {
    clean = clean.slice(1);
  }
  clean = clean.replace(/\/+/g, '/');
  return clean;
}

function splitPath(path) {
  const clean = normalizePath(path);
  return clean ? clean.split('/').filter(Boolean) : [];
}

function toBytes(data) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

export function createWorkspaceOpfsStore(options = {}) {
  const rootDirName = options.rootDirName || DEFAULT_ROOT_DIR;
  const workspaceId = options.workspaceId || DEFAULT_WORKSPACE_ID;
  const useSyncAccessHandle = options.useSyncAccessHandle !== false;
  const maxConcurrentHandles = options.maxConcurrentHandles ?? 4;

  let rootDir = null;
  let workspaceRoot = null;
  let workspaceDir = null;

  const syncAccessEnabled = !!useSyncAccessHandle
    && typeof FileSystemSyncAccessHandle !== 'undefined'
    && typeof WorkerGlobalScope !== 'undefined'
    && typeof self !== 'undefined'
    && self instanceof WorkerGlobalScope;
  const handleLimiter = syncAccessEnabled ? createLimiter(maxConcurrentHandles) : null;

  async function init() {
    if (workspaceDir) return;
    if (!navigator.storage?.getDirectory) {
      throw new Error('OPFS not available');
    }
    rootDir = await navigator.storage.getDirectory();
    workspaceRoot = await rootDir.getDirectoryHandle(rootDirName, { create: true });
    workspaceDir = await workspaceRoot.getDirectoryHandle(workspaceId, { create: true });
    log.info('WorkspaceOPFS', `Opened ${rootDirName}/${workspaceId}`);
  }

  async function ensureWorkspaceDir() {
    if (!workspaceDir) {
      await init();
    }
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
        },
      };
    } catch (error) {
      handleLimiter.release();
      if (error?.name === 'InvalidStateError' || error?.name === 'NotAllowedError') {
        return null;
      }
      throw error;
    }
  }

  async function resolveDir(pathParts, createDirs) {
    await ensureWorkspaceDir();
    let dir = workspaceDir;
    for (const part of pathParts) {
      if (!part) continue;
      dir = await dir.getDirectoryHandle(part, { create: createDirs });
    }
    return dir;
  }

  async function resolveFile(path, createDirs) {
    const parts = splitPath(path);
    if (parts.length === 0) {
      throw new Error('Invalid path');
    }
    const name = parts.pop();
    const dir = await resolveDir(parts, createDirs);
    return { dir, name };
  }

  async function readBlob(path) {
    await ensureWorkspaceDir();
    try {
      const { dir, name } = await resolveFile(path, false);
      const fileHandle = await dir.getFileHandle(name);
      const file = await fileHandle.getFile();
      return file;
    } catch (error) {
      if (error?.name === 'NotFoundError') return null;
      throw error;
    }
  }

  async function readText(path) {
    const blob = await readBlob(path);
    return blob ? blob.text() : null;
  }

  async function writeBlob(path, data) {
    await ensureWorkspaceDir();
    const { dir, name } = await resolveFile(path, true);
    const fileHandle = await dir.getFileHandle(name, { create: true });
    const blob = data instanceof Blob ? data : new Blob([data]);
    const access = await openSyncAccessHandle(fileHandle);
    if (access) {
      try {
        const buffer = await blob.arrayBuffer();
        const bytes = toBytes(buffer);
        access.handle.truncate(0);
        access.handle.write(bytes, { at: 0 });
        access.handle.flush();
      } finally {
        access.release();
      }
      return;
    }
    const writable = await fileHandle.createWritable();
    await writable.write(blob);
    await writable.close();
  }

  async function createWriteStream(path) {
    await ensureWorkspaceDir();
    const { dir, name } = await resolveFile(path, true);
    const fileHandle = await dir.getFileHandle(name, { create: true });
    const access = await openSyncAccessHandle(fileHandle);
    if (access) {
      let offset = 0;
      let closed = false;
      access.handle.truncate(0);
      return {
        write: async (chunk) => {
          if (closed) throw new Error('Write after close');
          const bytes = chunk instanceof Blob
            ? new Uint8Array(await chunk.arrayBuffer())
            : toBytes(chunk instanceof ArrayBuffer ? chunk : chunk);
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
        await writable.write(chunk);
      },
      close: async () => writable.close(),
      abort: async () => writable.abort(),
    };
  }

  async function list(prefix = '/') {
    await ensureWorkspaceDir();
    const results = [];
    const cleanPrefix = normalizePath(prefix || '/');
    const needle = cleanPrefix ? '/' + cleanPrefix : '/';

    async function walk(dir, base) {
      for await (const [name, handle] of dir.entries()) {
        const nextPath = base + '/' + name;
        if (handle.kind === 'file') {
          results.push(nextPath);
        } else if (handle.kind === 'directory') {
          await walk(handle, nextPath);
        }
      }
    }

    await walk(workspaceDir, '');
    if (needle === '/') return results;
    return results.filter((path) => path.startsWith(needle));
  }

  async function stat(path) {
    try {
      const { dir, name } = await resolveFile(path, false);
      const fileHandle = await dir.getFileHandle(name);
      const file = await fileHandle.getFile();
      return {
        path: '/' + normalizePath(path),
        size: file.size,
        updated: file.lastModified || Date.now(),
      };
    } catch (error) {
      if (error?.name === 'NotFoundError') return null;
      throw error;
    }
  }

  async function exists(path) {
    return !!(await stat(path));
  }

  async function remove(path) {
    const parts = splitPath(path);
    if (parts.length === 0) return false;
    const name = parts.pop();
    const dir = await resolveDir(parts, false);
    try {
      await dir.removeEntry(name);
      return true;
    } catch (error) {
      if (error?.name === 'NotFoundError') return false;
      throw error;
    }
  }

  async function mkdir(path) {
    const parts = splitPath(path);
    if (parts.length === 0) return;
    await resolveDir(parts, true);
  }

  return {
    init,
    readBlob,
    readText,
    writeBlob,
    createWriteStream,
    list,
    stat,
    exists,
    remove,
    mkdir,
  };
}
