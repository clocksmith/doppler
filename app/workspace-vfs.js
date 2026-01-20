import { log } from '../src/debug/index.js';
import { createWorkspaceOpfsStore } from './workspace-opfs.js';
import { createWorkspaceIdbStore, getWorkspaceDbName } from './workspace-idb.js';

export async function createWorkspaceVfs(options = {}) {
  const workspaceId = options.workspaceId || 'default';
  const rootDirName = options.rootDirName || 'workspaces';
  const backendPref = options.backend || 'auto';
  const useSyncAccessHandle = options.useSyncAccessHandle !== false;
  const maxConcurrentHandles = options.maxConcurrentHandles ?? 4;

  let backend = null;
  let backendType = null;

  if (backendPref !== 'indexeddb') {
    try {
      backend = createWorkspaceOpfsStore({
        rootDirName,
        workspaceId,
        useSyncAccessHandle,
        maxConcurrentHandles,
      });
      await backend.init();
      backendType = 'opfs';
    } catch (error) {
      if (backendPref === 'opfs') throw error;
      log.warn('WorkspaceVfs', 'OPFS unavailable, falling back to IndexedDB', {
        error: error?.message || String(error),
      });
    }
  }

  if (!backend) {
    const dbName = options.dbName || getWorkspaceDbName(workspaceId);
    const storeName = options.storeName || 'files';
    backend = createWorkspaceIdbStore({ dbName, storeName, workspaceId });
    await backend.init();
    backendType = 'indexeddb';
  }

  async function readText(path) {
    if (backend.readText) return backend.readText(path);
    const blob = await backend.readBlob(path);
    return blob ? blob.text() : null;
  }

  async function writeText(path, text) {
    const blob = new Blob([text], { type: 'text/plain' });
    await backend.writeBlob(path, blob);
  }

  return {
    backendType,
    workspaceId,
    rootDirName,
    readText,
    readBlob: (path) => backend.readBlob(path),
    writeText,
    writeBlob: (path, data) => backend.writeBlob(path, data),
    list: (prefix) => backend.list(prefix),
    stat: (path) => backend.stat(path),
    exists: (path) => backend.exists(path),
    remove: (path) => backend.remove(path),
    mkdir: (path) => backend.mkdir(path),
  };
}
