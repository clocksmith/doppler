import { log } from '../src/debug/index.js';
import { createWorkspaceOpfsStore } from './workspace-opfs.js';
import { createWorkspaceIdbStore, getWorkspaceDbName } from './workspace-idb.js';

async function buildStore(options) {
  const workspaceId = options.workspaceId || 'default';
  const rootDirName = options.rootDirName || 'workspaces';
  const backendType = options.backendType || 'auto';

  if (backendType === 'opfs' || backendType === 'auto') {
    try {
      const opfs = createWorkspaceOpfsStore({ rootDirName, workspaceId, useSyncAccessHandle: true });
      await opfs.init();
      return { store: opfs, backendType: 'opfs' };
    } catch (error) {
      if (backendType === 'opfs') throw error;
    }
  }

  const dbName = getWorkspaceDbName(workspaceId);
  const idb = createWorkspaceIdbStore({ dbName, workspaceId });
  await idb.init();
  return { store: idb, backendType: 'indexeddb' };
}

async function writeEntry(store, entry) {
  const file = await entry.handle.getFile();
  if (store.createWriteStream) {
    const writer = await store.createWriteStream(entry.path);
    if (writer) {
      for await (const chunk of file.stream()) {
        await writer.write(chunk);
      }
      await writer.close();
      return;
    }
  }
  await store.writeBlob(entry.path, file);
}

self.onmessage = async (event) => {
  const data = event.data || {};
  if (data.type !== 'import') return;

  try {
    const { store, backendType } = await buildStore(data);
    const entries = data.entries || [];
    const total = entries.length;
    let completed = 0;

    for (const entry of entries) {
      await writeEntry(store, entry);
      completed += 1;
      self.postMessage({
        type: 'progress',
        completed,
        total,
        path: entry.path,
      });
    }

    self.postMessage({
      type: 'done',
      total,
      workspaceId: data.workspaceId || 'default',
      backendType,
    });
  } catch (error) {
    log.error('WorkspaceWorker', 'Import failed', error);
    self.postMessage({
      type: 'error',
      message: error?.message || String(error),
    });
  }
};
