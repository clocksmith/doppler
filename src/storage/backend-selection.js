import { log } from '../debug/index.js';
import { isIndexedDBAvailable, isOPFSAvailable } from './quota.js';
import { createIdbStore } from './backends/idb-store.js';
import { createMemoryStore } from './backends/memory-store.js';
import { createOpfsStore } from './backends/opfs-store.js';

function resolveBackendType(config) {
  if (config.backend === 'opfs') {
    if (!isOPFSAvailable()) throw new Error('OPFS requested but not available');
    return 'opfs';
  }
  if (config.backend === 'indexeddb') {
    if (!isIndexedDBAvailable()) throw new Error('IndexedDB requested but not available');
    return 'indexeddb';
  }
  if (config.backend === 'memory') return 'memory';
  if (isOPFSAvailable()) return 'opfs';
  if (isIndexedDBAvailable()) return 'indexeddb';
  log.warn(
    'ShardManager',
    'No persistent storage available (OPFS/IndexedDB); falling back to in-memory storage. '
      + 'Model data will not persist across reloads.'
  );
  return 'memory';
}

export function selectStorageBackend(config, opfsPathConfig) {
  const type = resolveBackendType(config);
  if (type === 'opfs') {
    return {
      type,
      backend: createOpfsStore({
        opfsRootDir: opfsPathConfig.opfsRootDir,
        useSyncAccessHandle: config.opfs.useSyncAccessHandle,
        maxConcurrentHandles: config.opfs.maxConcurrentHandles,
      }),
    };
  }
  if (type === 'indexeddb') {
    return { type, backend: createIdbStore(config.indexeddb) };
  }
  return { type, backend: createMemoryStore(config.memory) };
}
