import { getRuntimeConfig } from '../config/runtime.js';
import { isIndexedDBAvailable, isOPFSAvailable } from './quota.js';
import { createIdbStore } from './backends/idb-store.js';

const LEGACY_OPFS_ROOTS = ['models'];
const SYSTEM_DIRS = new Set(['reports']);

async function scanOpfsDirectory(dirHandle) {
  let totalBytes = 0;
  let fileCount = 0;
  let shardCount = 0;
  let hasManifest = false;

  for await (const entry of dirHandle.values()) {
    if (entry.kind === 'file') {
      const file = await entry.getFile();
      totalBytes += file.size;
      fileCount += 1;
      if (entry.name === 'manifest.json') {
        hasManifest = true;
      }
      if (entry.name.startsWith('shard_') && entry.name.endsWith('.bin')) {
        shardCount += 1;
      }
    } else if (entry.kind === 'directory') {
      const child = await scanOpfsDirectory(entry);
      totalBytes += child.totalBytes;
      fileCount += child.fileCount;
      shardCount += child.shardCount;
      hasManifest = hasManifest || child.hasManifest;
    }
  }

  return {
    totalBytes,
    fileCount,
    shardCount,
    hasManifest,
  };
}

async function hasFile(dirHandle, filename) {
  try {
    await dirHandle.getFileHandle(filename);
    return true;
  } catch {
    return false;
  }
}

async function hasModelChild(dirHandle) {
  for await (const entry of dirHandle.values()) {
    if (entry.kind !== 'directory') continue;
    if (await hasFile(entry, 'manifest.json')) {
      return true;
    }
  }
  return false;
}

async function discoverOpfsRoots(rootHandle) {
  const roots = [];
  for await (const entry of rootHandle.values()) {
    if (entry.kind !== 'directory') continue;
    const hasRegistry = await hasFile(entry, 'models.json');
    if (hasRegistry || await hasModelChild(entry)) {
      roots.push(entry.name);
    }
  }
  return roots;
}

async function listOpfsRoots(runtime) {
  if (!isOPFSAvailable()) {
    return [];
  }
  const root = await navigator.storage.getDirectory();
  const configured = runtime?.loading?.opfsPath?.opfsRootDir || 'doppler-models';
  const roots = new Set([configured, ...LEGACY_OPFS_ROOTS]);
  const discovered = await discoverOpfsRoots(root);
  for (const name of discovered) {
    roots.add(name);
  }
  const resolved = [];
  for (const name of roots) {
    try {
      const handle = await root.getDirectoryHandle(name);
      resolved.push({ name, handle });
    } catch {
      // Root does not exist.
    }
  }
  return resolved;
}

async function listOpfsEntries(runtime) {
  if (!isOPFSAvailable()) {
    return { entries: [], systemEntries: [], opfsRoots: [] };
  }
  const roots = await listOpfsRoots(runtime);
  const entries = [];
  const systemEntries = [];

  for (const root of roots) {
    for await (const entry of root.handle.values()) {
      if (entry.kind !== 'directory') continue;
      const stats = await scanOpfsDirectory(entry);
      const base = {
        modelId: entry.name,
        backend: 'opfs',
        root: root.name,
        totalBytes: stats.totalBytes,
        fileCount: stats.fileCount,
        shardCount: stats.shardCount,
        hasManifest: stats.hasManifest,
      };
      if (SYSTEM_DIRS.has(entry.name)) {
        systemEntries.push({
          ...base,
          kind: 'system',
          label: entry.name,
        });
      } else {
        entries.push(base);
      }
    }
  }

  return {
    entries,
    systemEntries,
    opfsRoots: roots.map((root) => root.name),
  };
}

async function listIndexedDbEntries(runtime) {
  if (!isIndexedDBAvailable()) {
    return [];
  }
  const store = createIdbStore(runtime.loading.storage.backend.indexeddb);
  const modelIds = await store.listModels();
  const entries = [];

  for (const modelId of modelIds) {
    const stats = await store.getModelStats(modelId);
    entries.push({
      modelId,
      backend: 'indexeddb',
      totalBytes: stats.totalBytes,
      fileCount: stats.fileCount,
      shardCount: stats.shardCount,
      hasManifest: stats.hasManifest,
    });
  }

  return entries;
}

export async function listStorageInventory() {
  const runtime = getRuntimeConfig();
  const opfsResult = await listOpfsEntries(runtime);
  const idbEntries = await listIndexedDbEntries(runtime);

  return {
    entries: [...opfsResult.entries, ...idbEntries],
    systemEntries: opfsResult.systemEntries,
    opfsRoots: opfsResult.opfsRoots,
    backendAvailability: {
      opfs: isOPFSAvailable(),
      indexeddb: isIndexedDBAvailable(),
    },
  };
}

export async function deleteStorageEntry(entry) {
  if (!entry?.modelId) {
    return false;
  }
  const runtime = getRuntimeConfig();

  if (entry.backend === 'opfs') {
    if (!isOPFSAvailable()) {
      throw new Error('OPFS not available in this browser');
    }
    const rootName = entry.root || runtime?.loading?.opfsPath?.opfsRootDir || 'doppler-models';
    const root = await navigator.storage.getDirectory();
    let opfsRoot = null;
    try {
      opfsRoot = await root.getDirectoryHandle(rootName, { create: false });
    } catch {
      return false;
    }
    try {
      await opfsRoot.removeEntry(entry.modelId, { recursive: true });
      return true;
    } catch (error) {
      if (error?.name === 'NotFoundError') {
        return false;
      }
      throw error;
    }
  }

  if (entry.backend === 'indexeddb') {
    if (!isIndexedDBAvailable()) {
      throw new Error('IndexedDB not available in this browser');
    }
    const store = createIdbStore(runtime.loading.storage.backend.indexeddb);
    return store.deleteModel(entry.modelId);
  }

  return false;
}
