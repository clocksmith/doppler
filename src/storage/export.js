import { getRuntimeConfig } from '../config/runtime.js';
import { isIndexedDBAvailable, isOPFSAvailable } from './quota.js';
import { createOpfsStore } from './backends/opfs-store.js';
import { createIdbStore } from './backends/idb-store.js';

function requireDirectoryHandle(handle) {
  if (!handle || typeof handle.getFileHandle !== 'function') {
    throw new Error('exportModelToDirectory requires a FileSystemDirectoryHandle destination');
  }
}

function normalizeModelId(modelId) {
  // Keep consistent with shard-manager normalization.
  return String(modelId).replace(/[^a-zA-Z0-9_-]/g, '_');
}

function sanitizeFilename(name) {
  // Conservative and cross-platform (avoid nested dirs / invalid chars).
  return String(name).replace(/[\\/:*?"<>|]/g, '_');
}

function resolveChunkBytes(runtime, chunkBytesOverride) {
  const runtimeDefault = runtime?.loading?.storage?.backend?.streaming?.readChunkBytes ?? (4 * 1024 * 1024);
  const val = chunkBytesOverride ?? runtimeDefault;
  return Number.isFinite(val) && val > 0 ? Math.floor(val) : (4 * 1024 * 1024);
}

function resolveExportStore(runtime) {
  const mode = runtime?.loading?.storage?.backend?.backend ?? 'auto';
  if (mode === 'opfs') {
    if (!isOPFSAvailable()) {
      throw new Error('OPFS requested but not available');
    }
    const opfsCfg = runtime?.loading?.storage?.backend?.opfs ?? {};
    const rootCfg = runtime?.loading?.opfsPath?.opfsRootDir ?? 'doppler-models';
    return {
      backend: 'opfs',
      store: createOpfsStore({
        opfsRootDir: rootCfg,
        useSyncAccessHandle: Boolean(opfsCfg.useSyncAccessHandle),
        maxConcurrentHandles: opfsCfg.maxConcurrentHandles ?? 2,
      }),
    };
  }
  if (mode === 'indexeddb') {
    if (!isIndexedDBAvailable()) {
      throw new Error('IndexedDB requested but not available');
    }
    const cfg = runtime?.loading?.storage?.backend?.indexeddb;
    return { backend: 'indexeddb', store: createIdbStore(cfg) };
  }

  // auto: prefer OPFS then IDB
  if (isOPFSAvailable()) {
    const opfsCfg = runtime?.loading?.storage?.backend?.opfs ?? {};
    const rootCfg = runtime?.loading?.opfsPath?.opfsRootDir ?? 'doppler-models';
    return {
      backend: 'opfs',
      store: createOpfsStore({
        opfsRootDir: rootCfg,
        useSyncAccessHandle: Boolean(opfsCfg.useSyncAccessHandle),
        maxConcurrentHandles: opfsCfg.maxConcurrentHandles ?? 2,
      }),
    };
  }
  if (isIndexedDBAvailable()) {
    const cfg = runtime?.loading?.storage?.backend?.indexeddb;
    return { backend: 'indexeddb', store: createIdbStore(cfg) };
  }
  throw new Error('No supported storage backend available for export (opfs, indexeddb).');
}

export async function exportModelToDirectory(modelId, destinationDir, options = {}) {
  if (!modelId || typeof modelId !== 'string') {
    throw new Error('exportModelToDirectory requires modelId');
  }
  requireDirectoryHandle(destinationDir);

  const runtime = getRuntimeConfig();
  const chunkBytes = resolveChunkBytes(runtime, options.chunkBytes);
  const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;

  const { backend, store } = resolveExportStore(runtime);
  const safeModelId = normalizeModelId(modelId);

  await store.init();
  await store.openModel(safeModelId, { create: false });
  onProgress?.({ stage: 'open', modelId, backend });

  const files = await store.listFiles();
  files.sort((a, b) => a.localeCompare(b));
  onProgress?.({ stage: 'list', modelId, backend, fileCount: files.length });

  let exported = 0;
  try {
    for (const srcNameRaw of files) {
      const srcName = String(srcNameRaw);
      const dstName = sanitizeFilename(srcName);

      onProgress?.({
        stage: 'file_start',
        modelId,
        backend,
        filename: srcName,
        index: exported,
        total: files.length,
      });

      const outHandle = await destinationDir.getFileHandle(dstName, { create: true });
      const writable = await outHandle.createWritable();

      let written = 0;
      try {
        if (typeof store.readFileRangeStream === 'function') {
          for await (const chunk of store.readFileRangeStream(srcName, 0, null, { chunkBytes })) {
            await writable.write(chunk);
            written += chunk.byteLength ?? chunk.length ?? 0;
            onProgress?.({ stage: 'file_progress', modelId, filename: srcName, writtenBytes: written });
          }
        } else {
          const ab = await store.readFile(srcName);
          const bytes = ab instanceof Uint8Array ? ab : new Uint8Array(ab);
          await writable.write(bytes);
          written = bytes.byteLength;
        }
        await writable.close();
      } catch (err) {
        try {
          await writable.abort();
        } catch {}
        throw err;
      }

      exported += 1;
      onProgress?.({
        stage: 'file_done',
        modelId,
        backend,
        filename: srcName,
        writtenBytes: written,
        index: exported,
        total: files.length,
      });
    }
  } finally {
    await store.cleanup();
  }

  onProgress?.({ stage: 'done', modelId, backend, fileCount: files.length });
  return { modelId, fileCount: files.length };
}
