import { getRuntimeConfig } from '../config/runtime.js';
import { selectStorageBackend } from './backend-selection.js';
import { normalizeModelId } from './normalize-model-id.js';

function requireDirectoryHandle(handle) {
  if (!handle || typeof handle.getFileHandle !== 'function') {
    throw new Error('exportModelToDirectory requires a FileSystemDirectoryHandle destination');
  }
}

function sanitizeFilename(name) {
  // Conservative and cross-platform (avoid nested dirs / invalid chars).
  return String(name).replace(/[\\/:*?"<>|]/g, '_');
}

function resolveChunkBytes(runtime, chunkBytesOverride) {
  const value = chunkBytesOverride ?? runtime.loading.storage.backend.streaming.readChunkBytes;
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error('exportModelToDirectory requires a positive integer chunkBytes');
  }
  return value;
}

export async function exportModelToDirectory(modelId, destinationDir, options = {}) {
  if (!modelId || typeof modelId !== 'string') {
    throw new Error('exportModelToDirectory requires modelId');
  }
  requireDirectoryHandle(destinationDir);

  const runtime = getRuntimeConfig();
  const chunkBytes = resolveChunkBytes(runtime, options.chunkBytes);
  const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;

  const { type: backend, backend: store } = selectStorageBackend(
    runtime.loading.storage.backend,
    runtime.loading.opfsPath
  );
  if (backend === 'memory') {
    throw new Error('exportModelToDirectory cannot reopen in-memory storage; select the persistent backend containing the model.');
  }
  const safeModelId = normalizeModelId(modelId);

  try {
    await store.init();
    await store.openModel(safeModelId, { create: false });
    onProgress?.({ stage: 'open', modelId, backend });

    const files = await store.listFiles();
    files.sort((a, b) => a.localeCompare(b));
    onProgress?.({ stage: 'list', modelId, backend, fileCount: files.length });

    let exported = 0;
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
    onProgress?.({ stage: 'done', modelId, backend, fileCount: files.length });
    return { modelId, fileCount: files.length };
  } finally {
    await store.cleanup();
  }
}
