

import { getRuntimeConfig } from '../config/runtime.js';
import { isIndexedDBAvailable, isOPFSAvailable } from '../storage/quota.js';
import { createIdbStore } from '../storage/backends/idb-store.js';
import { createFileTensorSource } from './tensor-source-file.js';
import { createHttpTensorSource } from './tensor-source-http.js';

const TEMP_DIR = 'temp-downloads';
const TEMP_MODEL_PREFIX = '__temp_download__';

function inferNameFromUrl(url) {
  try {
    const parsed = new URL(url, typeof window !== 'undefined' ? window.location.href : undefined);
    const pathname = parsed.pathname || '';
    const part = pathname.split('/').filter(Boolean).pop();
    return part || 'remote';
  } catch {
    const parts = String(url).split('/');
    return parts[parts.length - 1] || 'remote';
  }
}

function sanitizeFilename(name) {
  return name.replace(/[^a-zA-Z0-9._-]/g, '_');
}

function buildTempFilename(name) {
  const safe = sanitizeFilename(name);
  const stamp = Date.now().toString(36);
  const rand = Math.random().toString(36).slice(2, 8);
  return `${stamp}-${rand}-${safe}`;
}

async function streamDownload(url, options, onChunk) {
  const { headers, signal } = options;
  const response = await fetch(url, { headers, signal });
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status}`);
  }

  let totalBytes = 0;
  if (response.body && response.body.getReader) {
    const reader = response.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        totalBytes += value.byteLength;
        await onChunk(value);
      }
    }
  } else {
    const buffer = new Uint8Array(await response.arrayBuffer());
    totalBytes = buffer.byteLength;
    await onChunk(buffer);
  }

  return {
    totalBytes,
    contentLength: response.headers.get('content-length'),
  };
}

async function downloadToOpfs(url, options = {}) {
  const runtime = getRuntimeConfig();
  const root = await navigator.storage.getDirectory();
  const opfsRoot = await root.getDirectoryHandle(runtime.loading.opfsPath.opfsRootDir, { create: true });
  const tempDir = await opfsRoot.getDirectoryHandle(TEMP_DIR, { create: true });

  const originalName = options.name || inferNameFromUrl(url);
  const tempName = buildTempFilename(originalName);
  const fileHandle = await tempDir.getFileHandle(tempName, { create: true });
  const writable = await fileHandle.createWritable();

  const result = await streamDownload(url, options, async (chunk) => {
    await writable.write(chunk);
  });

  await writable.close();
  const file = await fileHandle.getFile();
  const source = createFileTensorSource(file);
  const size = file.size || result.totalBytes;

  return {
    source: {
      ...source,
      name: originalName,
      sourceType: 'download-opfs',
      cleanup: async () => {
        await tempDir.removeEntry(tempName);
      },
    },
    size,
  };
}

async function downloadToIdb(url, options = {}) {
  const runtime = getRuntimeConfig();
  const idbConfig = runtime.loading.storage.backend.indexeddb;
  const store = createIdbStore(idbConfig);
  const originalName = options.name || inferNameFromUrl(url);
  const tempName = buildTempFilename(originalName);
  const modelId = `${TEMP_MODEL_PREFIX}${tempName}`;

  await store.openModel(modelId, { create: true });
  const stream = await store.createWriteStream(tempName);

  const result = await streamDownload(url, options, async (chunk) => {
    await stream.write(chunk);
  });

  await stream.close();

  let cached = null;
  const readAll = async () => {
    if (!cached) {
      cached = new Uint8Array(await store.readFile(tempName));
    }
    return cached;
  };

  const size = cached ? cached.byteLength : Number.parseInt(result.contentLength || '0', 10) || result.totalBytes;

  return {
    source: {
      sourceType: 'download-idb',
      name: originalName,
      size,
      readRange: async (offset, length) => {
        if (!Number.isFinite(offset) || !Number.isFinite(length) || length <= 0) {
          return new ArrayBuffer(0);
        }
        const bytes = await readAll();
        const start = Math.max(0, offset);
        const end = Math.min(start + length, bytes.byteLength);
        return bytes.slice(start, end).buffer;
      },
      readAll: async () => {
        const bytes = await readAll();
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      },
      close: async () => {
        return;
      },
      getAuxFiles: async () => {
        return {};
      },
      cleanup: async () => {
        await store.deleteModel(modelId);
        await store.cleanup();
      },
    },
    size,
  };
}

export async function createDownloadTensorSource(url, options = {}) {
  if (isOPFSAvailable()) {
    return downloadToOpfs(url, options);
  }
  if (isIndexedDBAvailable()) {
    return downloadToIdb(url, options);
  }
  throw new Error('No storage backend available for download fallback');
}

export async function createRemoteTensorSource(url, options = {}) {
  try {
    const source = await createHttpTensorSource(url, options);
    return { source, size: source.size, supportsRange: true };
  } catch (_error) {
    const downloaded = await createDownloadTensorSource(url, options);
    return { ...downloaded, supportsRange: false };
  }
}
