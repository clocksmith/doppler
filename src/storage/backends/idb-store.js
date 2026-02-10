import { isIndexedDBAvailable } from '../quota.js';

function requestToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function transactionDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error('IndexedDB transaction aborted'));
  });
}

function buildFileKey(modelId, filename) {
  return `file:${modelId}:${filename}`;
}

function buildManifestKey(modelId) {
  return `manifest:${modelId}`;
}

function buildTokenizerKey(modelId) {
  return `tokenizer:${modelId}`;
}

function buildModelKey(modelId) {
  return `model:${modelId}`;
}

export function createIdbStore(config) {
  const cfg = config ?? {};
  const {
    dbName = 'doppler-models',
    shardStore = 'shards',
    metaStore = 'meta',
    chunkSizeBytes,
  } = cfg;
  const chunkSizeBytesResolved = Number.isFinite(chunkSizeBytes) && chunkSizeBytes > 0
    ? Math.floor(chunkSizeBytes)
    : (4 * 1024 * 1024);

  let db = null;
  let currentModelId = null;

  async function init() {
    if (!isIndexedDBAvailable()) {
      throw new Error('IndexedDB not available in this browser');
    }
    if (db) return;
    db = await new Promise((resolve, reject) => {
      const request = indexedDB.open(dbName, 1);
      request.onerror = () => reject(new Error('Failed to open IndexedDB'));
      request.onupgradeneeded = (event) => {
        const database = event.target.result;
        if (!database.objectStoreNames.contains(shardStore)) {
          const store = database.createObjectStore(shardStore, { keyPath: ['modelId', 'filename', 'chunkIndex'] });
          store.createIndex('modelId', 'modelId', { unique: false });
          store.createIndex('modelFile', ['modelId', 'filename'], { unique: false });
        }
        if (!database.objectStoreNames.contains(metaStore)) {
          database.createObjectStore(metaStore, { keyPath: 'key' });
        }
      };
      request.onsuccess = () => resolve(request.result);
    });
  }

  async function openModel(modelId, options = {}) {
    await init();
    currentModelId = modelId;
    const create = options.create !== false;
    if (create) {
      const tx = db.transaction(metaStore, 'readwrite');
      const store = tx.objectStore(metaStore);
      store.put({ key: buildModelKey(modelId), value: true });
      await transactionDone(tx);
      return null;
    }
    const existing = await readMeta(buildModelKey(modelId));
    if (!existing) {
      throw new Error('Model not found');
    }
    return null;
  }

  function getCurrentModelId() {
    return currentModelId;
  }

  function requireModel() {
    if (!currentModelId) {
      throw new Error('No model open. Call openModelStore first.');
    }
  }

  async function readMeta(key) {
    const tx = db.transaction(metaStore, 'readonly');
    const store = tx.objectStore(metaStore);
    const result = await requestToPromise(store.get(key));
    await transactionDone(tx);
    return result?.value ?? null;
  }

  async function writeMeta(key, value) {
    const tx = db.transaction(metaStore, 'readwrite');
    const store = tx.objectStore(metaStore);
    store.put({ key, value });
    await transactionDone(tx);
  }

  async function readFile(filename) {
    requireModel();
    const fileKey = buildFileKey(currentModelId, filename);
    const fileMeta = await readMeta(fileKey);
    if (!fileMeta) {
      throw new Error(`File not found: ${filename}`);
    }

    const { size, chunkCount } = fileMeta;
    const buffer = new Uint8Array(size);
    let offset = 0;
    const tx = db.transaction(shardStore, 'readonly');
    const store = tx.objectStore(shardStore);

    for (let i = 0; i < chunkCount; i++) {
      const entry = await requestToPromise(store.get([currentModelId, filename, i]));
      if (!entry?.data) {
        throw new Error(`Missing chunk ${i} for ${filename}`);
      }
      const chunk = new Uint8Array(entry.data);
      buffer.set(chunk, offset);
      offset += chunk.byteLength;
    }

    await transactionDone(tx);
    return buffer.buffer;
  }

  async function readFileRange(filename, offset = 0, length = null) {
    requireModel();
    const fileKey = buildFileKey(currentModelId, filename);
    const fileMeta = await readMeta(fileKey);
    if (!fileMeta) {
      throw new Error(`File not found: ${filename}`);
    }

    const startRaw = Number(offset);
    const start = Number.isFinite(startRaw) ? Math.max(0, Math.floor(startRaw)) : 0;
    const end = length == null
      ? fileMeta.size
      : Math.min(fileMeta.size, start + Math.max(0, Number.isFinite(Number(length)) ? Math.floor(Number(length)) : 0));
    const want = Math.max(0, end - start);

    if (want === 0) {
      return new Uint8Array(0).buffer;
    }

    const tx = db.transaction(shardStore, 'readonly');
    const store = tx.objectStore(shardStore);

    // We still have to walk chunks sequentially because chunk sizes may vary (streaming writes).
    const out = new Uint8Array(want);
    let outOff = 0;
    let inOff = 0;

    for (let i = 0; i < fileMeta.chunkCount; i++) {
      const entry = await requestToPromise(store.get([currentModelId, filename, i]));
      if (!entry?.data) {
        throw new Error(`Missing chunk ${i} for ${filename}`);
      }
      const chunk = new Uint8Array(entry.data);
      const chunkStart = inOff;
      const chunkEnd = inOff + chunk.byteLength;
      inOff = chunkEnd;

      if (chunkEnd <= start) continue;
      if (chunkStart >= end) break;

      const takeStart = Math.max(0, start - chunkStart);
      const takeEnd = Math.min(chunk.byteLength, end - chunkStart);
      const slice = chunk.subarray(takeStart, takeEnd);
      out.set(slice, outOff);
      outOff += slice.byteLength;
      if (outOff >= want) break;
    }

    await transactionDone(tx);
    return out.buffer;
  }

  async function* readFileRangeStream(filename, offset = 0, length = null, options = {}) {
    requireModel();
    const fileKey = buildFileKey(currentModelId, filename);
    const fileMeta = await readMeta(fileKey);
    if (!fileMeta) {
      throw new Error(`File not found: ${filename}`);
    }

    const rawChunkBytes = options?.chunkBytes;
    const chunkBytes = Number.isFinite(rawChunkBytes) && rawChunkBytes > 0
      ? Math.floor(rawChunkBytes)
      : chunkSizeBytesResolved;
    const startRaw = Number(offset);
    const start = Number.isFinite(startRaw) ? Math.max(0, Math.floor(startRaw)) : 0;
    const end = length == null
      ? fileMeta.size
      : Math.min(fileMeta.size, start + Math.max(0, Number.isFinite(Number(length)) ? Math.floor(Number(length)) : 0));

    if (end <= start) return;

    let inOff = 0;
    let emitted = 0;

    for (let i = 0; i < fileMeta.chunkCount; i++) {
      // Important: do NOT keep a single IndexedDB transaction open across `yield`.
      // Transactions can auto-close when the event loop advances, which would make
      // subsequent requests fail (TransactionInactiveError).
      const tx = db.transaction(shardStore, 'readonly');
      const store = tx.objectStore(shardStore);
      const entry = await requestToPromise(store.get([currentModelId, filename, i]));
      await transactionDone(tx);
      if (!entry?.data) {
        throw new Error(`Missing chunk ${i} for ${filename}`);
      }
      const chunk = new Uint8Array(entry.data);
      const chunkStart = inOff;
      const chunkEnd = inOff + chunk.byteLength;
      inOff = chunkEnd;

      if (chunkEnd <= start) continue;
      if (chunkStart >= end) break;

      const takeStart = Math.max(0, start - chunkStart);
      const takeEnd = Math.min(chunk.byteLength, end - chunkStart);
      let view = chunk.subarray(takeStart, takeEnd);

      // Further split into smaller chunks if requested (controls writeBuffer / IO granularity).
      for (let at = 0; at < view.byteLength; at += chunkBytes) {
        const part = view.subarray(at, Math.min(view.byteLength, at + chunkBytes));
        emitted += part.byteLength;
        yield part.slice(0);
        if (start + emitted >= end) break;
      }
    }
  }

  async function readText(filename) {
    try {
      const buffer = await readFile(filename);
      return new TextDecoder().decode(buffer);
    } catch (error) {
      if (error.message?.includes('not found')) {
        return null;
      }
      throw error;
    }
  }

  async function deleteFile(filename) {
    requireModel();
    const tx = db.transaction([shardStore, metaStore], 'readwrite');
    const shardStoreRef = tx.objectStore(shardStore);
    const metaStoreRef = tx.objectStore(metaStore);
    const range = IDBKeyRange.bound([currentModelId, filename, 0], [currentModelId, filename, Number.MAX_SAFE_INTEGER]);
    shardStoreRef.delete(range);
    metaStoreRef.delete(buildFileKey(currentModelId, filename));
    await transactionDone(tx);
    return true;
  }

  async function writeFile(filename, data) {
    requireModel();
    await deleteFile(filename);
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
    const chunkCount = Math.ceil(bytes.byteLength / chunkSizeBytesResolved);
    const tx = db.transaction([shardStore, metaStore], 'readwrite');
    const shardStoreRef = tx.objectStore(shardStore);
    const metaStoreRef = tx.objectStore(metaStore);

    for (let i = 0; i < chunkCount; i++) {
      const start = i * chunkSizeBytesResolved;
      const end = Math.min(start + chunkSizeBytesResolved, bytes.byteLength);
      const chunk = bytes.slice(start, end);
      shardStoreRef.put({
        modelId: currentModelId,
        filename,
        chunkIndex: i,
        data: chunk,
      });
    }

    metaStoreRef.put({
      key: buildFileKey(currentModelId, filename),
      value: { size: bytes.byteLength, chunkCount },
    });
    metaStoreRef.put({ key: buildModelKey(currentModelId), value: true });
    await transactionDone(tx);
  }

  async function createWriteStream(filename) {
    requireModel();
    await deleteFile(filename);
    let chunkIndex = 0;
    let totalBytes = 0;

    return {
      write: async (chunk) => {
        const bytes = chunk instanceof ArrayBuffer ? new Uint8Array(chunk) : chunk;
        const tx = db.transaction(shardStore, 'readwrite');
        const store = tx.objectStore(shardStore);
        store.put({
          modelId: currentModelId,
          filename,
          chunkIndex,
          data: bytes.slice(0),
        });
        await transactionDone(tx);
        chunkIndex += 1;
        totalBytes += bytes.byteLength;
      },
      close: async () => {
        const tx = db.transaction(metaStore, 'readwrite');
        const store = tx.objectStore(metaStore);
        store.put({
          key: buildFileKey(currentModelId, filename),
          value: { size: totalBytes, chunkCount: chunkIndex },
        });
        store.put({ key: buildModelKey(currentModelId), value: true });
        await transactionDone(tx);
      },
      abort: async () => {
        await deleteFile(filename);
      },
    };
  }

  async function listFiles() {
    requireModel();
    const tx = db.transaction(metaStore, 'readonly');
    const store = tx.objectStore(metaStore);
    const results = [];
    const prefix = buildFileKey(currentModelId, '');
    const request = store.openCursor();
    await new Promise((resolve, reject) => {
      request.onerror = () => reject(request.error);
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (!cursor) {
          resolve();
          return;
        }
        const key = cursor.key;
        if (typeof key === 'string' && key.startsWith(prefix)) {
          results.push(key.substring(prefix.length));
        }
        cursor.continue();
      };
    });
    await transactionDone(tx);
    return results;
  }

  async function listModels() {
    await init();
    const tx = db.transaction(metaStore, 'readonly');
    const store = tx.objectStore(metaStore);
    const results = [];
    const request = store.openCursor();
    await new Promise((resolve, reject) => {
      request.onerror = () => reject(request.error);
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (!cursor) {
          resolve();
          return;
        }
        const key = cursor.key;
        if (typeof key === 'string' && key.startsWith('model:')) {
          results.push(key.substring('model:'.length));
        }
        cursor.continue();
      };
    });
    await transactionDone(tx);
    return results;
  }

  async function getModelStats(modelId) {
    await init();
    const tx = db.transaction(metaStore, 'readonly');
    const store = tx.objectStore(metaStore);
    const prefix = buildFileKey(modelId, '');
    let totalBytes = 0;
    let fileCount = 0;
    let shardCount = 0;
    let hasManifest = false;

    const request = store.openCursor();
    await new Promise((resolve, reject) => {
      request.onerror = () => reject(request.error);
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (!cursor) {
          resolve();
          return;
        }
        const key = cursor.key;
        if (typeof key === 'string' && key.startsWith(prefix)) {
          const filename = key.substring(prefix.length);
          const meta = cursor.value?.value;
          const size = meta?.size ?? 0;
          totalBytes += size;
          fileCount += 1;
          if (filename === 'manifest.json') {
            hasManifest = true;
          }
          if (filename.startsWith('shard_') && filename.endsWith('.bin')) {
            shardCount += 1;
          }
        }
        cursor.continue();
      };
    });
    await transactionDone(tx);
    return {
      totalBytes,
      fileCount,
      shardCount,
      hasManifest,
    };
  }

  async function deleteModel(modelId) {
    await init();
    const tx = db.transaction([shardStore, metaStore], 'readwrite');
    const shardStoreRef = tx.objectStore(shardStore);
    const metaStoreRef = tx.objectStore(metaStore);
    const range = IDBKeyRange.bound([modelId, '', 0], [modelId, '\uffff', Number.MAX_SAFE_INTEGER]);
    shardStoreRef.delete(range);

    const request = metaStoreRef.openCursor();
    await new Promise((resolve, reject) => {
      request.onerror = () => reject(request.error);
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (!cursor) {
          resolve();
          return;
        }
        const key = cursor.key;
        if (typeof key === 'string' && key.startsWith(`file:${modelId}:`)) {
          cursor.delete();
        }
        if (key === buildManifestKey(modelId) || key === buildTokenizerKey(modelId) || key === buildModelKey(modelId)) {
          cursor.delete();
        }
        cursor.continue();
      };
    });
    await transactionDone(tx);
    if (currentModelId === modelId) {
      currentModelId = null;
    }
    return true;
  }

  async function writeManifest(text) {
    requireModel();
    await writeMeta(buildManifestKey(currentModelId), text);
  }

  async function readManifest() {
    requireModel();
    return readMeta(buildManifestKey(currentModelId));
  }

  async function writeTokenizer(text) {
    requireModel();
    await writeMeta(buildTokenizerKey(currentModelId), text);
  }

  async function readTokenizer() {
    requireModel();
    return readMeta(buildTokenizerKey(currentModelId));
  }

  async function cleanup() {
    db = null;
    currentModelId = null;
  }

  return {
    init,
    openModel,
    getCurrentModelId,
    readFile,
    readText,
    writeFile,
    createWriteStream,
    deleteFile,
    listFiles,
    listModels,
    deleteModel,
    writeManifest,
    readManifest,
    writeTokenizer,
    readTokenizer,
    getModelStats,
    cleanup,
  };
}
