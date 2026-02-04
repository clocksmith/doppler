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
  const {
    dbName,
    shardStore,
    metaStore,
    chunkSizeBytes,
  } = config;

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
    const chunkCount = Math.ceil(bytes.byteLength / chunkSizeBytes);
    const tx = db.transaction([shardStore, metaStore], 'readwrite');
    const shardStoreRef = tx.objectStore(shardStore);
    const metaStoreRef = tx.objectStore(metaStore);

    for (let i = 0; i < chunkCount; i++) {
      const start = i * chunkSizeBytes;
      const end = Math.min(start + chunkSizeBytes, bytes.byteLength);
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
