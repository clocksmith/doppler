import { log } from '../src/debug/index.js';

const DEFAULT_DB_PREFIX = 'doppler-workspace-';
const DEFAULT_STORE = 'files';

function normalizePath(path) {
  if (typeof path !== 'string') {
    throw new Error('Invalid path');
  }
  let clean = path.trim().replace(/\\/g, '/');
  if (!clean.startsWith('/')) {
    clean = '/' + clean;
  }
  clean = clean.replace(/\/+/g, '/');
  return clean;
}

function requestToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('IndexedDB request failed'));
  });
}

function transactionDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error('IndexedDB transaction failed'));
    tx.onabort = () => reject(tx.error || new Error('IndexedDB transaction aborted'));
  });
}

function normalizeWorkspaceId(workspaceId) {
  return String(workspaceId || 'default').replace(/[^a-zA-Z0-9_-]/g, '_');
}

export function getWorkspaceDbName(workspaceId) {
  return DEFAULT_DB_PREFIX + normalizeWorkspaceId(workspaceId);
}

export function createWorkspaceIdbStore(options = {}) {
  const workspaceId = options.workspaceId || 'default';
  const dbName = options.dbName || getWorkspaceDbName(workspaceId);
  const storeName = options.storeName || DEFAULT_STORE;
  let db = null;

  async function init() {
    if (db) return;
    if (typeof indexedDB === 'undefined') {
      throw new Error('IndexedDB not available');
    }
    db = await new Promise((resolve, reject) => {
      const request = indexedDB.open(dbName, 1);
      request.onerror = () => reject(request.error || new Error('Failed to open IndexedDB'));
      request.onupgradeneeded = (event) => {
        const database = event.target.result;
        if (!database.objectStoreNames.contains(storeName)) {
          database.createObjectStore(storeName, { keyPath: 'path' });
        }
      };
      request.onsuccess = () => resolve(request.result);
    });
    log.info('WorkspaceIDB', `Opened ${dbName}`);
  }

  async function readEntry(path) {
    await init();
    const cleanPath = normalizePath(path);
    const tx = db.transaction(storeName, 'readonly');
    const store = tx.objectStore(storeName);
    const result = await requestToPromise(store.get(cleanPath));
    await transactionDone(tx);
    return result || null;
  }

  async function readBlob(path) {
    const entry = await readEntry(path);
    return entry ? entry.blob : null;
  }

  async function readText(path) {
    const blob = await readBlob(path);
    return blob ? blob.text() : null;
  }

  async function writeBlob(path, data) {
    await init();
    const cleanPath = normalizePath(path);
    const blob = data instanceof Blob ? data : new Blob([data]);
    const entry = {
      path: cleanPath,
      blob,
      size: blob.size,
      updated: Date.now(),
    };
    const tx = db.transaction(storeName, 'readwrite');
    const store = tx.objectStore(storeName);
    store.put(entry);
    await transactionDone(tx);
  }

  async function list(prefix = '/') {
    await init();
    const cleanPrefix = normalizePath(prefix);
    const tx = db.transaction(storeName, 'readonly');
    const store = tx.objectStore(storeName);
    const keys = await requestToPromise(store.getAllKeys());
    await transactionDone(tx);
    return (keys || []).filter((key) => typeof key === 'string' && key.startsWith(cleanPrefix));
  }

  async function stat(path) {
    const entry = await readEntry(path);
    if (!entry) return null;
    return {
      path: entry.path,
      size: entry.size,
      updated: entry.updated,
    };
  }

  async function exists(path) {
    return !!(await stat(path));
  }

  async function remove(path) {
    await init();
    const cleanPath = normalizePath(path);
    const tx = db.transaction(storeName, 'readwrite');
    const store = tx.objectStore(storeName);
    store.delete(cleanPath);
    await transactionDone(tx);
    return true;
  }

  async function mkdir() {
    return true;
  }

  return {
    init,
    readBlob,
    readText,
    writeBlob,
    list,
    stat,
    exists,
    remove,
    mkdir,
  };
}
