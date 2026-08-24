import { log } from '../../debug/index.js';
import { isIndexedDBAvailable } from '../quota.js';
import { DB_NAME, DB_VERSION, STORE_NAME } from '../download-types.js';
import { buildManifestVersionSet, normalizeSourceStats } from './integrity.js';

let database = null;

function isDatabaseClosingError(error) {
  const message = error?.message ?? '';
  return message.includes('database connection is closing')
    || error?.name === 'InvalidStateError';
}

export async function getDownloadStateDatabase() {
  if (database) return database;
  if (!isIndexedDBAvailable()) {
    log.warn('Downloader', 'IndexedDB unavailable, download resume will not work');
    return null;
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(new Error('Failed to open IndexedDB'));
    request.onsuccess = () => {
      database = request.result;
      resolve(database);
    };
    request.onupgradeneeded = (event) => {
      const openedDatabase = event.target.result;
      if (!openedDatabase.objectStoreNames.contains(STORE_NAME)) {
        const store = openedDatabase.createObjectStore(STORE_NAME, { keyPath: 'modelId' });
        store.createIndex('status', 'status', { unique: false });
      }
    };
  });
}

export async function saveDownloadState(state) {
  const openedDatabase = await getDownloadStateDatabase();
  if (!openedDatabase) return;

  try {
    await new Promise((resolve, reject) => {
      const tx = openedDatabase.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const request = store.put({
        ...state,
        completedShards: Array.from(state.completedShards),
      });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('Failed to save download state'));
    });
  } catch (error) {
    if (isDatabaseClosingError(error)) {
      database = null;
      log.warn('Downloader', 'IndexedDB unavailable, skipping download state save');
      return;
    }
    log.warn('Downloader', `Failed to save download state: ${error.message}`);
  }
}

export async function loadDownloadState(modelId) {
  const openedDatabase = await getDownloadStateDatabase();
  if (!openedDatabase) return null;

  try {
    return await new Promise((resolve, reject) => {
      const tx = openedDatabase.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const request = store.get(modelId);
      request.onsuccess = () => {
        const result = request.result;
        resolve(result
          ? {
            ...result,
            completedShards: new Set(result.completedShards),
            manifestVersionSet: typeof result.manifestVersionSet === 'string'
              ? result.manifestVersionSet
              : buildManifestVersionSet(result.manifest),
            sourceStats: normalizeSourceStats(result.sourceStats),
            lastSource: typeof result.lastSource === 'string' ? result.lastSource : null,
            lastSourcePath: typeof result.lastSourcePath === 'string' ? result.lastSourcePath : null,
          }
          : null);
      };
      request.onerror = () => reject(new Error('Failed to load download state'));
    });
  } catch (error) {
    if (isDatabaseClosingError(error)) {
      database = null;
      log.warn('Downloader', 'IndexedDB unavailable, skipping download state load');
      return null;
    }
    log.warn('Downloader', `Failed to load download state: ${error.message}`);
    return null;
  }
}

export async function deleteDownloadState(modelId) {
  const openedDatabase = await getDownloadStateDatabase();
  if (!openedDatabase) return;

  try {
    await new Promise((resolve, reject) => {
      const tx = openedDatabase.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const request = store.delete(modelId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('Failed to delete download state'));
    });
  } catch (error) {
    if (isDatabaseClosingError(error)) {
      database = null;
      log.warn('Downloader', 'IndexedDB unavailable, skipping download state delete');
      return;
    }
    log.warn('Downloader', `Failed to delete download state: ${error.message}`);
  }
}

export async function loadAllDownloadStates() {
  const openedDatabase = await getDownloadStateDatabase();
  if (!openedDatabase) return [];
  return new Promise((resolve, reject) => {
    const tx = openedDatabase.transaction(STORE_NAME, 'readonly');
    const request = tx.objectStore(STORE_NAME).getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(new Error('Failed to list downloads'));
  });
}
