function openCheckpointDB(options = {}) {
  const {
    dbName = 'doppler-training',
    storeName = 'checkpoints',
    version = 1,
  } = options;

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, version);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(storeName)) {
        db.createObjectStore(storeName);
      }
    };

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve({ db: request.result, storeName });
  });
}

export async function saveCheckpoint(key, data, options = {}) {
  const { db, storeName } = await openCheckpointDB(options);
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    const store = tx.objectStore(storeName);
    store.put(data, key);
  });
}

export async function loadCheckpoint(key, options = {}) {
  const { db, storeName } = await openCheckpointDB(options);
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    tx.onerror = () => reject(tx.error);
    const store = tx.objectStore(storeName);
    const request = store.get(key);
    request.onsuccess = () => resolve(request.result ?? null);
    request.onerror = () => reject(request.error);
  });
}
