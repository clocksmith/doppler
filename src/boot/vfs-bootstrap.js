import { log } from '../debug/index.js';

const DB_NAME = 'doppler-vfs-v0';
const STORE_NAME = 'files';
const DB_VERSION = 1;
const DEFAULT_TIMEOUT_MS = 4000;

const CONTENT_TYPES = {
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
};

function guessContentType(path) {
  for (const [ext, type] of Object.entries(CONTENT_TYPES)) {
    if (path.endsWith(ext)) return type;
  }
  return 'application/octet-stream';
}

function normalizePath(value) {
  if (!value) return null;
  if (value.startsWith('http://') || value.startsWith('https://')) return value;
  return value.startsWith('/') ? value : `/${value}`;
}

function withTimeout(promise, timeoutMs, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    promise
      .then((result) => {
        clearTimeout(timer);
        resolve(result);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

function openVfsDb({ timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  if (typeof indexedDB === 'undefined') {
    return Promise.reject(new Error('IndexedDB unavailable for VFS.'));
  }

  const openPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'path' });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Failed to open VFS database.'));
    request.onblocked = () => reject(new Error('VFS database open blocked by another connection.'));
  });

  return withTimeout(openPromise, timeoutMs, 'VFS database open');
}

function readEntry(db, path) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.get(path);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error || new Error(`Failed to read VFS entry: ${path}`));
  });
}

function writeEntry(db, entry) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.put(entry);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error || new Error(`Failed to write VFS entry: ${entry.path}`));
  });
}

export async function loadVfsManifest(manifestUrl) {
  const url = normalizePath(manifestUrl);
  if (!url) {
    throw new Error('VFS manifest url is required.');
  }

  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`VFS manifest fetch failed (${response.status}): ${url}`);
  }

  const manifest = await response.json();
  if (!manifest || typeof manifest !== 'object' || !Array.isArray(manifest.files)) {
    throw new Error('VFS manifest is missing files list.');
  }

  return manifest;
}

export async function seedVfsFromManifest(manifest, options = {}) {
  const files = Array.isArray(manifest?.files) ? manifest.files : null;
  if (!files) {
    throw new Error('VFS manifest files must be an array.');
  }

  const {
    preserve = true,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    onProgress = null,
  } = options;

  const db = await openVfsDb({ timeoutMs });

  let seeded = 0;
  let skipped = 0;
  const total = files.length;

  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    const path = normalizePath(file?.path || file?.url);
    if (!path) {
      throw new Error('VFS manifest entry missing path.');
    }

    if (preserve) {
      const existing = await readEntry(db, path);
      if (existing) {
        skipped += 1;
        if (onProgress) {
          onProgress({ path, index, total, status: 'skipped' });
        }
        continue;
      }
    }

    const url = normalizePath(file?.url || path);
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`VFS fetch failed (${response.status}): ${url}`);
    }

    const body = await response.arrayBuffer();
    const contentType = file?.contentType || response.headers.get('Content-Type') || guessContentType(path);
    await writeEntry(db, {
      path,
      body,
      contentType,
      size: body.byteLength,
      updatedAt: Date.now(),
    });
    seeded += 1;

    if (onProgress) {
      onProgress({ path, index, total, status: 'seeded' });
    }
  }

  log.info('VFS', `Seeded ${seeded} files, skipped ${skipped} (${total} total).`);

  return { total, seeded, skipped };
}
