import { log } from './debug/index.js';

const DB_NAME = 'doppler-vfs-v0';
const STORE_NAME = 'files';
const DB_VERSION = 1;
const DB_TIMEOUT_MS = 2000;

const EXTENSIONS = new Set(['.js', '.json', '.wgsl']);
const BYPASS_PATHS = new Set([
  '/bootstrap.js',
  '/sw-module-loader.js',
  '/config/vfs-manifest.json',
]);
const BYPASS_PREFIXES = [
  '/kernel-tests/',
  '/tests/',
  '/debug/',
];

const CONTENT_TYPES = {
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
};

let dbPromise = null;

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

function guessContentType(path) {
  for (const [ext, type] of Object.entries(CONTENT_TYPES)) {
    if (path.endsWith(ext)) return type;
  }
  return 'application/octet-stream';
}

function shouldHandlePath(path) {
  if (BYPASS_PATHS.has(path)) return false;
  for (const prefix of BYPASS_PREFIXES) {
    if (path.startsWith(prefix)) return false;
  }

  for (const ext of EXTENSIONS) {
    if (path.endsWith(ext)) return true;
  }
  return false;
}

function openDb() {
  if (dbPromise) return dbPromise;

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

  dbPromise = withTimeout(openPromise, DB_TIMEOUT_MS, 'VFS database open');
  return dbPromise;
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

async function respondFromVfs(path) {
  try {
    const db = await openDb();
    const entry = await readEntry(db, path);
    if (!entry) return null;

    const headers = new Headers();
    headers.set('Content-Type', entry.contentType || guessContentType(path));
    headers.set('Cache-Control', 'no-store');
    headers.set('X-VFS-Source', 'vfs');

    return new Response(entry.body, {
      status: 200,
      headers,
    });
  } catch (err) {
    log.warn('VFS', `VFS read failed: ${(err).message}`);
    return null;
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  const path = url.pathname;
  if (!shouldHandlePath(path)) return;

  event.respondWith((async () => {
    const response = await respondFromVfs(path);
    if (response) return response;
    return fetch(request, { cache: 'no-store' });
  })());
});
