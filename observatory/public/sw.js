// Offline worker for the front page. It keeps a copy of the files listed in
// offline-shell.json and answers from that copy only when the network fails,
// so a visitor who is online always gets the current files. Models are not
// kept here: the Doppler runtime saves those itself, in the browser's storage
// for this address.

const PREFIX = 'doppler-observatory-shell-';
const SHELL_URL = new URL('./offline-shell.json', self.location).href;

async function refreshShell() {
  const response = await fetch(SHELL_URL, { cache: 'no-store' });
  if (!response.ok) throw new Error(`offline-shell.json: HTTP ${response.status}`);
  const shell = await response.json();
  const name = `${PREFIX}${shell.version}`;
  if (!(await caches.has(name))) {
    // All or nothing: a copy that is missing files would fail later, offline.
    const cache = await caches.open(`${name}-filling`);
    await cache.addAll(shell.files.map((file) => new Request(new URL(file, self.location).href, { cache: 'no-store' })));
    const ready = await caches.open(name);
    for (const request of await cache.keys()) await ready.put(request, await cache.match(request));
    await caches.delete(`${name}-filling`);
  }
  for (const key of await caches.keys()) if (key.startsWith(PREFIX) && key !== name) await caches.delete(key);
}

self.addEventListener('install', (event) => { event.waitUntil(refreshShell().then(() => self.skipWaiting())); });
self.addEventListener('activate', (event) => { event.waitUntil(self.clients.claim()); });

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET' || new URL(request.url).origin !== self.location.origin) return;
  // Opening the page is the moment to notice that the listed files changed.
  if (request.mode === 'navigate') event.waitUntil(refreshShell().catch(() => {}));
  event.respondWith(fetch(request).catch(async () => {
    const kept = await caches.match(request, { ignoreSearch: request.mode === 'navigate' });
    if (kept) return kept;
    throw new TypeError(`Offline, and ${request.url} is not among the files kept for offline use.`);
  }));
});
