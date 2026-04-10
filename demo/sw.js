const CACHE_NAME = 'doppler-demo-shell-v2';
const APP_SHELL = [
  '/demo/index.html',
  '/demo/pwa-manifest.json',
  '/demo/favicon.svg',
  '/demo/styles/rd.css',
  '/demo/styles/rd-tokens.css',
  '/demo/styles/rd-primitives.css',
  '/demo/styles/rd-components.css',
  '/demo/ui/styles/app.css',
  '/demo/ui/token-press/styles.css',
  '/demo/ui/xray/styles.css',
  '/demo/demo.js',
  '/demo/boot.js',
  '/demo/core.js',
  '/demo/input.js',
  '/demo/models.js',
  '/demo/output.js',
  '/demo/pwa.js',
  '/demo/report.js',
  '/demo/settings.js',
  '/demo/examples.json',
  '/demo/assets/pwa/icon-192.png',
  '/demo/assets/pwa/icon-512.png',
  '/demo/assets/pwa/icon-maskable-512.png',
  '/demo/assets/pwa/shortcut-new-96.png',
  '/demo/assets/pwa/shortcut-xray-96.png',
  '/demo/assets/pwa/screenshot-desktop.png',
];

function isDemoAsset(url) {
  return url.origin === self.location.origin
    && (url.pathname.startsWith('/demo/') || url.pathname.startsWith('/src/'));
}

async function cacheShell() {
  const cache = await caches.open(CACHE_NAME);
  await cache.addAll(APP_SHELL);
}

async function networkFirstNavigation(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put('/demo/index.html', response.clone());
    }
    return response;
  } catch {
    return (await cache.match('/demo/index.html')) ?? Response.error();
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  const networkFetch = fetch(request)
    .then((response) => {
      if (response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => null);
  if (cached) {
    return cached;
  }
  return (await networkFetch) ?? Response.error();
}

self.addEventListener('install', (event) => {
  event.waitUntil(cacheShell().then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((key) => key !== CACHE_NAME)
        .map((key) => caches.delete(key))
    );
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') {
    return;
  }
  const url = new URL(request.url);
  if (!isDemoAsset(url)) {
    return;
  }
  if (request.mode === 'navigate') {
    event.respondWith(networkFirstNavigation(request));
    return;
  }
  event.respondWith(staleWhileRevalidate(request));
});
