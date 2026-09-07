importScripts('./application-assets.js');
const { cacheName, assets } = self.DOCUMENT_SEARCH_ASSETS;
const byUrl = new Map(assets.map(asset => [new URL(asset.path, self.registration.scope).href, asset]));
async function checked(response, asset) {
  if (!response?.ok) throw new Error(`Application asset unavailable: ${asset.path}`);
  const bytes = await response.clone().arrayBuffer();
  const actual = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), value => value.toString(16).padStart(2, '0')).join('');
  if (actual !== asset.sha256) throw new Error(`Application asset integrity failed: ${asset.path}`);
  return response;
}
self.addEventListener('install', event => event.waitUntil((async () => {
  const cache = await caches.open(cacheName);
  for (const [url, asset] of byUrl) {
    const existing = await cache.match(url);
    if (existing) { await checked(existing, asset); continue; }
    await cache.put(url, await checked(await fetch(url, { cache: 'no-store' }), asset));
  }
  await self.skipWaiting();
})()));
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  const asset = byUrl.get(event.request.url);
  if (asset && event.request.method === 'GET') event.respondWith((async () => {
    const response = await (await caches.open(cacheName)).match(event.request.url);
    return checked(response, asset);
  })());
});
