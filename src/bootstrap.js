import { log } from './debug/index.js';
import { loadVfsManifest, seedVfsFromManifest } from './boot/vfs-bootstrap.js';

const SW_URL = '/sw-module-loader.js';
const MANIFEST_URL = '/config/vfs-manifest.json';
const APP_ENTRY_URL = '/app/app.js';
const RELOAD_KEY = 'doppler_vfs_reload';

function parseFlags() {
  const params = new URLSearchParams(globalThis.location?.search || '');
  return {
    disableVfs: params.get('vfs') === '0',
    preserve: params.get('vfsPreserve') !== '0',
  };
}

function shouldReload() {
  try {
    if (!globalThis.sessionStorage) return false;
    if (sessionStorage.getItem(RELOAD_KEY)) return false;
    sessionStorage.setItem(RELOAD_KEY, '1');
    return true;
  } catch {
    return false;
  }
}

function waitForController(timeoutMs) {
  if (navigator.serviceWorker.controller) return Promise.resolve(true);

  return new Promise((resolve) => {
    let done = false;
    const timer = setTimeout(() => {
      if (done) return;
      done = true;
      resolve(false);
    }, timeoutMs);

    function onChange() {
      if (done) return;
      done = true;
      clearTimeout(timer);
      navigator.serviceWorker.removeEventListener('controllerchange', onChange);
      resolve(true);
    }

    navigator.serviceWorker.addEventListener('controllerchange', onChange);
  });
}

async function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) {
    log.warn('VFS', 'Service worker unavailable; skipping VFS bootstrap.');
    return false;
  }
  if (!globalThis.isSecureContext) {
    log.warn('VFS', 'Service worker requires a secure context; skipping VFS bootstrap.');
    return false;
  }

  await navigator.serviceWorker.register(SW_URL, { type: 'module', scope: '/' });

  const controlled = await waitForController(3000);
  if (!controlled && shouldReload()) {
    log.info('VFS', 'Reloading once to activate service worker control.');
    globalThis.location?.reload();
    return false;
  }

  if (!controlled) {
    log.warn('VFS', 'Service worker not controlling page; continuing without VFS.');
    return false;
  }

  return true;
}

async function hydrateVfs(preserve) {
  const manifest = await loadVfsManifest(MANIFEST_URL);
  return seedVfsFromManifest(manifest, { preserve });
}

async function main() {
  const flags = parseFlags();

  if (!flags.disableVfs) {
    try {
      const controlled = await registerServiceWorker();
      if (controlled) {
        await hydrateVfs(flags.preserve);
      }
    } catch (err) {
      log.warn('VFS', `Bootstrap failed: ${(err).message}`);
    }
  }

  await import(APP_ENTRY_URL);
}

main().catch((err) => {
  log.error('VFS', `Bootstrap crashed: ${(err).message}`);
});
