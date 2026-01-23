import { log } from './debug/index.js';
import { loadVfsManifest, seedVfsFromManifest } from './boot/vfs-bootstrap.js';

const RELOAD_KEY = 'doppler_vfs_reload';

function getBasePath() {
  if (typeof location === 'undefined') return '';
  const path = location.pathname || '';
  if (path === '/doppler' || path.startsWith('/doppler/')) return '/doppler';
  return '';
}

const BASE_PATH = getBasePath();

function withBase(path) {
  if (!path) return path;
  if (!BASE_PATH) return path;
  if (/^https?:\/\//.test(path)) return path;
  const normalized = path.startsWith('/') ? path : `/${path}`;
  if (normalized.startsWith(`${BASE_PATH}/`)) return normalized;
  return `${BASE_PATH}${normalized}`;
}

function applyBaseToManifest(manifest) {
  if (!BASE_PATH) return manifest;
  const files = (manifest.files || []).map((file) => {
    const next = { ...file };
    if (next.path && !/^https?:\/\//.test(next.path)) {
      next.path = withBase(next.path);
    }
    if (next.url && !/^https?:\/\//.test(next.url)) {
      next.url = withBase(next.url);
    }
    return next;
  });
  return { ...manifest, files };
}

const SW_URL = withBase('/sw.js');
const MANIFEST_URL = withBase('/config/vfs-manifest.json');
const APP_ENTRY_URL = withBase('/app/app.js');

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

  const scope = BASE_PATH ? `${BASE_PATH}/` : '/';
  await navigator.serviceWorker.register(SW_URL, { type: 'module', scope });

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
  const scopedManifest = applyBaseToManifest(manifest);
  return seedVfsFromManifest(scopedManifest, { preserve });
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
