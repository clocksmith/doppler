import { log } from './debug/index.js';

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

// No SW/VFS bootstrap in package+demo mode. Host app owns that integration.
const APP_ENTRY_URL = withBase('/demo/app.js');

async function main() {
  await import(APP_ENTRY_URL);
}

main().catch((err) => {
  log.error('Bootstrap', `Bootstrap crashed: ${err?.message || String(err)}`);
});
