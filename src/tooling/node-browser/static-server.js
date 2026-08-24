import { once } from 'node:events';
import { createReadStream } from 'node:fs';
import { promises as fs } from 'node:fs';
import { createServer } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_HOST = '127.0.0.1';
const SERVER_HOSTS = Object.freeze(['127.0.0.1', 'localhost', '0.0.0.0']);
const MIME_BY_EXTENSION = Object.freeze({
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
  '.wgsl': 'text/plain; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.txt': 'text/plain; charset=utf-8',
});

function contentTypeFor(filePath) {
  return MIME_BY_EXTENSION[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
}

function resolveStaticPath(rootDir, requestPath) {
  let decodedPath = '/';
  try {
    decodedPath = decodeURIComponent(requestPath || '/');
  } catch {
    return null;
  }
  const candidate = path.resolve(rootDir, decodedPath.replace(/^\/+/, '') || 'index.html');
  const normalizedRoot = path.resolve(rootDir);
  if (candidate !== normalizedRoot && !candidate.startsWith(`${normalizedRoot}${path.sep}`)) {
    return null;
  }
  return candidate;
}

function normalizeStaticMounts(mounts = []) {
  if (!Array.isArray(mounts)) {
    throw new Error('browser command: staticMounts must be an array.');
  }
  return mounts.map((mount, index) => {
    if (!mount || typeof mount !== 'object' || Array.isArray(mount)) {
      throw new Error(`browser command: staticMounts[${index}] must be an object.`);
    }
    const urlPrefix = String(mount.urlPrefix || '').trim();
    const rootDir = String(mount.rootDir || '').trim();
    if (!urlPrefix.startsWith('/')) {
      throw new Error(`browser command: staticMounts[${index}].urlPrefix must start with "/".`);
    }
    if (!rootDir) {
      throw new Error(`browser command: staticMounts[${index}].rootDir is required.`);
    }
    return {
      urlPrefix: urlPrefix.replace(/\/+$/u, '') || '/',
      rootDir: path.resolve(rootDir),
    };
  });
}

function findStaticRootForRequest(rootDir, mounts, requestPath) {
  const normalizedPath = String(requestPath || '/');
  let bestMount = null;
  for (const mount of mounts) {
    const prefix = mount.urlPrefix;
    if (normalizedPath !== prefix && !normalizedPath.startsWith(`${prefix}/`)) continue;
    if (!bestMount || prefix.length > bestMount.urlPrefix.length) bestMount = mount;
  }
  if (!bestMount) {
    return { effectiveRootDir: rootDir, effectivePath: normalizedPath };
  }
  const relativePath = normalizedPath.slice(bestMount.urlPrefix.length) || '/';
  return {
    effectiveRootDir: bestMount.rootDir,
    effectivePath: relativePath.startsWith('/') ? relativePath : `/${relativePath}`,
  };
}

async function resolveFileForRequest(rootDir, mounts, requestPath) {
  const { effectiveRootDir, effectivePath } = findStaticRootForRequest(rootDir, mounts, requestPath);
  const resolved = resolveStaticPath(effectiveRootDir, effectivePath);
  if (!resolved) return null;
  let stats;
  try {
    stats = await fs.stat(resolved);
  } catch {
    return null;
  }
  if (stats.isDirectory()) {
    const indexPath = path.join(resolved, 'index.html');
    try {
      const indexStats = await fs.stat(indexPath);
      return indexStats.isFile() ? { filePath: indexPath, size: indexStats.size } : null;
    } catch {
      return null;
    }
  }
  return stats.isFile() ? { filePath: resolved, size: stats.size } : null;
}

export async function createStaticFileServer(options = {}) {
  const rootDir = path.resolve(options.rootDir || fileURLToPath(new URL('../../..', import.meta.url)));
  const staticMounts = normalizeStaticMounts(options.staticMounts || []);
  const host = String(options.host || DEFAULT_HOST);
  const port = Number.isFinite(options.port) ? Math.max(0, Math.floor(options.port)) : 0;
  const server = createServer(async (req, res) => {
    const method = req.method || 'GET';
    if (method !== 'GET' && method !== 'HEAD') {
      res.statusCode = 405;
      res.end('Method Not Allowed');
      return;
    }
    let pathname = '/';
    try {
      pathname = new URL(req.url || '/', `http://${req.headers.host || host}`).pathname || '/';
    } catch {
      res.statusCode = 400;
      res.end('Bad Request');
      return;
    }
    const resolved = await resolveFileForRequest(rootDir, staticMounts, pathname);
    if (!resolved) {
      res.statusCode = 404;
      res.end('File not found');
      return;
    }
    res.statusCode = 200;
    res.setHeader('Content-Type', contentTypeFor(resolved.filePath));
    res.setHeader('Content-Length', resolved.size);
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    if (method === 'HEAD') {
      res.end();
      return;
    }
    const stream = createReadStream(resolved.filePath, {
      highWaterMark: resolved.size > 1024 * 1024 ? 1024 * 1024 : undefined,
    });
    stream.on('error', () => {
      if (!res.headersSent) res.statusCode = 500;
      res.end();
    });
    stream.pipe(res);
  });

  const tryListen = (listenHost) => new Promise((resolve, reject) => {
    const listener = listenHost == null ? server.listen(port) : server.listen(port, listenHost);
    listener.once('error', reject);
    listener.once('listening', () => resolve(listener));
  });
  const tryHosts = options.host == null ? [...SERVER_HOSTS, null] : [host];
  let lastError = null;
  for (const listenHost of tryHosts) {
    try {
      await tryListen(listenHost);
      break;
    } catch (error) {
      lastError = error;
      if (!['EACCES', 'EADDRINUSE', 'EPERM'].includes(error?.code)) throw error;
      server.close();
    }
  }
  if (lastError) throw lastError;

  const address = server.address();
  if (!address || typeof address !== 'object') {
    server.close();
    throw new Error('browser command: failed to resolve static server address.');
  }
  const resolvedHost = typeof address.address === 'string' ? address.address : DEFAULT_HOST;
  const effectiveHost = resolvedHost === '::' || resolvedHost === '0.0.0.0' ? DEFAULT_HOST : resolvedHost;
  return {
    baseUrl: `http://${effectiveHost}:${address.port}`,
    async close() {
      server.close();
      await once(server, 'close');
    },
  };
}
