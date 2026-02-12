import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { once } from 'node:events';
import {
  ensureCommandSupportedOnSurface,
  normalizeToolingCommandRequest,
} from './command-api.js';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_RUNNER_PATH = '/src/tooling/command-runner.html';
const DEFAULT_TIMEOUT_MS = 180_000;

const MIME_BY_EXTENSION = Object.freeze({
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
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
  const ext = path.extname(filePath).toLowerCase();
  return MIME_BY_EXTENSION[ext] || 'application/octet-stream';
}

function resolveStaticPath(rootDir, requestPath) {
  const normalizedPath = decodeURIComponent(requestPath || '/').replace(/^\/+/, '');
  const candidate = path.resolve(rootDir, normalizedPath || 'index.html');
  const normalizedRoot = path.resolve(rootDir);
  if (candidate !== normalizedRoot && !candidate.startsWith(`${normalizedRoot}${path.sep}`)) {
    return null;
  }
  return candidate;
}

async function resolveFileForRequest(rootDir, requestPath) {
  const resolved = resolveStaticPath(rootDir, requestPath);
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
      if (indexStats.isFile()) {
        return indexPath;
      }
    } catch {
      return null;
    }
    return null;
  }

  if (!stats.isFile()) return null;
  return resolved;
}

async function createStaticFileServer(options = {}) {
  const rootDir = path.resolve(
    options.rootDir || fileURLToPath(new URL('../../', import.meta.url))
  );
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
      const url = new URL(req.url || '/', `http://${req.headers.host || host}`);
      pathname = url.pathname || '/';
    } catch {
      res.statusCode = 400;
      res.end('Bad Request');
      return;
    }

    const filePath = await resolveFileForRequest(rootDir, pathname);
    if (!filePath) {
      res.statusCode = 404;
      res.end('File not found');
      return;
    }

    res.statusCode = 200;
    res.setHeader('Content-Type', contentTypeFor(filePath));
    if (method === 'HEAD') {
      res.end();
      return;
    }

    const stream = createReadStream(filePath);
    stream.on('error', () => {
      if (!res.headersSent) {
        res.statusCode = 500;
      }
      res.end();
    });
    stream.pipe(res);
  });

  server.listen(port, host);
  await once(server, 'listening');

  const address = server.address();
  if (!address || typeof address !== 'object') {
    server.close();
    throw new Error('browser command: failed to resolve static server address.');
  }

  const close = async () => {
    server.close();
    await once(server, 'close');
  };

  return {
    baseUrl: `http://${host}:${address.port}`,
    close,
  };
}

function normalizeHeadless(value) {
  if (value === undefined || value === null) return true;
  if (typeof value === 'boolean') {
    if (!value) {
      throw new Error('browser command: headed mode is not supported; headless must be true.');
    }
    return true;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true') return true;
    if (normalized === 'false') {
      throw new Error('browser command: headed mode is not supported; headless must be true.');
    }
  }
  throw new Error('browser command: headless must be true or false.');
}

function normalizeTimeoutMs(value) {
  if (value === undefined || value === null) return DEFAULT_TIMEOUT_MS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error('browser command: timeoutMs must be a positive number.');
  }
  return Math.floor(parsed);
}

function normalizeRunnerPath(value) {
  const raw = String(value || DEFAULT_RUNNER_PATH).trim();
  if (!raw.startsWith('/')) {
    return `/${raw}`;
  }
  return raw;
}

function normalizeBaseUrl(value) {
  if (value === undefined || value === null || value === '') {
    return null;
  }
  const raw = String(value).trim();
  if (!raw) return null;
  try {
    const url = new URL(raw);
    return url.toString().replace(/\/$/, '');
  } catch {
    throw new Error('browser command: baseUrl must be an absolute URL, for example http://127.0.0.1:8080');
  }
}

function browserLaunchArgs() {
  return [
    '--enable-unsafe-webgpu',
    '--enable-webgpu-developer-features',
    '--disable-dawn-features=disallow_unsafe_apis',
  ];
}

async function launchBrowser(chromium, launchOptions) {
  try {
    return await chromium.launch(launchOptions);
  } catch (error) {
    const message = error?.message || String(error);
    throw new Error(
      `browser command: failed to launch browser (${message}). Install Playwright browsers (npx playwright install) or pass --browser-channel chrome / --browser-executable.`
    );
  }
}

export async function runBrowserCommandInNode(commandRequest, options = {}) {
  const { request } = ensureCommandSupportedOnSurface(commandRequest, 'browser');

  if (request.keepPipeline) {
    throw new Error(
      'browser command relay does not support keepPipeline=true because pipeline objects are not serializable across process boundaries.'
    );
  }

  if (request.command === 'convert') {
    throw new Error('browser command relay does not support convert. Use --surface node for convert commands.');
  }

  const { chromium } = await import('playwright');
  const baseUrl = normalizeBaseUrl(options.baseUrl);
  const server = baseUrl
    ? null
    : await createStaticFileServer({
      rootDir: options.staticRootDir,
      host: options.host,
      port: options.port,
    }).catch((error) => {
      const message = error?.message || String(error);
      throw new Error(
        `browser command: failed to start static server (${message}). Pass --browser-base-url to reuse an existing server.`
      );
    });

  const launchOptions = {
    headless: normalizeHeadless(options.headless),
    args: browserLaunchArgs(),
  };

  if (options.channel) {
    launchOptions.channel = String(options.channel);
  }
  if (options.executablePath) {
    launchOptions.executablePath = String(options.executablePath);
  }

  const timeoutMs = normalizeTimeoutMs(options.timeoutMs);
  const runnerPath = normalizeRunnerPath(options.runnerPath);
  const resolvedBaseUrl = baseUrl || server.baseUrl;

  let browser = null;
  let context = null;
  try {
    browser = await launchBrowser(chromium, launchOptions);
    context = await browser.newContext();
    const page = await context.newPage();
    page.setDefaultTimeout(timeoutMs);
    const pageDiagnostics = [];

    if (typeof options.onConsole === 'function') {
      page.on('console', (message) => {
        options.onConsole({
          type: message.type(),
          text: message.text(),
        });
      });
    }

    page.on('pageerror', (error) => {
      pageDiagnostics.push(`pageerror: ${error?.message || String(error)}`);
    });
    page.on('requestfailed', (request) => {
      const failure = request.failure();
      pageDiagnostics.push(
        `requestfailed: ${request.url()} (${failure?.errorText || 'unknown error'})`
      );
    });

    await page.goto(`${resolvedBaseUrl}${runnerPath}`, { waitUntil: 'load' });
    try {
      await page.waitForFunction(() => window.__dopplerRunnerReady === true, null, {
        timeout: timeoutMs,
      });
    } catch (error) {
      const diagnostics = pageDiagnostics.length
        ? pageDiagnostics.slice(0, 10).join(' | ')
        : 'no page diagnostics captured';
      throw new Error(
        `browser command: runner did not become ready within ${timeoutMs}ms (${diagnostics}).`
      );
    }

    const response = await page.evaluate(async (payload) => {
      if (typeof window.__dopplerRunBrowserCommand !== 'function') {
        throw new Error('browser command runner is missing window.__dopplerRunBrowserCommand');
      }
      return window.__dopplerRunBrowserCommand(payload.request, payload.options || {});
    }, {
      request,
      options: {
        runtimeLoadOptions: options.runtimeLoadOptions || {},
      },
    });

    return response;
  } finally {
    if (context) {
      await context.close().catch(() => {});
    }
    if (browser) {
      await browser.close().catch(() => {});
    }
    if (server) {
      await server.close().catch(() => {});
    }
  }
}

export function normalizeNodeBrowserCommand(commandRequest) {
  return normalizeToolingCommandRequest(commandRequest);
}
