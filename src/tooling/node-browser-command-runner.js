import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { once } from 'node:events';
import {
  ensureCommandSupportedOnSurface,
  normalizeToolingCommandRequest,
} from './command-api.js';
import { normalizeToToolingCommandError } from './command-envelope.js';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_RUNNER_PATH = '/src/tooling/command-runner.html';
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_OPFS_CACHE_DIR = path.join(os.homedir(), '.cache', 'doppler', 'chromium-profile');
const DEFAULT_OPFS_CACHE_PORT = 19836;
const SERVER_HOSTS = Object.freeze(['127.0.0.1', 'localhost', '0.0.0.0']);
const DEFAULT_CHANNEL_ORDER = Object.freeze({
  darwin: ['chrome', 'chromium'],
  linux: ['chromium', 'chrome'],
  win32: ['chromium', 'chrome'],
});
const PERSISTENT_LAUNCH_ERROR_HINTS = Object.freeze([
  'Target page, context or browser has been closed',
  'bootstrap_check_in',
  'Permission denied',
  'org.chromium.Chromium.MachPortRendezvousServer',
]);

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
  const ext = path.extname(filePath).toLowerCase();
  return MIME_BY_EXTENSION[ext] || 'application/octet-stream';
}

function resolveStaticPath(rootDir, requestPath) {
  let decodedPath = '/';
  try {
    decodedPath = decodeURIComponent(requestPath || '/');
  } catch {
    return null;
  }
  const normalizedPath = decodedPath.replace(/^\/+/, '');
  const candidate = path.resolve(rootDir, normalizedPath || 'index.html');
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
    if (normalizedPath !== prefix && !normalizedPath.startsWith(`${prefix}/`)) {
      continue;
    }
    if (!bestMount || prefix.length > bestMount.urlPrefix.length) {
      bestMount = mount;
    }
  }

  if (!bestMount) {
    return {
      effectiveRootDir: rootDir,
      effectivePath: normalizedPath,
    };
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
      if (indexStats.isFile()) {
        return { filePath: indexPath, size: indexStats.size };
      }
    } catch {
      return null;
    }
    return null;
  }

  if (!stats.isFile()) return null;
  return { filePath: resolved, size: stats.size };
}

async function createStaticFileServer(options = {}) {
  const rootDir = path.resolve(
    options.rootDir || fileURLToPath(new URL('../../', import.meta.url))
  );
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
      const url = new URL(req.url || '/', `http://${req.headers.host || host}`);
      pathname = url.pathname || '/';
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
      if (!res.headersSent) {
        res.statusCode = 500;
      }
      res.end();
    });
    stream.pipe(res);
  });

  const tryListen = (listenHost) => new Promise((resolve, reject) => {
    const listener = listenHost == null ? server.listen(port) : server.listen(port, listenHost);
    listener.once('error', (error) => {
      reject(error);
    });
    listener.once('listening', () => {
      resolve(listener);
    });
  });

  const tryHosts = options.host == null ? [...SERVER_HOSTS, null] : [host];
  let lastError = null;
  for (const listenHost of tryHosts) {
    try {
      await tryListen(listenHost);
      break;
    } catch (error) {
      lastError = error;
      if (error?.code !== 'EACCES' && error?.code !== 'EADDRINUSE' && error?.code !== 'EPERM') {
        throw error;
      }
      server.close();
    }
  }

  if (lastError) {
    throw lastError;
  }

  const address = server.address();
  if (!address || typeof address !== 'object') {
    server.close();
    throw new Error('browser command: failed to resolve static server address.');
  }

  const resolvedHost = typeof address.address === 'string' ? address.address : DEFAULT_HOST;
  const effectiveHost = resolvedHost === '::' || resolvedHost === '0.0.0.0' ? DEFAULT_HOST : resolvedHost;

  const close = async () => {
    server.close();
    await once(server, 'close');
  };

  return {
    baseUrl: `http://${effectiveHost}:${address.port}`,
    close,
  };
}

function normalizeHeadless(value) {
  if (value === undefined || value === null) return true;
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true') return true;
    if (normalized === 'false') return false;
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

function formatLaunchErrorMessage(error) {
  if (error == null) return '';
  if (typeof error.message === 'string' && error.message.trim().length > 0) {
    return error.message;
  }
  return String(error);
}

function isRecoverablePersistentLaunchError(error) {
  const message = formatLaunchErrorMessage(error);
  return PERSISTENT_LAUNCH_ERROR_HINTS.some((hint) => message.includes(hint));
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

function normalizeBrowserArgs(value) {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) {
    throw new Error('browser command: browserArgs must be an array.');
  }

  return value.map((arg) => {
    if (arg === undefined || arg === null) {
      throw new Error('browser command: --browser-arg values must be strings.');
    }
    if (typeof arg !== 'string') {
      throw new Error('browser command: --browser-arg values must be strings.');
    }
    return arg.trim();
  }).filter((arg) => arg.length > 0);
}

const DEFAULT_WEBGPU_BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-dawn-features=disallow_unsafe_apis',
  '--ignore-gpu-blocklist',
]);
const CRASH_RECOVERY_BROWSER_ARGS = Object.freeze([
  '--disable-breakpad',
  '--disable-gpu-sandbox',
  '--no-sandbox',
]);

const PLATFORM_WEBGPU_ARGS = Object.freeze({
  darwin: Object.freeze(['--use-angle=metal']),
  linux: Object.freeze([
    '--use-angle=vulkan',
    '--enable-features=Vulkan',
    '--disable-vulkan-surface',
  ]),
  win32: Object.freeze([]),
});

function uniqueArgs(args) {
  return [...new Set(args)];
}

function asNonEmptyString(value) {
  if (value == null) return null;
  const normalized = String(value).trim();
  return normalized === '' ? null : normalized;
}

function normalizeWebgpuBackend(value) {
  const raw = asNonEmptyString(value);
  if (!raw) return null;
  const normalized = raw.toLowerCase();
  if (normalized.includes('metal')) return 'metal';
  if (normalized.includes('vulkan')) return 'vulkan';
  if (normalized.includes('d3d12')) return 'd3d12';
  if (normalized.includes('d3d11')) return 'd3d11';
  if (normalized.includes('opengl') || normalized === 'gl') return 'opengl';
  if (normalized.includes('swiftshader')) return 'swiftshader';
  return normalized;
}

function readFlagValue(args, flagName) {
  if (!Array.isArray(args)) return null;
  for (let i = 0; i < args.length; i += 1) {
    const token = String(args[i] ?? '');
    if (token === flagName) {
      return asNonEmptyString(args[i + 1]);
    }
    if (token.startsWith(`${flagName}=`)) {
      return asNonEmptyString(token.slice(flagName.length + 1));
    }
  }
  return null;
}

function inferWebgpuBackendFromArgs(args, hostPlatform) {
  const explicit = normalizeWebgpuBackend(readFlagValue(args, '--use-angle'));
  if (explicit) return explicit;
  const normalizedArgs = Array.isArray(args)
    ? args.map((value) => String(value ?? '').toLowerCase())
    : [];
  if (normalizedArgs.some((value) => value.includes('vulkan'))) return 'vulkan';
  if (normalizedArgs.some((value) => value.includes('metal'))) return 'metal';
  if (normalizedArgs.some((value) => value.includes('d3d12'))) return 'd3d12';
  if (normalizedArgs.some((value) => value.includes('d3d11'))) return 'd3d11';
  if (hostPlatform === 'darwin') return 'metal';
  if (hostPlatform === 'linux') return 'vulkan';
  if (hostPlatform === 'win32') return 'd3d12';
  return null;
}

function withCrashRecoveryArgs(args = []) {
  return uniqueArgs([...args, ...CRASH_RECOVERY_BROWSER_ARGS]);
}

function hasCrashRecoveryArgs(args = []) {
  const argSet = new Set(args);
  return CRASH_RECOVERY_BROWSER_ARGS.every((arg) => argSet.has(arg));
}

function browserLaunchArgs(extraArgs = []) {
  const platformArgs = PLATFORM_WEBGPU_ARGS[process.platform] ?? [];
  return uniqueArgs([...DEFAULT_WEBGPU_BROWSER_ARGS, ...platformArgs, ...extraArgs]);
}

function resolveDefaultChannels() {
  return DEFAULT_CHANNEL_ORDER[process.platform] ?? DEFAULT_CHANNEL_ORDER.linux;
}

async function launchBrowser(chromium, launchOptions, options = {}) {
  const explicitChannel = options.explicitChannel ?? false;
  const explicitExecutablePath = options.explicitExecutablePath ?? false;
  if (explicitChannel || explicitExecutablePath) {
    try {
      return await chromium.launch(launchOptions);
    } catch (error) {
      const message = error?.message || String(error);
      throw new Error(
        `browser command: failed to launch browser (${message}). Install Playwright browsers (npx playwright install) or pass --browser-channel chrome / --browser-executable.`
      );
    }
  }

  const tryLaunch = async (candidateLaunchOptions) => {
    const launchCandidateErrors = [];
    for (const channel of resolveDefaultChannels()) {
      try {
        return await chromium.launch({ ...candidateLaunchOptions, channel });
      } catch (error) {
        const message = error?.message || String(error);
        launchCandidateErrors.push(`${channel}: ${message}`);
      }
    }

    try {
      return await chromium.launch(candidateLaunchOptions);
    } catch (error) {
      const message = error?.message || String(error);
      const allErrors = launchCandidateErrors.length > 0
        ? `${message} | channel errors: ${launchCandidateErrors.join(' | ')}`
        : message;
      throw new Error(
        `browser command: failed to launch browser (${allErrors}). ` +
        `Tried default channels: ${resolveDefaultChannels().join(', ')}. ` +
        `Install Playwright browsers (npx playwright install) or pass --browser-channel / --browser-executable.`
      );
    }
  };

  const launchErrors = [];
  const attemptConfigs = hasCrashRecoveryArgs(launchOptions.args || [])
    ? [launchOptions]
    : [
      launchOptions,
      { ...launchOptions, args: withCrashRecoveryArgs(launchOptions.args || []) },
    ];

  for (const candidateLaunchOptions of attemptConfigs) {
    try {
      return await tryLaunch(candidateLaunchOptions);
    } catch (error) {
      const message = error?.message || String(error);
      launchErrors.push(message);

      if (isRecoverablePersistentLaunchError(error) && attemptConfigs.length === 2) {
        continue;
      }

      if (!isRecoverablePersistentLaunchError(error) || launchErrors.length >= 2) {
        throw error;
      }
    }
  }

  const retryMessage = launchErrors.join(' | ');
  if (isRecoverablePersistentLaunchError(retryMessage)) {
    throw new Error(
      `browser command: failed to launch browser with crash recovery enabled (${retryMessage}). ` +
      `Install Playwright browsers (npx playwright install) or pass --browser-channel / --browser-executable.`
    );
  }

  throw new Error(
    `browser command: failed to launch browser (${retryMessage}). ` +
    `Tried default channels: ${resolveDefaultChannels().join(', ')}. ` +
    `Install Playwright browsers (npx playwright install) or pass --browser-channel / --browser-executable.`
  );
}

async function launchPersistentBrowser(chromium, userDataDir, launchOptions, options = {}) {
  await fs.mkdir(userDataDir, { recursive: true });

  const explicitChannel = options.explicitChannel ?? false;
  const explicitExecutablePath = options.explicitExecutablePath ?? false;

  // launchPersistentContext returns a BrowserContext directly (no separate Browser object).
  const persistentOpts = { ...launchOptions };

  if (explicitChannel || explicitExecutablePath) {
    try {
      return await chromium.launchPersistentContext(userDataDir, persistentOpts);
    } catch (error) {
      const message = error?.message || String(error);
      throw new Error(
        `browser command: failed to launch persistent browser (${message}). Install Playwright browsers (npx playwright install) or pass --browser-channel chrome / --browser-executable.`
      );
    }
  }

  const launchErrors = [];
  const attemptConfigs = hasCrashRecoveryArgs(persistentOpts.args || [])
    ? [persistentOpts]
    : [
      persistentOpts,
      { ...persistentOpts, args: withCrashRecoveryArgs(persistentOpts.args || []) },
    ];

  for (const candidateLaunchOptions of attemptConfigs) {
    try {
      for (const channel of resolveDefaultChannels()) {
        try {
          return await chromium.launchPersistentContext(userDataDir, { ...candidateLaunchOptions, channel });
        } catch (error) {
          const message = error?.message || String(error);
          launchErrors.push(`${channel}: ${message}`);
        }
      }
      return await chromium.launchPersistentContext(userDataDir, candidateLaunchOptions);
    } catch (error) {
      const message = error?.message || String(error);
      launchErrors.push(message);
      if (isRecoverablePersistentLaunchError(error) && attemptConfigs.length === 2) {
        continue;
      }

      if (!isRecoverablePersistentLaunchError(error) || launchErrors.length >= 2) {
        throw error;
      }
    }
  }

  const retryMessage = launchErrors.join(' | ');
  if (isRecoverablePersistentLaunchError(retryMessage)) {
    throw new Error(
      `browser command: failed to launch persistent browser with crash recovery enabled (${retryMessage}). ` +
      `Tried default channels: ${resolveDefaultChannels().join(', ')}. ` +
      `Install Playwright browsers (npx playwright install) or pass --browser-channel / --browser-executable.`
    );
  }

  throw new Error(
    `browser command: failed to launch persistent browser (${retryMessage}). ` +
    `Tried default channels: ${resolveDefaultChannels().join(', ')}. ` +
    `Install Playwright browsers (npx playwright install) or pass --browser-channel / --browser-executable.`
  );
}

export async function runBrowserCommandInNode(commandRequest, options = {}) {
  let request = null;
  try {
    ({ request } = ensureCommandSupportedOnSurface(commandRequest, 'browser'));

    if (request.keepPipeline) {
      throw new Error(
        'browser command relay does not support keepPipeline=true because pipeline objects are not serializable across process boundaries.'
      );
    }

    if (request.command === 'convert') {
      throw new Error('browser command relay does not support convert. Use --surface node for convert commands.');
    }

  let useOpfsCache = options.opfsCache !== false;
  const userDataDir = options.userDataDir || DEFAULT_OPFS_CACHE_DIR;

  if (options.wipeCacheBeforeLaunch && useOpfsCache) {
    await fs.rm(userDataDir, { recursive: true, force: true }).catch(() => {});
  }

  const { chromium } = await import('playwright');
  const baseUrl = normalizeBaseUrl(options.baseUrl);
  // When OPFS caching is enabled, use a fixed port so the browser origin stays the same
  // across runs (OPFS is origin-scoped). Without this, random ports create new origins.
  const serverPort = options.port ?? (useOpfsCache ? DEFAULT_OPFS_CACHE_PORT : 0);
  const server = baseUrl
    ? null
    : await createStaticFileServer({
      rootDir: options.staticRootDir,
      staticMounts: options.staticMounts,
      host: options.host,
      port: serverPort,
    }).catch((error) => {
      const message = error?.message || String(error);
      throw new Error(
        `browser command: failed to start static server (${message}). Pass --browser-base-url to reuse an existing server.`
      );
    });

  const launchOptions = {
    headless: normalizeHeadless(options.headless),
    args: browserLaunchArgs(normalizeBrowserArgs(options.browserArgs)),
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
  const requestedLoadMode = request.loadMode;
  const requireOpfsLoad = requestedLoadMode === 'opfs';
  if (requireOpfsLoad && useOpfsCache === false) {
    throw new Error('browser command: loadMode=opfs requires OPFS cache support (remove --no-opfs-cache).');
  }
  if (!requestedLoadMode && request.modelUrl && useOpfsCache === false) {
    request = {
      ...request,
      loadMode: 'http',
    };
  }

    let browser = null;
    let context = null;
    try {
    if (useOpfsCache) {
      // Persistent context: OPFS data survives between runs.
      // launchPersistentContext returns a BrowserContext directly (no separate Browser).
      try {
        context = await launchPersistentBrowser(chromium, userDataDir, launchOptions, {
          explicitChannel: Boolean(options.channel),
          explicitExecutablePath: Boolean(options.executablePath),
        });
      } catch (error) {
        if (!isRecoverablePersistentLaunchError(error)) {
          throw error;
        }
        if (typeof options.onConsole === 'function') {
          options.onConsole({
            type: 'warning',
            text: '[browser] Persistent browser launch failed; retrying with a clean OPFS profile.',
          });
        }
        await fs.rm(userDataDir, { recursive: true, force: true }).catch(() => {});
        try {
          context = await launchPersistentBrowser(chromium, userDataDir, launchOptions, {
            explicitChannel: Boolean(options.channel),
            explicitExecutablePath: Boolean(options.executablePath),
          });
        } catch (retryError) {
          if (!isRecoverablePersistentLaunchError(retryError)) {
            throw retryError;
          }
          if (typeof options.onConsole === 'function') {
            options.onConsole({
              type: 'warning',
              text: '[browser] Persistent launch still failing; falling back to non-persistent mode.',
            });
          }
          if (requireOpfsLoad) {
            throw new Error(
              'browser command: loadMode=opfs requires persistent browser context; persistent launch failed.'
            );
          }
          useOpfsCache = false;
          if (request.loadMode === 'opfs') {
            request = {
              ...request,
              loadMode: 'http',
            };
          }
          browser = await launchBrowser(chromium, launchOptions, {
            explicitChannel: Boolean(options.channel),
            explicitExecutablePath: Boolean(options.executablePath),
          });
          context = await browser.newContext();
        }
      }
    } else {
      browser = await launchBrowser(chromium, launchOptions, {
        explicitChannel: Boolean(options.channel),
        explicitExecutablePath: Boolean(options.executablePath),
      });
      context = await browser.newContext();
    }

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

    const runnerUrl = new URL(runnerPath, resolvedBaseUrl);
    runnerUrl.searchParams.set('_dopplerRunner', String(Date.now()));
    await page.goto(runnerUrl.toString(), { waitUntil: 'load' });
    try {
      await page.waitForFunction(() => globalThis.__dopplerRunnerReady === true, null, {
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

    // OPFS cache: ensure model is cached before running the command.
    // On cache hit, strip modelUrl so the harness takes the fast OPFS path.
    if (useOpfsCache && request.modelId && request.modelUrl) {
      try {
        const cacheResult = await page.evaluate(async (payload) => {
          if (typeof globalThis.__dopplerEnsureCached !== 'function') {
            return { cached: false, error: '__dopplerEnsureCached not available' };
          }
          return globalThis.__dopplerEnsureCached(payload.modelId, payload.modelBaseUrl);
        }, {
          modelId: request.modelId,
          modelBaseUrl: request.modelUrl,
        });

        if (cacheResult.cached) {
          // Remove modelUrl so the harness loads from OPFS instead of HTTP.
          request = { ...request };
          delete request.modelUrl;
          request.loadMode = 'opfs';
        } else {
          if (requireOpfsLoad) {
            const cacheError = cacheResult?.error || 'model not cached';
            throw new Error(
              `[opfs-cache] loadMode=opfs requested but cache is unavailable for "${request.modelId || 'unknown-model'}": ${cacheError}`
            );
          }
          if (!requestedLoadMode) {
            request = { ...request, loadMode: 'http' };
          }
          if (cacheResult.error) {
            if (typeof options.onConsole === 'function') {
              options.onConsole({
                type: 'warning',
                text: `[opfs-cache] Cache check failed (${cacheResult.error}), falling back to HTTP`,
              });
            }
          }
        }
      } catch (error) {
        if (requireOpfsLoad) {
          throw new Error(
            `[opfs-cache] loadMode=opfs requested but cache priming failed: ${error?.message || error}`
          );
        }
        if (!requestedLoadMode && request.modelUrl) {
          request = { ...request, loadMode: 'http' };
        }

        // OPFS cache is best-effort; fall back to HTTP on any error.
        if (typeof options.onConsole === 'function') {
          options.onConsole({
            type: 'warning',
            text: `[opfs-cache] Error (${error?.message || error}), falling back to HTTP`,
          });
        }
      }
    }

    const response = await page.evaluate(async (payload) => {
      if (typeof globalThis.__dopplerRunBrowserCommand !== 'function') {
        throw new Error('browser command runner is missing globalThis.__dopplerRunBrowserCommand');
      }
      return globalThis.__dopplerRunBrowserCommand(payload.request, payload.options || {});
    }, {
      request,
      options: {
        runtimeLoadOptions: options.runtimeLoadOptions || {},
      },
    });

    const result = response?.result;
    if (result && typeof result === 'object' && !Array.isArray(result)) {
      const cpuInfo = typeof os.cpus === 'function' ? os.cpus() : null;
      const hostEnvironment = {
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version,
        osRelease: typeof os.release === 'function' ? os.release() : null,
        cpuModel: Array.isArray(cpuInfo) && cpuInfo.length > 0 ? asNonEmptyString(cpuInfo[0]?.model) : null,
      };
      const webgpuBackend = inferWebgpuBackendFromArgs(launchOptions.args, hostEnvironment.platform);
      const env = result.env && typeof result.env === 'object' ? result.env : {};
      const deviceInfo = result.deviceInfo && typeof result.deviceInfo === 'object'
        ? result.deviceInfo
        : {};
      result.env = {
        ...env,
        webgpuBackend: normalizeWebgpuBackend(env.webgpuBackend)
          || normalizeWebgpuBackend(env.gpuBackend)
          || normalizeWebgpuBackend(env.graphicsBackend)
          || webgpuBackend,
      };
      const existingEnvironment = result.environment && typeof result.environment === 'object'
        ? result.environment
        : {};
      result.environment = {
        ...existingEnvironment,
        host: {
          ...(existingEnvironment.host && typeof existingEnvironment.host === 'object' ? existingEnvironment.host : {}),
          platform: asNonEmptyString(existingEnvironment?.host?.platform) || hostEnvironment.platform,
          arch: asNonEmptyString(existingEnvironment?.host?.arch) || hostEnvironment.arch,
          nodeVersion: asNonEmptyString(existingEnvironment?.host?.nodeVersion) || hostEnvironment.nodeVersion,
          osRelease: asNonEmptyString(existingEnvironment?.host?.osRelease) || hostEnvironment.osRelease,
          cpuModel: asNonEmptyString(existingEnvironment?.host?.cpuModel) || hostEnvironment.cpuModel,
        },
        browser: {
          ...(existingEnvironment.browser && typeof existingEnvironment.browser === 'object' ? existingEnvironment.browser : {}),
          userAgent: asNonEmptyString(existingEnvironment?.browser?.userAgent) || asNonEmptyString(env.browserUserAgent),
          platform: asNonEmptyString(existingEnvironment?.browser?.platform) || asNonEmptyString(env.browserPlatform),
          language: asNonEmptyString(existingEnvironment?.browser?.language) || asNonEmptyString(env.browserLanguage),
          vendor: asNonEmptyString(existingEnvironment?.browser?.vendor) || asNonEmptyString(env.browserVendor),
          executable: asNonEmptyString(existingEnvironment?.browser?.executable) || asNonEmptyString(options.executablePath),
          channel: asNonEmptyString(existingEnvironment?.browser?.channel) || asNonEmptyString(options.channel),
        },
        gpu: {
          ...(existingEnvironment.gpu && typeof existingEnvironment.gpu === 'object' ? existingEnvironment.gpu : {}),
          api: asNonEmptyString(existingEnvironment?.gpu?.api) || 'webgpu',
          backend: normalizeWebgpuBackend(existingEnvironment?.gpu?.backend)
            || normalizeWebgpuBackend(env.webgpuBackend)
            || webgpuBackend,
          vendor: asNonEmptyString(existingEnvironment?.gpu?.vendor) || asNonEmptyString(deviceInfo.vendor),
          architecture: asNonEmptyString(existingEnvironment?.gpu?.architecture) || asNonEmptyString(deviceInfo.architecture),
          device: asNonEmptyString(existingEnvironment?.gpu?.device) || asNonEmptyString(deviceInfo.device),
          description: asNonEmptyString(existingEnvironment?.gpu?.description) || asNonEmptyString(deviceInfo.description),
          hasF16: typeof existingEnvironment?.gpu?.hasF16 === 'boolean'
            ? existingEnvironment.gpu.hasF16
            : (typeof deviceInfo.hasF16 === 'boolean' ? deviceInfo.hasF16 : null),
          hasSubgroups: typeof existingEnvironment?.gpu?.hasSubgroups === 'boolean'
            ? existingEnvironment.gpu.hasSubgroups
            : (typeof deviceInfo.hasSubgroups === 'boolean' ? deviceInfo.hasSubgroups : null),
          hasTimestampQuery: typeof existingEnvironment?.gpu?.hasTimestampQuery === 'boolean'
            ? existingEnvironment.gpu.hasTimestampQuery
            : (typeof deviceInfo.hasTimestampQuery === 'boolean' ? deviceInfo.hasTimestampQuery : null),
        },
        runtime: {
          ...(existingEnvironment.runtime && typeof existingEnvironment.runtime === 'object' ? existingEnvironment.runtime : {}),
          library: asNonEmptyString(existingEnvironment?.runtime?.library) || asNonEmptyString(env.library) || 'doppler',
          version: asNonEmptyString(existingEnvironment?.runtime?.version) || asNonEmptyString(env.version),
          surface: asNonEmptyString(existingEnvironment?.runtime?.surface) || asNonEmptyString(env.runtime) || 'browser',
          device: asNonEmptyString(existingEnvironment?.runtime?.device) || asNonEmptyString(env.device),
          dtype: asNonEmptyString(existingEnvironment?.runtime?.dtype) || asNonEmptyString(env.dtype),
          requestedDtype: asNonEmptyString(existingEnvironment?.runtime?.requestedDtype) || asNonEmptyString(env.requestedDtype),
          executionProviderMode: asNonEmptyString(existingEnvironment?.runtime?.executionProviderMode)
            || asNonEmptyString(env.executionProviderMode),
          cacheMode: asNonEmptyString(existingEnvironment?.runtime?.cacheMode)
            || asNonEmptyString(result.cacheMode)
            || asNonEmptyString(result?.timing?.cacheMode),
          loadMode: asNonEmptyString(existingEnvironment?.runtime?.loadMode)
            || asNonEmptyString(result.loadMode)
            || asNonEmptyString(result?.timing?.loadMode),
        },
      };
    }

      return response;
    } catch (error) {
      throw normalizeToToolingCommandError(error, {
        surface: 'browser',
        request,
      });
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
  } catch (error) {
    throw normalizeToToolingCommandError(error, {
      surface: 'browser',
      request,
    });
  }
}

export function normalizeNodeBrowserCommand(commandRequest) {
  return normalizeToolingCommandRequest(commandRequest);
}
