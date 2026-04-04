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
import {
  isToolingSuccessEnvelope,
  normalizeToToolingCommandError,
} from './command-envelope.js';

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

function resolveLocalFileModelPath(modelUrl) {
  return resolveLocalFileUrlPath(modelUrl, 'request.modelUrl');
}

function resolveLocalFileUrlPath(urlValue, fieldLabel) {
  const normalized = asNonEmptyString(urlValue);
  if (!normalized || !normalized.startsWith('file://')) {
    return null;
  }
  try {
    return fileURLToPath(normalized);
  } catch (error) {
    throw new Error(
      `browser command: ${fieldLabel} must be a valid file:// URL when provided explicitly; ` +
      `got ${JSON.stringify(urlValue)} (${error?.message || error}).`
    );
  }
}

async function createLocalFileRelayMount(filePath, fieldLabel, urlPrefixRoot, options = {}) {
  if (options.baseUrl) {
    throw new Error(
      `browser command: explicit local file:// ${fieldLabel} requires the relay-owned static server. ` +
      'Remove run.browser.baseUrl or use a hosted URL instead.'
    );
  }

  let stats;
  try {
    stats = await fs.stat(filePath);
  } catch (error) {
    throw new Error(
      `browser command: explicit local ${fieldLabel} "${options.originalUrl}" is not accessible: ${error?.message || error}.`
    );
  }

  const expectDirectory = options.kind === 'directory';
  if (expectDirectory && !stats.isDirectory()) {
    throw new Error(
      `browser command: explicit local file:// ${fieldLabel} must point to a directory; got "${options.originalUrl}".`
    );
  }
  if (!expectDirectory && !stats.isFile()) {
    throw new Error(
      `browser command: explicit local file:// ${fieldLabel} must point to a file; got "${options.originalUrl}".`
    );
  }

  const fileName = path.basename(filePath) || options.fallbackName || 'asset';
  const mountName = encodeURIComponent(fileName);
  const mountPrefix = `${urlPrefixRoot}/${mountName}-${process.pid}-${Date.now()}`;
  if (expectDirectory) {
    return {
      url: mountPrefix,
      staticMount: {
        urlPrefix: mountPrefix,
        rootDir: filePath,
      },
    };
  }

  return {
    url: `${mountPrefix}/${encodeURIComponent(fileName)}`,
    staticMount: {
      urlPrefix: mountPrefix,
      rootDir: path.dirname(filePath),
    },
  };
}

export async function resolveLocalFileModelUrlForBrowserRelay(request, options = {}) {
  const localModelPath = resolveLocalFileModelPath(request?.modelUrl);
  const localRuntimeConfigPath = resolveLocalFileUrlPath(request?.runtimeConfigUrl, 'request.runtimeConfigUrl');
  if (options.staticMounts != null && !Array.isArray(options.staticMounts)) {
    throw new Error('browser command: staticMounts must be an array.');
  }
  const staticMounts = Array.isArray(options.staticMounts) ? [...options.staticMounts] : [];
  if (!localModelPath && !localRuntimeConfigPath) {
    return {
      relayRequest: request,
      staticMounts,
    };
  }
  let relayRequest = request;
  const relayStaticMounts = [...staticMounts];

  if (localModelPath) {
    const modelRelayMount = await createLocalFileRelayMount(localModelPath, 'modelUrl', '/__doppler_local_model', {
      ...options,
      originalUrl: request.modelUrl,
      kind: 'directory',
      fallbackName: 'model',
    });
    relayRequest = {
      ...relayRequest,
      modelUrl: modelRelayMount.url,
    };
    relayStaticMounts.push(modelRelayMount.staticMount);
  }

  if (localRuntimeConfigPath) {
    const runtimeConfigRelayMount = await createLocalFileRelayMount(
      localRuntimeConfigPath,
      'runtimeConfigUrl',
      '/__doppler_local_runtime_config',
      {
        ...options,
        originalUrl: request.runtimeConfigUrl,
        kind: 'file',
        fallbackName: 'runtime-config.json',
      }
    );
    relayRequest = {
      ...relayRequest,
      runtimeConfigUrl: runtimeConfigRelayMount.url,
    };
    relayStaticMounts.push(runtimeConfigRelayMount.staticMount);
  }

  return {
    relayRequest,
    staticMounts: relayStaticMounts,
  };
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

function createPersistentContextRequiredError(requestedLoadMode, cause = null) {
  const baseMessage = requestedLoadMode === 'opfs'
    ? 'browser command: loadMode=opfs requires persistent browser context; persistent launch failed.'
    : 'browser command: persistent browser context is required when OPFS cache is enabled; persistent launch failed.';
  const causeMessage = asNonEmptyString(cause?.message || cause);
  return new Error(
    `${baseMessage} Re-run with run.browser.opfsCache=false to use a non-persistent browser session.${causeMessage ? ` (${causeMessage})` : ''}`
  );
}

export function finalizeBrowserRelayResponse(response, request) {
  if (!isToolingSuccessEnvelope(response)) {
    throw new Error('browser command: runner returned an invalid success envelope.');
  }
  return {
    ...response,
    request,
  };
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
  let sourceRequest = null;
  try {
    ({ request } = ensureCommandSupportedOnSurface(commandRequest, 'browser'));
    sourceRequest = request;

    if (request.keepPipeline) {
      throw new Error(
        'browser command relay does not support keepPipeline=true because pipeline objects are not serializable across process boundaries.'
      );
    }

    if (request.command === 'convert') {
      throw new Error('browser command relay does not support convert. Use --surface node for convert commands.');
    }

    let useOpfsCache = options.opfsCache !== false;
    let relayRequest = request;
    const userDataDir = options.userDataDir || DEFAULT_OPFS_CACHE_DIR;
    const baseUrl = normalizeBaseUrl(options.baseUrl);
    const localModelRelayState = await resolveLocalFileModelUrlForBrowserRelay(relayRequest, {
      baseUrl,
      staticMounts: options.staticMounts,
    });
    relayRequest = localModelRelayState.relayRequest;
    const relayStaticMounts = localModelRelayState.staticMounts;

    if (options.wipeCacheBeforeLaunch && useOpfsCache) {
      await fs.rm(userDataDir, { recursive: true, force: true }).catch(() => {});
    }

    const { chromium } = await import('playwright');
    // When OPFS caching is enabled, use a fixed port so the browser origin stays the same
    // across runs (OPFS is origin-scoped). Without this, random ports create new origins.
    const serverPort = options.port ?? (useOpfsCache ? DEFAULT_OPFS_CACHE_PORT : 0);
    const server = baseUrl
      ? null
      : await createStaticFileServer({
        rootDir: options.staticRootDir,
        staticMounts: relayStaticMounts,
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
    const requestedLoadMode = sourceRequest.loadMode;
    const requireOpfsLoad = requestedLoadMode === 'opfs';
    const allowOpfsPromotion = requestedLoadMode == null;
    if (requireOpfsLoad && useOpfsCache === false) {
      throw new Error('browser command: loadMode=opfs requires OPFS cache support (remove --no-opfs-cache).');
    }
    if (requireOpfsLoad && sourceRequest.modelUrl && !sourceRequest.modelId) {
      throw new Error(
        'browser command: loadMode=opfs requires modelId when modelUrl is provided so the relay can verify and load the cached OPFS artifact.'
      );
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
            throw createPersistentContextRequiredError(requestedLoadMode, retryError);
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

      let effectiveRequest = sourceRequest;
      const shouldPrimeOpfsCache = useOpfsCache
        && relayRequest.modelId
        && relayRequest.modelUrl
        && (requireOpfsLoad || allowOpfsPromotion);

      if (shouldPrimeOpfsCache) {
        try {
          const cacheResult = await page.evaluate(async (payload) => {
            if (typeof globalThis.__dopplerEnsureCached !== 'function') {
              return { cached: false, error: '__dopplerEnsureCached not available' };
            }
            return globalThis.__dopplerEnsureCached(payload.modelId, payload.modelBaseUrl);
          }, {
            modelId: relayRequest.modelId,
            modelBaseUrl: relayRequest.modelUrl,
          });

          if (cacheResult.cached) {
            relayRequest = { ...relayRequest, loadMode: 'opfs' };
            delete relayRequest.modelUrl;
            if (allowOpfsPromotion) {
              effectiveRequest = {
                ...sourceRequest,
                loadMode: 'opfs',
              };
            }
          } else if (requireOpfsLoad) {
            const cacheError = cacheResult?.error || 'model not cached';
            throw new Error(
              `[opfs-cache] model cache is unavailable for "${relayRequest.modelId || 'unknown-model'}": ${cacheError}.`
            );
          }
        } catch (error) {
          if (requireOpfsLoad) {
            throw new Error(
              `[opfs-cache] cache priming failed: ${error?.message || error}.`
            );
          }
        }
      }

      const response = await page.evaluate(async (payload) => {
        if (typeof globalThis.__dopplerRunBrowserCommand !== 'function') {
          throw new Error('browser command runner is missing globalThis.__dopplerRunBrowserCommand');
        }
        return globalThis.__dopplerRunBrowserCommand(payload.request, payload.options || {});
      }, {
        request: relayRequest,
        options: {
          runtimeLoadOptions: options.runtimeLoadOptions || {},
        },
      });

      return finalizeBrowserRelayResponse(response, effectiveRequest);
    } catch (error) {
      throw normalizeToToolingCommandError(error, {
        surface: 'browser',
        request: sourceRequest,
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
      request: sourceRequest,
    });
  }
}

export function normalizeNodeBrowserCommand(commandRequest) {
  return normalizeToolingCommandRequest(commandRequest);
}
