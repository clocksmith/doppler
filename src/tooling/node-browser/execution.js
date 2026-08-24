import { promises as fs } from 'node:fs';
import {
  ensureCommandSupportedOnSurface,
  normalizeToolingCommandRequest,
} from '../command-api.js';
import { normalizeToToolingCommandError } from '../command-envelope.js';
import {
  DEFAULT_CLEANUP_TIMEOUT_MS,
  DEFAULT_OPFS_CACHE_DIR,
  DEFAULT_OPFS_CACHE_PORT,
  browserLaunchArgs,
  createPersistentContextRequiredError,
  createStaticFileServer,
  finalizeBrowserRelayResponse,
  formatBrowserEvaluationError,
  isRecoverablePersistentLaunchError,
  launchBrowser,
  launchPersistentBrowser,
  normalizeBaseUrl,
  normalizeBrowserArgs,
  normalizeHeadless,
  normalizeRunnerPath,
  normalizeTimeoutMs,
  resolveLocalFileModelUrlForBrowserRelay,
  runBrowserCleanupWithTimeout,
  runBrowserCommandEvaluationWithTimeout,
  terminateBrowserProcess,
} from './transport.js';

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
          `browser command: failed to start static server (${message}). Set run.browser.baseUrl to reuse an existing server.`
        );
      });

    const timeoutMs = normalizeTimeoutMs(options.timeoutMs);
    const launchOptions = {
      headless: normalizeHeadless(options.headless),
      args: browserLaunchArgs(normalizeBrowserArgs(options.browserArgs)),
      timeout: timeoutMs,
    };

    if (options.channel) {
      launchOptions.channel = String(options.channel);
    }
    if (options.executablePath) {
      launchOptions.executablePath = String(options.executablePath);
    }

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
            if (typeof options.afterOpfsCachePrime === 'function') {
              await options.afterOpfsCachePrime({
                page,
                modelId: relayRequest.modelId,
                cacheResult,
              });
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

      const response = await runBrowserCommandEvaluationWithTimeout(() => page.evaluate(async (payload) => {
        const serializeError = (error, depth = 0) => {
          if (!error || typeof error !== 'object') {
            return {
              name: null,
              message: String(error || 'Unknown browser error'),
              stack: null,
              code: null,
              details: null,
              retryable: null,
              cause: null,
            };
          }
          return {
            name: typeof error.name === 'string' ? error.name : null,
            message: typeof error.message === 'string' ? error.message : String(error),
            stack: typeof error.stack === 'string' ? error.stack : null,
            code: typeof error.code === 'string' ? error.code : null,
            details: error.details && typeof error.details === 'object' ? error.details : null,
            retryable: typeof error.retryable === 'boolean' ? error.retryable : null,
            cause: depth < 2 ? serializeError(error.cause, depth + 1) : null,
          };
        };
        if (typeof globalThis.__dopplerRunBrowserCommand !== 'function') {
          throw new Error('browser command runner is missing globalThis.__dopplerRunBrowserCommand');
        }
        try {
          return await globalThis.__dopplerRunBrowserCommand(payload.request, payload.options || {});
        } catch (error) {
          return {
            __dopplerBrowserError: serializeError(error),
          };
        }
      }, {
        request: relayRequest,
        options: {
          runtimeLoadOptions: options.runtimeLoadOptions || {},
        },
      }), timeoutMs);

      if (response?.__dopplerBrowserError) {
        throw formatBrowserEvaluationError(response.__dopplerBrowserError);
      }

      return finalizeBrowserRelayResponse(response, effectiveRequest);
    } catch (error) {
      throw normalizeToToolingCommandError(error, {
        surface: 'browser',
        request: sourceRequest,
      });
    } finally {
      const cleanupTimeoutMs = Math.min(timeoutMs, DEFAULT_CLEANUP_TIMEOUT_MS);
      if (browser) {
        const browserClosed = await runBrowserCleanupWithTimeout(() => browser.close(), cleanupTimeoutMs);
        if (!browserClosed) {
          terminateBrowserProcess(browser);
        }
      } else if (context) {
        await runBrowserCleanupWithTimeout(() => context.close(), cleanupTimeoutMs);
      }
      if (server) {
        await runBrowserCleanupWithTimeout(() => server.close(), cleanupTimeoutMs);
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
