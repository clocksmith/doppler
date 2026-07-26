import { log } from '../../debug/index.js';
import { createPipeline } from '../../generation/index.js';
import { listQuickstartModels } from '../doppler-registry.js';
import {
  createDefaultNodeLoadProgressLogger,
  fetchManifestPayloadFromBaseUrl,
  resolveManifestArtifactSource,
  resolveLoadProgressHandlers,
  resolveModelSource,
  sha256ManifestText,
} from './model-source.js';
import { assertSupportedGenerationOptions, createModelHandle } from './model-session.js';
import {
  createHttpArtifactStorageContext,
  createNodeFileArtifactStorageContext,
} from '../../storage/artifact-storage-context.js';
import {
  ensureModelCachedSource,
  loadPersistentModelSource,
} from '../../storage/model-cache.js';
import { deleteModel } from '../../storage/shard-manager.js';
import {
  listRegisteredModels,
  registerModel,
  removeRegisteredModel,
} from '../../storage/registry.js';
import { isNodeRuntime } from '../../utils/runtime-env.js';
import { resolveManifestGpuResidentEmbeddingLimitError } from '../../loader/embedding-limit-preflight.js';
import { createDopplerLoader } from '../../loader/doppler-loader.js';
import { initDevice } from '../../gpu/device.js';
import { createScopedModelSession } from './scoped-session.js';

function emitLoadProgress(callback, phase, percent, message) {
  if (typeof callback !== 'function') return;
  callback({ phase, percent, message });
}

function assertDopplerOptions(options, apiName) {
  if (!options || typeof options !== 'object') {
    return;
  }
  if (
    options.runtimeConfig !== undefined
    || options.runtimeProfile !== undefined
    || options.runtimeConfigUrl !== undefined
    || options.cache !== undefined
  ) {
    throw new Error(
      `${apiName} does not accept load-affecting options. Use doppler.load(model, options) instead.`
    );
  }
}

function assertPersistentCacheMode(cache) {
  if (cache === undefined || cache === false || cache === 'opfs') {
    if (cache === 'opfs' && isNodeRuntime()) {
      throw new Error('doppler.load options.cache="opfs" is browser-only.');
    }
    return;
  }
  throw new Error('doppler.load options.cache must be false or "opfs".');
}

function emitPersistentCacheProgress(callback, event) {
  const percent = Number.isFinite(event?.percent)
    ? Math.max(0, Math.min(100, Number(event.percent)))
    : 0;
  emitLoadProgress(
    callback,
    'cache',
    15 + Math.round(percent * 0.1),
    event?.message || 'Preparing persistent model cache'
  );
}

export async function resolvePersistentBrowserLoadSource(
  loadSource,
  cache,
  onProgress = null,
  cacheSource = ensureModelCachedSource
) {
  assertPersistentCacheMode(cache);
  if (cache !== 'opfs') {
    return loadSource;
  }
  if (isNodeRuntime()) {
    throw new Error('doppler.load options.cache="opfs" is browser-only.');
  }
  const storageBaseUrl = loadSource?.storageBaseUrl ?? loadSource?.baseUrl;
  const storageManifest = loadSource?.storageManifest ?? loadSource?.manifest;
  const storageManifestText = loadSource?.storageManifestText
    ?? (storageManifest ? JSON.stringify(storageManifest) : null);
  const storageModelId = storageManifest?.modelId;
  if (!storageBaseUrl || !storageManifest || !storageManifestText || !storageModelId) {
    throw new Error(
      'doppler.load options.cache="opfs" requires a URL-backed artifact with a manifest modelId.'
    );
  }
  const expectedManifestHash = await sha256ManifestText(storageManifestText);
  const cached = await cacheSource(
    storageModelId,
    storageBaseUrl,
    (event) => emitPersistentCacheProgress(onProgress, event),
    { expectedManifestHash }
  );
  return {
    ...loadSource,
    storageContext: cached.storageContext,
    storage: cached.storageContext,
    persistentCache: {
      backend: cached.storageBackend,
      state: cached.cacheState,
      fromCache: cached.fromCache,
      manifestHash: cached.manifestHash,
      totalBytes: cached.totalBytes,
    },
  };
}

async function resolveCachedNodeQuickstartSource(resolved, manifestPayload, onProgress) {
  if (!isNodeRuntime()) {
    return null;
  }
  const { resolveNodeQuickstartCachedSource } = await import('./node-quickstart-cache.js');
  return resolveNodeQuickstartCachedSource(resolved, manifestPayload, {
    onProgress,
  });
}

async function resolveNodeArtifactStorageContext(loadSource) {
  if (!isNodeRuntime() || !loadSource?.cache || !loadSource?.baseUrl || !loadSource?.manifest) {
    return null;
  }
  return createNodeFileArtifactStorageContext(
    loadSource.storageBaseUrl ?? loadSource.baseUrl,
    loadSource.storageManifest ?? loadSource.manifest
  );
}

function resolveArtifactStorageContext(loadSource) {
  const providedStorageContext = loadSource?.storageContext ?? loadSource?.storage;
  if (providedStorageContext && typeof providedStorageContext === 'object') {
    return providedStorageContext;
  }
  const baseUrl = loadSource?.storageBaseUrl ?? loadSource?.baseUrl;
  const manifest = loadSource?.storageManifest ?? loadSource?.manifest;
  if (!baseUrl || !manifest) {
    return null;
  }
  return createNodeFileArtifactStorageContext(baseUrl, manifest)
    ?? createHttpArtifactStorageContext(baseUrl, manifest);
}

export function createDopplerRuntimeService({
  ensureWebGPUAvailable,
  defaultLoadProgressLogger = null,
} = {}) {
  if (typeof ensureWebGPUAvailable !== 'function') {
    throw new Error('createDopplerRuntimeService requires ensureWebGPUAvailable.');
  }

  const convenienceModelCache = new Map();
  const inFlightLoadCache = new Map();

  function clearModelCache() {
    convenienceModelCache.clear();
    inFlightLoadCache.clear();
    log.debug('doppler', 'Model cache cleared');
  }

  async function load(model, options = {}) {
    assertPersistentCacheMode(options.cache);
    const { userProgress, pipelineProgress } = resolveLoadProgressHandlers(options, defaultLoadProgressLogger);

    emitLoadProgress(userProgress, 'resolve', 5, 'Resolving model');
    const resolved = await resolveModelSource(model);
    await ensureWebGPUAvailable();

    emitLoadProgress(userProgress, 'manifest', 15, 'Fetching manifest');
    const persistentSource = options.cache === 'opfs' && !isNodeRuntime()
      ? await loadPersistentModelSource(resolved.modelId)
      : null;
    const manifestPayload = persistentSource
      ? {
        text: persistentSource.manifestText,
        manifest: persistentSource.manifest,
        manifestHash: persistentSource.manifestHash,
      }
      : resolved.manifest
      ? await (async () => {
        const text = resolved.manifestText ?? JSON.stringify(resolved.manifest);
        const manifestHash = await sha256ManifestText(text);
        if (resolved.manifestHash && resolved.manifestHash !== manifestHash) {
          throw new Error(
            `Resolved manifest hash mismatch: expected ${resolved.manifestHash}, got ${manifestHash}.`
          );
        }
        return { text, manifest: resolved.manifest, manifestHash };
      })()
      : await fetchManifestPayloadFromBaseUrl(resolved.baseUrl);
    const resolvedArtifactSource = persistentSource
      ? {
        ...resolved,
        manifest: persistentSource.manifest,
        manifestText: persistentSource.manifestText,
        storageContext: persistentSource.storageContext,
        persistentCache: {
          backend: 'opfs',
          state: 'verified-hit',
          fromCache: true,
          manifestHash: persistentSource.manifestHash,
          totalBytes: persistentSource.totalBytes,
        },
      }
      : await resolveManifestArtifactSource(resolved, manifestPayload);
    await initDevice();
    const embeddingLimitError = resolveManifestGpuResidentEmbeddingLimitError(
      resolvedArtifactSource.manifest,
      {
        storageManifest: resolvedArtifactSource.storageManifest,
        runtimeConfig: options.runtimeConfig,
      }
    );
    if (embeddingLimitError) {
      throw embeddingLimitError;
    }
    const cachedResolved = manifestPayload.manifest?.weightsRef == null
      ? await resolveCachedNodeQuickstartSource(
        resolvedArtifactSource,
        manifestPayload,
        userProgress
      )
      : null;
    const loadSource = await resolvePersistentBrowserLoadSource(
      cachedResolved ?? resolvedArtifactSource,
      persistentSource ? false : options.cache,
      userProgress
    );
    if (options.cache === 'opfs' && !isNodeRuntime()) {
      await registerModel({
        modelId: resolved.modelId,
        manifestHash: manifestPayload.manifestHash,
        backend: 'opfs',
      });
    }
    const nodeStorageContext = await resolveNodeArtifactStorageContext(loadSource);
    const storageContext = nodeStorageContext ?? resolveArtifactStorageContext(loadSource);
    await storageContext?.preflight?.();

    const effectiveBaseUrl = loadSource.storageBaseUrl ?? loadSource.baseUrl;
    const isolatedLoader = options.isolatedLoader === true
      ? createDopplerLoader(options.runtimeConfig?.loading)
      : null;
    emitLoadProgress(userProgress, 'load', 25, 'Loading weights');
    let pipeline;
    try {
      pipeline = await createPipeline(loadSource.manifest, {
        baseUrl: effectiveBaseUrl ?? undefined,
        storage: storageContext ?? undefined,
        runtimeConfig: options.runtimeConfig,
        loader: isolatedLoader ?? undefined,
        ownsLoader: Boolean(isolatedLoader),
        onProgress: pipelineProgress
          ? (progress) => emitLoadProgress(
            pipelineProgress,
            'load',
            Math.max(25, Math.min(99, Math.round(progress.percent))),
            progress.message || 'Loading weights'
          )
          : undefined,
      });
    } catch (error) {
      try {
        await storageContext?.close?.();
      } catch {
        // Preserve the original pipeline construction error.
      }
      throw error;
    }

    emitLoadProgress(userProgress, 'ready', 100, 'Model ready');
    return createModelHandle(pipeline, {
      ...resolved,
      manifestHash: manifestPayload.manifestHash,
      persistentCache: loadSource.persistentCache ?? null,
    });
  }

  async function getCachedModel(model, options = {}) {
    const resolved = await resolveModelSource(model);
    const cacheKey = resolved.modelId;
    const cached = convenienceModelCache.get(cacheKey);
    if (cached?.loaded) {
      return cached;
    }
    if (cached && !cached.loaded) {
      convenienceModelCache.delete(cacheKey);
    }
    if (!inFlightLoadCache.has(cacheKey)) {
      inFlightLoadCache.set(cacheKey, load(model, options).then((instance) => {
        convenienceModelCache.set(cacheKey, instance);
        inFlightLoadCache.delete(cacheKey);
        return instance;
      }).catch((error) => {
        inFlightLoadCache.delete(cacheKey);
        throw error;
      }));
    }
    return inFlightLoadCache.get(cacheKey);
  }

  async function open(model, options = {}) {
    return createScopedModelSession(await load(model, options));
  }

  async function generate(model, input, options = {}) {
    const {
      cache,
      isolatedLoader,
      onProgress,
      runtimeConfig,
      ...generationOptions
    } = options;
    const session = await open(model, {
      cache,
      isolatedLoader,
      onProgress,
      runtimeConfig,
    });
    try {
      return await session.generate(input, generationOptions);
    } finally {
      await session.close();
    }
  }

  async function* dopplerGenerate(prompt, options = {}) {
    if (!options || typeof options !== 'object' || options.model == null) {
      throw new Error('doppler() requires options.model.');
    }
    assertDopplerOptions(options, 'doppler()');
    assertSupportedGenerationOptions(options);
    const model = await getCachedModel(options.model, { onProgress: options.onProgress });
    yield* model.generate(prompt, options);
  }

  function doppler(prompt, options) {
    return dopplerGenerate(prompt, options);
  }

  doppler.load = load;
  doppler.open = open;
  doppler.generate = generate;
  doppler.text = async function text(prompt, options = {}) {
    if (!options || typeof options !== 'object' || options.model == null) {
      throw new Error('doppler.text() requires options.model.');
    }
    assertDopplerOptions(options, 'doppler.text()');
    assertSupportedGenerationOptions(options);
    const model = await getCachedModel(options.model, { onProgress: options.onProgress });
    return model.generateText(prompt, options);
  };
  doppler.chat = function chat(messages, options = {}) {
    if (!options || typeof options !== 'object' || options.model == null) {
      throw new Error('doppler.chat() requires options.model.');
    }
    assertDopplerOptions(options, 'doppler.chat()');
    assertSupportedGenerationOptions(options);
    return (async function* run() {
      const model = await getCachedModel(options.model, { onProgress: options.onProgress });
      yield* model.chat(messages, options);
    }());
  };
  doppler.chatText = async function chatText(messages, options = {}) {
    if (!options || typeof options !== 'object' || options.model == null) {
      throw new Error('doppler.chatText() requires options.model.');
    }
    assertDopplerOptions(options, 'doppler.chatText()');
    assertSupportedGenerationOptions(options);
    const model = await getCachedModel(options.model, { onProgress: options.onProgress });
    return model.chatText(messages, options);
  };
  doppler.evict = async function evict(model) {
    const resolved = await resolveModelSource(model);
    const cached = convenienceModelCache.get(resolved.modelId);
    if (!cached) {
      return false;
    }
    await cached.unload();
    convenienceModelCache.delete(resolved.modelId);
    return true;
  };
  doppler.evictAll = async function evictAll() {
    const cached = [...convenienceModelCache.values()];
    convenienceModelCache.clear();
    await Promise.allSettled(cached.map((entry) => entry.unload()));
  };
  doppler.listModels = async function listModels() {
    const models = await listQuickstartModels();
    return models.map((entry) => entry.modelId);
  };
  doppler.listModelDetails = async function listModelDetails() {
    return listQuickstartModels();
  };
  doppler.listPersistentModels = async function listPersistentModels() {
    if (isNodeRuntime()) {
      return [];
    }
    return listRegisteredModels();
  };
  doppler.removePersistentModel = async function removePersistentModel(model) {
    if (isNodeRuntime()) {
      throw new Error('doppler.removePersistentModel() is browser-only.');
    }
    const resolved = await resolveModelSource(model);
    await doppler.evict(resolved.modelId);
    const removed = await deleteModel(resolved.modelId);
    await removeRegisteredModel(resolved.modelId);
    return removed;
  };

  return {
    doppler,
    load,
    open,
    generate,
    clearModelCache,
    resolveLoadProgressHandlers(options = {}) {
      return resolveLoadProgressHandlers(options, defaultLoadProgressLogger);
    },
    createDefaultNodeLoadProgressLogger,
  };
}
