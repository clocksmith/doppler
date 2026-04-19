import { log } from '../../debug/index.js';
import { createPipeline } from '../../generation/index.js';
import { listQuickstartModels } from '../doppler-registry.js';
import {
  createDefaultNodeLoadProgressLogger,
  fetchManifestPayloadFromBaseUrl,
  resolveLoadProgressHandlers,
  resolveModelSource,
} from './model-source.js';
import { assertSupportedGenerationOptions, createModelHandle } from './model-session.js';
import { isNodeRuntime } from '../../utils/runtime-env.js';

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
  ) {
    throw new Error(
      `${apiName} does not accept load-affecting options. Use doppler.load(model, options) instead.`
    );
  }
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
  const { createNodeFileArtifactStorageContext } = await import('../../storage/artifact-storage-context.js');
  return createNodeFileArtifactStorageContext(loadSource.baseUrl, loadSource.manifest);
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
    const { userProgress, pipelineProgress } = resolveLoadProgressHandlers(options, defaultLoadProgressLogger);

    emitLoadProgress(userProgress, 'resolve', 5, 'Resolving model');
    const resolved = await resolveModelSource(model);
    await ensureWebGPUAvailable();

    emitLoadProgress(userProgress, 'manifest', 15, 'Fetching manifest');
    const manifestPayload = resolved.manifest
      ? { text: JSON.stringify(resolved.manifest), manifest: resolved.manifest }
      : await fetchManifestPayloadFromBaseUrl(resolved.baseUrl);
    const cachedResolved = await resolveCachedNodeQuickstartSource(
      resolved,
      manifestPayload,
      userProgress
    );
    const loadSource = cachedResolved ?? {
      ...resolved,
      manifest: manifestPayload.manifest,
    };
    const storageContext = await resolveNodeArtifactStorageContext(loadSource);

    emitLoadProgress(userProgress, 'load', 25, 'Loading weights');
    const pipeline = await createPipeline(loadSource.manifest, {
      baseUrl: loadSource.baseUrl ?? undefined,
      storage: storageContext ?? undefined,
      runtimeConfig: options.runtimeConfig,
      onProgress: pipelineProgress
        ? (progress) => emitLoadProgress(
          pipelineProgress,
          'load',
          Math.max(25, Math.min(99, Math.round(progress.percent))),
          progress.message || 'Loading weights'
        )
        : undefined,
    });

    emitLoadProgress(userProgress, 'ready', 100, 'Model ready');
    return createModelHandle(pipeline, resolved);
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

  return {
    doppler,
    load,
    clearModelCache,
    resolveLoadProgressHandlers(options = {}) {
      return resolveLoadProgressHandlers(options, defaultLoadProgressLogger);
    },
    createDefaultNodeLoadProgressLogger,
  };
}
