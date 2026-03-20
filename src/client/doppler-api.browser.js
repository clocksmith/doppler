import { loadLoRAFromManifest, loadLoRAFromUrl } from '../adapters/lora-loader.js';
import { log } from '../debug/index.js';
import { getManifestUrl, parseManifest } from '../formats/rdrr/index.js';
import { createPipeline } from '../generation/index.js';
import { getKernelCapabilities } from '../gpu/device.js';
import { formatChatMessages } from '../inference/pipelines/text/chat-format.js';
import { buildQuickstartModelBaseUrl, listQuickstartModels, resolveQuickstartModel } from './doppler-registry.js';

const convenienceModelCache = new Map();
const inFlightLoadCache = new Map();

function emitLoadProgress(callback, phase, percent, message) {
  if (typeof callback !== 'function') return;
  callback({ phase, percent, message });
}

async function ensureWebGPUAvailable() {
  if (typeof globalThis.navigator !== 'undefined' && globalThis.navigator?.gpu) {
    return;
  }
  throw new Error('WebGPU is unavailable. Run in a WebGPU-capable browser.');
}

export function createDefaultNodeLoadProgressLogger() {
  return (event) => {
    const message = typeof event?.message === 'string' ? event.message.trim() : '';
    if (!message) return;
    log.info('doppler', message);
  };
}

export function resolveLoadProgressHandlers(options = {}) {
  const onProgress = typeof options?.onProgress === 'function' ? options.onProgress : null;
  if (onProgress) {
    return {
      userProgress: onProgress,
      pipelineProgress: onProgress,
    };
  }
  return {
    userProgress: null,
    pipelineProgress: null,
  };
}

async function fetchManifestFromBaseUrl(baseUrl) {
  const response = await fetch(getManifestUrl(baseUrl));
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest from ${baseUrl}: ${response.status}`);
  }
  return parseManifest(await response.text());
}

async function resolveModelSource(model) {
  if (typeof model === 'string') {
    const entry = await resolveQuickstartModel(model);
    return {
      modelId: entry.modelId,
      baseUrl: buildQuickstartModelBaseUrl(entry),
      manifest: null,
    };
  }
  if (model && typeof model === 'object' && typeof model.url === 'string' && model.url.trim().length > 0) {
    return {
      modelId: model.url.trim(),
      baseUrl: model.url.trim(),
      manifest: null,
    };
  }
  if (model && typeof model === 'object' && model.manifest && typeof model.manifest === 'object') {
    const manifest = model.manifest;
    const modelId = typeof manifest.modelId === 'string' && manifest.modelId.length > 0
      ? manifest.modelId
      : 'manifest';
    return {
      modelId,
      baseUrl: typeof model.baseUrl === 'string' && model.baseUrl.length > 0 ? model.baseUrl : null,
      manifest,
    };
  }
  throw new Error('doppler.load expects a quickstart registry id, { url }, or { manifest, baseUrl? }.');
}

function countTokens(pipeline, text) {
  if (!text || typeof text !== 'string') return 0;
  try {
    return pipeline?.tokenizer?.encode(text)?.length ?? 0;
  } catch {
    return 0;
  }
}

function resolveChatPromptForUsage(pipeline, messages) {
  const templateType = pipeline?.manifest?.inference?.chatTemplate?.enabled === false
    ? null
    : (pipeline?.manifest?.inference?.chatTemplate?.type ?? null);
  try {
    return formatChatMessages(messages, templateType);
  } catch {
    return messages.map((message) => String(message?.content ?? '')).join('\n');
  }
}

async function collectText(iterable) {
  let output = '';
  for await (const token of iterable) {
    output += token;
  }
  return output;
}

function createModelHandle(pipeline, resolved) {
  return {
    generate(prompt, options = {}) {
      return pipeline.generate(prompt, options);
    },
    async generateText(prompt, options = {}) {
      return collectText(pipeline.generate(prompt, options));
    },
    chat(messages, options = {}) {
      return pipeline.generate(messages, options);
    },
    async chatText(messages, options = {}) {
      const content = await collectText(pipeline.generate(messages, options));
      const promptText = resolveChatPromptForUsage(pipeline, messages);
      const promptTokens = countTokens(pipeline, promptText);
      const completionTokens = countTokens(pipeline, content);
      return {
        content,
        usage: {
          promptTokens,
          completionTokens,
          totalTokens: promptTokens + completionTokens,
        },
      };
    },
    async loadLoRA(adapter) {
      const lora = typeof adapter === 'string'
        ? await loadLoRAFromUrl(adapter)
        : await loadLoRAFromManifest(adapter);
      pipeline.setLoRAAdapter(lora);
    },
    async unloadLoRA() {
      pipeline.setLoRAAdapter(null);
    },
    async unload() {
      await pipeline.unload();
    },
    get activeLoRA() {
      return pipeline.getActiveLoRA()?.name ?? null;
    },
    get loaded() {
      return pipeline.isLoaded === true;
    },
    get modelId() {
      return resolved.modelId;
    },
    get manifest() {
      return pipeline.manifest;
    },
    get deviceInfo() {
      return getKernelCapabilities()?.adapterInfo ?? null;
    },
    advanced: {
      prefillKV(prompt, options = {}) {
        return pipeline.prefillKVOnly(prompt, options);
      },
      prefillWithLogits(prompt, options = {}) {
        return pipeline.prefillWithLogits(prompt, options);
      },
      decodeStepLogits(currentIds, options = {}) {
        return pipeline.decodeStepLogits(currentIds, options);
      },
      generateWithPrefixKV(prefix, prompt, options = {}) {
        return pipeline.generateWithPrefixKV(prefix, prompt, options);
      },
    },
  };
}

export async function load(model, options = {}) {
  const { userProgress, pipelineProgress } = resolveLoadProgressHandlers(options);

  emitLoadProgress(userProgress, 'resolve', 5, 'Resolving model');
  const resolved = await resolveModelSource(model);
  await ensureWebGPUAvailable();

  emitLoadProgress(userProgress, 'manifest', 15, 'Fetching manifest');
  const manifest = resolved.manifest ?? await fetchManifestFromBaseUrl(resolved.baseUrl);

  emitLoadProgress(userProgress, 'load', 25, 'Loading weights');
  const pipeline = await createPipeline(manifest, {
    baseUrl: resolved.baseUrl ?? undefined,
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
  assertNoLoadAffectingOptions('doppler()', options);
  const model = await getCachedModel(options.model, { onProgress: options.onProgress });
  yield* model.generate(prompt, options);
}

export function doppler(prompt, options) {
  return dopplerGenerate(prompt, options);
}

doppler.load = load;

function assertNoLoadAffectingOptions(apiName, options) {
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

doppler.text = async function text(prompt, options = {}) {
  if (!options || typeof options !== 'object' || options.model == null) {
    throw new Error('doppler.text() requires options.model.');
  }
  assertNoLoadAffectingOptions('doppler.text()', options);
  const model = await getCachedModel(options.model, { onProgress: options.onProgress });
  return model.generateText(prompt, options);
};

doppler.chat = function chat(messages, options = {}) {
  if (!options || typeof options !== 'object' || options.model == null) {
    throw new Error('doppler.chat() requires options.model.');
  }
  assertNoLoadAffectingOptions('doppler.chat()', options);
  return (async function* run() {
    const model = await getCachedModel(options.model, { onProgress: options.onProgress });
    yield* model.chat(messages, options);
  }());
};

doppler.chatText = async function chatText(messages, options = {}) {
  if (!options || typeof options !== 'object' || options.model == null) {
    throw new Error('doppler.chatText() requires options.model.');
  }
  assertNoLoadAffectingOptions('doppler.chatText()', options);
  const model = await getCachedModel(options.model, { onProgress: options.onProgress });
  return model.chatText(messages, options);
};

doppler.evict = async function evict(model) {
  const resolved = await resolveModelSource(model);
  const cacheKey = resolved.modelId;
  const cached = convenienceModelCache.get(cacheKey);
  if (!cached) return false;
  await cached.unload();
  convenienceModelCache.delete(cacheKey);
  return true;
};

doppler.evictAll = async function evictAll() {
  const cachedModels = Array.from(convenienceModelCache.values());
  convenienceModelCache.clear();
  await Promise.allSettled(cachedModels.map((model) => model.unload()));
};

doppler.listModels = async function listModels() {
  const models = await listQuickstartModels();
  return models.map((entry) => entry.modelId);
};
