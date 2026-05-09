import { LORA_MODULE_ALIASES } from '../inference/pipelines/text/lora.js';

async function collectText(iterable) {
  let result = '';
  for await (const chunk of iterable) {
    if (typeof chunk === 'string') {
      result += chunk;
    } else if (chunk && typeof chunk.text === 'string') {
      result += chunk.text;
    }
  }
  return result;
}

function getPipelineModelId(pipeline, resolved = {}) {
  return String(
    resolved.modelId ||
    pipeline?.manifest?.meta?.modelId ||
    pipeline?.manifest?.modelId ||
    ''
  );
}

function asFloat32Array(value, label) {
  if (value instanceof Float32Array) return value;
  if (Array.isArray(value) || ArrayBuffer.isView(value)) {
    const out = new Float32Array(value);
    if (out.length > 0) return out;
  }
  throw new Error(`wrapPipelineAsDreamProvider: ${label} must be a non-empty numeric array.`);
}

function normalizeTargetModule(moduleName) {
  const key = String(moduleName || '').trim();
  const normalized = LORA_MODULE_ALIASES[key];
  if (!normalized) {
    throw new Error(`wrapPipelineAsDreamProvider: unsupported LoRA target module "${moduleName}".`);
  }
  return normalized;
}

function normalizeDreamLayerMap(layers, rank, alpha) {
  if (!(layers instanceof Map)) {
    throw new Error('wrapPipelineAsDreamProvider: attachLoraAdapter requires layers as a Map.');
  }
  const normalizedLayers = new Map();
  for (const [layerKey, moduleMap] of layers.entries()) {
    const layerIndex = Number(layerKey);
    if (!Number.isInteger(layerIndex) || layerIndex < 0) {
      throw new Error(`wrapPipelineAsDreamProvider: invalid LoRA layer index "${layerKey}".`);
    }
    if (!moduleMap || typeof moduleMap !== 'object' || Array.isArray(moduleMap)) {
      throw new Error(`wrapPipelineAsDreamProvider: layer ${layerIndex} module map must be an object.`);
    }
    const normalizedModules = {};
    for (const [moduleKey, weights] of Object.entries(moduleMap)) {
      const moduleName = normalizeTargetModule(moduleKey);
      const a = asFloat32Array(weights?.a, `layer ${layerIndex} ${moduleName}.a`);
      const b = asFloat32Array(weights?.b, `layer ${layerIndex} ${moduleName}.b`);
      normalizedModules[moduleName] = {
        a,
        b,
        rank,
        alpha,
        scale: rank > 0 ? alpha / rank : 1,
      };
    }
    if (Object.keys(normalizedModules).length === 0) {
      throw new Error(`wrapPipelineAsDreamProvider: layer ${layerIndex} has no LoRA modules.`);
    }
    normalizedLayers.set(layerIndex, normalizedModules);
  }
  if (normalizedLayers.size === 0) {
    throw new Error('wrapPipelineAsDreamProvider: attachLoraAdapter requires at least one layer.');
  }
  return normalizedLayers;
}

function normalizeDreamLoraAdapter(adapter) {
  const adapterId = String(adapter?.adapterId || adapter?.name || '').trim();
  if (!adapterId) {
    throw new Error('wrapPipelineAsDreamProvider: attachLoraAdapter requires adapterId.');
  }
  const rank = Number(adapter?.rank);
  if (!Number.isInteger(rank) || rank <= 0) {
    throw new Error('wrapPipelineAsDreamProvider: attachLoraAdapter requires positive integer rank.');
  }
  const alpha = Number.isFinite(Number(adapter?.alpha))
    ? Number(adapter.alpha)
    : Number(adapter?.scale) * rank;
  if (!Number.isFinite(alpha)) {
    throw new Error('wrapPipelineAsDreamProvider: attachLoraAdapter requires alpha or scale.');
  }
  const targetModules = Array.isArray(adapter?.targetModules)
    ? adapter.targetModules.map(normalizeTargetModule)
    : [];
  return {
    name: adapterId,
    version: adapter?.version,
    baseModel: adapter?.baseModel || adapter?.baseModelId,
    rank,
    alpha,
    targetModules,
    layers: normalizeDreamLayerMap(adapter.layers, rank, alpha),
  };
}

/**
 * Adapts a raw Doppler pipeline into the model handle shape that
 * createDopplerProvider() expects, without triggering a load.
 *
 * @param {Object} pipeline - A loaded Doppler pipeline (has generate, manifest, etc.)
 * @param {{ modelId?: string, manifest?: Object, deviceInfo?: Object }} [resolved]
 * @returns {{ loaded: boolean, modelId: string, manifest: Object|null, deviceInfo: Object|null, generateText(prompt: unknown, opts?: Object): Promise<string>, unload(): Promise<void> }}
 */
export function wrapPipelineAsHandle(pipeline, resolved = {}) {
  if (!pipeline || typeof pipeline.generate !== 'function') {
    throw new Error('wrapPipelineAsHandle requires a loaded pipeline with a generate() method.');
  }

  return {
    get loaded() {
      return pipeline.isLoaded === true;
    },
    get modelId() {
      return getPipelineModelId(pipeline, resolved);
    },
    get manifest() {
      return pipeline.manifest || resolved.manifest || null;
    },
    get deviceInfo() {
      return resolved.deviceInfo || null;
    },
    get supportsEmbedding() {
      const manifest = pipeline.manifest || resolved.manifest || null;
      return manifest?.modelType === 'embedding'
        || manifest?.inference?.supportsEmbedding === true;
    },
    get supportsTranscription() {
      const manifest = pipeline.manifest || resolved.manifest || null;
      return manifest?.inference?.supportsTranscription === true
        && pipeline.audioCapable === true;
    },
    get supportsVision() {
      const manifest = pipeline.manifest || resolved.manifest || null;
      return manifest?.inference?.supportsVision === true
        && pipeline.visionCapable === true;
    },
    async generateText(prompt, opts = {}) {
      return collectText(pipeline.generate(prompt, opts));
    },
    async embed(prompt, options = {}) {
      return pipeline.embed(prompt, options);
    },
    async embedBatch(prompts, options = {}) {
      return pipeline.embedBatch(prompts, options);
    },
    async embedImage(args = {}) {
      return pipeline.embedImage(args);
    },
    async embedAudio(args = {}) {
      return pipeline.embedAudio(args);
    },
    async transcribeImage(args = {}) {
      return pipeline.transcribeImage(args);
    },
    async transcribeAudio(args = {}) {
      return pipeline.transcribeAudio(args);
    },
    async transcribeVideo(args = {}) {
      return pipeline.transcribeVideo(args);
    },
    async unload() {
      if (typeof pipeline.unload === 'function') {
        await pipeline.unload();
      }
    },
  };
}

export function wrapPipelineAsDreamProvider(pipeline, resolved = {}) {
  if (!pipeline || typeof pipeline.generate !== 'function') {
    throw new Error('wrapPipelineAsDreamProvider requires a loaded pipeline with a generate() method.');
  }
  if (typeof pipeline.setLoRAAdapter !== 'function') {
    throw new Error('wrapPipelineAsDreamProvider requires pipeline.setLoRAAdapter(adapter).');
  }
  if (typeof pipeline.getActiveLoRA !== 'function') {
    throw new Error('wrapPipelineAsDreamProvider requires pipeline.getActiveLoRA().');
  }

  const adapters = new Map();

  return {
    get modelId() {
      return getPipelineModelId(pipeline, resolved);
    },
    get manifest() {
      return pipeline.manifest || resolved.manifest || null;
    },
    get backend() {
      return 'doppler';
    },
    get device() {
      return resolved.device || pipeline.gpuContext?.device || pipeline.device || null;
    },
    async attachLoraAdapter(adapter) {
      const normalized = normalizeDreamLoraAdapter(adapter);
      adapters.set(normalized.name, normalized);
      pipeline.setLoRAAdapter(normalized);
      return {
        adapterId: normalized.name,
        rank: normalized.rank,
        alpha: normalized.alpha,
        scale: normalized.rank > 0 ? normalized.alpha / normalized.rank : 1,
        targetModules: normalized.targetModules,
        layerCount: normalized.layers.size,
      };
    },
    async detachLoraAdapter(adapterId = null) {
      const id = adapterId == null ? null : String(adapterId);
      if (id && !adapters.has(id)) {
        throw new Error(`wrapPipelineAsDreamProvider: cannot detach unknown LoRA adapter "${id}".`);
      }
      if (id) adapters.delete(id);
      if (!id) adapters.clear();
      pipeline.setLoRAAdapter(null);
      return { detached: true, adapterId: id };
    },
    async generate(request = {}) {
      const payload = typeof request === 'string' ? { prompt: request } : request;
      const prompt = payload?.prompt;
      if (prompt == null) {
        throw new Error('wrapPipelineAsDreamProvider.generate requires prompt.');
      }
      const loraAdapterId = payload?.loraAdapterId == null ? null : String(payload.loraAdapterId);
      const samplingOptions = payload?.samplingOptions && typeof payload.samplingOptions === 'object'
        ? payload.samplingOptions
        : {};
      const targetAdapter = loraAdapterId ? adapters.get(loraAdapterId) : null;
      if (loraAdapterId && !targetAdapter) {
        throw new Error(`wrapPipelineAsDreamProvider.generate requested unknown LoRA adapter "${loraAdapterId}".`);
      }
      const previousAdapter = pipeline.getActiveLoRA();
      pipeline.setLoRAAdapter(targetAdapter);
      try {
        const text = await collectText(pipeline.generate(prompt, samplingOptions));
        return {
          text,
          useLora: Boolean(targetAdapter),
          baseModelId: getPipelineModelId(pipeline, resolved),
          loraAdapterId,
        };
      } finally {
        pipeline.setLoRAAdapter(previousAdapter || null);
      }
    },
  };
}
