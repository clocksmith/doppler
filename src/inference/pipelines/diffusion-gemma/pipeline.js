import { log } from '../../../debug/index.js';
import { applyPipelineContexts, restorePipelineContexts } from '../context.js';
import { createInitializedPipeline } from '../factory.js';
import { registerPipeline } from '../registry.js';
import { InferencePipeline } from '../text.js';
import { initTokenizerFromManifest } from '../text/model-load.js';
import { parseDiffusionGemmaConfig } from './config.js';
import { createSeededRandom, denoiseCanvas, denoiseCanvasWithStatsProvider } from './sampling.js';

function createSystemRandom() {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    return () => {
      const value = new Uint32Array(1);
      crypto.getRandomValues(value);
      return value[0] / 0x100000000;
    };
  }
  return Math.random;
}

function normalizeInputIds(value, tokenizer, contextLabel) {
  if (ArrayBuffer.isView(value)) {
    return Int32Array.from(value);
  }
  if (Array.isArray(value)) {
    return Int32Array.from(value);
  }
  if (typeof value === 'string') {
    if (!tokenizer || typeof tokenizer.encode !== 'function') {
      throw new Error(`${contextLabel}: tokenizer.encode is required for string prompts.`);
    }
    const encoded = tokenizer.encode(value);
    return Int32Array.from(ArrayBuffer.isView(encoded) ? Array.from(encoded) : encoded);
  }
  if (value == null) {
    return new Int32Array(0);
  }
  throw new Error(`${contextLabel}: prompt/inputIds must be a string, array, or typed array.`);
}

function resolveRandom(options) {
  if (typeof options.random === 'function') {
    return options.random;
  }
  if (options.seed !== undefined && options.seed !== null) {
    return createSeededRandom(options.seed);
  }
  return createSystemRandom();
}

function resolveMaxNewTokens(config, value) {
  if (value === undefined || value === null) {
    return config.maxNewTokens;
  }
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`DiffusionGemma maxNewTokens must be a positive integer; got ${String(value)}.`);
  }
  return value;
}

function resolveCanvasLength(config, value) {
  if (value === undefined || value === null) {
    return config.canvasLength;
  }
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`DiffusionGemma canvasLength must be a positive integer; got ${String(value)}.`);
  }
  if (value > config.canvasLength) {
    throw new Error(
      `DiffusionGemma canvasLength ${value} exceeds manifest canvasLength ${config.canvasLength}.`
    );
  }
  return value;
}

function resolveMaxDenoisingSteps(config, value) {
  if (value === undefined || value === null) {
    return config.maxDenoisingSteps;
  }
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`DiffusionGemma maxDenoisingSteps must be a positive integer; got ${String(value)}.`);
  }
  if (value > config.maxDenoisingSteps) {
    throw new Error(
      `DiffusionGemma maxDenoisingSteps ${value} exceeds manifest maxDenoisingSteps ${config.maxDenoisingSteps}.`
    );
  }
  return value;
}

function decodeToken(tokenizer, tokenId, options) {
  if (typeof options.decodeToken === 'function') {
    return options.decodeToken(tokenId);
  }
  if (tokenizer && typeof tokenizer.decode === 'function') {
    return tokenizer.decode([tokenId]);
  }
  return String(tokenId);
}

function createStats(modelLoadMs = 0) {
  return {
    canvasesGenerated: 0,
    tokensGenerated: 0,
    denoiseSteps: 0,
    modelLoadMs,
    totalTimeMs: 0,
    prefillTimeMs: 0,
    decodeTimeMs: 0,
    prefillTokens: 0,
    decodeTokens: 0,
    tokensPerForward: 0,
    stopReason: null,
    stopTokenId: null,
  };
}

function releaseSelfConditioningState(state) {
  if (state && !ArrayBuffer.isView(state) && typeof state.release === 'function') {
    state.release();
  }
}

export class DiffusionGemmaPipeline {
  constructor() {
    this.manifest = null;
    this.config = null;
    this.runtimeConfig = null;
    this.runtimeOverrides = null;
    this.tokenizer = null;
    this.logitsProvider = null;
    this.diffusionGemmaContext = null;
    this.pipelineContexts = null;
    this.corePipeline = null;
    this.ownsCorePipeline = false;
    this.isLoaded = false;
    this.stats = createStats();
  }

  async initialize(contexts = {}) {
    const { runtimeConfig } = applyPipelineContexts(this, contexts, {
      assignGpuContext: true,
      assignUseGPU: true,
      assignMemoryContext: true,
      assignStorageContext: true,
    });
    this.runtimeConfig = runtimeConfig;
    this.pipelineContexts = contexts;
    this.runtimeOverrides = contexts.runtimeConfig == null
      ? null
      : (typeof structuredClone === 'function'
        ? structuredClone(contexts.runtimeConfig)
        : JSON.parse(JSON.stringify(contexts.runtimeConfig)));
    this.diffusionGemmaContext = contexts.diffusionGemma ?? null;
    this.logitsProvider = this.diffusionGemmaContext?.logitsProvider ?? null;
    if (typeof this.logitsProvider !== 'function') {
      const providedCore = this.diffusionGemmaContext?.corePipeline ?? null;
      this.corePipeline = providedCore ?? new InferencePipeline();
      this.ownsCorePipeline = providedCore == null;
      if (typeof this.corePipeline.initialize === 'function') {
        await this.corePipeline.initialize(contexts);
      }
    }
  }

  async loadModel(manifest) {
    const loadStart = performance.now();
    this.manifest = manifest;
    this.config = parseDiffusionGemmaConfig(manifest);
    if (this.corePipeline) {
      await this.corePipeline.loadModel(manifest);
    }
    this.tokenizer = this.diffusionGemmaContext?.tokenizer
      ?? this.corePipeline?.tokenizer
      ?? await initTokenizerFromManifest(
        manifest,
        this.baseUrl,
        this.storageContext
      );
    this.isLoaded = true;
    this.stats = createStats(performance.now() - loadStart);
    log.info(
      'DiffusionGemma',
      `Loaded block-diffusion contract: canvas=${this.config.canvasLength}, ` +
      `steps=${this.config.maxDenoisingSteps}, vocab=${this.config.vocabSize}`
    );
  }

  assertReady() {
    if (!this.isLoaded || !this.config) {
      throw new Error('DiffusionGemma pipeline is not loaded.');
    }
  }

  resolveCoreOptions(options) {
    return {
      ...options,
      __internalGenerate: true,
      useChatTemplate: false,
    };
  }

  async resetCoreEncoder(inputIds, options) {
    if (!this.corePipeline) {
      throw new Error('DiffusionGemma internal core pipeline is not initialized.');
    }
    this.corePipeline.resetToSeqLen(0);
    if (inputIds.length === 0) {
      return;
    }
    await this.corePipeline.prefillKVOnly('', {
      ...this.resolveCoreOptions(options),
      inputIds: Array.from(inputIds),
    });
  }

  async appendCoreEncoderTokens(tokenIds, options) {
    if (!this.corePipeline) {
      throw new Error('DiffusionGemma internal core pipeline is not initialized.');
    }
    if (tokenIds.length === 0) {
      return;
    }
    await this.corePipeline.prefillKVOnly('', {
      ...this.resolveCoreOptions(options),
      inputIds: Array.from(tokenIds),
    });
  }

  async generateTokenIds(prompt, options = {}) {
    this.assertReady();
    const runStart = performance.now();
    const modelLoadMs = this.stats.modelLoadMs;
    this.stats = createStats(modelLoadMs);
    let logitsProvider = options.logitsProvider ?? this.logitsProvider;

    const inputIds = normalizeInputIds(
      options.inputIds ?? prompt,
      this.tokenizer,
      'DiffusionGemma generateTokenIds'
    );
    const random = resolveRandom(options);
    const maxNewTokens = resolveMaxNewTokens(this.config, options.maxNewTokens);
    const runConfig = {
      ...this.config,
      canvasLength: resolveCanvasLength(this.config, options.canvasLength),
      maxDenoisingSteps: resolveMaxDenoisingSteps(this.config, options.maxDenoisingSteps),
    };
    const generated = [];
    let currentInputIds = Int32Array.from(inputIds);
    let selfConditioningLogits = options.selfConditioningLogits ?? null;
    const eos = new Set(runConfig.eosTokenIds);
    const maxCanvases = Math.ceil(maxNewTokens / runConfig.canvasLength);
    const useInternalCore = typeof logitsProvider !== 'function';
    const coreOptions = this.resolveCoreOptions(options);
    let prefillTimeMs = 0;
    let decodeTimeMs = 0;
    let prefillTokens = 0;
    let stopReason = 'max_new_tokens';
    let stopTokenId = null;
    if (useInternalCore) {
      const prefillStart = performance.now();
      await this.resetCoreEncoder(currentInputIds, coreOptions);
      prefillTimeMs += performance.now() - prefillStart;
      prefillTokens += currentInputIds.length;
      logitsProvider = ({ canvas, selfConditioningLogits: logits }) =>
        this.corePipeline.computeDiffusionGemmaCanvasLogits(
          { canvas, selfConditioningLogits: logits },
          coreOptions
        );
    }

    for (let canvasIndex = 0; canvasIndex < maxCanvases; canvasIndex += 1) {
      const decodeStart = performance.now();
      const useStatsProvider = useInternalCore
        && typeof this.corePipeline?.computeDiffusionGemmaCanvasStep === 'function';
      const result = useStatsProvider
        ? await denoiseCanvasWithStatsProvider(runConfig, {
          statsProvider: ({ canvas, selfConditioningLogits: logits, temperature }) =>
            this.corePipeline.computeDiffusionGemmaCanvasStep(
              { canvas, selfConditioningLogits: logits, temperature },
              coreOptions
            ),
          random,
          initialCanvas: canvasIndex === 0 ? (options.initialCanvas ?? null) : null,
          selfConditioningLogits: canvasIndex === 0 ? selfConditioningLogits : null,
          canvasIndex,
          inputIds: currentInputIds,
        })
        : await denoiseCanvas(runConfig, {
          logitsProvider,
          random,
          initialCanvas: canvasIndex === 0 ? (options.initialCanvas ?? null) : null,
          selfConditioningLogits: canvasIndex === 0 ? selfConditioningLogits : null,
          canvasIndex,
          inputIds: currentInputIds,
        });
      releaseSelfConditioningState(result.selfConditioningLogits);
      decodeTimeMs += performance.now() - decodeStart;
      const canvasTokens = result.argmaxCanvas ?? result.canvas;
      this.stats.canvasesGenerated += 1;
      this.stats.denoiseSteps += result.stepsRun ?? runConfig.maxDenoisingSteps;

      const nextInputIds = new Int32Array(currentInputIds.length + canvasTokens.length);
      nextInputIds.set(currentInputIds, 0);
      nextInputIds.set(canvasTokens, currentInputIds.length);
      currentInputIds = nextInputIds;

      let hitEos = false;
      for (let i = 0; i < canvasTokens.length && generated.length < maxNewTokens; i += 1) {
        const tokenId = canvasTokens[i];
        if (eos.has(tokenId)) {
          stopReason = 'eos';
          stopTokenId = tokenId;
          hitEos = true;
          break;
        }
        if (tokenId !== runConfig.padTokenId) {
          generated.push(tokenId);
          this.stats.tokensGenerated += 1;
        }
      }
      if (hitEos) {
        break;
      }

      if (useInternalCore && generated.length < maxNewTokens && canvasIndex + 1 < maxCanvases) {
        const appendStart = performance.now();
        await this.appendCoreEncoderTokens(canvasTokens, coreOptions);
        prefillTimeMs += performance.now() - appendStart;
        prefillTokens += canvasTokens.length;
      }
    }

    this.stats.totalTimeMs = performance.now() - runStart;
    this.stats.prefillTimeMs = prefillTimeMs;
    this.stats.decodeTimeMs = decodeTimeMs;
    this.stats.prefillTokens = prefillTokens;
    this.stats.decodeTokens = generated.length;
    this.stats.tokensGenerated = generated.length;
    this.stats.tokensPerForward = this.stats.denoiseSteps > 0
      ? generated.length / this.stats.denoiseSteps
      : 0;
    this.stats.stopReason = stopReason;
    this.stats.stopTokenId = stopTokenId;
    return Int32Array.from(generated);
  }

  async *generate(prompt, options = {}) {
    const tokenIds = await this.generateTokenIds(prompt, options);
    for (const tokenId of tokenIds) {
      yield decodeToken(this.tokenizer, tokenId, options);
    }
  }

  getStats() {
    return { ...this.stats };
  }

  async unload() {
    this.manifest = null;
    this.config = null;
    this.tokenizer = null;
    this.logitsProvider = null;
    this.diffusionGemmaContext = null;
    if (this.ownsCorePipeline && this.corePipeline && typeof this.corePipeline.unload === 'function') {
      await this.corePipeline.unload();
    }
    this.corePipeline = null;
    this.ownsCorePipeline = false;
    this.isLoaded = false;
    restorePipelineContexts(this);
  }
}

async function createDiffusionGemmaPipeline(manifest, contexts = {}) {
  return createInitializedPipeline(DiffusionGemmaPipeline, manifest, contexts);
}

registerPipeline('diffusion_gemma', createDiffusionGemmaPipeline);
