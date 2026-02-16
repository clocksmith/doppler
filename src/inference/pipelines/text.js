
import { getDevice } from '../../gpu/device.js';
import { getBufferPool as getGlobalBufferPool } from '../../memory/buffer-pool.js';
import { log } from '../../debug/index.js';
import { setActiveKernelPath } from '../../config/kernel-path-loader.js';
import { configurePerfGuards } from '../../gpu/perf-guards.js';
import { MoERouter } from '../moe-router.js';
import { DecodeBufferManager } from '../decode-buffers.js';
import { DecodeRing } from '../decode-ring.js';
import { applyPipelineContexts } from './context.js';
import { createInitializedPipeline } from './factory.js';

// Pipeline sub-modules
import { PipelineState } from './text/state.js';
import { PipelineGenerator } from './text/generator.js';
import { parseModelConfig } from './text/config.js';
import {
  initRoPEFrequencies,
  createKVCache,
  loadWeights,
  initMoERouter,
  initSpeculativeDecoder,
  fuseQKVWeights,
  initEmulation,
  destroyEmulation,
} from './text/init.js';
import {
  runKernelWarmup,
  resolveAndActivateKernelPath,
  initTokenizerFromManifestPreset,
} from './text/model-load.js';
import { applyPipelineDebugConfig } from './text/debug-utils.js';
import { resolveLayerPipeline } from './text/layer-plan.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { registerPipeline, getPipelineFactory } from './registry.js';



// ============================================================================
// Main Inference Pipeline Class
// ============================================================================

export class InferencePipeline extends PipelineState {
  
  generator;

  // Progress callback
  
  _onProgress = null;

  
  _preloadedWeights = null;

  constructor() {
    super();
    this.generator = new PipelineGenerator(this);
    this.decodeBuffers = new DecodeBufferManager();
    this.decodeRing = new DecodeRing();
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================

  
  async initialize(contexts = {}) {
    const { runtimeConfig, sharedDebug } = applyPipelineContexts(this, contexts, {
      assignGpuContext: true,
      assignUseGPU: true,
      assignMemoryContext: true,
      assignStorageContext: true,
    });
    this.runtimeConfig = runtimeConfig;
    applyPipelineDebugConfig(sharedDebug?.pipeline);
    configurePerfGuards(sharedDebug?.perfGuards);

    if (contexts.runtime?.kernelPath) {
      this.runtimeKernelPath = contexts.runtime.kernelPath;
    }

    this.emulation = await initEmulation(this.runtimeConfig);

    this.debug = sharedDebug?.pipeline?.enabled === true;
    log.debug('Pipeline', 'Initialized', { useGPU: this.useGPU, debug: this.debug });
  }

  
  async loadModel(manifest) {
    this.manifest = manifest;
    this.decodeRing?.release();
    // Pass runtime model overrides to merge with manifest inference config
    const modelOverrides =  (this.runtimeConfig.inference.modelOverrides);
    this.modelConfig = parseModelConfig(manifest, modelOverrides);
    this.useTiedEmbeddings = this.modelConfig.useTiedEmbeddings;
    this.embeddingVocabSize = this.modelConfig.embeddingVocabSize;
    this.embeddingTranspose = this.modelConfig.embeddingTranspose;

    await runKernelWarmup({
      useGPU: this.useGPU,
      kernelWarmup: this.runtimeConfig.shared?.kernelWarmup,
      modelConfig: this.modelConfig,
    });

    const kernelPathState = resolveAndActivateKernelPath({
      manifest,
      runtimeKernelPath: this.runtimeKernelPath,
      runtimeConfig: this.runtimeConfig,
      modelConfig: this.modelConfig,
    });
    this.resolvedKernelPath = kernelPathState.resolvedKernelPath;
    this.kernelPathSource = kernelPathState.kernelPathSource;
    this.runtimeConfig = kernelPathState.runtimeConfig;

    this._resolveLayerPipeline();

    const cfg = this.modelConfig;
    const moeStr = cfg.useMoE ? `, MoE(${cfg.numExperts}x${cfg.moeTopK})` : '';
    const kernelInfo = this.resolvedKernelPath ? `kernelPath=${this.resolvedKernelPath.id}` : 'kernelPath=none';
    log.info('Pipeline', `${cfg.numLayers}L/${cfg.hiddenSize}H/${cfg.numHeads}heads (${cfg.headDim}dim)${moeStr}, ${kernelInfo}`);

    this.tokenizer = await initTokenizerFromManifestPreset(manifest, this.baseUrl);
    const tokenizerVocabSize = this.tokenizer.getVocabSize();
    if (Number.isFinite(tokenizerVocabSize) && tokenizerVocabSize > 0) {
      if (tokenizerVocabSize !== this.modelConfig.vocabSize) {
        log.info('Pipeline', `Tokenizer vocabSize=${tokenizerVocabSize} differs from model=${this.modelConfig.vocabSize}, using model size`);
      }
    }

    // Initialize KV cache
    this.kvCache = createKVCache(this.modelConfig, this.useGPU, this.debug, this.runtimeConfig.inference.kvcache);

    // Initialize MoE router if needed
    if (this.modelConfig.useMoE) {
      this.moeRouter = new MoERouter({
        numExperts: this.modelConfig.numExperts,
        topK: this.modelConfig.moeTopK,
        hiddenSize: this.modelConfig.hiddenSize,
        normalizeWeights: this.runtimeConfig.inference.moe.routing.normalizeWeights,
      });
    }

    // Initialize speculative decoder
    if (manifest.draftModel) {
      this.speculativeDecoder = initSpeculativeDecoder(
        manifest,
        this.runtimeConfig.inference.speculative
      );
    }

    // Load weights
    await this._loadWeights();

    // Initialize RoPE frequencies
    await this._initRoPE();

    this.isLoaded = true;
    log.info('Pipeline', 'Model loaded successfully');
  }

  
  async _loadWeights() {
    const result = this._preloadedWeights || await loadWeights(
       (this.manifest),
       (this.modelConfig),
      {
        storageContext: this.storageContext ?? undefined,
        loadingConfig: this.runtimeConfig.loading,
        baseUrl: this.baseUrl ?? undefined,
        onProgress: ( info) => {
          if (info.stage !== 'layers' && info.stage !== 'shards') {
            log.verbose('Loader', `${info.stage}: ${Math.round(info.progress * 100)}%${info.message ? ` - ${info.message}` : ''}`);
          }
          if (this._onProgress) {
            this._onProgress({
              percent: info.progress * 100,
              message: info.message,
              stage: info.stage,
              layer: info.layer,
              total: info.total,
            });
          }
        },
      }
    );

    result.layerWeights.forEach((w, k) => this.weights.set(k, w));
    this.weights.set('embed', result.embeddings);
    this.weights.set('lm_head', result.lmHead);
    this.weights.set('final_norm', result.finalNorm);

    this.layerRouterWeights = result.layerRouterWeights;

    this.dopplerLoader = getDopplerLoader(this.runtimeConfig.loading);

    if ( (this.modelConfig).useMoE && this.moeRouter) {
      this.moeRouter = initMoERouter(
         (this.modelConfig),
        this.runtimeConfig.inference.moe.routing,
        result.layerWeights
      );
    }

    if (this.useGPU && this.modelConfig) {
      fuseQKVWeights(result.layerWeights, this.modelConfig);
    }

    if (this.useGPU && this.modelConfig) {
      this.decodeBuffers?.ensureBuffers({
        hiddenSize: this.modelConfig.hiddenSize,
        intermediateSize: this.modelConfig.intermediateSize,
        activationDtype: this.runtimeConfig.inference.compute.activationDtype,
        enablePingPong: true,
      });
    }
  }

  
  setPreloadedWeights(weights) {
    this._preloadedWeights = weights;
  }

  
  async _initRoPE() {
    const config =  (this.modelConfig);
    const maxSeqLen = config.maxSeqLen;
    const ropeBuffers = await initRoPEFrequencies({
      headDim: config.headDim,
      maxSeqLen,
      ropeTheta: config.ropeTheta,
      ropeLocalTheta: config.ropeLocalTheta,
      ropeScale: config.ropeScale,
      ropeScalingType: config.ropeScalingType,
      ropeScaling: config.ropeScaling,
    }, this.useGPU);
    this.ropeFreqsCos = ropeBuffers.cos;
    this.ropeFreqsSin = ropeBuffers.sin;
    this.ropeLocalCos = ropeBuffers.localCos ?? null;
    this.ropeLocalSin = ropeBuffers.localSin ?? null;
  }

  
  _resolveLayerPipeline() {
    if (!this.modelConfig) return;
    const runtimePlan = this.runtimeConfig.inference.pipeline ?? null;
    const modelPlan = this.modelConfig.layerPipeline ?? null;
    this.layerPipelinePlan = resolveLayerPipeline(modelPlan, runtimePlan, this.modelConfig.numLayers);
    if (this.layerPipelinePlan) {
      log.info(
        'Pipeline',
        `Layer pipeline plan enabled (source=${this.layerPipelinePlan.source}, steps=${this.layerPipelinePlan.steps.length}, overrides=${this.layerPipelinePlan.overrides.length})`
      );
    }
  }

  // ==========================================================================
  // Generation Delegates
  // ==========================================================================

  
  generate(prompt, options = {}) {
    return this.generator.generate(prompt, options);
  }

  decodeStepLogits(currentIds, options = {}) {
    return this.generator.decodeStepLogits(currentIds, options);
  }

  advanceWithToken(tokenId, options = {}) {
    return this.generator.advanceWithToken(tokenId, options);
  }

  advanceWithTokenAndEmbedding(tokenId, options = {}) {
    return this.generator.advanceWithTokenAndEmbedding(tokenId, options);
  }

  
  prefillKVOnly(prompt, options = {}) {
    return this.generator.prefillKVOnly(prompt, options);
  }

  prefillWithEmbedding(prompt, options = {}) {
    return this.generator.prefillWithEmbedding(prompt, options);
  }

  async embed(prompt, options = {}) {
    const result = await this.prefillWithEmbedding(prompt, options);
    return {
      embedding: result.embedding,
      tokens: result.tokens,
      seqLen: result.seqLen,
      embeddingMode: result.embeddingMode,
    };
  }

  async embedBatch(prompts, options = {}) {
    if (!Array.isArray(prompts)) {
      throw new Error('embedBatch expects an array of prompts');
    }
    const outputs = [];
    for (const prompt of prompts) {
      outputs.push(await this.embed(prompt, options));
      this.reset();
    }
    return outputs;
  }

  prefillWithLogits(prompt, options = {}) {
    return this.generator.prefillWithLogits(prompt, options);
  }

  
  applyKVCacheSnapshot(snapshot) {
    this.kvCache = snapshot.cache.clone();
    if (this.useGPU && this.kvCache) {
      const device = getDevice();
      if (device) {
        this.kvCache.setGPUContext({ device });
      }
    }
    this.currentSeqLen = snapshot.seqLen;
  }

  
  generateWithPrefixKV(prefix, prompt, options = {}) {
    return this.generator.generateWithPrefixKV(prefix, prompt, options);
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  
  getStats() {
    const stats = { ...this.stats };
    const ringStats = this.decodeRing?.getStats();
    if (ringStats) {
      stats.decodeRing = ringStats;
    }
    return stats;
  }

  
  getBatchingStats() {
    return { ...this.batchingStats };
  }

  
  getMemoryStats() {
    
    const stats = { used: 0 };

    try {
      const poolStats = getGlobalBufferPool().getStats();
      stats.pool = poolStats;
      stats.used += poolStats.currentBytesAllocated || 0;
    } catch {
      // Buffer pool not initialized yet
    }

    if (this.kvCache) {
      const kvStats = this.kvCache.getMemoryStats();
      stats.kvCache = kvStats;
      stats.used += kvStats.allocated || 0;
    }

    if (this.emulation?.config?.statsEnabled) {
      stats.emulation = this.emulation.getStats();
    }

    return stats;
  }

  
  getKVCacheStats() {
    if (!this.kvCache) return null;
    const { seqLen, maxSeqLen } = this.kvCache.getMemoryStats();
    return { seqLen, maxSeqLen };
  }

  
  getBufferPool() {
    try {
      return getGlobalBufferPool();
    } catch {
      return null;
    }
  }

  
  async unload() {
    await destroyEmulation(this.emulation);
    this.emulation = null;
    this.decodeRing?.release();
    this.kvCache?.clear();
    this.weights.clear();
    this.expertWeights.clear();
    this.lora = null;
    setActiveKernelPath(null, 'none');
    this.isLoaded = false;
    this.currentSeqLen = 0;
    log.info('Pipeline', 'Unloaded');
  }

  
  setLoRAAdapter(adapter) {
    this.lora = adapter;
  }

  
  getActiveLoRA() {
    return this.lora;
  }

  
  reset() {
    this.kvCache?.clear();
    this.currentSeqLen = 0;
    this.decodeStepCount = 0;
    this.debugFlags = {};
    this.decodeBuffers?.resetPingPong();
    this.decodeRing?.reset();
    // Reset stats
    this.stats.tokensGenerated = 0;
    this.stats.totalTimeMs = 0;
    this.stats.prefillTimeMs = 0;
    this.stats.decodeTimeMs = 0;
    this.stats.gpuTimePrefillMs = undefined;
    this.stats.gpuTimeDecodeMs = undefined;
    this.stats.decodeProfileSteps = [];
    this.stats.attentionInputs = [];
  }

  
  releaseGPUResources() {
    this.decodeBuffers?.release();
    this.decodeRing?.release();
  }
}

// ============================================================================
// Factory Function
// ============================================================================


async function createTransformerPipeline(manifest, contexts = {}) {
  return createInitializedPipeline(InferencePipeline, manifest, contexts);
}

registerPipeline('transformer', createTransformerPipeline);

export class EmbeddingPipeline extends InferencePipeline {
  async *generate() {
    throw new Error('Embedding pipeline does not support token generation. Use embed() or prefillWithEmbedding().');
  }
}

async function createEmbeddingPipeline(manifest, contexts = {}) {
  return createInitializedPipeline(EmbeddingPipeline, manifest, contexts);
}

registerPipeline('embedding', createEmbeddingPipeline);

export async function createPipeline(manifest, contexts = {}) {
  const modelType = manifest?.modelType;
  if (typeof modelType !== 'string' || modelType.length === 0) {
    throw new Error('Manifest is missing modelType. Re-convert the model with modelType set.');
  }
  let factory = getPipelineFactory(modelType);

  if (!factory && modelType === 'diffusion') {
    await import('./diffusion/pipeline.js');
    factory = getPipelineFactory(modelType);
  }

  if (!factory && modelType === 'energy') {
    await import('./energy/pipeline.js');
    factory = getPipelineFactory(modelType);
  }

  if (!factory) {
    throw new Error(`No pipeline registered for modelType "${modelType}".`);
  }

  return factory(manifest, contexts);
}

export { InferencePipeline as Pipeline };
