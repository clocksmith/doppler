

import { getDevice, setDevice } from '../gpu/device.js';
import { getBufferPool as getGlobalBufferPool } from '../memory/buffer-pool.js';
import { markWarmed as markKernelCacheWarmed } from '../gpu/kernel-selection-cache.js';
import { log, applyDebugConfig, setGPUDevice } from '../debug/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import { resolvePreset } from '../config/loader.js';
import {
  resolveKernelPath,
  getKernelPathStats,
  getKernelPathActivationDtype,
  setActiveKernelPath,
  applyKernelOverrides,
} from '../config/kernel-path-loader.js';
import { configurePerfGuards } from '../gpu/perf-guards.js';
import { autoTuneKernels, prewarmKernels } from '../gpu/kernels/index.js';
import { MoERouter } from './moe-router.js';
import { DecodeBufferManager } from './decode-buffers.js';
import { DecodeRing } from './decode-ring.js';

// Pipeline sub-modules
import { PipelineState } from './pipeline/state.js';
import { PipelineGenerator } from './pipeline/generator.js';
import { parseModelConfig } from './pipeline/config.js';
import {
  initRoPEFrequencies,
  createKVCache,
  initTokenizer,
  loadWeights,
  initMoERouter,
  initSpeculativeDecoder,
  fuseQKVWeights,
  initEmulation,
  destroyEmulation,
} from './pipeline/init.js';
import { applyPipelineDebugConfig } from './pipeline/debug-utils.js';
import { resolveLayerPipeline } from './pipeline/layer-plan.js';
import { getDopplerLoader } from '../loader/doppler-loader.js';
import { registerPipeline, getPipelineFactory } from './pipeline/registry.js';



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
    if (contexts.runtimeConfig) {
      this.runtimeConfig = setRuntimeConfig(contexts.runtimeConfig);
    } else {
      this.runtimeConfig = getRuntimeConfig();
    }
    const sharedDebug = this.runtimeConfig.shared.debug;
    applyDebugConfig(sharedDebug);
    applyPipelineDebugConfig(sharedDebug.pipeline);
    configurePerfGuards(sharedDebug.perfGuards);

    if (contexts.gpu?.device) {
      this.gpuContext = { device: contexts.gpu.device };
      this.useGPU = true;
      setDevice(contexts.gpu.device);
      setGPUDevice(contexts.gpu.device);
    }
    if (contexts.memory) this.memoryContext = contexts.memory;
    if (contexts.storage) this.storageContext = contexts.storage;
    if (contexts.baseUrl) this.baseUrl = contexts.baseUrl;

    if (contexts.runtime?.kernelPath) {
      this.runtimeKernelPath = contexts.runtime.kernelPath;
    }
    if (contexts.onProgress) this._onProgress = contexts.onProgress;

    if (!contexts.gpu?.device) {
      const device = getDevice();
      if (device) setGPUDevice(device);
    }

    this.emulation = await initEmulation(this.runtimeConfig);

    this.debug = sharedDebug.pipeline.enabled === true;
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

    const kernelWarmup = this.runtimeConfig.shared?.kernelWarmup;
    if (this.useGPU && kernelWarmup?.prewarm) {
      const mode = kernelWarmup.prewarmMode ?? 'parallel';
      log.info('Pipeline', `Kernel prewarm enabled (mode=${mode})`);
      try {
        await prewarmKernels({ mode });
        markKernelCacheWarmed();
      } catch (e) {
        log.warn('Pipeline', `Kernel prewarm failed: ${ (e).message}`);
      }
    }
    if (this.useGPU && kernelWarmup?.autoTune) {
      log.info('Pipeline', 'Kernel auto-tune enabled');
      try {
        await autoTuneKernels(this.modelConfig);
        markKernelCacheWarmed();
      } catch (e) {
        log.warn('Pipeline', `Kernel auto-tune failed: ${ (e).message}`);
      }
    }

    // Kernel path resolution
    log.debug('Pipeline', `kernelPath sources: runtime=${this.runtimeKernelPath}, config=${this.runtimeConfig.inference.kernelPath}, model=${this.modelConfig.kernelPath}`);
    
    let kernelPathSource = 'none';
    const kernelPathRef = this.runtimeKernelPath
      ?? this.runtimeConfig.inference.kernelPath
      ?? this.modelConfig.kernelPath
      ??  (manifest.optimizations)?.kernelPath;
    this.resolvedKernelPath = null;

    if (kernelPathRef) {
      kernelPathSource = this.runtimeKernelPath
        ? 'runtime'
        : this.runtimeConfig.inference.kernelPath
          ? 'config'
          : this.modelConfig.kernelPath
            ? 'model'
            : 'manifest';
      try {
        this.resolvedKernelPath = resolveKernelPath(kernelPathRef);
        
        // Apply runtime kernel overrides if configured
        if (this.runtimeConfig.inference.kernelOverrides) {
          log.info('Pipeline', 'Applying kernel path overrides', this.runtimeConfig.inference.kernelOverrides);
          this.resolvedKernelPath = applyKernelOverrides(
            this.resolvedKernelPath,
            this.runtimeConfig.inference.kernelOverrides
          );
        }

        const stats = getKernelPathStats(this.resolvedKernelPath);
        log.info('Pipeline', `KernelPath: ${this.resolvedKernelPath.id} (${stats.decodeSteps} decode steps, ${stats.uniqueKernels} kernels)`);
      } catch (e) {
        this.resolvedKernelPath = null;
        log.warn('Pipeline', `Failed to resolve kernel path '${kernelPathRef}': ${ (e).message}`);
      }
    } else {
      log.info('Pipeline', 'KernelPath: none (no kernel path configured)');
    }

    this.kernelPathSource = kernelPathSource;
    setActiveKernelPath(this.resolvedKernelPath, kernelPathSource);

    const kernelPathActivationDtype = getKernelPathActivationDtype(this.resolvedKernelPath);
    if (kernelPathActivationDtype) {
      const currentActivation = this.runtimeConfig.inference.compute.activationDtype;
      const currentKV = this.runtimeConfig.inference.kvcache.kvDtype;
      const nextInference = {
        ...this.runtimeConfig.inference,
        compute: { ...this.runtimeConfig.inference.compute },
        kvcache: { ...this.runtimeConfig.inference.kvcache },
      };
      let updated = false;

      if (currentActivation !== kernelPathActivationDtype) {
        log.info(
          'Pipeline',
          `KernelPath ${this.resolvedKernelPath?.id ?? 'unknown'} overrides activationDtype=${currentActivation} -> ${kernelPathActivationDtype}`
        );
        nextInference.compute.activationDtype = kernelPathActivationDtype;
        updated = true;
      }

      if (currentKV !== kernelPathActivationDtype) {
        log.info(
          'Pipeline',
          `KernelPath ${this.resolvedKernelPath?.id ?? 'unknown'} overrides kvDtype=${currentKV} -> ${kernelPathActivationDtype}`
        );
        nextInference.kvcache.kvDtype = kernelPathActivationDtype;
        updated = true;
      }

      if (updated) {
        this.runtimeConfig = setRuntimeConfig({ ...this.runtimeConfig, inference: nextInference });
      }
    }

    this._resolveLayerPipeline();

    const cfg = this.modelConfig;
    const moeStr = cfg.useMoE ? `, MoE(${cfg.numExperts}x${cfg.moeTopK})` : '';
    const kernelInfo = this.resolvedKernelPath ? `kernelPath=${this.resolvedKernelPath.id}` : 'kernelPath=none';
    log.info('Pipeline', `${cfg.numLayers}L/${cfg.hiddenSize}H/${cfg.numHeads}heads (${cfg.headDim}dim)${moeStr}, ${kernelInfo}`);

    // Initialize tokenizer with preset fallback hints
    const presetId = manifest.inference?.presetId;
    if (!presetId) {
      throw new Error(
        `Manifest "${manifest.modelId ?? 'unknown'}" is missing inference.presetId. ` +
        'Re-convert the model using the latest converter.'
      );
    }
    const preset = resolvePreset(presetId);
    this.tokenizer = await initTokenizer(manifest, {
      baseUrl: this.baseUrl ?? undefined,
      presetTokenizer: preset?.tokenizer,
    });
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
  const pipeline = new InferencePipeline();
  await pipeline.initialize(contexts);
  await pipeline.loadModel(manifest);
  return pipeline;
}

registerPipeline('transformer', createTransformerPipeline);

export async function createPipeline(manifest, contexts = {}) {
  const modelType = manifest?.modelType || 'transformer';
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
