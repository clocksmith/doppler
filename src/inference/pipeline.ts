/**
 * pipeline.ts - Main Inference Pipeline (Thin Orchestrator)
 *
 * This module orchestrates inference by delegating to specialized modules:
 * - state.ts: Holds model configuration, weights, and runtime state
 * - generator.ts: Handles token generation loops and decoding
 * - init.ts: Initialization, weight loading, KV cache, RoPE
 *
 * The pipeline maintains state and coordinates the flow from input tokens to generated output.
 *
 * @module inference/pipeline
 */

import { getDevice, getKernelCapabilities } from '../gpu/device.js';
import { getBufferPool as getGlobalBufferPool } from '../gpu/buffer-pool.js';
import { markWarmed as markKernelCacheWarmed } from '../gpu/kernel-selection-cache.js';
import { log, applyDebugConfig, setGPUDevice } from '../debug/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import {
  resolveKernelPath,
  getKernelPathStats,
  autoSelectKernelPath,
  setActiveKernelPath,
  type KernelPathSource,
} from '../config/kernel-path-loader.js';
import { DEFAULT_KVCACHE_CONFIG, type KernelPathRef, type ModelInferenceOverrides } from '../config/schema/index.js';
import { MoERouter } from './moe-router.js';

// Pipeline sub-modules
import { PipelineState } from './pipeline/state.js';
import { PipelineGenerator } from './pipeline/generator.js';
import { parseModelConfig, type Manifest } from './pipeline/config.js';
import {
  initRoPEFrequencies,
  createKVCache,
  initTokenizer,
  loadWeights,
  type WeightLoadResult,
  initMoERouter,
  initSpeculativeDecoder,
  fuseQKVWeights,
  type PipelineContexts,
} from './pipeline/init.js';
import { applyPipelineDebugConfig } from './pipeline/debug-utils.js';
import { resolveLayerPipeline } from './pipeline/layer-plan.js';
import { getDopplerLoader } from '../loader/doppler-loader.js';

// Re-export types for external use
import type { GenerateOptions, KVCacheSnapshot, LayerWeights, ExpertWeights, RouterWeights, GenerationResult, PipelineStats, BatchingStats } from './pipeline/types.js';
import type { LoRAAdapter } from './pipeline/lora.js';
export type { GenerateOptions, KVCacheSnapshot, LayerWeights, ExpertWeights, RouterWeights, GenerationResult, PipelineStats, BatchingStats };
export { PipelineContexts };

// ============================================================================
// Main Inference Pipeline Class
// ============================================================================

export class InferencePipeline extends PipelineState {
  private generator: PipelineGenerator;

  // Progress callback
  private _onProgress: ((progress: { percent: number; message?: string; stage?: string; layer?: number; total?: number }) => void) | null = null;
  private _preloadedWeights: WeightLoadResult | null = null;

  constructor() {
    super();
    this.generator = new PipelineGenerator(this);
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================

  async initialize(contexts: PipelineContexts = {}): Promise<void> {
    if (contexts.runtimeConfig) {
      this.runtimeConfig = setRuntimeConfig(contexts.runtimeConfig);
    } else {
      this.runtimeConfig = getRuntimeConfig();
    }
    applyDebugConfig(this.runtimeConfig.debug, { respectUrlParams: true });
    applyPipelineDebugConfig(this.runtimeConfig.debug.pipeline);

    if (contexts.gpu?.device) {
      this.gpuContext = { device: contexts.gpu.device };
      this.useGPU = true;
    }
    if (contexts.memory) this.memoryContext = contexts.memory;
    if (contexts.storage) this.storageContext = contexts.storage;
    if (contexts.baseUrl) this.baseUrl = contexts.baseUrl;

    if (contexts.runtime?.debug) this.debug = true;
    if (contexts.runtime?.kernelPath) {
      this.runtimeKernelPath = contexts.runtime.kernelPath;
    }
    if (contexts.onProgress) this._onProgress = contexts.onProgress;

    const device = getDevice();
    if (device) setGPUDevice(device);

    log.debug('Pipeline', 'Initialized', { useGPU: this.useGPU, debug: this.debug });
  }

  async loadModel(manifest: Manifest): Promise<void> {
    this.manifest = manifest;
    // Pass runtime model overrides to merge with manifest inference config
    const modelOverrides = this.runtimeConfig.inference.modelOverrides as ModelInferenceOverrides | undefined;
    this.modelConfig = parseModelConfig(manifest, modelOverrides);

    if (manifest.optimizations?.debug || manifest.runtime?.debug) this.debug = true;

    // Kernel path resolution
    log.debug('Pipeline', `kernelPath sources: runtime=${this.runtimeKernelPath}, config=${this.runtimeConfig.inference.kernelPath}, model=${this.modelConfig.kernelPath}`);
    let kernelPathSource: KernelPathSource = 'none';
    const kernelPathRef = this.runtimeKernelPath
      ?? this.runtimeConfig.inference.kernelPath
      ?? this.modelConfig.kernelPath
      ?? (manifest.optimizations as { kernelPath?: KernelPathRef } | undefined)?.kernelPath;

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
        const stats = getKernelPathStats(this.resolvedKernelPath);
        log.info('Pipeline', `KernelPath: ${this.resolvedKernelPath.id} (${stats.decodeSteps} decode steps, ${stats.uniqueKernels} kernels)`);
      } catch (e) {
        log.warn('Pipeline', `Failed to resolve kernel path '${kernelPathRef}': ${(e as Error).message}`);
      }
    } else {
      // Auto-select kernel path
      kernelPathSource = 'auto';
      try {
        const quantization = manifest.quantization ?? manifest.quantizationInfo?.weights ?? null;
        const modelFamily = manifest.architecture ?? 'gemma2';
        const gpuCaps = getKernelCapabilities();
        const capabilities = {
          hasSubgroups: gpuCaps.hasSubgroups,
          hasF16: gpuCaps.hasF16,
        };
        this.resolvedKernelPath = autoSelectKernelPath(quantization, modelFamily, capabilities);
        const stats = getKernelPathStats(this.resolvedKernelPath);
        log.info('Pipeline', `KernelPath (auto): ${this.resolvedKernelPath.id} (${stats.decodeSteps} decode steps, ${stats.uniqueKernels} kernels)`);
      } catch (e) {
        log.warn('Pipeline', `Failed to auto-select kernel path: ${(e as Error).message}`);
      }
    }

    this.kernelPathSource = kernelPathSource;
    setActiveKernelPath(this.resolvedKernelPath, kernelPathSource);

    this._resolveLayerPipeline();

    const cfg = this.modelConfig;
    const moeStr = cfg.useMoE ? `, MoE(${cfg.numExperts}x${cfg.moeTopK || 2})` : '';
    const kernelInfo = this.resolvedKernelPath ? `kernelPath=${this.resolvedKernelPath.id}` : 'kernelPath=none';
    log.info('Pipeline', `${cfg.numLayers}L/${cfg.hiddenSize}H/${cfg.numHeads}heads (${cfg.headDim}dim)${moeStr}, ${kernelInfo}`);

    // Initialize tokenizer
    this.tokenizer = await initTokenizer(manifest, this.baseUrl ?? undefined);
    const tokenizerVocabSize = this.tokenizer.getVocabSize();
    if (Number.isFinite(tokenizerVocabSize) && tokenizerVocabSize > 0) {
      if (tokenizerVocabSize !== this.modelConfig.vocabSize) {
        log.info('Pipeline', `Tokenizer vocabSize=${tokenizerVocabSize} differs from model=${this.modelConfig.vocabSize}, using model size`);
      }
    }

    // Initialize KV cache
    this.kvCache = createKVCache(this.modelConfig, this.useGPU, this.debug, this.runtimeConfig.kvcache);

    // Initialize MoE router if needed
    if (this.modelConfig.useMoE) {
      this.moeRouter = new MoERouter({
        numExperts: this.modelConfig.numExperts,
        topK: this.modelConfig.moeTopK || 2,
        hiddenSize: this.modelConfig.hiddenSize,
        normalizeWeights: true,
      });
    }

    // Initialize speculative decoder
    if (manifest.draftModel) {
      this.speculativeDecoder = initSpeculativeDecoder(manifest);
    }

    // Load weights
    await this._loadWeights();

    // Initialize RoPE frequencies
    await this._initRoPE();

    this.isLoaded = true;
    log.info('Pipeline', 'Model loaded successfully');
  }

  private async _loadWeights(): Promise<void> {
    const result = this._preloadedWeights || await loadWeights(
      this.manifest!,
      this.modelConfig!,
      {
        storageContext: this.storageContext ?? undefined,
        loadingConfig: this.runtimeConfig.loading,
        baseUrl: this.baseUrl ?? undefined,
        onProgress: (info: { stage: string; progress: number; message?: string; layer?: number; total?: number; shard?: number; totalShards?: number }) => {
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
        verifyHashes: false,
      }
    );

    result.layerWeights.forEach((w, k) => this.weights.set(k, w));
    this.weights.set('embed', result.embeddings);
    this.weights.set('lm_head', result.lmHead);
    this.weights.set('final_norm', result.finalNorm);

    this.useTiedEmbeddings = result.useTiedEmbeddings;
    this.embeddingVocabSize = result.embeddingVocabSize;
    this.embeddingTranspose = result.embeddingTranspose;
    this.layerRouterWeights = result.layerRouterWeights;

    this.dopplerLoader = getDopplerLoader(this.runtimeConfig.loading);

    if (this.modelConfig!.useMoE && this.moeRouter) {
      this.moeRouter = initMoERouter(this.modelConfig!, result.layerWeights);
    }

    if (this.useGPU && this.modelConfig) {
      fuseQKVWeights(result.layerWeights, this.modelConfig);
    }

    if (this.useGPU && this.modelConfig) {
      this.decodeBuffers?.ensureBuffers({
        hiddenSize: this.modelConfig.hiddenSize,
        intermediateSize: this.modelConfig.intermediateSize,
        enablePingPong: true,
      });
    }
  }

  setPreloadedWeights(weights: WeightLoadResult): void {
    this._preloadedWeights = weights;
  }

  private async _initRoPE(): Promise<void> {
    const config = this.modelConfig!;
    const ropeBuffers = await initRoPEFrequencies({
      headDim: config.headDim,
      maxSeqLen: config.maxSeqLen || DEFAULT_KVCACHE_CONFIG.maxSeqLen,
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

  private _resolveLayerPipeline(): void {
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

  generate(prompt: string, options: GenerateOptions = {}): AsyncGenerator<string, void, void> {
    return this.generator.generate(prompt, options);
  }

  prefillKVOnly(prompt: string, options: GenerateOptions = {}): Promise<KVCacheSnapshot> {
    return this.generator.prefillKVOnly(prompt, options);
  }

  applyKVCacheSnapshot(snapshot: KVCacheSnapshot): void {
    this.kvCache = snapshot.cache.clone();
    if (this.useGPU && this.kvCache) {
      const device = getDevice();
      if (device) {
        this.kvCache.setGPUContext({ device });
      }
    }
    this.currentSeqLen = snapshot.seqLen;
  }

  generateWithPrefixKV(
    prefix: KVCacheSnapshot,
    prompt: string,
    options: GenerateOptions = {}
  ): AsyncGenerator<string, void, void> {
    return this.generator.generateWithPrefixKV(prefix, prompt, options);
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  getStats(): PipelineStats {
    return { ...this.stats };
  }

  getBatchingStats(): BatchingStats {
    return { ...this.batchingStats };
  }

  getMemoryStats(): {
    used: number;
    pool?: { currentBytesAllocated?: number; peakBytesAllocated?: number; activeBuffers?: number; pooledBuffers?: number };
    kvCache?: { allocated?: number; used?: number; seqLen?: number; maxSeqLen?: number };
  } {
    const stats: {
      used: number;
      pool?: { currentBytesAllocated?: number; peakBytesAllocated?: number; activeBuffers?: number; pooledBuffers?: number };
      kvCache?: { allocated?: number; used?: number; seqLen?: number; maxSeqLen?: number };
    } = { used: 0 };

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

    return stats;
  }

  getKVCacheStats(): { seqLen: number; maxSeqLen: number } | null {
    if (!this.kvCache) return null;
    const { seqLen, maxSeqLen } = this.kvCache.getMemoryStats();
    return { seqLen, maxSeqLen };
  }

  getBufferPool(): ReturnType<typeof getGlobalBufferPool> | null {
    try {
      return getGlobalBufferPool();
    } catch {
      return null;
    }
  }

  async unload(): Promise<void> {
    this.kvCache?.clear();
    this.weights.clear();
    this.expertWeights.clear();
    this.lora = null;
    setActiveKernelPath(null, 'none');
    this.isLoaded = false;
    this.currentSeqLen = 0;
    log.info('Pipeline', 'Unloaded');
  }

  setLoRAAdapter(adapter: LoRAAdapter | null): void {
    this.lora = adapter;
  }

  getActiveLoRA(): LoRAAdapter | null {
    return this.lora;
  }

  reset(): void {
    this.kvCache?.clear();
    this.currentSeqLen = 0;
    this.decodeStepCount = 0;
    this.debugFlags = {};
    this.decodeBuffers?.resetPingPong();
    // Reset stats
    this.stats.tokensGenerated = 0;
    this.stats.totalTimeMs = 0;
    this.stats.prefillTimeMs = 0;
    this.stats.decodeTimeMs = 0;
    this.stats.gpuTimePrefillMs = undefined;
    this.stats.gpuTimeDecodeMs = undefined;
  }

  releaseGPUResources(): void {
    this.decodeBuffers?.release();
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export async function createPipeline(manifest: Manifest, contexts: PipelineContexts = {}): Promise<InferencePipeline> {
  // Use manifest's quantizationInfo.compute as default activationDtype
  const manifestComputeDtype = manifest.quantizationInfo?.compute;
  if (manifestComputeDtype && !contexts.runtimeConfig?.inference?.compute?.activationDtype) {
    const computeToActivation: Record<string, 'f16' | 'f32'> = {
      'f16': 'f16',
      'bf16': 'f16',
      'f32': 'f32',
    };
    const activationDtype = computeToActivation[manifestComputeDtype];
    if (activationDtype) {
      contexts = {
        ...contexts,
        runtimeConfig: {
          ...contexts.runtimeConfig,
          inference: {
            ...contexts.runtimeConfig?.inference,
            compute: {
              ...contexts.runtimeConfig?.inference?.compute,
              activationDtype,
            },
          },
        },
      };
    }
  }

  const pipeline = new InferencePipeline();
  await pipeline.initialize(contexts);
  await pipeline.loadModel(manifest);
  return pipeline;
}

export { InferencePipeline as Pipeline };
