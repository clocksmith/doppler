/**
 * DopplerLoader - Core Model Loader
 * Phase 1: Foundation
 *
 * Orchestrates the complete model loading pipeline:
 * - Storage: Load shards from OPFS
 * - Memory: Stage in heap (Memory64 or segmented)
 * - GPU: Transfer to VRAM for compute
 *
 * @module loader/doppler-loader
 */

import { getMemoryCapabilities, type MemoryCapabilities } from '../memory/capability.js';
import { detectUnifiedMemory } from '../memory/unified-detect.js';
import { HeapManager, getHeapManager } from '../memory/heap-manager.js';
import {
  initOPFS,
  openModelDirectory,
  loadShard as loadShardFromOPFS,
  verifyIntegrity,
  loadManifestFromOPFS,
} from '../storage/shard-manager.js';
import {
  parseManifest,
  isMoE,
  type RDRRManifest,
} from '../storage/rdrr-format.js';
import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import { releaseBuffer } from '../gpu/buffer-pool.js';
import {
  type WeightBuffer,
  type WeightLayout,
  type CpuWeightBuffer,
} from '../gpu/weight-buffer.js';
import { ExpertCache, getExpertCache, type CacheStats } from './expert-cache.js';
import type { ExpertWeights } from './weights.js';
import type { LoRAAdapter } from '../inference/pipeline/lora.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';

// Import types and utilities from split modules
import type {
  TensorLocation,
  LayerWeights,
  LoadProgress,
  LoadOptions,
  CustomShardLoader,
  CustomShardLoaderOptions,
  LoaderStats,
  KernelCapabilities,
  ModelConfig,
} from './loader-types.js';

import { findAlternativeTensorName } from './dtype-utils.js';

import { ShardCache, createShardCache } from './shard-cache.js';
import type { LoadingConfigSchema } from '../config/schema/loading.schema.js';
import { validateManifestInference } from '../config/schema/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

// Import helper modules for refactored logic
import { buildTensorLocations } from './shard-resolver.js';
import {
  configureQ4KStrategy,
  needsNormWeightOffset,
  resolveWeightLayout,
  shouldStreamLargeWeight,
} from './manifest-config.js';
import { MemoryMonitor } from './memory-monitor.js';
import {
  type TensorLoadConfig,
  loadTensorToGPU,
  loadTensorToCPU,
} from './tensor-loader.js';
import { loadEmbeddings, type EmbeddingLoaderContext } from './embedding-loader.js';
import { loadLayer, type LayerLoaderContext } from './layer-loader.js';
import { loadFinalWeights, type FinalWeightsContext } from './final-weights-loader.js';
import {
  loadExpert as loadExpertFromModule,
  prefetchExperts as prefetchExpertsFromModule,
  predictNextLayerExperts as predictNextLayerExpertsFromModule,
  type ExpertLoaderContext,
} from './expert-loader.js';
import { loadLoRAWeights as loadLoRAWeightsFromModule } from './lora-loader.js';
import { assembleShardData } from './tensor-reader.js';

// Re-export types for backward compatibility
export type {
  TensorLocation,
  LayerWeights,
  LoadProgress,
  LoadOptions,
  CustomShardLoader,
  CustomShardLoaderOptions,
  LoaderStats,
} from './loader-types.js';

// ============================================================================
// DopplerLoader Class
// ============================================================================

/**
 * DopplerLoader class
 */
export class DopplerLoader {
  // Capabilities
  memoryCapabilities: MemoryCapabilities | null = null;
  gpuCapabilities: KernelCapabilities | null = null;
  isUnifiedMemory = false;

  // Manifest and model info
  manifest: RDRRManifest | null = null;
  modelId: string | null = null;
  isMoE = false;

  // Loaded state
  isLoaded = false;
  embeddings: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null = null;
  layers = new Map<number, LayerWeights>();
  experts = new Map<string, ExpertWeights>();
  lmHead: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null = null;
  finalNorm: GPUBuffer | Float32Array | null = null;

  // Memory management
  heapManager: HeapManager | null = null;
  gpuBuffers = new Set<GPUBuffer>();

  // Expert cache for MoE models (LRU eviction)
  expertCache: ExpertCache | null = null;

  // Loading state
  loadedShards = new Set<number>();
  tensorLocations = new Map<string, TensorLocation>();

  // Shard cache (LRU with request deduplication)
  shardCache: ShardCache;

  // Loading configuration
  private loadingConfig: LoadingConfigSchema = getRuntimeConfig().loading;

  // Fused Q4_K matmul: skip dequantization for matmul weights, use fused kernel
  useFusedQ4K = false;

  // Q4K layout from manifest: 'column_wise' means weights are pre-transposed
  q4kLayout: 'flat' | 'row_wise' | 'column_wise' | null = null;
  keepF32Weights = false;

  // Internal tracking
  private _normOffsetLogged = false;
  private _normOffsetDebugLogged = false;
  private _memoryMonitor: MemoryMonitor | null = null;
  private _tensorsJsonUrl: string | null = null;

  constructor(loadingConfig?: LoadingConfigSchema) {
    this.loadingConfig = loadingConfig ?? getRuntimeConfig().loading;
    this.shardCache = createShardCache(
      this.loadingConfig.shardCache.opfsEntries,
      this.loadingConfig.shardCache
    );
  }

  setLoadingConfig(config: LoadingConfigSchema): void {
    this.loadingConfig = config;
    this.shardCache.configure({
      loadingConfig: config.shardCache,
      maxEntries: config.shardCache.opfsEntries,
    });
    if (this.manifest) {
      this.shardCache.configureForModel(this.manifest, this.shardCache.hasCustomLoader);
    }
    if (this.expertCache) {
      this.expertCache.configure(config.expertCache);
    }
  }

  private _getMemoryState() {
    return {
      shardCacheBytes: this.shardCache.totalBytes,
      shardCount: this.shardCache.size,
      layerCount: this.layers.size,
      gpuBufferCount: this.gpuBuffers.size,
    };
  }

  private _startMemoryLogging(): void {
    const logIntervalMs = this.loadingConfig.memoryManagement.logIntervalMs;
    this._memoryMonitor = new MemoryMonitor(logIntervalMs);
    this._memoryMonitor.start(() => this._getMemoryState());
  }

  private _stopMemoryLogging(phase: 'complete' | 'failed' = 'complete'): void {
    if (this._memoryMonitor) {
      this._memoryMonitor.stop(phase, () => this._getMemoryState());
      this._memoryMonitor = null;
    }
  }

  setCustomShardLoader(loadShardFn: CustomShardLoader, options: CustomShardLoaderOptions = {}): void {
    this.shardCache.setCustomLoader(loadShardFn, options.verify !== false);
  }

  setTensorsJsonUrl(url: string | null): void {
    this._tensorsJsonUrl = url;
  }

  private async _loadShard(shardIndex: number): Promise<ArrayBuffer> {
    return this.shardCache.load(shardIndex);
  }

  async init(): Promise<void> {
    log.info('Loader', 'Initializing...');

    this.memoryCapabilities = await getMemoryCapabilities();
    const unifiedInfo = await detectUnifiedMemory();
    this.isUnifiedMemory = unifiedInfo.isUnified;

    const device = await initDevice();
    if (!device) {
      throw new Error('Failed to initialize WebGPU device');
    }
    this.gpuCapabilities = getKernelCapabilities();

    this.heapManager = getHeapManager();
    await this.heapManager.init();

    this.expertCache = getExpertCache();

    await initOPFS();

    const caps = [
      this.gpuCapabilities.hasF16 ? 'f16' : null,
      this.gpuCapabilities.hasSubgroups ? 'subgroups' : null,
      this.memoryCapabilities.hasMemory64 ? 'mem64' : null,
      this.isUnifiedMemory ? 'unified' : null,
    ].filter(Boolean).join(', ');
    log.info('Loader', `Initialized (${caps})`);
  }

  setManifest(manifest: RDRRManifest): void {
    this.manifest = manifest;
    const config = manifest.config as ModelConfig | undefined;
    this.isMoE = manifest.moeConfig != null || (config?.num_local_experts ?? 0) > 1;
    this.shardCache.configureForModel(this.manifest, this.shardCache.hasCustomLoader);
    this._configureQ4KStrategy();
    debugTrace.loader('Manifest set externally');
  }

  async loadLoRAWeights(manifest: RDRRManifest): Promise<LoRAAdapter> {
    const prevManifest = this.manifest;
    const prevLocations = new Map(this.tensorLocations);

    this.manifest = manifest;
    // We must rebuild locations so _loadTensor finds them
    await this._buildTensorLocations();

    try {
      return await loadLoRAWeightsFromModule(
        manifest,
        (name, toGPU, silent) => this._loadTensor(name, toGPU, silent)
      );
    } finally {
      this.manifest = prevManifest;
      this.tensorLocations = prevLocations;
    }
  }

  private _configureQ4KStrategy(): void {
    const config = configureQ4KStrategy(this.manifest, this.gpuCapabilities);
    this.useFusedQ4K = config.useFusedQ4K;
    this.q4kLayout = config.q4kLayout;
    this.keepF32Weights = config.keepF32Weights;
  }

  private _resolveWeightLayout(location: TensorLocation, name: string): WeightLayout {
    return resolveWeightLayout(location, name);
  }

  private _shouldStreamLargeWeight(name: string, location: TensorLocation, label: string): boolean {
    return shouldStreamLargeWeight(name, location, label, this.gpuCapabilities, this.keepF32Weights);
  }

  async load(modelId: string, options: LoadOptions = {}): Promise<ModelConfig> {
    const { onProgress = null, verifyHashes = true } = options;

    if (!this.heapManager) {
      await this.init();
    }

    const hasExistingModelState =
      this.isLoaded ||
      this.modelId !== null ||
      this.tensorLocations.size > 0 ||
      this.shardCache.size > 0 ||
      this.layers.size > 0 ||
      this.experts.size > 0 ||
      this.gpuBuffers.size > 0;

    const preservedManifest = this.shardCache.hasCustomLoader ? this.manifest : null;

    if (hasExistingModelState) {
      await this.unload();
    }

    if (preservedManifest) {
      this.manifest = preservedManifest;
    }

    log.info('Loader', `Loading: ${modelId}`);
    this.modelId = modelId;

    this._startMemoryLogging();

    if (!this.shardCache.hasCustomLoader) {
      await openModelDirectory(modelId);
      const manifestJson = await loadManifestFromOPFS();
      this.manifest = parseManifest(manifestJson);
    }

    if (!this.manifest) {
      throw new Error('No manifest available. Set manifest via setManifest() or ensure OPFS has the model.');
    }

    validateManifestInference(this.manifest);

    this._configureQ4KStrategy();

    const config = this.manifest.config as ModelConfig | undefined;
    this.isMoE = this.manifest.moeConfig != null ||
                 (config?.num_local_experts ?? 0) > 1 ||
                 isMoE();

    this.shardCache.configureForModel(this.manifest, this.shardCache.hasCustomLoader);

    if (!this.isMoE && !this.isUnifiedMemory) {
      log.warn('Loader', 'Dense model on discrete GPU - performance limited. Consider MoE model.');
    }

    if (verifyHashes && !this.shardCache.hasCustomLoader) {
      const integrity = await verifyIntegrity();
      if (!integrity.valid) {
        throw new Error(
          `Model integrity check failed. ` +
          `Missing shards: ${integrity.missingShards.length}, ` +
          `Corrupt shards: ${integrity.corruptShards.length}`
        );
      }
    }

    await this._buildTensorLocations();

    const totalBytes = (this.manifest.shards || []).reduce((sum, s) => sum + (s.size || 0), 0);
    const totalShards = this.manifest.shards?.length || 0;
    const loadStartTime = Date.now();
    let bytesLoaded = 0;
    let shardsLoaded = 0;

    const reportProgress = (stage: LoadProgress['stage'], baseProgress: number, detail?: string): void => {
      if (!onProgress) return;
      const elapsed = (Date.now() - loadStartTime) / 1000;
      const speed = elapsed > 0 ? bytesLoaded / elapsed : 0;
      const speedStr = speed > 0 ? `${formatBytes(speed)}/s` : '';
      const message = detail ||
        `${formatBytes(bytesLoaded)} / ${formatBytes(totalBytes)} ${speedStr ? `• ${speedStr}` : ''}`;
      onProgress({
        stage,
        progress: baseProgress,
        shard: shardsLoaded,
        totalShards,
        bytesLoaded,
        totalBytes,
        bytesPerSecond: speed,
        message,
      });
    };

    if (onProgress) {
      onProgress({ stage: 'manifest', progress: 0.05, message: 'Parsing manifest...' });
    }

    const loadedShardIndices = new Set<number>();
    let inLayerPhase = false;
    const originalLoadShard = this._loadShard.bind(this);
    this._loadShard = async (shardIndex: number): Promise<ArrayBuffer> => {
      const shardInfo = this.manifest?.shards?.[shardIndex];
      const shardSize = shardInfo?.size || 0;
      const data = await originalLoadShard(shardIndex);

      if (!loadedShardIndices.has(shardIndex)) {
        loadedShardIndices.add(shardIndex);
        bytesLoaded += shardSize;
        shardsLoaded++;
        if (!inLayerPhase) {
          const pct = 0.1 + Math.min(bytesLoaded / totalBytes, 1.0) * 0.7;
          const elapsed = (Date.now() - loadStartTime) / 1000;
          const speed = elapsed > 0 ? bytesLoaded / elapsed : 0;
          const sourceInfo = this.shardCache.lastSource;
          const sourceStr = sourceInfo ? sourceInfo.source : 'unknown';
          const elapsedStr = sourceInfo && sourceInfo.elapsed > 0 ? ` ${sourceInfo.elapsed.toFixed(2)}s` : '';
          if (onProgress) {
            onProgress({
              stage: 'shards',
              progress: pct,
              shard: shardsLoaded,
              totalShards,
              bytesLoaded,
              totalBytes,
              bytesPerSecond: speed,
              message: `Shard ${shardIndex}: ${sourceStr} (${formatBytes(shardSize)}${elapsedStr})`,
            });
          }
        }
      }
      return data;
    };

    let loadError: unknown = null;
    try {
      reportProgress('shards', 0.1, 'Loading embeddings...');
      await this._loadEmbeddings(onProgress);

      const numLayers = config?.num_hidden_layers ||
                        config?.blockCount ||
                        config?.text_config?.num_hidden_layers ||
                        config?.n_layer ||
                        (this.manifest.architecture as { numLayers?: number } | undefined)?.numLayers ||
                        32;
      log.info('Loader', `Layers: 0-${numLayers - 1}`);

      inLayerPhase = true;
      const layersStartTime = performance.now();

      for (let l = 0; l < numLayers; l++) {
        const layerStart = performance.now();
        await this._loadLayer(l, onProgress);
        const layerElapsed = ((performance.now() - layerStart) / 1000).toFixed(2);
        log.verbose('Loader', `  Layer ${l}: ${layerElapsed}s`);

        await new Promise(r => setTimeout(r, 0));

        const { flushIntervalLayers, flushThresholdBytes, gpuQueueFlushLayers } = this.loadingConfig.memoryManagement;
        const cacheBytes = this.shardCache.totalBytes;
        const shouldFlushCache = !this.shardCache.hasCustomLoader && l > 0 && (l % flushIntervalLayers === 0 || cacheBytes > flushThresholdBytes);
        if (shouldFlushCache) {
          this.shardCache.clear();
        }
        if (l > 0 && l % gpuQueueFlushLayers === 0) {
          const device = getDevice();
          if (device) {
            await device.queue.onSubmittedWorkDone();
          }
        }

        if (onProgress) {
          const layerFraction = (l + 1) / numLayers;
          const layerProgress = 0.80 + layerFraction * 0.05;
          onProgress({
            stage: 'layers',
            layer: l + 1,
            total: numLayers,
            progress: layerProgress,
            shard: shardsLoaded,
            totalShards,
            bytesLoaded,
            totalBytes,
            bytesPerSecond: 0,
            message: `Layer ${l + 1}/${numLayers}`,
          });
        }
      }

      const layersTotalTime = ((performance.now() - layersStartTime) / 1000).toFixed(2);
      log.info('Loader', `Layers: ${numLayers} complete (${layersTotalTime}s)`);

      reportProgress('gpu_transfer', 0.85, 'Loading final weights...');
      await this._loadFinalWeights(onProgress);

      if (onProgress) {
        onProgress({ stage: 'complete', progress: 1.0 });
      }

      this.isLoaded = true;
      const totalTime = ((Date.now() - loadStartTime) / 1000).toFixed(2);
      const avgSpeed = formatBytes(bytesLoaded / (Date.now() - loadStartTime) * 1000);
      log.info('Loader', `Complete: ${formatBytes(bytesLoaded)} in ${totalTime}s (${avgSpeed}/s)`);

      this.shardCache.clear();

      return (this.manifest.config as ModelConfig) || {};
    } catch (error) {
      loadError = error;
    } finally {
      this._loadShard = originalLoadShard;
      if (this._memoryMonitor) {
        this._stopMemoryLogging(loadError ? 'failed' : 'complete');
      }
    }

    if (loadError) {
      await this.unload();
      if (preservedManifest) {
        this.manifest = preservedManifest;
      }
      throw loadError;
    }
    return (this.manifest?.config as ModelConfig) || {};
  }

  private async _buildTensorLocations(): Promise<void> {
    this.tensorLocations.clear();
    if (!this.manifest) return;

    const locations = await buildTensorLocations(this.manifest, {
      tensorsJsonUrl: this._tensorsJsonUrl,
      hasCustomLoader: this.shardCache.hasCustomLoader,
    });

    for (const [name, loc] of locations) {
      this.tensorLocations.set(name, loc);
    }
  }

  private async _loadTensor(
    name: string,
    toGPU = true,
    silent = false
  ): Promise<GPUBuffer | WeightBuffer | Float32Array | Uint8Array | null> {
    const location = this.tensorLocations.get(name);
    if (!location) {
      const altName = findAlternativeTensorName(name, this.tensorLocations);
      if (altName) {
        return this._loadTensor(altName, toGPU, silent);
      }
      if (!silent) {
        log.warn('Loader', `Tensor not found: ${name}`);
      }
      return null;
    }

    if (name.includes('attn_k') || name.includes('k_proj')) {
      debugTrace.loader(`Loading ${name}: shape=${JSON.stringify(location.shape)}, size=${location.size}, dtype=${location.dtype}, spans=${!!location.spans}`);
    }

    const shardData = await this._assembleShardData(location, name);

    if (toGPU) {
      const device = getDevice();
      if (!device) {
        log.warn('Loader', 'GPU device not available; falling back to CPU');
        return loadTensorToCPU(shardData, location);
      }

      const config: TensorLoadConfig = {
        useFusedQ4K: this.useFusedQ4K,
        keepF32Weights: this.keepF32Weights,
        q4kLayout: this.q4kLayout,
        gpuCapabilities: this.gpuCapabilities,
      };

      const result = await loadTensorToGPU(shardData, location, name, config);

      for (const buffer of result.allocatedBuffers) {
        this.gpuBuffers.add(buffer);
      }

      return result.data;
    }

    return loadTensorToCPU(shardData, location);
  }

  private async _assembleShardData(location: TensorLocation, name: string): Promise<Uint8Array> {
    return assembleShardData(location, name, (idx) => this._loadShard(idx));
  }

  private _needsNormWeightOffset(): boolean {
    const result = needsNormWeightOffset(this.manifest);
    if (result && !this._normOffsetLogged) {
      this._normOffsetLogged = true;
    }
    return result;
  }

  private async _loadEmbeddings(_onProgress: ((progress: LoadProgress) => void) | null): Promise<void> {
    const ctx: EmbeddingLoaderContext = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this._loadTensor(name, toGPU, silent),
      shouldStreamLargeWeight: (name, loc, label) => this._shouldStreamLargeWeight(name, loc, label),
      resolveWeightLayout: (loc, name) => this._resolveWeightLayout(loc, name),
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
    };

    this.embeddings = await loadEmbeddings(ctx);
  }

  private async _loadLayer(
    layerIdx: number,
    _onProgress: ((progress: LoadProgress) => void) | null
  ): Promise<void> {
    const ctx: LayerLoaderContext = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this._loadTensor(name, toGPU, silent),
      needsNormWeightOffset: () => this._needsNormWeightOffset(),
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      isMoE: this.isMoE,
      isExpertLayer: (idx) => this._isExpertLayer(idx),
    };

    const weights = await loadLayer(ctx, layerIdx);
    this.layers.set(layerIdx, weights);
  }

  private _isExpertLayer(_layerIdx: number): boolean {
    return this.isMoE;
  }

  prefetchExperts(nextLayerIdx: number, expertIndices: number[]): void {
    prefetchExpertsFromModule(this._getExpertLoaderContext(), nextLayerIdx, expertIndices, this.isMoE);
  }

  predictNextLayerExperts(currentExperts: number[]): number[] {
    return predictNextLayerExpertsFromModule(currentExperts);
  }

  async loadExpert(layerIdx: number, expertIdx: number): Promise<ExpertWeights> {
    return loadExpertFromModule(this._getExpertLoaderContext(), layerIdx, expertIdx);
  }

  private _getExpertLoaderContext(): ExpertLoaderContext {
    return {
      manifest: this.manifest,
      loadTensor: (name, toGPU, silent) => this._loadTensor(name, toGPU, silent),
      loadShard: (idx) => this._loadShard(idx),
      shardCache: this.shardCache,
      expertCache: this.expertCache,
      experts: this.experts,
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
    };
  }

  private async _loadFinalWeights(_onProgress: ((progress: LoadProgress) => void) | null): Promise<void> {
    const ctx: FinalWeightsContext = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this._loadTensor(name, toGPU, silent),
      needsNormWeightOffset: () => this._needsNormWeightOffset(),
      shouldStreamLargeWeight: (name, loc, label) => this._shouldStreamLargeWeight(name, loc, label),
      resolveWeightLayout: (loc, name) => this._resolveWeightLayout(loc, name),
      embeddings: this.embeddings,
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      normOffsetDebugLogged: this._normOffsetDebugLogged,
    };

    const result = await loadFinalWeights(ctx);
    this.finalNorm = result.finalNorm;
    this.lmHead = result.lmHead;
    this._normOffsetDebugLogged = result.normOffsetDebugLogged;
  }

  getLayerWeights(layerIdx: number): LayerWeights | null {
    return this.layers.get(layerIdx) || null;
  }

  getConfig(): ModelConfig {
    return (this.manifest?.config as ModelConfig) || {};
  }

  canRunDense(): boolean {
    return this.isUnifiedMemory;
  }

  getStats(): LoaderStats {
    const expertCacheCount = this.expertCache?.getStats().expertCount || 0;
    return {
      modelId: this.modelId,
      isLoaded: this.isLoaded,
      isMoE: this.isMoE,
      isUnifiedMemory: this.isUnifiedMemory,
      layersLoaded: this.layers.size,
      expertsLoaded: this.experts.size + expertCacheCount,
      gpuBuffers: this.gpuBuffers.size,
    };
  }

  getExpertCacheStats(): CacheStats | null {
    return this.expertCache?.getStats() || null;
  }

  async unload(): Promise<void> {
    debugTrace.loader(' Unloading model...');

    if (this._memoryMonitor) {
      this._stopMemoryLogging('complete');
    }

    for (const buffer of this.gpuBuffers) {
      releaseBuffer(buffer);
    }
    this.gpuBuffers.clear();

    if (this.expertCache) {
      this.expertCache.clear();
    }

    this.embeddings = null;
    this.layers.clear();
    this.experts.clear();
    this.lmHead = null;
    this.finalNorm = null;
    this.manifest = null;
    this.modelId = null;
    this.loadedShards.clear();
    this.isLoaded = false;
    this.tensorLocations.clear();
    this.shardCache.clear();
    this._normOffsetLogged = false;

    debugTrace.loader(' Model unloaded');
  }
}

let globalLoader: DopplerLoader | null = null;

export function getDopplerLoader(loadingConfig?: LoadingConfigSchema): DopplerLoader {
  if (!globalLoader) {
    globalLoader = new DopplerLoader(loadingConfig);
  } else if (loadingConfig) {
    globalLoader.setLoadingConfig(loadingConfig);
  }
  return globalLoader;
}

export function createDopplerLoader(loadingConfig?: LoadingConfigSchema): DopplerLoader {
  return new DopplerLoader(loadingConfig);
}

export default DopplerLoader;