

import { getMemoryCapabilities } from '../memory/capability.js';
import { detectUnifiedMemory } from '../memory/unified-detect.js';
import { getHeapManager } from '../memory/heap-manager.js';
import {
  initStorage,
  openModelStore,
  verifyIntegrity,
  loadManifestFromStore,
} from '../storage/shard-manager.js';
import { parseManifest } from '../storage/rdrr-format.js';
import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import { releaseBuffer } from '../memory/buffer-pool.js';
import { getExpertCache } from './expert-cache.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';


import { createShardCache } from './shard-cache.js';
import { validateManifestInference } from '../config/schema/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

// Import helper modules for refactored logic
import { buildTensorLocations } from './shard-resolver.js';
import {
  needsNormWeightOffset,
  resolveWeightLayout,
  shouldStreamLargeWeight,
} from './manifest-config.js';
import { MemoryMonitor } from './memory-monitor.js';
import {
  loadTensorToGPU,
  loadTensorToCPU,
} from './tensor-loader.js';
import { loadEmbeddings } from './embedding-loader.js';
import { loadLayer } from './layer-loader.js';
import { loadFinalWeights } from './final-weights-loader.js';
import {
  loadExpert as loadExpertFromModule,
  prefetchExperts as prefetchExpertsFromModule,
  predictNextLayerExperts as predictNextLayerExpertsFromModule,
} from './expert-loader.js';
import { loadLoRAWeights as loadLoRAWeightsFromModule } from './lora-loader.js';
import { assembleShardData } from './tensor-reader.js';

// Re-export types for backward compatibility
export {
  // Types are in .d.ts file
} from './loader-types.js';

// ============================================================================
// DopplerLoader Class
// ============================================================================


export class DopplerLoader {
  // Capabilities
  
  memoryCapabilities = null;
  
  gpuCapabilities = null;
  
  isUnifiedMemory = false;

  // Manifest and model info
  
  manifest = null;
  
  modelId = null;
  
  isMoE = false;

  // Loaded state
  
  isLoaded = false;
  
  embeddings = null;
  
  layers = new Map();
  
  experts = new Map();
  
  lmHead = null;
  
  finalNorm = null;

  // Memory management
  
  heapManager = null;
  
  gpuBuffers = new Set();

  // Expert cache for MoE models (LRU eviction)
  
  expertCache = null;

  // Loading state
  
  loadedShards = new Set();
  
  tensorLocations = new Map();

  // Shard cache (LRU with request deduplication)
  
  shardCache;

  // Loading configuration
  
  #loadingConfig;

  // Fused Q4_K matmul: skip dequantization for matmul weights, use fused kernel
  
  useFusedQ4K = false;

  // Q4K layout from manifest: 'column_wise' means weights are pre-transposed
  
  q4kLayout = null;
  
  keepF32Weights = false;

  // Internal tracking
  
  #normOffsetLogged = false;
  
  #normOffsetDebugLogged = false;
  
  #memoryMonitor = null;
  
  #tensorsJsonUrl = null;
  
  #loadShardOverride = null;

  
  constructor(loadingConfig) {
    this.#loadingConfig = loadingConfig ?? getRuntimeConfig().loading;
    this.shardCache = createShardCache(
      this.#loadingConfig.shardCache.opfsEntries,
      this.#loadingConfig.shardCache
    );
  }

  
  setLoadingConfig(config) {
    this.#loadingConfig = config;
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

  
  setQ4KConfig(config) {
    this.useFusedQ4K = config.useFusedQ4K;
    this.q4kLayout = config.q4kLayout;
    this.keepF32Weights = config.keepF32Weights;
  }

  
  #getMemoryState() {
    return {
      shardCacheBytes: this.shardCache.totalBytes,
      shardCount: this.shardCache.size,
      layerCount: this.layers.size,
      gpuBufferCount: this.gpuBuffers.size,
    };
  }

  #startMemoryLogging() {
    const logIntervalMs = this.#loadingConfig.memoryManagement.logIntervalMs;
    this.#memoryMonitor = new MemoryMonitor(logIntervalMs);
    this.#memoryMonitor.start(() => this.#getMemoryState());
  }

  
  #stopMemoryLogging(phase = 'complete') {
    if (this.#memoryMonitor) {
      this.#memoryMonitor.stop(phase, () => this.#getMemoryState());
      this.#memoryMonitor = null;
    }
  }

  
  setCustomShardLoader(loadShardFn, options = {}) {
    this.shardCache.setCustomLoader(loadShardFn, options.verify !== false);
  }

  
  setTensorsJsonUrl(url) {
    this.#tensorsJsonUrl = url;
  }

  
  async #loadShard(shardIndex) {
    return this.shardCache.load(shardIndex);
  }

  
  #getLoadShard() {
    return this.#loadShardOverride ?? ((idx) => this.#loadShard(idx));
  }

  
  async init() {
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

    if (!this.shardCache.hasCustomLoader) {
      await initStorage();
    }

    const caps = [
      this.gpuCapabilities.hasF16 ? 'f16' : null,
      this.gpuCapabilities.hasSubgroups ? 'subgroups' : null,
      this.memoryCapabilities.hasMemory64 ? 'mem64' : null,
      this.isUnifiedMemory ? 'unified' : null,
    ].filter(Boolean).join(', ');
    log.info('Loader', `Initialized (${caps})`);
  }

  
  setManifest(manifest) {
    this.manifest = manifest;
    const config =  (manifest.config);
    const moeConfig = manifest.moeConfig;
    this.isMoE = moeConfig != null && (moeConfig.numExperts ?? 0) > 1;
    if (!this.isMoE && (config?.num_local_experts ?? 0) > 1) {
      throw new Error(
        `Manifest "${manifest.modelId ?? 'unknown'}" missing moeConfig for MoE model. Re-convert with moeConfig.`
      );
    }
    this.shardCache.configureForModel(this.manifest, this.shardCache.hasCustomLoader);
    debugTrace.loader('Manifest set externally');
  }

  
  async loadLoRAWeights(manifest) {
    const prevManifest = this.manifest;
    const prevLocations = new Map(this.tensorLocations);

    this.manifest = manifest;
    // We must rebuild locations so _loadTensor finds them
    await this.#buildTensorLocations();

    try {
      return await loadLoRAWeightsFromModule(
        manifest,
        (name, toGPU, silent) => this.#loadTensor(name, toGPU, silent)
      );
    } finally {
      this.manifest = prevManifest;
      this.tensorLocations = prevLocations;
    }
  }

  
  #resolveWeightLayout(location, name) {
    return resolveWeightLayout(location, name);
  }

  
  #shouldStreamLargeWeight(name, location, label) {
    return shouldStreamLargeWeight(name, location, label, this.gpuCapabilities, this.keepF32Weights);
  }

  
  async load(modelId, options = {}) {
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

    this.#startMemoryLogging();

    if (!this.shardCache.hasCustomLoader) {
      await openModelStore(modelId);
      const manifestJson = await loadManifestFromStore();
      this.manifest = parseManifest(manifestJson);
    }

    if (!this.manifest) {
      throw new Error('No manifest available. Set manifest via setManifest() or ensure OPFS has the model.');
    }

    validateManifestInference(this.manifest);

    const config =  (this.manifest.config);
    const moeConfig = this.manifest.moeConfig;
    this.isMoE = moeConfig != null && (moeConfig.numExperts ?? 0) > 1;
    if (!this.isMoE && (config?.num_local_experts ?? 0) > 1) {
      throw new Error(
        `Manifest "${this.manifest.modelId ?? 'unknown'}" missing moeConfig for MoE model. Re-convert with moeConfig.`
      );
    }

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

    await this.#buildTensorLocations();

    const totalBytes = (this.manifest.shards || []).reduce((sum, s) => sum + (s.size || 0), 0);
    const totalShards = this.manifest.shards?.length || 0;
    const loadStartTime = Date.now();
    let bytesLoaded = 0;
    let shardsLoaded = 0;

    
    const reportProgress = (stage, baseProgress, detail) => {
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

    
    const loadedShardIndices = new Set();
    let inLayerPhase = false;
    const originalLoadShard = (shardIndex) => this.#loadShard(shardIndex);

    
    this.#loadShardOverride = async (shardIndex) => {
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

    
    let loadError = null;
    try {
      reportProgress('shards', 0.1, 'Loading embeddings...');
      await this.#loadEmbeddings(onProgress);

      const numLayers = config?.num_hidden_layers ||
                        config?.blockCount ||
                        config?.text_config?.num_hidden_layers ||
                        config?.n_layer ||
                         (this.manifest.architecture)?.numLayers ||
                        32;
      log.info('Loader', `Layers: 0-${numLayers - 1}`);

      inLayerPhase = true;
      const layersStartTime = performance.now();

      for (let l = 0; l < numLayers; l++) {
        const layerStart = performance.now();
        await this.#loadLayer(l, onProgress);
        const layerElapsed = ((performance.now() - layerStart) / 1000).toFixed(2);
        log.verbose('Loader', `  Layer ${l}: ${layerElapsed}s`);

        await new Promise(r => setTimeout(r, 0));

        const { flushIntervalLayers, flushThresholdBytes, gpuQueueFlushLayers } = this.#loadingConfig.memoryManagement;
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
      await this.#loadFinalWeights(onProgress);

      if (onProgress) {
        onProgress({ stage: 'complete', progress: 1.0 });
      }

      this.isLoaded = true;
      const totalTime = ((Date.now() - loadStartTime) / 1000).toFixed(2);
      const avgSpeed = formatBytes(bytesLoaded / (Date.now() - loadStartTime) * 1000);
      log.info('Loader', `Complete: ${formatBytes(bytesLoaded)} in ${totalTime}s (${avgSpeed}/s)`);

      this.shardCache.clear();

      return  (this.manifest.config) || {};
    } catch (error) {
      loadError = error;
    } finally {
      this.#loadShardOverride = null;
      if (this.#memoryMonitor) {
        this.#stopMemoryLogging(loadError ? 'failed' : 'complete');
      }
    }

    if (loadError) {
      await this.unload();
      if (preservedManifest) {
        this.manifest = preservedManifest;
      }
      throw loadError;
    }
    return  (this.manifest?.config) || {};
  }

  
  async #buildTensorLocations() {
    this.tensorLocations.clear();
    if (!this.manifest) return;

    const locations = await buildTensorLocations(this.manifest, {
      tensorsJsonUrl: this.#tensorsJsonUrl,
      hasCustomLoader: this.shardCache.hasCustomLoader,
    });

    for (const [name, loc] of locations) {
      this.tensorLocations.set(name, loc);
    }
  }

  
  async #loadTensor(name, toGPU = true, silent = false) {
    const location = this.tensorLocations.get(name);
    if (!location) {
      if (!silent) {
        log.warn('Loader', `Tensor not found: ${name}`);
      }
      return null;
    }

    if (name.includes('attn_k') || name.includes('k_proj')) {
      debugTrace.loader(`Loading ${name}: shape=${JSON.stringify(location.shape)}, size=${location.size}, dtype=${location.dtype}, spans=${!!location.spans}`);
    }

    const shardData = await this.#assembleShardData(location, name);

    if (toGPU) {
      const device = getDevice();
      if (!device) {
        log.warn('Loader', 'GPU device not available; falling back to CPU');
        return loadTensorToCPU(shardData, location);
      }

      
      const config = {
        useFusedQ4K: this.useFusedQ4K,
        keepF32Weights: this.keepF32Weights,
        q4kLayout: this.q4kLayout,
        gpuCapabilities: this.gpuCapabilities,
        allowF32UpcastNonMatmul: this.#loadingConfig?.allowF32UpcastNonMatmul ?? false,
      };

      const result = await loadTensorToGPU(shardData, location, name, config);

      for (const buffer of result.allocatedBuffers) {
        this.gpuBuffers.add(buffer);
      }

      return result.data;
    }

    return loadTensorToCPU(shardData, location);
  }

  
  async #assembleShardData(location, name) {
    const loadShard = this.#getLoadShard();
    return assembleShardData(location, name, loadShard);
  }

  
  #needsNormWeightOffset() {
    const result = needsNormWeightOffset(this.manifest);
    if (result && !this.#normOffsetLogged) {
      this.#normOffsetLogged = true;
    }
    return result;
  }

  
  async #loadEmbeddings(_onProgress) {
    
    const ctx = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this.#loadTensor(name, toGPU, silent),
      shouldStreamLargeWeight: (name, loc, label) => this.#shouldStreamLargeWeight(name, loc, label),
      resolveWeightLayout: (loc, name) => this.#resolveWeightLayout(loc, name),
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
    };

    this.embeddings = await loadEmbeddings(ctx);
  }

  
  async #loadLayer(layerIdx, _onProgress) {
    
    const ctx = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this.#loadTensor(name, toGPU, silent),
      needsNormWeightOffset: () => this.#needsNormWeightOffset(),
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      isMoE: this.isMoE,
      isExpertLayer: (idx) => this.#isExpertLayer(idx),
    };

    const weights = await loadLayer(ctx, layerIdx);
    this.layers.set(layerIdx, weights);
  }

  
  #isExpertLayer(_layerIdx) {
    return this.isMoE;
  }

  
  prefetchExperts(nextLayerIdx, expertIndices) {
    prefetchExpertsFromModule(this.#getExpertLoaderContext(), nextLayerIdx, expertIndices, this.isMoE);
  }

  
  predictNextLayerExperts(currentExperts) {
    return predictNextLayerExpertsFromModule(currentExperts);
  }

  
  async loadExpert(layerIdx, expertIdx) {
    return loadExpertFromModule(this.#getExpertLoaderContext(), layerIdx, expertIdx);
  }

  
  #getExpertLoaderContext() {
    const loadShard = this.#getLoadShard();
    return {
      manifest: this.manifest,
      loadTensor: (name, toGPU, silent) => this.#loadTensor(name, toGPU, silent),
      loadShard,
      shardCache: this.shardCache,
      expertCache: this.expertCache,
      experts: this.experts,
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
    };
  }

  
  async #loadFinalWeights(_onProgress) {
    const tieWordEmbeddings = this.manifest?.inference?.output?.tieWordEmbeddings;
    if (tieWordEmbeddings == null) {
      const modelId = this.manifest?.modelId ?? 'unknown';
      throw new Error(
        `Manifest "${modelId}" is missing inference.output.tieWordEmbeddings. ` +
        'Re-convert the model with a complete manifest.inference config.'
      );
    }

    
    const ctx = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this.#loadTensor(name, toGPU, silent),
      needsNormWeightOffset: () => this.#needsNormWeightOffset(),
      shouldStreamLargeWeight: (name, loc, label) => this.#shouldStreamLargeWeight(name, loc, label),
      resolveWeightLayout: (loc, name) => this.#resolveWeightLayout(loc, name),
      embeddings: this.embeddings,
      tieWordEmbeddings,
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      normOffsetDebugLogged: this.#normOffsetDebugLogged,
    };

    const result = await loadFinalWeights(ctx);
    this.finalNorm = result.finalNorm;
    this.lmHead = result.lmHead;
    this.#normOffsetDebugLogged = result.normOffsetDebugLogged;
  }

  
  getLayerWeights(layerIdx) {
    return this.layers.get(layerIdx) || null;
  }

  
  getConfig() {
    return  (this.manifest?.config) || {};
  }

  
  canRunDense() {
    return this.isUnifiedMemory;
  }

  
  getStats() {
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

  
  getExpertCacheStats() {
    return this.expertCache?.getStats() || null;
  }

  
  async unload() {
    debugTrace.loader(' Unloading model...');

    if (this.#memoryMonitor) {
      this.#stopMemoryLogging('complete');
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
    this.#normOffsetLogged = false;

    debugTrace.loader(' Model unloaded');
  }
}


let globalLoader = null;


export function getDopplerLoader(loadingConfig) {
  if (!globalLoader) {
    globalLoader = new DopplerLoader(loadingConfig);
  } else if (loadingConfig) {
    globalLoader.setLoadingConfig(loadingConfig);
  }
  return globalLoader;
}


export function createDopplerLoader(loadingConfig) {
  return new DopplerLoader(loadingConfig);
}

export default DopplerLoader;
