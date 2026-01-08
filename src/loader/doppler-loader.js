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

import { getMemoryCapabilities } from '../memory/capability.js';
import { detectUnifiedMemory } from '../memory/unified-detect.js';
import { getHeapManager } from '../memory/heap-manager.js';
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
} from '../storage/rdrr-format.js';
import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import { releaseBuffer } from '../gpu/buffer-pool.js';
import { getExpertCache } from './expert-cache.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';

import { findAlternativeTensorName } from './dtype-utils.js';

import { createShardCache } from './shard-cache.js';
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

/**
 * DopplerLoader class
 */
export class DopplerLoader {
  // Capabilities
  /** @type {import('../memory/capability.js').MemoryCapabilities | null} */
  memoryCapabilities = null;
  /** @type {import('./loader-types.js').KernelCapabilities | null} */
  gpuCapabilities = null;
  /** @type {boolean} */
  isUnifiedMemory = false;

  // Manifest and model info
  /** @type {import('../storage/rdrr-format.js').RDRRManifest | null} */
  manifest = null;
  /** @type {string | null} */
  modelId = null;
  /** @type {boolean} */
  isMoE = false;

  // Loaded state
  /** @type {boolean} */
  isLoaded = false;
  /** @type {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | import('../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | null} */
  embeddings = null;
  /** @type {Map<number, import('./loader-types.js').LayerWeights>} */
  layers = new Map();
  /** @type {Map<string, import('./weights.js').ExpertWeights>} */
  experts = new Map();
  /** @type {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | import('../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | null} */
  lmHead = null;
  /** @type {GPUBuffer | Float32Array | null} */
  finalNorm = null;

  // Memory management
  /** @type {import('../memory/heap-manager.js').HeapManager | null} */
  heapManager = null;
  /** @type {Set<GPUBuffer>} */
  gpuBuffers = new Set();

  // Expert cache for MoE models (LRU eviction)
  /** @type {import('./expert-cache.js').ExpertCache | null} */
  expertCache = null;

  // Loading state
  /** @type {Set<number>} */
  loadedShards = new Set();
  /** @type {Map<string, import('./loader-types.js').TensorLocation>} */
  tensorLocations = new Map();

  // Shard cache (LRU with request deduplication)
  /** @type {import('./shard-cache.js').ShardCache} */
  shardCache;

  // Loading configuration
  /** @type {import('../config/schema/loading.schema.js').LoadingConfigSchema} */
  #loadingConfig;

  // Fused Q4_K matmul: skip dequantization for matmul weights, use fused kernel
  /** @type {boolean} */
  useFusedQ4K = false;

  // Q4K layout from manifest: 'column_wise' means weights are pre-transposed
  /** @type {'flat' | 'row_wise' | 'column_wise' | null} */
  q4kLayout = null;
  /** @type {boolean} */
  keepF32Weights = false;

  // Internal tracking
  /** @type {boolean} */
  #normOffsetLogged = false;
  /** @type {boolean} */
  #normOffsetDebugLogged = false;
  /** @type {MemoryMonitor | null} */
  #memoryMonitor = null;
  /** @type {string | null} */
  #tensorsJsonUrl = null;

  /**
   * @param {import('../config/schema/loading.schema.js').LoadingConfigSchema} [loadingConfig]
   */
  constructor(loadingConfig) {
    this.#loadingConfig = loadingConfig ?? getRuntimeConfig().loading;
    this.shardCache = createShardCache(
      this.#loadingConfig.shardCache.opfsEntries,
      this.#loadingConfig.shardCache
    );
  }

  /**
   * @param {import('../config/schema/loading.schema.js').LoadingConfigSchema} config
   */
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

  /**
   * @returns {{ shardCacheBytes: number; shardCount: number; layerCount: number; gpuBufferCount: number }}
   */
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

  /**
   * @param {'complete' | 'failed'} [phase='complete']
   */
  #stopMemoryLogging(phase = 'complete') {
    if (this.#memoryMonitor) {
      this.#memoryMonitor.stop(phase, () => this.#getMemoryState());
      this.#memoryMonitor = null;
    }
  }

  /**
   * @param {import('./loader-types.js').CustomShardLoader} loadShardFn
   * @param {import('./loader-types.js').CustomShardLoaderOptions} [options={}]
   */
  setCustomShardLoader(loadShardFn, options = {}) {
    this.shardCache.setCustomLoader(loadShardFn, options.verify !== false);
  }

  /**
   * @param {string | null} url
   */
  setTensorsJsonUrl(url) {
    this.#tensorsJsonUrl = url;
  }

  /**
   * @param {number} shardIndex
   * @returns {Promise<ArrayBuffer>}
   */
  async #loadShard(shardIndex) {
    return this.shardCache.load(shardIndex);
  }

  /**
   * @returns {Promise<void>}
   */
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

    await initOPFS();

    const caps = [
      this.gpuCapabilities.hasF16 ? 'f16' : null,
      this.gpuCapabilities.hasSubgroups ? 'subgroups' : null,
      this.memoryCapabilities.hasMemory64 ? 'mem64' : null,
      this.isUnifiedMemory ? 'unified' : null,
    ].filter(Boolean).join(', ');
    log.info('Loader', `Initialized (${caps})`);
  }

  /**
   * @param {import('../storage/rdrr-format.js').RDRRManifest} manifest
   */
  setManifest(manifest) {
    this.manifest = manifest;
    const config = /** @type {import('./loader-types.js').ModelConfig | undefined} */ (manifest.config);
    this.isMoE = manifest.moeConfig != null || (config?.num_local_experts ?? 0) > 1;
    this.shardCache.configureForModel(this.manifest, this.shardCache.hasCustomLoader);
    this.#configureQ4KStrategy();
    debugTrace.loader('Manifest set externally');
  }

  /**
   * @param {import('../storage/rdrr-format.js').RDRRManifest} manifest
   * @returns {Promise<import('../inference/pipeline/lora.js').LoRAAdapter>}
   */
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

  #configureQ4KStrategy() {
    const config = configureQ4KStrategy(this.manifest, this.gpuCapabilities);
    this.useFusedQ4K = config.useFusedQ4K;
    this.q4kLayout = config.q4kLayout;
    this.keepF32Weights = config.keepF32Weights;
  }

  /**
   * @param {import('./loader-types.js').TensorLocation} location
   * @param {string} name
   * @returns {import('../gpu/weight-buffer.js').WeightLayout}
   */
  #resolveWeightLayout(location, name) {
    return resolveWeightLayout(location, name);
  }

  /**
   * @param {string} name
   * @param {import('./loader-types.js').TensorLocation} location
   * @param {string} label
   * @returns {boolean}
   */
  #shouldStreamLargeWeight(name, location, label) {
    return shouldStreamLargeWeight(name, location, label, this.gpuCapabilities, this.keepF32Weights);
  }

  /**
   * @param {string} modelId
   * @param {import('./loader-types.js').LoadOptions} [options={}]
   * @returns {Promise<import('./loader-types.js').ModelConfig>}
   */
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
      await openModelDirectory(modelId);
      const manifestJson = await loadManifestFromOPFS();
      this.manifest = parseManifest(manifestJson);
    }

    if (!this.manifest) {
      throw new Error('No manifest available. Set manifest via setManifest() or ensure OPFS has the model.');
    }

    validateManifestInference(this.manifest);

    this.#configureQ4KStrategy();

    const config = /** @type {import('./loader-types.js').ModelConfig | undefined} */ (this.manifest.config);
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

    await this.#buildTensorLocations();

    const totalBytes = (this.manifest.shards || []).reduce((sum, s) => sum + (s.size || 0), 0);
    const totalShards = this.manifest.shards?.length || 0;
    const loadStartTime = Date.now();
    let bytesLoaded = 0;
    let shardsLoaded = 0;

    /**
     * @param {import('./loader-types.js').LoadProgress['stage']} stage
     * @param {number} baseProgress
     * @param {string} [detail]
     */
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

    /** @type {Set<number>} */
    const loadedShardIndices = new Set();
    let inLayerPhase = false;
    const originalLoadShard = this.#loadShard.bind(this);

    /**
     * @param {number} shardIndex
     * @returns {Promise<ArrayBuffer>}
     */
    this.#loadShard = async (shardIndex) => {
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

    /** @type {unknown} */
    let loadError = null;
    try {
      reportProgress('shards', 0.1, 'Loading embeddings...');
      await this.#loadEmbeddings(onProgress);

      const numLayers = config?.num_hidden_layers ||
                        config?.blockCount ||
                        config?.text_config?.num_hidden_layers ||
                        config?.n_layer ||
                        /** @type {{ numLayers?: number } | undefined} */ (this.manifest.architecture)?.numLayers ||
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

      return /** @type {import('./loader-types.js').ModelConfig} */ (this.manifest.config) || {};
    } catch (error) {
      loadError = error;
    } finally {
      this.#loadShard = originalLoadShard;
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
    return /** @type {import('./loader-types.js').ModelConfig} */ (this.manifest?.config) || {};
  }

  /**
   * @returns {Promise<void>}
   */
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

  /**
   * @param {string} name
   * @param {boolean} [toGPU=true]
   * @param {boolean} [silent=false]
   * @returns {Promise<GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | Float32Array | Uint8Array | null>}
   */
  async #loadTensor(name, toGPU = true, silent = false) {
    const location = this.tensorLocations.get(name);
    if (!location) {
      const altName = findAlternativeTensorName(name, this.tensorLocations);
      if (altName) {
        return this.#loadTensor(altName, toGPU, silent);
      }
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

      /** @type {import('./tensor-loader.js').TensorLoadConfig} */
      const config = {
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

  /**
   * @param {import('./loader-types.js').TensorLocation} location
   * @param {string} name
   * @returns {Promise<Uint8Array>}
   */
  async #assembleShardData(location, name) {
    return assembleShardData(location, name, (idx) => this.#loadShard(idx));
  }

  /**
   * @returns {boolean}
   */
  #needsNormWeightOffset() {
    const result = needsNormWeightOffset(this.manifest);
    if (result && !this.#normOffsetLogged) {
      this.#normOffsetLogged = true;
    }
    return result;
  }

  /**
   * @param {((progress: import('./loader-types.js').LoadProgress) => void) | null} _onProgress
   * @returns {Promise<void>}
   */
  async #loadEmbeddings(_onProgress) {
    /** @type {import('./embedding-loader.js').EmbeddingLoaderContext} */
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

  /**
   * @param {number} layerIdx
   * @param {((progress: import('./loader-types.js').LoadProgress) => void) | null} _onProgress
   * @returns {Promise<void>}
   */
  async #loadLayer(layerIdx, _onProgress) {
    /** @type {import('./layer-loader.js').LayerLoaderContext} */
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

  /**
   * @param {number} _layerIdx
   * @returns {boolean}
   */
  #isExpertLayer(_layerIdx) {
    return this.isMoE;
  }

  /**
   * @param {number} nextLayerIdx
   * @param {number[]} expertIndices
   */
  prefetchExperts(nextLayerIdx, expertIndices) {
    prefetchExpertsFromModule(this.#getExpertLoaderContext(), nextLayerIdx, expertIndices, this.isMoE);
  }

  /**
   * @param {number[]} currentExperts
   * @returns {number[]}
   */
  predictNextLayerExperts(currentExperts) {
    return predictNextLayerExpertsFromModule(currentExperts);
  }

  /**
   * @param {number} layerIdx
   * @param {number} expertIdx
   * @returns {Promise<import('./weights.js').ExpertWeights>}
   */
  async loadExpert(layerIdx, expertIdx) {
    return loadExpertFromModule(this.#getExpertLoaderContext(), layerIdx, expertIdx);
  }

  /**
   * @returns {import('./expert-loader.js').ExpertLoaderContext}
   */
  #getExpertLoaderContext() {
    return {
      manifest: this.manifest,
      loadTensor: (name, toGPU, silent) => this.#loadTensor(name, toGPU, silent),
      loadShard: (idx) => this.#loadShard(idx),
      shardCache: this.shardCache,
      expertCache: this.expertCache,
      experts: this.experts,
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
    };
  }

  /**
   * @param {((progress: import('./loader-types.js').LoadProgress) => void) | null} _onProgress
   * @returns {Promise<void>}
   */
  async #loadFinalWeights(_onProgress) {
    /** @type {import('./final-weights-loader.js').FinalWeightsContext} */
    const ctx = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this.#loadTensor(name, toGPU, silent),
      needsNormWeightOffset: () => this.#needsNormWeightOffset(),
      shouldStreamLargeWeight: (name, loc, label) => this.#shouldStreamLargeWeight(name, loc, label),
      resolveWeightLayout: (loc, name) => this.#resolveWeightLayout(loc, name),
      embeddings: this.embeddings,
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      normOffsetDebugLogged: this.#normOffsetDebugLogged,
    };

    const result = await loadFinalWeights(ctx);
    this.finalNorm = result.finalNorm;
    this.lmHead = result.lmHead;
    this.#normOffsetDebugLogged = result.normOffsetDebugLogged;
  }

  /**
   * @param {number} layerIdx
   * @returns {import('./loader-types.js').LayerWeights | null}
   */
  getLayerWeights(layerIdx) {
    return this.layers.get(layerIdx) || null;
  }

  /**
   * @returns {import('./loader-types.js').ModelConfig}
   */
  getConfig() {
    return /** @type {import('./loader-types.js').ModelConfig} */ (this.manifest?.config) || {};
  }

  /**
   * @returns {boolean}
   */
  canRunDense() {
    return this.isUnifiedMemory;
  }

  /**
   * @returns {import('./loader-types.js').LoaderStats}
   */
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

  /**
   * @returns {import('./expert-cache.js').CacheStats | null}
   */
  getExpertCacheStats() {
    return this.expertCache?.getStats() || null;
  }

  /**
   * @returns {Promise<void>}
   */
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

/** @type {DopplerLoader | null} */
let globalLoader = null;

/**
 * @param {import('../config/schema/loading.schema.js').LoadingConfigSchema} [loadingConfig]
 * @returns {DopplerLoader}
 */
export function getDopplerLoader(loadingConfig) {
  if (!globalLoader) {
    globalLoader = new DopplerLoader(loadingConfig);
  } else if (loadingConfig) {
    globalLoader.setLoadingConfig(loadingConfig);
  }
  return globalLoader;
}

/**
 * @param {import('../config/schema/loading.schema.js').LoadingConfigSchema} [loadingConfig]
 * @returns {DopplerLoader}
 */
export function createDopplerLoader(loadingConfig) {
  return new DopplerLoader(loadingConfig);
}

export default DopplerLoader;
