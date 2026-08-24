

import { getMemoryCapabilities } from '../memory/capability.js';
import { detectUnifiedMemory } from '../memory/unified-detect.js';
import { getHeapManager } from '../memory/heap-manager.js';
import {
  initStorage,
  openModelStore,
  verifyIntegrity,
  loadManifestFromStore,
  loadAuxFile,
  computeHash,
} from '../storage/shard-manager.js';
import { clearManifest, parseManifest, setManifest as setCurrentManifest } from '../formats/rdrr/index.js';
import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import {
  PersistentBufferSet,
  acquireBuffer,
  isBufferActive,
  releaseBuffer,
  forceBufferPoolReclaim,
} from '../memory/buffer-pool.js';
import { getExpertCache } from './experts/expert-cache.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { isGpuBufferInstance, isWeightBuffer } from '../gpu/weight-buffer.js';

import { createShardCache } from './shard-cache.js';
import { validateManifestInference } from '../config/schema/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

// Import helper modules for refactored logic
import { buildTensorLocations } from './shard-resolver.js';
import {
  needsNormWeightOffset,
  resolveWeightLayout,
  requiresCpuF16ToF32MatmulMaterialization,
  shouldStreamLargeWeight,
} from './manifest-config.js';
import { MemoryMonitor } from './memory-monitor.js';
import {
  loadTensorToGPU,
  loadTensorToCPU,
  isLiteRTAffineInt4FusedEligible,
} from './tensors/tensor-loader.js';
import { detectMoE } from './model-load-validation.js';
import { getTensorShardIndices } from './tensor-shard-indices.js';
import { loadEmbeddings } from './embedding-loader.js';
import { loadPerLayerInputWeights } from './per-layer-input-loader.js';
import { loadLayer } from './layer-loader.js';
import { loadFinalWeights } from './final-weights-loader.js';
import { cloneJsonValue } from '../formats/clone-json.js';
import { assertFunctionalDescriptorManifest } from '../formats/rdrr/functional-descriptor.js';
import {
  loadExpert as loadExpertFromModule,
  prefetchExperts as prefetchExpertsFromModule,
  predictNextLayerExperts as predictNextLayerExpertsFromModule,
} from './experts/expert-loader.js';
import { assembleShardData } from './tensors/tensor-reader.js';
import { hasSourceTransform } from './tensors/source-transform.js';
import {
  load as loadImpl,
  _loadTensor as _loadTensorImpl,
  _getDescriptorShardFiles as _getDescriptorShardFilesImpl,
  _loadDescriptorShardFile as _loadDescriptorShardFileImpl,
  _assertDescriptorHash as _assertDescriptorHashImpl,
  _assembleFunctionalDescriptorData as _assembleFunctionalDescriptorDataImpl,
  _assembleShardData as _assembleShardDataImpl,
  _shouldStreamUploadToGPU as _shouldStreamUploadToGPUImpl,
  _assembleShardDataToGpuBuffer as _assembleShardDataToGpuBufferImpl,
  _loadEmbeddings as _loadEmbeddingsImpl,
  _loadLayer as _loadLayerImpl,
  _loadFinalWeights as _loadFinalWeightsImpl,
} from './model-executor.js';

let loraModulePromise = null;

async function getExperimentalLoRAModule() {
  loraModulePromise ??= import('../experimental/adapters/lora-loader.js');
  return loraModulePromise;
}

function isDirectSourceRuntimeManifest(manifest) {
  const sourceRuntime = manifest?.metadata?.sourceRuntime;
  const mode = typeof sourceRuntime?.mode === 'string'
    ? sourceRuntime.mode.trim().toLowerCase()
    : '';
  const sourceKind = typeof sourceRuntime?.sourceKind === 'string'
    ? sourceRuntime.sourceKind.trim().toLowerCase()
    : '';
  return mode === 'direct-source' && sourceKind !== 'rdrr';
}

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

  lmHeadBias = null;
  finalNormBias = null;
  
  finalNorm = null;

  embeddingPostprocessor = null;

  diffusionGemmaSelfConditioning = null;

  perLayerInputWeights = null;

  // Memory management
  
  heapManager = null;
  
  gpuBuffers = new PersistentBufferSet();

  // Expert cache for MoE models (LRU eviction)
  
  expertCache = null;

  // Loading state
  
  loadedShards = new Set();
  
  tensorLocations = new Map();

  loadTiming = null;

  // Shard cache (LRU with request deduplication)
  
  shardCache;

  // Loading configuration
  
  _loadingConfig;
  _loaderDebug = null;

  _perLayerInputSession = null;
  _loadAuxiliaryFile = null;

  // Fused Q4_K matmul: skip dequantization for matmul weights, use fused kernel
  
  useFusedQ4K = false;

  // Q4K layout: 'col' = dequant fallback, 'row' = fused kernel (optimal)
  
  q4kLayout = null;
  
  keepF32Weights = false;
  keepBF16Weights = false;

  q4kMaterializationMode = 'dense';
  q4kFusedRoles = [];

  // Internal tracking
  
  _normOffsetLogged = false;
  
  _normOffsetDebugLogged = false;
  
  _memoryMonitor = null;
  
  _tensorsJsonUrl = null;
  _loadTensorsJson = null;
  
  _loadShardOverride = null;

  _layerShardMap = new Map();

  constructor(loadingConfig) {
    if (!loadingConfig) {
      log.debug('Loader', 'No explicit loadingConfig provided; falling back to getRuntimeConfig().loading');
    }
    this._loadingConfig = loadingConfig ?? getRuntimeConfig().loading;
    this.shardCache = createShardCache(
      this._loadingConfig.shardCache.opfsEntries,
      this._loadingConfig.shardCache
    );
  }

  
  setLoadingConfig(config) {
    this._loadingConfig = config;
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

  setLoaderDebugConfig(loaderDebug) {
    this._loaderDebug = loaderDebug ?? null;
  }

  setPerLayerInputSession(sessionConfig) {
    this._perLayerInputSession = sessionConfig ?? null;
  }

  
  setQ4KConfig(config) {
    this.useFusedQ4K = config.useFusedQ4K;
    this.q4kLayout = config.q4kLayout;
    this.keepF32Weights = config.keepF32Weights;
    this.keepBF16Weights = config.keepBF16Weights === true;
    this.q4kMaterializationMode = config.q4kMaterializationMode ?? 'dense';
    this.q4kFusedRoles = Array.isArray(config.q4kFusedRoles)
      ? [...new Set(config.q4kFusedRoles.map((role) => String(role)).filter(Boolean))].sort()
      : [];
  }

  
  _getMemoryState() {
    return {
      shardCacheBytes: this.shardCache.totalBytes,
      shardCount: this.shardCache.size,
      layerCount: this.layers.size,
      gpuBufferCount: this.gpuBuffers.size,
    };
  }

  _startMemoryLogging() {
    const logIntervalMs = this._loadingConfig.memoryManagement.logIntervalMs;
    this._memoryMonitor = new MemoryMonitor(logIntervalMs);
    this._memoryMonitor.start(() => this._getMemoryState());
  }

  _assertResidentBudget(phase) {
    const budgetConfig = this._loadingConfig?.memoryManagement?.budget;
    if (!budgetConfig || budgetConfig.enabled !== true) {
      return;
    }
    const maxResidentBytes = Number(budgetConfig.maxResidentBytes);
    if (!Number.isFinite(maxResidentBytes) || maxResidentBytes <= 0) {
      return;
    }
    if (typeof process === 'undefined' || typeof process.memoryUsage !== 'function') {
      return;
    }
    const rssBytes = process.memoryUsage().rss;
    if (rssBytes <= maxResidentBytes) {
      return;
    }
    throw new Error(
      `Loader resident memory budget exceeded during ${phase}: ` +
      `rss=${formatBytes(rssBytes)}, max=${formatBytes(maxResidentBytes)}. ` +
      'Lower the model working set or raise runtime.loading.memoryManagement.budget.maxResidentBytes.'
    );
  }

  
  _stopMemoryLogging(phase = 'complete') {
    if (this._memoryMonitor) {
      this._memoryMonitor.stop(phase, () => this._getMemoryState());
      this._memoryMonitor = null;
    }
  }

  
  setCustomShardLoader(loadShardFn, options = {}) {
    this.shardCache.setCustomLoader(loadShardFn, options.verify !== false, {
      loadRange: options.loadShardRange ?? null,
      streamRange: options.streamShardRange ?? null,
    });
    this._loadAuxiliaryFile = typeof options.loadAuxiliaryFile === 'function'
      ? options.loadAuxiliaryFile
      : null;
  }

  setAuxiliaryFileLoader(loadAuxiliaryFile) {
    this._loadAuxiliaryFile = typeof loadAuxiliaryFile === 'function' ? loadAuxiliaryFile : null;
  }

  
  setTensorsJsonUrl(url) {
    this._tensorsJsonUrl = url;
  }

  setTensorsJsonLoader(loadTensorsJson) {
    this._loadTensorsJson = typeof loadTensorsJson === 'function' ? loadTensorsJson : null;
  }

  
  async _loadShard(shardIndex, options) {
    return this.shardCache.load(shardIndex, options);
  }

  
  _getLoadShard() {
    return this._loadShardOverride ?? ((idx, options) => this._loadShard(idx, options));
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
    setCurrentManifest(manifest);
    this.isMoE = detectMoE(manifest);
    this.shardCache.setManifest(this.manifest);
    this.shardCache.configureForModel(this.manifest, this.shardCache.hasCustomLoader);
    debugTrace.loader('Manifest set externally');
  }

  
  async loadLoRAWeights(manifest) {
    const prevManifest = this.manifest;
    const prevLocations = new Map(this.tensorLocations);
    const prevLayerShardMap = new Map(this._layerShardMap);

    try {
      const { loadLoRAWeights } = await getExperimentalLoRAModule();
      this.manifest = manifest;
      // We must rebuild locations so _loadTensor finds them
      await this._buildTensorLocations();
      this._logWeightBreakdown();

      return await loadLoRAWeights(
        manifest,
        (name, toGPU, silent) => this._loadTensor(name, toGPU, silent)
      );
    } finally {
      // Always restore previous state, even if buildTensorLocations or the
      // LoRA load itself throws, to avoid leaving the loader in an
      // inconsistent intermediate state.
      this.manifest = prevManifest;
      this.tensorLocations = prevLocations;
      this._layerShardMap = prevLayerShardMap;
    }
  }

  
  _resolveWeightLayout(location) {
    return resolveWeightLayout(location);
  }

  
  _shouldStreamLargeWeight(name, location, label) {
    const manifestOverrides = this.manifest?.inference?.largeWeights?.gpuResidentOverrides ?? null;
    return shouldStreamLargeWeight(name, location, label, this.gpuCapabilities, this.keepF32Weights, manifestOverrides);
  }

  
  async load(modelId, options = {}) {
    return loadImpl.call(this, modelId, options);
  }

  
  async _buildTensorLocations() {
    this.tensorLocations.clear();
    if (!this.manifest) {
      this._layerShardMap.clear();
      return;
    }

    const locations = await buildTensorLocations(this.manifest, {
      tensorsJsonUrl: this._tensorsJsonUrl,
      loadTensorsJson: this._loadTensorsJson,
      hasCustomLoader: this.shardCache.hasCustomLoader,
    });

    for (const [name, loc] of locations) {
      this.tensorLocations.set(name, loc);
    }

    this._buildLayerShardMap();
  }

  _buildLayerShardMap() {
    this._layerShardMap.clear();

    for (const [, location] of this.tensorLocations) {
      const layerIdx = getLayerIndexFromGroup(location.group);
      if (layerIdx == null || isExpertGroup(location.group)) {
        continue;
      }

      let shards = this._layerShardMap.get(layerIdx);
      if (!shards) {
        shards = new Set();
        this._layerShardMap.set(layerIdx, shards);
      }

      for (const shardIndex of getTensorShardIndices(location)) {
        shards.add(shardIndex);
      }
    }
  }

  _logWeightBreakdown() {
    if (this.tensorLocations.size === 0) return;

    let totalBytes = 0;
    let expertBytes = 0;

    for (const [, location] of this.tensorLocations) {
      const size = location.size || 0;
      totalBytes += size;
      if (location.role === 'expert') {
        expertBytes += size;
      }
    }

    if (expertBytes > 0) {
      const denseBytes = totalBytes - expertBytes;
      debugTrace.loader(
        `Weights: dense=${formatBytes(denseBytes)}, experts=${formatBytes(expertBytes)} (total=${formatBytes(totalBytes)})`
      );
    }
  }

  _prefetchLayerShards(layerIdx) {
    const prefetch = this._loadingConfig.prefetch;
    if (!prefetch?.enabled) return;
    // Range-capable custom loaders are expected to serve fine-grained tensor reads.
    // Whole-shard prefetch defeats that contract and can force invalid >4 GiB reads
    // for direct-source SafeTensors bundles.
    if (this.shardCache.hasCustomLoader && this.shardCache.canStreamRanges) {
      if (prefetch.allowRangeLoaderPrefetch !== true) return;
      if (isDirectSourceRuntimeManifest(this.manifest)) {
        throw new Error(
          'runtime.loading.prefetch.allowRangeLoaderPrefetch is only supported for RDRR shard manifests; ' +
          'direct-source artifacts must use range reads.'
        );
      }
      if (!Array.isArray(this.manifest?.shards) || this.manifest.shards.length === 0) {
        throw new Error(
          'runtime.loading.prefetch.allowRangeLoaderPrefetch requires a manifest with explicit RDRR shards.'
        );
      }
    }

    const layersAhead = prefetch.layersAhead;
    if (!Number.isFinite(layersAhead) || layersAhead <= 0) return;
    if (this._layerShardMap.size === 0) return;

    const maxShards = prefetch.maxShards;
    const hasLimit = maxShards > 0;
    let scheduled = 0;
    const loadShard = this._getLoadShard();

    for (let idx = layerIdx + 1; idx <= layerIdx + layersAhead; idx++) {
      const shards = this._layerShardMap.get(idx);
      if (!shards) continue;

      for (const shardIndex of shards) {
        if (this.shardCache.has(shardIndex)) continue;

        loadShard(shardIndex, { priority: 'low' }).catch(() => {});
        scheduled++;
        if (hasLimit && scheduled >= maxShards) return;
      }
    }
  }

  
  async _loadTensor(name, toGPU = true, silent = false) {
    return _loadTensorImpl.call(this, name, toGPU, silent);
  }

  _isFunctionalDescriptorLocation(location) {
    return String(location?.dtype || '').trim().toUpperCase() === 'FUNCTIONAL_DESCRIPTOR';
  }

  _getDescriptorShardFiles(location, name) {
    return _getDescriptorShardFilesImpl.call(this, location, name);
  }

  async _loadDescriptorShardFile(file, name) {
    return _loadDescriptorShardFileImpl.call(this, file, name);
  }

  async _assertDescriptorHash(location, name, descriptorShards) {
    return _assertDescriptorHashImpl.call(this, location, name, descriptorShards);
  }

  async _assembleFunctionalDescriptorData(location, name) {
    return _assembleFunctionalDescriptorDataImpl.call(this, location, name);
  }

  
  async _assembleShardData(location, name, options = {}) {
    return _assembleShardDataImpl.call(this, location, name, options);
  }

  _shouldStreamUploadToGPU(location) {
    return _shouldStreamUploadToGPUImpl.call(this, location);
  }

  async _assembleShardDataToGpuBuffer(location, name) {
    return _assembleShardDataToGpuBufferImpl.call(this, location, name);
  }

  
  _needsNormWeightOffset() {
    const result = needsNormWeightOffset(this.manifest);
    if (result && !this._normOffsetLogged) {
      this._normOffsetLogged = true;
    }
    return result;
  }

  
  async _loadEmbeddings(_onProgress) {
    return _loadEmbeddingsImpl.call(this, _onProgress);
  }

  
  async _loadLayer(layerIdx, _onProgress) {
    return _loadLayerImpl.call(this, layerIdx, _onProgress);
  }

  
  _isExpertLayer(_layerIdx) {
    return this.isMoE;
  }

  
  prefetchExperts(nextLayerIdx, expertIndices) {
    prefetchExpertsFromModule(this._getExpertLoaderContext(), nextLayerIdx, expertIndices, this.isMoE);
  }

  
  predictNextLayerExperts(currentExperts) {
    return predictNextLayerExpertsFromModule(currentExperts);
  }

  
  async loadExpert(layerIdx, expertIdx) {
    return loadExpertFromModule(this._getExpertLoaderContext(), layerIdx, expertIdx);
  }

  
  _getExpertLoaderContext() {
    const loadShard = this._getLoadShard();
    return {
      manifest: this.manifest,
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this._loadTensor(name, toGPU, silent),
      loadShard,
      shardCache: this.shardCache,
      expertCache: this.expertCache,
      experts: this.experts,
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
    };
  }

  
  async _loadFinalWeights(_onProgress) {
    return _loadFinalWeightsImpl.call(this, _onProgress);
  }

  
  getLayerWeights(layerIdx) {
    return this.layers.get(layerIdx) || null;
  }

  async loadTensor(name, toGPU = true, silent = false) {
    return this._loadTensor(name, toGPU, silent);
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
      loadTiming: this.getLoadTiming(),
    };
  }

  getLoadTiming() {
    return this.loadTiming ? cloneJsonValue(this.loadTiming) : null;
  }

  
  getExpertCacheStats() {
    return this.expertCache?.getStats() || null;
  }

  
  async unload() {
    debugTrace.loader(' Unloading model...');

    if (this._memoryMonitor) {
      this._stopMemoryLogging('complete');
    }

    const releaseCandidate = (value) => {
      if (!value) return;
      const gpuBuffer = isWeightBuffer(value)
        ? value.buffer
        : (isGpuBufferInstance(value) ? value : null);
      if (!gpuBuffer) return;
      try {
        if (isBufferActive(gpuBuffer)) {
          releaseBuffer(gpuBuffer);
        } else {
          gpuBuffer.destroy();
        }
      } catch {
        // Ignore already released/destroyed buffers.
      }
      this.gpuBuffers.delete(gpuBuffer);
    };
    const releaseSplitCandidate = (value) => {
      const sections = value?.gpuSplitWeight?.sections;
      if (!Array.isArray(sections)) {
        return;
      }
      for (const section of sections) {
        releaseCandidate(section?.buffer ?? null);
      }
      value.gpuSplitWeight = null;
    };

    for (const buffer of this.gpuBuffers) {
      releaseCandidate(buffer);
    }
    releaseSplitCandidate(this.lmHead);
    this.gpuBuffers.clear();

    if (this.expertCache) {
      this.expertCache.clear();
    }

    for (const packed of this.experts.values()) {
      if (!packed || typeof packed !== 'object') continue;
      releaseCandidate(packed.gate);
      releaseCandidate(packed.up);
      releaseCandidate(packed.down);
      releaseCandidate(packed.gateUpBlocks);
      releaseCandidate(packed.gateUpScales);
      releaseCandidate(packed.gateUpBias);
      releaseCandidate(packed.downBlocks);
      releaseCandidate(packed.downScales);
      releaseCandidate(packed.downBias);
    }

    forceBufferPoolReclaim();

    this.embeddings = null;
    this.layers.clear();
    this.experts.clear();
    this.lmHead = null;
    this.lmHeadBias = null;
    this.finalNormBias = null;
    this.finalNorm = null;
    this.embeddingPostprocessor = null;
    this.perLayerInputWeights = null;
    this.manifest = null;
    clearManifest();
    this.modelId = null;
    this.loadedShards.clear();
    this.isLoaded = false;
    this.loadTiming = null;
    this.tensorLocations.clear();
    this._layerShardMap.clear();
    this.shardCache.clear();
    this._normOffsetLogged = false;

    debugTrace.loader(' Model unloaded');
  }
}

function getLayerIndexFromGroup(group) {
  if (!group) return null;
  const match = /^layer\.(\d+)/.exec(group);
  if (!match) return null;
  const layerIdx = Number(match[1]);
  return Number.isFinite(layerIdx) ? layerIdx : null;
}

function isExpertGroup(group) {
  if (!group) return false;
  return group.includes('.expert.') || group.includes('.shared_expert');
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
