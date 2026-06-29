

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
import { isWeightBuffer } from '../gpu/weight-buffer.js';


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
  isLiteRTAffineInt4FusedEligible,
} from './tensors/tensor-loader.js';
import { annotateWeightLoadError } from '../inference/pipelines/text/load-errors.js';
import { loadEmbeddings } from './embedding-loader.js';
import { loadPerLayerInputWeights } from './per-layer-input-loader.js';
import { loadLayer } from './layer-loader.js';
import { loadFinalWeights } from './final-weights-loader.js';
import { cloneJsonValue } from '../utils/clone-json.js';
import { assertFunctionalDescriptorManifest } from '../formats/rdrr/functional-descriptor.js';
import {
  loadExpert as loadExpertFromModule,
  prefetchExperts as prefetchExpertsFromModule,
  predictNextLayerExperts as predictNextLayerExpertsFromModule,
} from './experts/expert-loader.js';
import { assembleShardData } from './tensors/tensor-reader.js';
import { hasSourceTransform } from './tensors/source-transform.js';

let loraModulePromise = null;

async function getExperimentalLoRAModule() {
  loraModulePromise ??= import('../experimental/adapters/lora-loader.js');
  return loraModulePromise;
}

function hasExpertGroups(manifest) {
  if (!manifest?.groups) return false;
  return Object.keys(manifest.groups).some((groupId) => groupId.includes('.expert.'));
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

/**
 * Detect whether a manifest describes a Mixture-of-Experts model.
 * Validates that manifests with expert groups also have moeConfig.
 * Used by both setManifest() and load() to avoid duplicating MoE detection logic.
 */
function detectMoE(manifest) {
  const moeConfig = manifest.moeConfig;
  const isMoE = moeConfig != null && (moeConfig.numExperts ?? 0) > 1;
  if (!isMoE && hasExpertGroups(manifest)) {
    throw new Error(
      `Manifest "${manifest.modelId ?? 'unknown'}" missing moeConfig for MoE model. Re-convert with moeConfig.`
    );
  }
  return isMoE;
}

function isGpuBufferInstance(value) {
  return typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer;
}

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function roundLoadTimingMs(value) {
  return Number.isFinite(value) ? Number(value.toFixed(3)) : null;
}

function createLoadTiming(modelId, hasCustomLoader) {
  return {
    schemaVersion: 1,
    source: 'doppler-loader',
    modelId: typeof modelId === 'string' ? modelId : null,
    status: 'running',
    customShardLoader: hasCustomLoader === true,
    byteAccountingMode: hasCustomLoader === true
      ? 'custom-loader-progress-unavailable'
      : 'full-shard-progress',
    totalBytes: null,
    totalShards: null,
    bytesLoaded: 0,
    shardsLoaded: 0,
    bytesPerSecond: null,
    phasesMs: {
      preflight: null,
      tensorLocations: null,
      embeddings: null,
      layers: null,
      finalWeights: null,
      cleanup: null,
    },
    layers: {
      count: null,
      totalMs: null,
      meanMs: null,
      maxMs: null,
      maxLayer: null,
    },
    totalMs: null,
    failedPhase: null,
    error: null,
  };
}

function finishLoadPhase(loadTiming, phase, startMs) {
  if (!loadTiming?.phasesMs || !phase) return;
  loadTiming.phasesMs[phase] = roundLoadTimingMs(nowMs() - startMs);
}

function finishLoadTiming(loadTiming, status, startMs, error = null, failedPhase = null) {
  if (!loadTiming) return;
  const totalMs = nowMs() - startMs;
  loadTiming.status = status;
  loadTiming.totalMs = roundLoadTimingMs(totalMs);
  loadTiming.bytesPerSecond = totalMs > 0 && Number.isFinite(loadTiming.bytesLoaded)
    ? Math.round((loadTiming.bytesLoaded / totalMs) * 1000)
    : null;
  if (status === 'failed') {
    loadTiming.failedPhase = failedPhase;
    loadTiming.error = error?.message ?? String(error ?? 'unknown load error');
  }
}

function getTensorShardIndices(location) {
  if (Array.isArray(location?.spans) && location.spans.length > 0) {
    return Array.from(new Set(location.spans.map((span) => span?.shardIndex).filter(Number.isInteger)));
  }
  return Number.isInteger(location?.shardIndex) ? [location.shardIndex] : [];
}

function annotateTensorLoadError(error, name, location, details = {}) {
  return annotateWeightLoadError(error, {
    tensorName: name,
    tensorRole: location?.role ?? null,
    tensorDtype: location?.dtype ?? null,
    tensorShape: Array.isArray(location?.shape) ? [...location.shape] : null,
    tensorSizeBytes: Number.isFinite(location?.size) ? location.size : null,
    tensorShardIndices: getTensorShardIndices(location),
    ...details,
  });
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
  
  #loadingConfig;
  #loaderDebug = null;

  #perLayerInputSession = null;
  #loadAuxiliaryFile = null;

  // Fused Q4_K matmul: skip dequantization for matmul weights, use fused kernel
  
  useFusedQ4K = false;

  // Q4K layout: 'col' = dequant fallback, 'row' = fused kernel (optimal)
  
  q4kLayout = null;
  
  keepF32Weights = false;

  q4kMaterializationMode = 'dense';

  q4kFusedRoles = [];

  // Internal tracking
  
  #normOffsetLogged = false;
  
  #normOffsetDebugLogged = false;
  
  #memoryMonitor = null;
  
  #tensorsJsonUrl = null;
  #loadTensorsJson = null;
  
  #loadShardOverride = null;

  #layerShardMap = new Map();

  
  constructor(loadingConfig) {
    if (!loadingConfig) {
      log.debug('Loader', 'No explicit loadingConfig provided; falling back to getRuntimeConfig().loading');
    }
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

  setLoaderDebugConfig(loaderDebug) {
    this.#loaderDebug = loaderDebug ?? null;
  }

  setPerLayerInputSession(sessionConfig) {
    this.#perLayerInputSession = sessionConfig ?? null;
  }

  
  setQ4KConfig(config) {
    this.useFusedQ4K = config.useFusedQ4K;
    this.q4kLayout = config.q4kLayout;
    this.keepF32Weights = config.keepF32Weights;
    this.q4kMaterializationMode = config.q4kMaterializationMode ?? 'dense';
    this.q4kFusedRoles = Array.isArray(config.q4kFusedRoles)
      ? [...new Set(config.q4kFusedRoles.map((role) => String(role)).filter(Boolean))].sort()
      : [];
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

  #assertResidentBudget(phase) {
    const budgetConfig = this.#loadingConfig?.memoryManagement?.budget;
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

  
  #stopMemoryLogging(phase = 'complete') {
    if (this.#memoryMonitor) {
      this.#memoryMonitor.stop(phase, () => this.#getMemoryState());
      this.#memoryMonitor = null;
    }
  }

  
  setCustomShardLoader(loadShardFn, options = {}) {
    this.shardCache.setCustomLoader(loadShardFn, options.verify !== false, {
      loadRange: options.loadShardRange ?? null,
      streamRange: options.streamShardRange ?? null,
    });
    this.#loadAuxiliaryFile = typeof options.loadAuxiliaryFile === 'function'
      ? options.loadAuxiliaryFile
      : null;
  }

  setAuxiliaryFileLoader(loadAuxiliaryFile) {
    this.#loadAuxiliaryFile = typeof loadAuxiliaryFile === 'function' ? loadAuxiliaryFile : null;
  }

  
  setTensorsJsonUrl(url) {
    this.#tensorsJsonUrl = url;
  }

  setTensorsJsonLoader(loadTensorsJson) {
    this.#loadTensorsJson = typeof loadTensorsJson === 'function' ? loadTensorsJson : null;
  }

  
  async #loadShard(shardIndex, options) {
    return this.shardCache.load(shardIndex, options);
  }

  
  #getLoadShard() {
    return this.#loadShardOverride ?? ((idx, options) => this.#loadShard(idx, options));
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
    const prevLayerShardMap = new Map(this.#layerShardMap);

    try {
      const { loadLoRAWeights } = await getExperimentalLoRAModule();
      this.manifest = manifest;
      // We must rebuild locations so _loadTensor finds them
      await this.#buildTensorLocations();
      this.#logWeightBreakdown();

      return await loadLoRAWeights(
        manifest,
        (name, toGPU, silent) => this.#loadTensor(name, toGPU, silent)
      );
    } finally {
      // Always restore previous state, even if buildTensorLocations or the
      // LoRA load itself throws, to avoid leaving the loader in an
      // inconsistent intermediate state.
      this.manifest = prevManifest;
      this.tensorLocations = prevLocations;
      this.#layerShardMap = prevLayerShardMap;
    }
  }

  
  #resolveWeightLayout(location) {
    return resolveWeightLayout(location);
  }

  
  #shouldStreamLargeWeight(name, location, label) {
    const manifestOverrides = this.manifest?.inference?.largeWeights?.gpuResidentOverrides ?? null;
    return shouldStreamLargeWeight(name, location, label, this.gpuCapabilities, this.keepF32Weights, manifestOverrides);
  }

  
  async load(modelId, options = {}) {
    const { onProgress = null, verifyHashes } = options;
    if (verifyHashes == null) {
      throw new Error('Loader.load requires explicit verifyHashes (runtime.loading.shardCache.verifyHashes).');
    }

    if (!this.heapManager) {
      await this.init();
    }

    // Check order matters: isLoaded is the fast-path indicator; modelId catches
    // partial loads that set the ID before completing; tensorLocations/shardCache
    // detect interrupted builds; layers/experts/gpuBuffers catch residual GPU
    // state from a prior model that was never fully unloaded.
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
    const loadTimingStart = nowMs();
    let activeLoadPhase = 'preflight';
    let phaseStart = loadTimingStart;
    this.loadTiming = createLoadTiming(modelId, this.shardCache.hasCustomLoader);

    this.#startMemoryLogging();
    this.#assertResidentBudget('load start');

    if (!this.shardCache.hasCustomLoader) {
      await openModelStore(modelId);
      const manifestJson = await loadManifestFromStore();
      this.manifest = parseManifest(manifestJson);
    }

    if (!this.manifest) {
      throw new Error('No manifest available. Set manifest via setManifest() or ensure OPFS has the model.');
    }

    validateManifestInference(this.manifest);

    this.isMoE = detectMoE(this.manifest);

    this.shardCache.configureForModel(this.manifest, this.shardCache.hasCustomLoader);

    if (!this.isMoE && !this.isUnifiedMemory) {
      log.warn('Loader', 'Dense model on discrete GPU - performance limited. Consider MoE model.');
    }

    if (!this.shardCache.hasCustomLoader) {
      const integrity = await verifyIntegrity({ checkHashes: false });
      if (!integrity.valid) {
        throw new Error(
          `Artifact contract preflight failed for "${this.manifest?.modelId ?? modelId}". ` +
          `Missing shards: ${integrity.missingShards.length}, ` +
          `corrupt shards: ${integrity.corruptShards.length}. ` +
          'Re-import, re-download, or provide a manifest with a valid weightsRef.'
        );
      }
    }

    const totalBytes = (this.manifest.shards || []).reduce((sum, s) => sum + (s.size || 0), 0);
    const totalShards = this.manifest.shards?.length || 0;
    this.loadTiming.totalBytes = totalBytes;
    this.loadTiming.totalShards = totalShards;
    finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);

    activeLoadPhase = 'tensorLocations';
    phaseStart = nowMs();
    await this.#buildTensorLocations();
    finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);

    const loadStartTime = Date.now();
    let bytesLoaded = 0;
    let shardsLoaded = 0;

    
    const reportProgress = (stage, baseProgress, detail) => {
      if (!onProgress || typeof onProgress !== 'function') return;
      const elapsed = (Date.now() - loadStartTime) / 1000;
      const speed = elapsed > 0 ? bytesLoaded / elapsed : 0;
      const speedStr = speed > 0 ? `${formatBytes(speed)}/s` : '';
      const message = detail ||
        `${formatBytes(bytesLoaded)} / ${formatBytes(totalBytes)} ${speedStr ? `- ${speedStr}` : ''}`;
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
    const originalLoadShard = (shardIndex, options) => this.#loadShard(shardIndex, options);

    
    this.#loadShardOverride = async (shardIndex, options) => {
      const shardInfo = this.manifest?.shards?.[shardIndex];
      const shardSize = shardInfo?.size || 0;
      const shardName = shardInfo?.filename ?? `index=${shardIndex}`;
      let data;
      try {
        data = await originalLoadShard(shardIndex, options);
      } catch (error) {
        const modelId = this.manifest?.modelId ?? 'unknown';
        const shardUrl = shardInfo?.url ?? shardInfo?.path ?? 'unknown';
        const sizeStr = shardSize > 0 ? `, size=${formatBytes(shardSize)}` : '';
        log.error(
          'Loader',
          `Failed to load shard ${shardIndex}/${totalShards} "${shardName}" ` +
          `for model "${modelId}" (url=${shardUrl}${sizeStr}): ${error.message}`
        );
        throw error;
      }

      if (!loadedShardIndices.has(shardIndex)) {
        loadedShardIndices.add(shardIndex);
        bytesLoaded += shardSize;
        shardsLoaded++;
        this.loadTiming.bytesLoaded = bytesLoaded;
        this.loadTiming.shardsLoaded = shardsLoaded;
        if (!inLayerPhase) {
          const pct = 0.1 + Math.min(bytesLoaded / totalBytes, 1.0) * 0.7;
          const elapsed = (Date.now() - loadStartTime) / 1000;
          const speed = elapsed > 0 ? bytesLoaded / elapsed : 0;
          const sourceInfo = this.shardCache.lastSource;
          const sourceStr = sourceInfo
            ? [sourceInfo.source, sourceInfo.mode, sourceInfo.path].filter(Boolean).join('/')
            : 'unknown';
          const fallbackStr = sourceInfo?.fallback && sourceInfo.fallback !== 'none'
            ? ` fallback=${sourceInfo.fallback}`
            : '';
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
              message: `Shard ${shardIndex}: ${sourceStr} (${formatBytes(shardSize)}${elapsedStr}${fallbackStr})`,
            });
          }
        }
      }
      return data;
    };

    
    let loadError = null;
    try {
      reportProgress('shards', 0.1, 'Loading embeddings...');
      activeLoadPhase = 'embeddings';
      phaseStart = nowMs();
      await this.#loadEmbeddings(onProgress);
      finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);
      this.#assertResidentBudget('embeddings');

      const resolveNumLayers = (value) => {
        const normalized = Number(value);
        if (!Number.isInteger(normalized) || normalized <= 0) {
          return 0;
        }
        return normalized;
      };

      const manifestConfig = this.manifest.config;
      const layerCountCandidates = [
        manifestConfig?.num_hidden_layers,
        manifestConfig?.blockCount,
        manifestConfig?.text_config?.num_hidden_layers,
        manifestConfig?.n_layer,
        this.manifest.architecture?.numLayers,
      ];
      const numLayers = layerCountCandidates
        .map(resolveNumLayers)
        .find((count) => Number.isInteger(count) && count > 0);

      if (!Number.isInteger(numLayers)) {
        throw new Error(
          `Manifest "${this.manifest.modelId ?? 'unknown'}" missing or invalid layer count. ` +
          `Expected one of manifest.config.num_hidden_layers/blockCount/text_config.num_hidden_layers/n_layer ` +
          `or manifest.architecture.numLayers.`
        );
      }

      log.info('Loader', `Layers: 0-${numLayers - 1}`);

      inLayerPhase = true;
      activeLoadPhase = 'layers';
      const layersStartTime = performance.now();
      let layerTotalMs = 0;
      let maxLayerMs = 0;
      let maxLayer = null;

      for (let l = 0; l < numLayers; l++) {
        const layerStart = performance.now();
        const layerPromise = this.#loadLayer(l, onProgress);
        this.#prefetchLayerShards(l);
        await layerPromise;
        const layerElapsedMs = performance.now() - layerStart;
        layerTotalMs += layerElapsedMs;
        if (layerElapsedMs > maxLayerMs) {
          maxLayerMs = layerElapsedMs;
          maxLayer = l;
        }
        const layerElapsed = (layerElapsedMs / 1000).toFixed(2);
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
        this.#assertResidentBudget(`layer ${l + 1}`);
      }

      const layersTotalTime = ((performance.now() - layersStartTime) / 1000).toFixed(2);
      this.loadTiming.layers = {
        count: numLayers,
        totalMs: roundLoadTimingMs(layerTotalMs),
        meanMs: numLayers > 0 ? roundLoadTimingMs(layerTotalMs / numLayers) : null,
        maxMs: maxLayer == null ? null : roundLoadTimingMs(maxLayerMs),
        maxLayer,
      };
      finishLoadPhase(this.loadTiming, activeLoadPhase, layersStartTime);
      log.info('Loader', `Layers: ${numLayers} complete (${layersTotalTime}s)`);

      reportProgress('gpu_transfer', 0.85, 'Loading final weights...');
      activeLoadPhase = 'finalWeights';
      phaseStart = nowMs();
      await this.#loadFinalWeights(onProgress);
      finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);
      this.#assertResidentBudget('final weights');

      if (onProgress) {
        onProgress({ stage: 'complete', progress: 1.0 });
      }

      this.isLoaded = true;
      const totalTime = ((Date.now() - loadStartTime) / 1000).toFixed(2);
      const avgSpeed = formatBytes(bytesLoaded / (Date.now() - loadStartTime) * 1000);
      log.info('Loader', `Complete: ${formatBytes(bytesLoaded)} in ${totalTime}s (${avgSpeed}/s)`);

      activeLoadPhase = 'cleanup';
      phaseStart = nowMs();
      this.shardCache.clear();
      finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);
      finishLoadTiming(this.loadTiming, 'complete', loadTimingStart);

      return  (this.manifest.config) || {};
    } catch (error) {
      loadError = error;
      finishLoadTiming(this.loadTiming, 'failed', loadTimingStart, error, activeLoadPhase);
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
    if (!this.manifest) {
      this.#layerShardMap.clear();
      return;
    }

    const locations = await buildTensorLocations(this.manifest, {
      tensorsJsonUrl: this.#tensorsJsonUrl,
      loadTensorsJson: this.#loadTensorsJson,
      hasCustomLoader: this.shardCache.hasCustomLoader,
    });

    for (const [name, loc] of locations) {
      this.tensorLocations.set(name, loc);
    }

    this.#buildLayerShardMap();
  }

  #buildLayerShardMap() {
    this.#layerShardMap.clear();

    for (const [, location] of this.tensorLocations) {
      const layerIdx = getLayerIndexFromGroup(location.group);
      if (layerIdx == null || isExpertGroup(location.group)) {
        continue;
      }

      let shards = this.#layerShardMap.get(layerIdx);
      if (!shards) {
        shards = new Set();
        this.#layerShardMap.set(layerIdx, shards);
      }

      for (const shardIndex of getTensorShardIndices(location)) {
        shards.add(shardIndex);
      }
    }
  }

  #logWeightBreakdown() {
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

  #prefetchLayerShards(layerIdx) {
    const prefetch = this.#loadingConfig.prefetch;
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
    if (this.#layerShardMap.size === 0) return;

    const maxShards = prefetch.maxShards;
    const hasLimit = maxShards > 0;
    let scheduled = 0;
    const loadShard = this.#getLoadShard();

    for (let idx = layerIdx + 1; idx <= layerIdx + layersAhead; idx++) {
      const shards = this.#layerShardMap.get(idx);
      if (!shards) continue;

      for (const shardIndex of shards) {
        if (this.shardCache.has(shardIndex)) continue;

        loadShard(shardIndex, { priority: 'low' }).catch(() => {});
        scheduled++;
        if (hasLimit && scheduled >= maxShards) return;
      }
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

    const streamedUpload = toGPU && this.#shouldStreamUploadToGPU(location);
    let shardData;
    try {
      const preserveRawSourceBytes = toGPU && isLiteRTAffineInt4FusedEligible(location, {
        gpuCapabilities: this.gpuCapabilities,
      });
      shardData = this.#isFunctionalDescriptorLocation(location)
        ? await this.#assembleFunctionalDescriptorData(location, name)
        : streamedUpload
        ? await this.#assembleShardDataToGpuBuffer(location, name)
        : await this.#assembleShardData(location, name, {
            materializeSourceTransform: !preserveRawSourceBytes,
          });
    } catch (error) {
      throw annotateTensorLoadError(error, name, location, {
        tensorLoadStage: streamedUpload ? 'streamShardToGpuBuffer' : 'assembleShardData',
        toGPU,
        streamedUpload,
      });
    }

    if (toGPU) {
      const device = getDevice();
      if (!device) {
        log.warn('Loader', 'GPU device not available; falling back to CPU');
        if (isGpuBufferInstance(shardData)) {
          releaseBuffer(shardData);
          shardData = await this.#assembleShardData(location, name);
        }
        return loadTensorToCPU(shardData, location, name);
      }

      
      const allowF32UpcastNonMatmul = this.#loadingConfig?.allowF32UpcastNonMatmul;
      if (allowF32UpcastNonMatmul == null) {
        throw new Error('runtime.loading.allowF32UpcastNonMatmul is required.');
      }
      const config = {
        useFusedQ4K: this.useFusedQ4K,
        q4kMaterializationMode: this.q4kMaterializationMode,
        q4kFusedRoles: this.q4kFusedRoles,
        keepF32Weights: this.keepF32Weights,
        q4kLayout: this.q4kLayout,
        loaderDebug: this.#loaderDebug,
        gpuCapabilities: this.gpuCapabilities,
        allowF32UpcastNonMatmul,
      };

      let result;
      try {
        result = await loadTensorToGPU(shardData, location, name, config);
      } catch (error) {
        if (isGpuBufferInstance(shardData)) {
          releaseBuffer(shardData);
        }
        throw annotateTensorLoadError(error, name, location, {
          tensorLoadStage: 'materializeTensorToGPU',
          toGPU: true,
          streamedUpload,
        });
      }

      for (const buffer of result.allocatedBuffers) {
        this.gpuBuffers.add(buffer);
      }

      return result.data;
    }

    if (isGpuBufferInstance(shardData)) {
      // Shouldn't happen (streaming is only used for toGPU), but keep this leak-proof.
      releaseBuffer(shardData);
      shardData = await this.#assembleShardData(location, name);
    }
    return loadTensorToCPU(shardData, location, name);
  }

  #isFunctionalDescriptorLocation(location) {
    return String(location?.dtype || '').trim().toUpperCase() === 'FUNCTIONAL_DESCRIPTOR';
  }

  #getDescriptorShardFiles(location, name) {
    if (!location?.descriptorManifest) {
      throw new Error(
        `[DopplerLoader] FUNCTIONAL_DESCRIPTOR tensor "${name}" is missing descriptorManifest.`
      );
    }
    const manifest = assertFunctionalDescriptorManifest(
      location.descriptorManifest,
      `FUNCTIONAL_DESCRIPTOR tensor "${name}" descriptorManifest`
    );
    const components = manifest.components;
    if (!components || typeof components !== 'object') {
      throw new Error(
        `[DopplerLoader] FUNCTIONAL_DESCRIPTOR tensor "${name}" descriptorManifest.components is required.`
      );
    }
    const files = [
      components.kronecker_sum?.shard_file,
      components.coordinate_inr?.shard_file,
      components.sparse_outliers?.shard_file,
    ];
    if (files.some((file) => typeof file !== 'string' || file.trim().length === 0)) {
      throw new Error(
        `[DopplerLoader] FUNCTIONAL_DESCRIPTOR tensor "${name}" must declare kronecker, SIREN, and sparse shard_file values.`
      );
    }
    return files.map((file) => file.trim());
  }

  async #loadDescriptorShardFile(file, name) {
    const loadAuxiliaryFile = this.#loadAuxiliaryFile;
    const payload = loadAuxiliaryFile
      ? await loadAuxiliaryFile(file)
      : await loadAuxFile(file);
    if (payload == null) {
      throw new Error(
        `[DopplerLoader] Descriptor shard "${file}" for tensor "${name}" was not found.`
      );
    }
    if (payload instanceof Uint8Array) {
      return payload;
    }
    if (ArrayBuffer.isView(payload)) {
      return new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
    }
    if (payload instanceof ArrayBuffer) {
      return new Uint8Array(payload);
    }
    throw new Error(
      `[DopplerLoader] Descriptor shard "${file}" for tensor "${name}" must load as ArrayBuffer or Uint8Array.`
    );
  }

  async #assertDescriptorHash(location, name, descriptorShards) {
    const descriptorHash = location?.descriptorManifest?.descriptor_hash;
    if (typeof descriptorHash !== 'string' || !descriptorHash.trim()) {
      return;
    }
    const match = /^sha256:([a-f0-9]{64})$/i.exec(descriptorHash.trim());
    if (!match) {
      throw new Error(
        `[DopplerLoader] FUNCTIONAL_DESCRIPTOR tensor "${name}" descriptor_hash must be sha256:<64 hex chars>.`
      );
    }
    const totalBytes = Array.from(descriptorShards.values()).reduce((sum, bytes) => sum + bytes.byteLength, 0);
    const combined = new Uint8Array(totalBytes);
    let offset = 0;
    for (const bytes of descriptorShards.values()) {
      combined.set(bytes, offset);
      offset += bytes.byteLength;
    }
    const actual = await computeHash(combined, 'sha256');
    if (actual.toLowerCase() !== match[1].toLowerCase()) {
      throw new Error(
        `[DopplerLoader] FUNCTIONAL_DESCRIPTOR tensor "${name}" descriptor hash mismatch. ` +
        `Expected ${descriptorHash}, got sha256:${actual}.`
      );
    }
  }

  async #assembleFunctionalDescriptorData(location, name) {
    const shardFiles = this.#getDescriptorShardFiles(location, name);
    const descriptorShards = new Map();
    for (const file of shardFiles) {
      descriptorShards.set(file, await this.#loadDescriptorShardFile(file, name));
    }
    await this.#assertDescriptorHash(location, name, descriptorShards);
    const data = new Uint8Array(0);
    Object.defineProperty(data, 'descriptorShards', {
      value: descriptorShards,
      enumerable: false,
    });
    return data;
  }

  
  async #assembleShardData(location, name, options = {}) {
    const loadShard = this.#getLoadShard();
    const loadShardRange = (idx, offset, length) => this.shardCache.loadRange(idx, offset, length);
    const data = await assembleShardData(location, name, loadShard, loadShardRange, options);
    const companions = Array.isArray(location?.storage?.companions)
      ? location.storage.companions
      : [];
    if (companions.length === 0) {
      return data;
    }
    const storageCompanions = {};
    for (const companion of companions) {
      const companionLocation = this.tensorLocations.get(companion.tensorId);
      if (!companionLocation) {
        throw new Error(
          `[DopplerLoader] Tensor "${name}" storage companion "${companion.tensorId}" for role "${companion.role}" was not found.`
        );
      }
      storageCompanions[companion.role] = {
        tensorId: companion.tensorId,
        location: companionLocation,
        bytes: await assembleShardData(companionLocation, companion.tensorId, loadShard, loadShardRange),
      };
    }
    Object.defineProperty(data, 'storageCompanions', {
      value: storageCompanions,
      enumerable: false,
    });
    return data;
  }

  #shouldStreamUploadToGPU(location) {
    if (this.#isFunctionalDescriptorLocation(location)) return false;
    if (!location?.size || location.size <= 0) return false;
    if (hasSourceTransform(location)) return false;
    if (Array.isArray(location?.storage?.companions) && location.storage.companions.length > 0) return false;
    if (this.shardCache.hasCustomLoader && !this.shardCache.canStreamRanges) return false;
    const chunkBytes = this.#loadingConfig?.storage?.backend?.streaming?.readChunkBytes ?? 0;
    if (!Number.isFinite(chunkBytes) || chunkBytes <= 0) return false;
    // Always stream multi-span tensors to avoid loading whole shards + assembling on CPU.
    if (location.spans && location.spans.length > 0) {
      return true;
    }
    // Conservative default: only stream "large" single-span tensors to avoid turning
    // OPFS into many small random reads that can be slower than whole-shard caching.
    const minStreamBytes = Math.max(16 * 1024 * 1024, chunkBytes * 4);
    return location.size >= minStreamBytes;
  }

  async #assembleShardDataToGpuBuffer(location, name) {
    const device = getDevice();
    if (!device) {
      throw new Error('GPU device not available');
    }
    const rawChunkBytes = Number(this.#loadingConfig?.storage?.backend?.streaming?.readChunkBytes);
    const chunkBytes = Number.isFinite(rawChunkBytes) && rawChunkBytes > 0
      ? Math.floor(rawChunkBytes)
      : 1;

    // queue.writeBuffer requires 4-byte aligned sizes; we pad the buffer.
    const alignedSize = Math.ceil(location.size / 4) * 4;
    const raw = acquireBuffer(alignedSize, undefined, `raw_${name}`);
    let complete = false;

    try {
      let dstOffset = 0;
      let pendingBytes = null;
      const writeAlignedChunk = (bytes) => {
        if (bytes.byteLength === 0) return;
        device.queue.writeBuffer(raw, dstOffset, bytes, 0, bytes.byteLength);
        dstOffset += bytes.byteLength;
      };
      const uploadChunk = (bytes) => {
        let merged = bytes;
        if (pendingBytes && pendingBytes.byteLength > 0) {
          merged = new Uint8Array(pendingBytes.byteLength + bytes.byteLength);
          merged.set(pendingBytes, 0);
          merged.set(bytes, pendingBytes.byteLength);
          pendingBytes = null;
        }
        const alignedLength = merged.byteLength - (merged.byteLength % 4);
        if (alignedLength > 0) {
          writeAlignedChunk(merged.subarray(0, alignedLength));
        }
        const remainder = merged.byteLength - alignedLength;
        pendingBytes = remainder > 0 ? merged.slice(alignedLength) : null;
      };
      const streamRange = (idx, offset, length) => this.shardCache.streamRange(idx, offset, length, { chunkBytes });

      if (location.spans) {
        for (const span of location.spans) {
          for await (const chunk of streamRange(span.shardIndex, span.offset, span.size)) {
            uploadChunk(chunk);
          }
        }
      } else {
        for await (const chunk of streamRange(location.shardIndex, location.offset, location.size)) {
          uploadChunk(chunk);
        }
      }

      if (pendingBytes && pendingBytes.byteLength > 0) {
        const padded = new Uint8Array(4);
        padded.set(pendingBytes, 0);
        writeAlignedChunk(padded);
        dstOffset -= (4 - pendingBytes.byteLength);
        pendingBytes = null;
      }

      if (dstOffset !== location.size) {
        throw new Error(
          `Stream upload short read for "${name}": got=${dstOffset}, expected=${location.size}.`
        );
      }
      complete = true;
      return raw;
    } finally {
      if (!complete) {
        releaseBuffer(raw);
      }
    }
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
      loadShardRange: (index, offset, length) => this.shardCache.loadRange(index, offset, length),
      shouldStreamLargeWeight: (name, loc, label) => this.#shouldStreamLargeWeight(name, loc, label),
      resolveWeightLayout: (loc) => this.#resolveWeightLayout(loc),
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      // Keep embedding weights in F32 when manifest quantization requires it.
      // gather.wgsl reads embeddings as f32; downcasting here corrupts reads.
      preserveF32Embeddings: String(this.manifest?.quantizationInfo?.embeddings ?? '').toLowerCase() === 'f32',
      hostHasShaderF16: this.gpuCapabilities?.hasF16 ?? null,
      embeddingKernel: this.manifest?.inference?.execution?.kernels?.embed ?? null,
    };

    this.embeddings = await loadEmbeddings(ctx);
    this.perLayerInputWeights = await loadPerLayerInputWeights({
      modelId: this.manifest?.modelId ?? null,
      tensorLocations: this.tensorLocations,
      gpuBuffers: this.gpuBuffers,
      loadTensor: (name, toGPU, silent) => this.#loadTensor(name, toGPU, silent),
      shouldStreamLargeWeight: (name, loc, label) => this.#shouldStreamLargeWeight(name, loc, label),
      loadShardRange: (index, offset, length) => this.shardCache.loadRange(index, offset, length),
      resolveWeightLayout: (loc) => this.#resolveWeightLayout(loc),
      perLayerInputSession: this.#perLayerInputSession,
    }, this.manifest?.architecture ?? null);
  }

  
  async #loadLayer(layerIdx, _onProgress) {
    const textConfig = (
      this.manifest?.config?.text_config
      && typeof this.manifest.config.text_config === 'object'
      && !Array.isArray(this.manifest.config.text_config)
    )
      ? this.manifest.config.text_config
      : this.manifest?.config ?? null;

    
    const ctx = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this.#loadTensor(name, toGPU, silent),
      needsNormWeightOffset: () => this.#needsNormWeightOffset(),
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      isMoE: this.isMoE,
      isExpertLayer: (idx) => this.#isExpertLayer(idx),
      loadDenseFfnForMoeLayers: this.manifest?.inference?.ffn?.branchMode === 'dense_plus_moe',
      numHeads: this.manifest?.architecture?.numAttentionHeads ?? null,
      numKVHeads: this.manifest?.architecture?.numKeyValueHeads ?? null,
      headDim: this.manifest?.architecture?.headDim ?? null,
      hiddenSize: this.manifest?.architecture?.hiddenSize ?? null,
      linearNumKeyHeads: textConfig?.linear_num_key_heads ?? this.manifest?.architecture?.linearNumKeyHeads ?? null,
      linearNumValueHeads: textConfig?.linear_num_value_heads ?? this.manifest?.architecture?.linearNumValueHeads ?? null,
      linearKeyHeadDim: textConfig?.linear_key_head_dim ?? this.manifest?.architecture?.linearKeyHeadDim ?? null,
      linearValueHeadDim: textConfig?.linear_value_head_dim ?? this.manifest?.architecture?.linearValueHeadDim ?? null,
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
      tensorLocations: this.tensorLocations,
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
      loadShardRange: (index, offset, length) => this.shardCache.loadRange(index, offset, length),
      needsNormWeightOffset: () => this.#needsNormWeightOffset(),
      shouldStreamLargeWeight: (name, loc, label) => this.#shouldStreamLargeWeight(name, loc, label),
      resolveWeightLayout: (loc) => this.#resolveWeightLayout(loc),
      embeddings: this.embeddings,
      embeddingPostprocessor: this.manifest?.inference?.output?.embeddingPostprocessor ?? null,
      diffusionGemmaSelfConditioning: this.manifest?.inference?.diffusionGemma?.selfConditioning === true,
      modelType: this.manifest?.modelType ?? null,
      tieWordEmbeddings,
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      normOffsetDebugLogged: this.#normOffsetDebugLogged,
    };

    const result = await loadFinalWeights(ctx);
    this.finalNorm = result.finalNorm;
    this.lmHead = result.lmHead;
    this.embeddingPostprocessor = result.embeddingPostprocessor;
    this.diffusionGemmaSelfConditioning = result.diffusionGemmaSelfConditioning;
    this.#normOffsetDebugLogged = result.normOffsetDebugLogged;
  }

  
  getLayerWeights(layerIdx) {
    return this.layers.get(layerIdx) || null;
  }

  /**
   * Load a tensor by name. Public interface for extension loaders (e.g., vision).
   */
  async loadTensor(name, toGPU = true, silent = false) {
    return this.#loadTensor(name, toGPU, silent);
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

    if (this.#memoryMonitor) {
      this.#stopMemoryLogging('complete');
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
    this.#layerShardMap.clear();
    this.shardCache.clear();
    this.#normOffsetLogged = false;

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
