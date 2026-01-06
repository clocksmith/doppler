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
  loadTensorsFromOPFS,
  computeHash,
} from '../storage/shard-manager.js';
import {
  parseManifest,
  parseTensorMap,
  getShardInfo,
  getShardCount,
  isMoE,
  getShardsForExpert,
  getTensorsForExpert,
  getExpertBytes,
  TENSORS_FILENAME,
  type RDRRManifest,
  type ShardInfo,
  type HashAlgorithm,
  type TensorMap,
} from '../storage/rdrr-format.js';
import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../gpu/buffer-pool.js';
import { dequantize, dequantizeQ6K, castF32ToF16, castF16ToF32, runBF16ToF16 } from '../gpu/kernel-selector.js';
import { createTensor } from '../gpu/tensor.js';
import {
  createWeightBuffer,
  createCpuWeightBuffer,
  isWeightBuffer,
  isCpuWeightBuffer,
  getWeightDtype,
  getLayout,
  type WeightBuffer,
  type WeightDtype,
  type WeightLayout,
  type CpuWeightBuffer,
} from '../gpu/weight-buffer.js';
import { ExpertCache, getExpertCache, type CacheStats } from './expert-cache.js';
import type { ExpertWeights } from './weights.js';
import { getKernelPlanQ4KStrategy, getKernelPlanSource, getKernelPlanStrict } from '../config/kernel-plan.js';
import { LORA_MODULE_ALIASES, type LoRAAdapter, type LoRAModuleName } from '../inference/pipeline/lora.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { getBufferPool } from '../gpu/buffer-pool.js';
import { QK_K, Q4K_BLOCK_BYTES, Q6K_BLOCK_BYTES } from './quantization-constants.js';

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
  ShardSourceInfo,
} from './loader-types.js';

import {
  f16ToF32,
  convertBF16ToF32GPU,
  shouldDequantizeToF16,
  isEmbeddingWeight,
  applyBufferLayout,
  applyNormWeightOffset,
  findAlternativeTensorName,
} from './dtype-utils.js';

import { ShardCache, createShardCache } from './shard-cache.js';
import type { LoadingConfigSchema } from '../config/schema/loading.schema.js';
import { DTYPE_SIZES } from '../config/schema/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

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

export type { ExpertWeights } from './weights.js';

// ============================================================================
// LoRA Parsing Helpers
// ============================================================================

interface ParsedLoRATensorName {
  layer: number;
  module: LoRAModuleName;
  kind: 'a' | 'b';
}

const parseLoRATensorName = (name: string): ParsedLoRATensorName | null => {
  const match = name.match(/layers?\.?(\d+)\.([^\.]+)\.lora_([ab])/i);
  if (!match) return null;
  const layer = parseInt(match[1], 10);
  const rawModule = match[2].toLowerCase();
  const module = LORA_MODULE_ALIASES[rawModule];
  if (!module) return null;
  const kind = match[3].toLowerCase() === 'a' ? 'a' : 'b';
  return { layer, module, kind };
};

const toFloat32 = (value: GPUBuffer | Float32Array | Uint8Array | ArrayBuffer | WeightBuffer | CpuWeightBuffer): Float32Array => {
  if (value instanceof Float32Array) return value;
  if (value instanceof ArrayBuffer) return new Float32Array(value);
  if (value instanceof Uint8Array) {
    return new Float32Array(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));
  }
  if (isCpuWeightBuffer(value)) {
    return value.data;
  }
  // WeightBuffer: should not happen for LoRA loading (toGPU=false), but handle for type safety
  if (isWeightBuffer(value)) {
    throw new Error('LoRA tensor load returned WeightBuffer - expected CPU array');
  }
  throw new Error('LoRA tensor load returned unsupported buffer type');
};

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
  // Default is false; opt-in via manifest kernel plan.
  useFusedQ4K = false;

  // Q4K layout from manifest: 'column_wise' means weights are pre-transposed
  // When set, dequantized matmul weights need layout='column' for transposeB=false
  q4kLayout: 'flat' | 'row_wise' | 'column_wise' | null = null;
  keepF32Weights = false;

  // Internal tracking
  private _normOffsetLogged = false;
  private _normOffsetDebugLogged = false;
  private _memoryLogInterval: ReturnType<typeof setInterval> | null = null;
  private _loadStartTime = 0;
  private _tensorsJsonUrl: string | null = null;

  constructor(loadingConfig?: LoadingConfigSchema) {
    this.loadingConfig = loadingConfig ?? getRuntimeConfig().loading;
    this.shardCache = createShardCache(
      this.loadingConfig.shardCache.opfsEntries,
      this.loadingConfig.shardCache
    );
  }

  /**
   * Update loading configuration and related caches.
   */
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

  /**
   * Log comprehensive memory stats during loading
   */
  private _logMemoryStats(phase: string): void {
    const elapsed = ((performance.now() - this._loadStartTime) / 1000).toFixed(1);
    const stats: string[] = [`[${elapsed}s] Memory (${phase}):`];

    // JS Heap (Chrome only)
    const perfMemory = (performance as Performance & {
      memory?: {
        usedJSHeapSize?: number;
        totalJSHeapSize?: number;
        jsHeapSizeLimit?: number;
      };
    }).memory;

    if (perfMemory) {
      const usedHeap = perfMemory.usedJSHeapSize || 0;
      const totalHeap = perfMemory.totalJSHeapSize || 0;
      const heapLimit = perfMemory.jsHeapSizeLimit || 0;
      stats.push(`Heap=${formatBytes(usedHeap)}/${formatBytes(totalHeap)} (limit=${formatBytes(heapLimit)})`);
    }

    // GPU buffer pool stats
    try {
      const pool = getBufferPool();
      const poolStats = pool.getStats();
      stats.push(`GPU=${formatBytes(poolStats.currentBytesAllocated)} (${poolStats.activeBuffers} active, ${poolStats.pooledBuffers} pooled, peak=${formatBytes(poolStats.peakBytesAllocated)})`);
    } catch {
      // Buffer pool not initialized yet
    }

    // Shard cache stats
    stats.push(`ShardCache=${formatBytes(this.shardCache.totalBytes)} (${this.shardCache.size} shards)`);

    // Loaded model state
    stats.push(`Layers=${this.layers.size}, GPUBuffers=${this.gpuBuffers.size}`);

    log.info('Loader', stats.join(' | '));
  }

  /**
   * Start periodic memory logging during load
   */
  private _startMemoryLogging(): void {
    this._loadStartTime = performance.now();
    this._logMemoryStats('start');
    // Log memory periodically during loading (configurable, default 30s)
    const logIntervalMs = this.loadingConfig.memoryManagement.logIntervalMs;
    this._memoryLogInterval = setInterval(() => {
      this._logMemoryStats('loading');
    }, logIntervalMs);
  }

  /**
   * Stop periodic memory logging
   */
  private _stopMemoryLogging(): void {
    if (this._memoryLogInterval) {
      clearInterval(this._memoryLogInterval);
      this._memoryLogInterval = null;
    }
    this._logMemoryStats('complete');
  }

  /**
   * Set custom shard loader (e.g., for Native Bridge)
   */
  setCustomShardLoader(loadShardFn: CustomShardLoader, options: CustomShardLoaderOptions = {}): void {
    this.shardCache.setCustomLoader(loadShardFn, options.verify !== false);
  }

  /**
   * Set URL for loading tensors.json via HTTP (for test harnesses with custom shard loaders)
   */
  setTensorsJsonUrl(url: string | null): void {
    this._tensorsJsonUrl = url;
  }

  /**
   * Load shard using ShardCache (handles caching and request deduplication)
   */
  private async _loadShard(shardIndex: number): Promise<ArrayBuffer> {
    return this.shardCache.load(shardIndex);
  }

  /**
   * Initialize loader and detect capabilities
   */
  async init(): Promise<void> {
    log.info('Loader', 'Initializing...');

    // Detect memory capabilities
    this.memoryCapabilities = await getMemoryCapabilities();
    const unifiedInfo = await detectUnifiedMemory();
    this.isUnifiedMemory = unifiedInfo.isUnified;

    // Initialize GPU
    const device = await initDevice();
    if (!device) {
      throw new Error('Failed to initialize WebGPU device');
    }
    this.gpuCapabilities = getKernelCapabilities();

    // Initialize heap manager
    this.heapManager = getHeapManager();
    await this.heapManager.init();

    // Initialize expert cache for MoE models
    this.expertCache = getExpertCache();

    // Initialize OPFS
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
   * Set manifest directly (for bridge/external loading)
   */
  setManifest(manifest: RDRRManifest): void {
    this.manifest = manifest;
    const config = manifest.config as ModelConfig | undefined;
    this.isMoE = manifest.moeConfig != null || (config?.num_local_experts ?? 0) > 1;
    this.shardCache.configureForModel(this.manifest, this.shardCache.hasCustomLoader);
    this._configureQ4KStrategy();
    debugTrace.loader('Manifest set externally');
  }

  /**
   * Load LoRA weights from an adapter manifest (RDRR format).
   */
  async loadLoRAWeights(manifest: RDRRManifest): Promise<LoRAAdapter> {
    const isLoRA = manifest.adapterType === 'lora' || manifest.modelType === 'lora' || !!manifest.loraConfig;
    if (!isLoRA) {
      throw new Error('Manifest is not a LoRA adapter');
    }
    if (!manifest.loraConfig) {
      throw new Error('LoRA manifest missing loraConfig');
    }

    const prevManifest = this.manifest;
    const prevLocations = new Map(this.tensorLocations);

    this.manifest = manifest;
    this._buildTensorLocations();

    try {
      const adapter: LoRAAdapter = {
        name: manifest.modelId,
        version: typeof manifest.version === 'string' ? manifest.version : String(manifest.version),
        baseModel: manifest.baseModel,
        rank: manifest.loraConfig.rank,
        alpha: manifest.loraConfig.alpha,
        targetModules: manifest.loraConfig.targetModules as LoRAModuleName[] | undefined,
        layers: new Map(),
      };

      for (const name of this.tensorLocations.keys()) {
        const parsed = parseLoRATensorName(name);
        if (!parsed) continue;
        const tensor = await this._loadTensor(name, false, true);
        if (!tensor) continue;
        const data = toFloat32(tensor);

        const layer = adapter.layers.get(parsed.layer) || {};
        const scale = adapter.rank > 0 ? adapter.alpha / adapter.rank : 1;
        if (!layer[parsed.module]) {
          layer[parsed.module] = {
            a: new Float32Array(0),
            b: new Float32Array(0),
            rank: adapter.rank,
            alpha: adapter.alpha,
            scale,
          };
        }

        if (parsed.kind === 'a') {
          layer[parsed.module].a = data;
        } else {
          layer[parsed.module].b = data;
        }

        adapter.layers.set(parsed.layer, layer);
      }

      return adapter;
    } finally {
      this.manifest = prevManifest;
      this.tensorLocations = prevLocations;
    }
  }

  private _configureQ4KStrategy(): void {
    const q4kStrategy = getKernelPlanQ4KStrategy();
    const planSource = getKernelPlanSource();
    const strict = getKernelPlanStrict();
    const q4kLayout = (this.manifest?.config as { q4kLayout?: string } | undefined)?.q4kLayout;

    // Default to fused Q4K when subgroups are available (4x memory savings)
    // Explicit 'dequant_f16' or 'dequant_f32' opts out of fused path
    const caps = this.gpuCapabilities || getKernelCapabilities();
    const hasSubgroups = caps?.hasSubgroups ?? false;
    const wantsFused = q4kStrategy === 'fused_q4k';
    const wantsDequant = q4kStrategy === 'dequant_f16' || q4kStrategy === 'dequant_f32';
    let useFused = q4kStrategy === 'auto' ? hasSubgroups && !wantsDequant : wantsFused;

    if (typeof window !== 'undefined' && (window as any).DOPPLER_DISABLE_FUSED_Q4K) {
      useFused = false;
    }
    if (q4kLayout === 'column_wise') {
      useFused = false;
    }
    if (wantsFused && !useFused) {
      const message = `Q4K fused requested but unavailable (subgroups=${hasSubgroups}, layout=${q4kLayout ?? 'default'}).`;
      if (strict) {
        throw new Error(message);
      }
      log.warn('Loader', message);
    }

    this.useFusedQ4K = useFused;
    this.q4kLayout = (q4kLayout as 'flat' | 'row_wise' | 'column_wise') ?? null;
    this.keepF32Weights = q4kStrategy === 'dequant_f32';

    debugTrace.loader(`Q4K strategy: fused=${this.useFusedQ4K}, strategy=${q4kStrategy}, source=${planSource}, layout=${q4kLayout ?? 'default'}, subgroups=${hasSubgroups}, keepF32=${this.keepF32Weights}`);
  }

  // ==========================================================================
  // Large Weight Handling (Embeddings / LM Head)
  // ==========================================================================

  private _getLargeWeightConfig() {
    return getRuntimeConfig().inference.largeWeights;
  }

  private _getLargeWeightMaxBytes(): number | null {
    const config = this._getLargeWeightConfig();
    if (!config?.enabled) return null;
    const device = getDevice();
    if (!device) return null;
    const safety = Math.min(Math.max(config.safetyRatio ?? 0.9, 0.1), 1);
    const maxBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    return Math.floor(maxBinding * safety);
  }

  private _estimateMatmulWeightBytes(
    name: string,
    location: TensorLocation
  ): { bytes: number; dtype: WeightDtype } | null {
    if (!location.shape || location.shape.length === 0) return null;
    const numElements = location.shape.reduce((a, b) => a * b, 1);
    if (!Number.isFinite(numElements) || numElements <= 0) return null;

    const caps = this.gpuCapabilities || getKernelCapabilities();
    const hasF16 = caps?.hasF16 ?? false;
    const isMatmulWeight = shouldDequantizeToF16(name);

    let dtype: WeightDtype = 'f32';
    switch (location.dtype) {
      case 'F16':
        dtype = 'f16';
        break;
      case 'BF16':
        dtype = hasF16 && isMatmulWeight ? 'f16' : 'f32';
        break;
      case 'Q4_K':
      case 'Q4_K_M':
        dtype = (hasF16 && isMatmulWeight && !this.keepF32Weights) ? 'f16' : 'f32';
        break;
      case 'Q6_K':
        dtype = 'f16';
        break;
      default:
        dtype = location.dtype === 'F32' ? 'f32' : 'f32';
        break;
    }

    const bytesPerElement = DTYPE_SIZES[dtype === 'f16' ? 'f16' : 'f32'];
    return { bytes: numElements * bytesPerElement, dtype };
  }

  private _resolveWeightLayout(location: TensorLocation, name: string): WeightLayout {
    if (location.layout === 'column') return 'column';
    if (isEmbeddingWeight(name) && location.shape?.length === 2) {
      const [dim0, dim1] = location.shape;
      if (dim0 < dim1) {
        return 'column';
      }
    }
    return 'row';
  }

  private _shouldStreamLargeWeight(name: string, location: TensorLocation, label: string): boolean {
    const maxBytes = this._getLargeWeightMaxBytes();
    if (!maxBytes) return false;
    const estimate = this._estimateMatmulWeightBytes(name, location);
    if (!estimate) return false;
    if (estimate.bytes <= maxBytes) return false;
    const canStream = location.dtype === 'F16' || location.dtype === 'F32' || location.dtype === 'BF16';
    if (!canStream) {
      log.warn(
        'Loader',
        `${label} weight "${name}" (${formatBytes(estimate.bytes)}) exceeds GPU binding limit (${formatBytes(maxBytes)}) ` +
        `but dtype ${location.dtype} cannot be streamed. Regenerate with F16/F32 weights.`
      );
      return false;
    }
    log.warn(
      'Loader',
      `${label} weight "${name}" (${formatBytes(estimate.bytes)}) exceeds GPU binding limit (${formatBytes(maxBytes)}). ` +
      'Using CPU-backed streaming.'
    );
    return true;
  }

  /**
   * Load model from OPFS or external source
   */
  async load(modelId: string, options: LoadOptions = {}): Promise<ModelConfig> {
    const { onProgress = null, verifyHashes = true } = options;

    if (!this.heapManager) {
      await this.init();
    }

    // Avoid cross-model contamination when reusing the global loader instance.
    const hasExistingModelState =
      this.isLoaded ||
      this.modelId !== null ||
      this.tensorLocations.size > 0 ||
      this.shardCache.size > 0 ||
      this.layers.size > 0 ||
      this.experts.size > 0 ||
      this.gpuBuffers.size > 0;

    // Preserve manifest if set externally (for custom shard loader)
    const preservedManifest = this.shardCache.hasCustomLoader ? this.manifest : null;

    if (hasExistingModelState) {
      await this.unload();
    }

    // Restore preserved manifest after unload
    if (preservedManifest) {
      this.manifest = preservedManifest;
    }

    log.info('Loader', `Loading: ${modelId}`);
    this.modelId = modelId;

    // Start periodic memory logging
    this._startMemoryLogging();

    // If using custom shard loader (bridge), manifest should be set externally
    if (!this.shardCache.hasCustomLoader) {
      await openModelDirectory(modelId);
      const manifestJson = await loadManifestFromOPFS();
      this.manifest = parseManifest(manifestJson);
    }

    if (!this.manifest) {
      throw new Error('No manifest available. Set manifest via setManifest() or ensure OPFS has the model.');
    }

    this._configureQ4KStrategy();

    // Check model type
    const config = this.manifest.config as ModelConfig | undefined;
    this.isMoE = this.manifest.moeConfig != null ||
                 (config?.num_local_experts ?? 0) > 1 ||
                 isMoE();

    // Configure shard cache size based on model type
    this.shardCache.configureForModel(this.manifest, this.shardCache.hasCustomLoader);

    // Enforce dense/MoE gating based on hardware
    if (!this.isMoE && !this.isUnifiedMemory) {
      log.warn('Loader', 'Dense model on discrete GPU - performance limited. Consider MoE model.');
    }

    // Verify integrity if requested (only for OPFS path)
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

    // Build tensor location map from manifest (or external tensors.json)
    await this._buildTensorLocations();

    // Calculate total bytes for progress tracking
    const totalBytes = (this.manifest.shards || []).reduce((sum, s) => sum + (s.size || 0), 0);
    const totalShards = this.manifest.shards?.length || 0;
    const loadStartTime = Date.now();
    let bytesLoaded = 0;
    let shardsLoaded = 0;

    // Helper to format bytes
    const formatBytes = (bytes: number): string => {
      if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
      if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
      if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${bytes} B`;
    };

    // Helper to calculate speed
    const getSpeed = (): number => {
      const elapsed = (Date.now() - loadStartTime) / 1000;
      return elapsed > 0 ? bytesLoaded / elapsed : 0;
    };

    // Helper to report detailed progress
    const reportProgress = (stage: LoadProgress['stage'], baseProgress: number, detail?: string): void => {
      if (!onProgress) return;
      const speed = getSpeed();
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

    // Track shard loading with progress (only count each shard once)
    // Suppress shard progress during layer phase to avoid noisy interleaved logs
    const loadedShardIndices = new Set<number>();
    let inLayerPhase = false;
    const originalLoadShard = this._loadShard.bind(this);
    this._loadShard = async (shardIndex: number): Promise<ArrayBuffer> => {
      const shardInfo = this.manifest?.shards?.[shardIndex];
      const shardSize = shardInfo?.size || 0;
      const data = await originalLoadShard(shardIndex);

      // Only count bytes and report progress for first load of each shard
      if (!loadedShardIndices.has(shardIndex)) {
        loadedShardIndices.add(shardIndex);
        bytesLoaded += shardSize;
        shardsLoaded++;
        // Only show shard progress before layer phase
        if (!inLayerPhase) {
          const pct = 0.1 + Math.min(bytesLoaded / totalBytes, 1.0) * 0.7;
          const speed = getSpeed();
          // Include source info from _lastShardSource (matches console log)
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

    // Load embeddings (always needed)
    reportProgress('shards', 0.1, 'Loading embeddings...');
    await this._loadEmbeddings(onProgress);

    // Load layers
    const numLayers = config?.num_hidden_layers ||
                      config?.blockCount ||
                      config?.text_config?.num_hidden_layers ||
                      config?.n_layer ||
                      (this.manifest.architecture as { numLayers?: number } | undefined)?.numLayers ||
                      32;
    log.info('Loader', `Layers: 0-${numLayers - 1}`);

    inLayerPhase = true;  // Suppress shard progress during layer loading
    const layersStartTime = performance.now();

    for (let l = 0; l < numLayers; l++) {
      const layerStart = performance.now();
      await this._loadLayer(l, onProgress);
      const layerElapsed = ((performance.now() - layerStart) / 1000).toFixed(2);
      log.verbose('Loader', `  Layer ${l}: ${layerElapsed}s`);

      // Yield to event loop after each layer to allow GC of temporary ArrayBuffers.
      // Without this, tight async loops accumulate garbage (420 tensors × ~15MB = 6GB).
      await new Promise(r => setTimeout(r, 0));

      // Periodically flush shard cache during OPFS loading to reduce JS heap pressure.
      // Dense models access shards sequentially, so keeping old shards cached wastes memory.
      // Skip for network loading (customLoadShard) - re-fetching is expensive.
      const { flushIntervalLayers, flushThresholdBytes, gpuQueueFlushLayers } = this.loadingConfig.memoryManagement;
      const cacheBytes = this.shardCache.totalBytes;
      const shouldFlushCache = !this.shardCache.hasCustomLoader && l > 0 && (l % flushIntervalLayers === 0 || cacheBytes > flushThresholdBytes);
      if (shouldFlushCache) {
        this.shardCache.clear();
      }
      // Flush GPU queue periodically to release Chrome's internal staging memory
      if (l > 0 && l % gpuQueueFlushLayers === 0) {
        const device = getDevice();
        if (device) {
          await device.queue.onSubmittedWorkDone();
        }
      }

      if (onProgress) {
        // Layers phase: progress from 80% to 85% based on layer count
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

    // Load final norm and LM head
    reportProgress('gpu_transfer', 0.85, 'Loading final weights...');
    await this._loadFinalWeights(onProgress);

    if (onProgress) {
      onProgress({ stage: 'complete', progress: 1.0 });
    }

    this.isLoaded = true;
    const totalTime = ((Date.now() - loadStartTime) / 1000).toFixed(2);
    const avgSpeed = formatBytes(bytesLoaded / (Date.now() - loadStartTime) * 1000);
    log.info('Loader', `Complete: ${formatBytes(bytesLoaded)} in ${totalTime}s (${avgSpeed}/s)`);

    // Clear shard cache after loading - weights are now in GPU memory
    // This frees JS heap memory that was holding ArrayBuffer copies of model shards
    this.shardCache.clear();

    // Stop periodic memory logging and log final stats
    this._stopMemoryLogging();

    return (this.manifest.config as ModelConfig) || {};
  }

  /**
   * Build tensor location map from manifest (v1: external tensors.json) or inline tensors (legacy)
   */
  private async _buildTensorLocations(): Promise<void> {
    this.tensorLocations.clear();

    // v1 format: load external tensors.json
    if (this.manifest?.tensorsFile) {
      debugTrace.loader(`Loading external tensor map: ${this.manifest.tensorsFile}`);

      let tensorsJsonRaw: string | null = null;

      // Try OPFS first (for downloaded models)
      if (!this.shardCache.hasCustomLoader) {
        tensorsJsonRaw = await loadTensorsFromOPFS();
      }

      // Try HTTP if we have a tensors URL set (for HTTP-based testing)
      if (!tensorsJsonRaw && this._tensorsJsonUrl) {
        try {
          const resp = await fetch(this._tensorsJsonUrl);
          if (resp.ok) {
            tensorsJsonRaw = await resp.text();
            debugTrace.loader(`Loaded tensors.json via HTTP: ${this._tensorsJsonUrl}`);
          }
        } catch (e) {
          log.warn('Loader', `Failed to load tensors.json from ${this._tensorsJsonUrl}: ${(e as Error).message}`);
        }
      }

      if (tensorsJsonRaw) {
        // Parse the tensor map (returns TensorMap with 'shard' property from rdrr format)
        const tensorsJson = parseTensorMap(tensorsJsonRaw);
        for (const [name, info] of Object.entries(tensorsJson)) {
          this.tensorLocations.set(name, {
            shardIndex: info.shard,  // Map 'shard' to 'shardIndex'
            offset: info.offset,
            size: info.size,
            shape: info.shape,
            dtype: info.dtype,
            spans: info.spans,
            layout: info.layout,
            originalShape: info.originalShape,
          });
        }
        debugTrace.loader(`Loaded ${this.tensorLocations.size} tensors from tensors.json`);
        return;
      }
    }

    // Legacy format: inline tensors in manifest
    if (!this.manifest?.tensors) {
      log.warn('Loader', 'No tensor locations in manifest');
      return;
    }

    for (const [name, info] of Object.entries(this.manifest.tensors)) {
      const tensorInfo = info as {
        shard?: number;
        shardIndex?: number;
        offset: number;
        size: number;
        shape: number[];
        dtype: string;
        spans?: Array<{ shardIndex: number; offset: number; size: number }>;
        layout?: 'row' | 'column';
        originalShape?: number[];
      };
      this.tensorLocations.set(name, {
        shardIndex: tensorInfo.shardIndex ?? tensorInfo.shard ?? 0,
        offset: tensorInfo.offset,
        size: tensorInfo.size,
        shape: tensorInfo.shape,
        dtype: tensorInfo.dtype,
        spans: tensorInfo.spans,
        layout: tensorInfo.layout,  // Column-major if pre-transposed
        originalShape: tensorInfo.originalShape,
      });
    }
    debugTrace.loader(`Tensor map: ${this.tensorLocations.size} tensors (inline)`);
  }

  /**
   * Load a tensor by name
   */
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

    // Debug: Log tensor loading details for attention weights
    if (name.includes('attn_k') || name.includes('k_proj')) {
      debugTrace.loader(`Loading ${name}: shape=${JSON.stringify(location.shape)}, size=${location.size}, dtype=${location.dtype}, spans=${!!location.spans}`);
    }

    // Fast path for multi-shard tensors when uploading to GPU
    if (location.spans && toGPU) {
      debugTrace.loader(`Loading tensor "${name}" via spans path (${location.spans.length} spans, dtype=${location.dtype})`);
      const device = getDevice();
      if (!device) {
        log.warn('Loader', ' GPU device not available; falling back to CPU assembly');
      } else {
        // Quantized tensors
        if (location.dtype === 'Q4_K_M' || location.dtype === 'Q4_K') {
          const caps = this.gpuCapabilities || getKernelCapabilities();
          const isMatmulWeight = shouldDequantizeToF16(name);

          const isPackedQ4K =
            Array.isArray(location.shape) &&
            location.shape.length === 2 &&
            (() => {
              const [rows, cols] = location.shape;
              const expectedRowwise = rows * Math.ceil(cols / QK_K) * Q4K_BLOCK_BYTES;
              return location.size < expectedRowwise;
            })();

          // Fused Q4K path: keep raw quantized buffer for matmul weights
          // EXCLUDE embeddings - gather kernel doesn't support Q4K, only matmul does
          // Note: GGUF uses 'token_embd' (without second 'e'), HF uses 'embed_tokens'
          const isEmbedding = name.toLowerCase().includes('embd') ||
                              name.toLowerCase().includes('embed') ||
                              name.toLowerCase().includes('wte');
          if (this.useFusedQ4K && isMatmulWeight && !isEmbedding && caps?.hasSubgroups && !isPackedQ4K) {
            debugTrace.loader(`Loading Q4K weight: ${name} (size=${location.size})`);
            const q4kBuffer = acquireBuffer(location.size, undefined, `q4k_${name}`);
            let tensorOffset = 0;
            for (const span of location.spans) {
              const data = await this._loadShard(span.shardIndex);
              if (span.offset + span.size > data.byteLength) {
                throw new Error(
                  `[DopplerLoader] Shard ${span.shardIndex} too small for tensor "${name}" span.`
                );
              }
              const bytes = new Uint8Array(data, span.offset, span.size);
              device.queue.writeBuffer(q4kBuffer, tensorOffset, bytes);
              tensorOffset += span.size;
            }
            this.gpuBuffers.add(q4kBuffer);
            // Return WeightBuffer with q4k dtype, row layout (GGUF convention)
            return createWeightBuffer(q4kBuffer, 'q4k', 'row', location.shape, name);
          }

          // Standard dequant path
          if (this.useFusedQ4K && isMatmulWeight && caps?.hasSubgroups && isPackedQ4K) {
            const [rows, cols] = location.shape;
            const expectedRowwise = rows * Math.ceil(cols / QK_K) * Q4K_BLOCK_BYTES;
            debugTrace.loader(`Packed Q4K weight ${name} [${rows},${cols}] incompatible with fused matmul, using dequant`);
          }
          const quantBuffer = acquireBuffer(location.size, undefined, `quant_${name}`);
          let tensorOffset = 0;
          for (const span of location.spans) {
            const data = await this._loadShard(span.shardIndex);
            if (span.offset + span.size > data.byteLength) {
              throw new Error(
                `[DopplerLoader] Shard ${span.shardIndex} too small for tensor "${name}" span.`
              );
            }
            const bytes = new Uint8Array(data, span.offset, span.size);
            device.queue.writeBuffer(quantBuffer, tensorOffset, bytes);
            tensorOffset += span.size;
          }

          const numBlocks = Math.ceil(location.size / 144);
          let outputDtype: 'f16' | 'f32' = 'f32';
          if (isMatmulWeight) {
            outputDtype = this.keepF32Weights ? 'f32' : (caps?.hasF16 ? 'f16' : 'f32');
          }
          const dequantizedTensor = await dequantize(quantBuffer, numBlocks, { outputDtype });
          const dequantized = dequantizedTensor.buffer;

          releaseBuffer(quantBuffer);
          this.gpuBuffers.add(dequantized);

          // Handle weight layout based on q4kLayout setting:
          // - column_wise: weights were pre-transposed during conversion, use transposeB=false
          // - otherwise: GGUF convention (weights are [N,K]), use transposeB=true (default)
          const layout: WeightLayout = (this.q4kLayout === 'column_wise' && isMatmulWeight) ? 'column' : 'row';
          const dtype: WeightDtype = outputDtype;
          return createWeightBuffer(dequantized, dtype, layout, location.shape, name);
        }

        // Q6_K tensors (6-bit quantization)
        if (location.dtype === 'Q6_K') {
          debugTrace.loader(`Loading Q6_K tensor "${name}" via spans path (${location.spans.length} spans)`);
          const quantBuffer = acquireBuffer(location.size, undefined, `quant_${name}`);
          let tensorOffset = 0;
          for (const span of location.spans) {
            const data = await this._loadShard(span.shardIndex);
            if (span.offset + span.size > data.byteLength) {
              throw new Error(
                `[DopplerLoader] Shard ${span.shardIndex} too small for tensor "${name}" span.`
              );
            }
            const bytes = new Uint8Array(data, span.offset, span.size);
            device.queue.writeBuffer(quantBuffer, tensorOffset, bytes);
            tensorOffset += span.size;
          }

          const numBlocks = Math.floor(location.size / Q6K_BLOCK_BYTES);
          debugTrace.loader(`Dequantizing Q6_K ${name}: size=${location.size}, numBlocks=${numBlocks}, expectedOutput=${numBlocks * 256 * 2} (f16)`);
          const dequantizedTensor = await dequantizeQ6K(quantBuffer, numBlocks, { outputDtype: 'f16' });
          const dequantized = dequantizedTensor.buffer;
          debugTrace.loader(`Dequantized Q6_K ${name}: resultSize=${dequantized.size}`);

          // DEBUG: Sample dequantized values for embedding verification
          if (name.includes('embd') || name.includes('embed')) {
            const sampleSize = Math.min(256, dequantized.size);
            const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const enc = device.createCommandEncoder();
            enc.copyBufferToBuffer(dequantized, 0, staging, 0, sampleSize);
            device.queue.submit([enc.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            const u16 = new Uint16Array(staging.getMappedRange().slice(0));
            staging.unmap();
            staging.destroy();
            // Convert F16 to F32 for display
            const f32Samples: number[] = [];
            for (let i = 0; i < Math.min(16, u16.length); i++) {
              const h = u16[i];
              const sign = (h & 0x8000) >> 15;
              const exp = (h & 0x7C00) >> 10;
              const mant = h & 0x03FF;
              let f: number;
              if (exp === 0) { f = mant === 0 ? 0 : Math.pow(2, -14) * (mant / 1024); }
              else if (exp === 31) { f = mant === 0 ? Infinity : NaN; }
              else { f = Math.pow(2, exp - 15) * (1 + mant / 1024); }
              f32Samples.push(sign ? -f : f);
            }
            debugTrace.loader(`DEBUG embed dequant first16: [${f32Samples.map(x => x.toFixed(4)).join(', ')}]`);
          }

          releaseBuffer(quantBuffer);
          this.gpuBuffers.add(dequantized);
          // GGUF stores ALL weights transposed - use default transposeB=true (row layout)
          // Return WeightBuffer for matmul weights, raw GPUBuffer for norms/other
          const isQ6KMatmulWeight = shouldDequantizeToF16(name);
          if (isQ6KMatmulWeight) {
            return createWeightBuffer(dequantized, 'f16', 'row', location.shape, name);
          }
          return dequantized;
        }

        // BF16 tensors
        if (location.dtype === 'BF16') {
          debugTrace.loader(`Loading BF16 tensor "${name}" with ${location.spans.length} spans, total size=${location.size}`);
          const srcBuffer = acquireBuffer(location.size, undefined, `${name}_bf16`);
          let tensorOffset = 0;
          for (const span of location.spans) {
            const data = await this._loadShard(span.shardIndex);
            if (span.offset + span.size > data.byteLength) {
              throw new Error(
                `[DopplerLoader] Shard ${span.shardIndex} too small for tensor "${name}" span.`
              );
            }
            const bytes = new Uint8Array(data, span.offset, span.size);
            device.queue.writeBuffer(srcBuffer, tensorOffset, bytes);
            tensorOffset += span.size;
            debugTrace.loader(`Wrote span ${span.shardIndex}: offset=${tensorOffset}, first4bytes=[${bytes[0]}, ${bytes[1]}, ${bytes[2]}, ${bytes[3]}]`);
          }

          const numElements = location.size / 2;
          const caps = this.gpuCapabilities || getKernelCapabilities();
          const isMatmulWeight = shouldDequantizeToF16(name);

          try {
            // Ensure all writeBuffer operations are flushed to GPU before conversion
            await device.queue.onSubmittedWorkDone();

            // For matmul weights with F16 support: BF16 → F16 (no intermediate F32 buffer)
            // This is critical for lm_head (262k vocab) which dominates decode time
            if (caps?.hasF16 && isMatmulWeight) {
              const f16Tensor = await runBF16ToF16(srcBuffer, [numElements], name);
              releaseBuffer(srcBuffer);
              this.gpuBuffers.add(f16Tensor.buffer);
              debugTrace.loader(`BF16→F16 for matmul weight: ${name} (${numElements} elements, spans path)`);
              // Return WeightBuffer with f16 dtype and layout from manifest
              const layout: WeightLayout = location.layout === 'column' ? 'column' : 'row';
              return createWeightBuffer(f16Tensor.buffer, 'f16', layout, location.shape, name);
            }

            // Standard path: BF16 → F32
            debugTrace.loader(`Converting BF16→F32: ${numElements} elements`);
            const dstBuffer = await convertBF16ToF32GPU(srcBuffer, numElements, name);
            releaseBuffer(srcBuffer);
            if (dstBuffer instanceof GPUBuffer) {
              this.gpuBuffers.add(dstBuffer);
              // Return WeightBuffer for matmul weights, raw GPUBuffer for norms/other
              if (isMatmulWeight) {
                const layout: WeightLayout = location.layout === 'column' ? 'column' : 'row';
                return createWeightBuffer(dstBuffer, 'f32', layout, location.shape, name);
              }
              return applyBufferLayout(dstBuffer, location);
            }
            return dstBuffer;
          } catch (err) {
            log.error('Loader', 'BF16 conversion failed:', err);
            throw err;
          }
        }

        // Other dtypes (F16, F32)
        const buffer = acquireBuffer(location.size, undefined, name);
        let tensorOffset = 0;
        for (const span of location.spans) {
          const data = await this._loadShard(span.shardIndex);
          if (span.offset + span.size > data.byteLength) {
            throw new Error(
              `[DopplerLoader] Shard ${span.shardIndex} too small for tensor "${name}" span.`
            );
          }
          const bytes = new Uint8Array(data, span.offset, span.size);
          device.queue.writeBuffer(buffer, tensorOffset, bytes);
          tensorOffset += span.size;
        }
        this.gpuBuffers.add(buffer);

        // Determine dtype and layout for WeightBuffer
        const dtype: WeightDtype = location.dtype === 'F16' ? 'f16' : 'f32';
        const layout: WeightLayout = location.layout === 'column' ? 'column' : 'row';
        const isMatmulWeight = shouldDequantizeToF16(name);

        // Return WeightBuffer for matmul weights (preserves dtype/layout for matmul selection)
        if (isMatmulWeight) {
          return createWeightBuffer(buffer, dtype, layout, location.shape, name);
        }

        // Non-matmul weights (e.g., RMSNorm) are consumed as F32 in kernels.
        // If stored as F16, upcast to F32 to avoid interpreting f16 bits as f32.
        if (dtype === 'f16') {
          const numElements = location.shape.reduce((a, b) => a * b, 1);
          const inputTensor = createTensor(buffer, 'f16', [numElements], `${name}_f16`);
          const f32Tensor = await castF16ToF32(inputTensor);
          releaseBuffer(buffer);
          this.gpuBuffers.add(f32Tensor.buffer);
          return applyBufferLayout(f32Tensor.buffer, location);
        }

        // Note: norm weights don't need dtype tracking (kernel handles F32 internally)
        return applyBufferLayout(buffer, location);
      }
    }

    // Load shard data into CPU memory (single-shard or CPU path)
    // Use Uint8Array views to avoid copying 420+ tensors (~6GB of garbage)
    let shardView: Uint8Array;
    if (location.spans) {
      const chunks: Uint8Array[] = [];
      for (const span of location.spans) {
        const data = await this._loadShard(span.shardIndex);
        if (span.offset + span.size > data.byteLength) {
          throw new Error(
            `[DopplerLoader] Shard ${span.shardIndex} too small for tensor "${name}" span.`
          );
        }
        chunks.push(new Uint8Array(data, span.offset, span.size));
      }
      const totalSize = chunks.reduce((s, c) => s + c.length, 0);
      const combined = new Uint8Array(totalSize);
      let offset = 0;
      for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      shardView = combined;
    } else {
      const fullShard = await this._loadShard(location.shardIndex);
      // Use a view instead of slice() to avoid copying - saves ~6GB of garbage for 9B models
      shardView = new Uint8Array(fullShard, location.offset, location.size);
    }

    // Handle quantized data
    if (location.dtype === 'Q4_K_M' || location.dtype === 'Q4_K') {
      if (toGPU) {
        const device = getDevice();
        const caps = this.gpuCapabilities || getKernelCapabilities();
        const isMatmulWeight = shouldDequantizeToF16(name);

        const isPackedQ4K =
          Array.isArray(location.shape) &&
          location.shape.length === 2 &&
          (() => {
            const [rows, cols] = location.shape;
            const expectedRowwise = rows * Math.ceil(cols / QK_K) * Q4K_BLOCK_BYTES;
            return location.size < expectedRowwise;
          })();

        // Fused Q4K path: keep raw quantized buffer for matmul weights
        // This enables 2-3x speedup by doing dequant+matmul in one kernel pass
        // EXCLUDE embeddings - gather kernel doesn't support Q4K, only matmul does
        // Note: GGUF uses 'token_embd' (without second 'e'), HF uses 'embed_tokens'
        const isEmbedding = name.toLowerCase().includes('embd') ||
                            name.toLowerCase().includes('embed') ||
                            name.toLowerCase().includes('wte');
        if (this.useFusedQ4K && isMatmulWeight && !isEmbedding && caps?.hasSubgroups && !isPackedQ4K) {
          debugTrace.loader(`Loading Q4K weight (single-shard): ${name} (size=${location.size})`);
          const q4kBuffer = acquireBuffer(location.size, undefined, `q4k_${name}`);
          device!.queue.writeBuffer(q4kBuffer, 0, shardView as GPUAllowSharedBufferSource);
          this.gpuBuffers.add(q4kBuffer);
          // Return WeightBuffer with q4k dtype, row layout (GGUF convention)
          return createWeightBuffer(q4kBuffer, 'q4k', 'row', location.shape, name);
        }

        if (this.useFusedQ4K && isMatmulWeight && caps?.hasSubgroups && isPackedQ4K) {
          const [rows, cols] = location.shape;
          debugTrace.loader(`Packed Q4K weight ${name} [${rows},${cols}] incompatible with fused matmul, using dequant`);
        }

        // Standard dequant path: dequantize to f16 or f32
        const quantBuffer = acquireBuffer(location.size, undefined, `quant_${name}`);
        device!.queue.writeBuffer(quantBuffer, 0, shardView as GPUAllowSharedBufferSource);

        const numBlocks = Math.ceil(location.size / 144);
        let outputDtype: 'f16' | 'f32' = 'f32';
          if (isMatmulWeight) {
            outputDtype = this.keepF32Weights ? 'f32' : (caps?.hasF16 ? 'f16' : 'f32');
          }
        debugTrace.loader(`Dequantizing ${name}: size=${location.size}, numBlocks=${numBlocks}, outputDtype=${outputDtype}, expectedOutput=${numBlocks * 256 * (outputDtype === 'f16' ? 2 : 4)}`);
        const dequantizedTensor = await dequantize(quantBuffer, numBlocks, { outputDtype });
        const dequantized = dequantizedTensor.buffer;
        debugTrace.loader(`Dequantized ${name}: resultSize=${dequantized.size}`);

        releaseBuffer(quantBuffer);
        this.gpuBuffers.add(dequantized);

        // Handle weight layout based on q4kLayout setting:
        // - column_wise: weights were pre-transposed during conversion, use transposeB=false
        // - otherwise: GGUF convention (weights are [N,K]), use transposeB=true (default)
        const layout: WeightLayout = (this.q4kLayout === 'column_wise' && isMatmulWeight) ? 'column' : 'row';
        const dtype: WeightDtype = outputDtype;
        return createWeightBuffer(dequantized, dtype, layout, location.shape, name);
      }
      return shardView;
    }

    // Handle Q6_K data (single-shard path)
    if (location.dtype === 'Q6_K') {
      if (toGPU) {
        const device = getDevice();
        debugTrace.loader(`Loading Q6_K tensor "${name}" (single-shard), size=${location.size}`);
        const quantBuffer = acquireBuffer(location.size, undefined, `quant_${name}`);
        device!.queue.writeBuffer(quantBuffer, 0, shardView as GPUAllowSharedBufferSource);

        const numBlocks = Math.floor(location.size / Q6K_BLOCK_BYTES);
        debugTrace.loader(`Dequantizing Q6_K ${name}: size=${location.size}, numBlocks=${numBlocks}, expectedOutput=${numBlocks * 256 * 2} (f16)`);
        const dequantizedTensor = await dequantizeQ6K(quantBuffer, numBlocks, { outputDtype: 'f16' });
        const dequantized = dequantizedTensor.buffer;
        debugTrace.loader(`Dequantized Q6_K ${name}: resultSize=${dequantized.size}`);

        releaseBuffer(quantBuffer);
        this.gpuBuffers.add(dequantized);
        // GGUF stores ALL weights transposed - use default transposeB=true (row layout)
        // Return WeightBuffer for matmul weights, raw GPUBuffer for norms/other
        const isQ6KMatmulWeight = shouldDequantizeToF16(name);
        if (isQ6KMatmulWeight) {
          return createWeightBuffer(dequantized, 'f16', 'row', location.shape, name);
        }
        return dequantized;
      }
      return shardView;
    }

    // Handle BF16 data
    if (location.dtype === 'BF16') {
      if (toGPU) {
        const device = getDevice();
        const srcBuffer = acquireBuffer(shardView.byteLength, undefined, `${name}_bf16`);
        device!.queue.writeBuffer(srcBuffer, 0, shardView as GPUAllowSharedBufferSource);

        const numElements = shardView.byteLength / 2;
        const caps = this.gpuCapabilities || getKernelCapabilities();
        const isMatmulWeight = shouldDequantizeToF16(name);

        // For matmul weights with F16 support: BF16 → F16 (no intermediate F32 buffer)
        // This is critical for lm_head (262k vocab) which dominates decode time
        if (caps?.hasF16 && isMatmulWeight) {
          const f16Tensor = await runBF16ToF16(srcBuffer, [numElements], name);
          releaseBuffer(srcBuffer);
          this.gpuBuffers.add(f16Tensor.buffer);
          debugTrace.loader(`BF16→F16 for matmul weight: ${name} (${numElements} elements)`);
          // Return WeightBuffer with f16 dtype and layout from manifest
          const layout: WeightLayout = location.layout === 'column' ? 'column' : 'row';
          return createWeightBuffer(f16Tensor.buffer, 'f16', layout, location.shape, name);
        }

        // Standard path: BF16 → F32
        const dstBuffer = await convertBF16ToF32GPU(srcBuffer, numElements, name);
        releaseBuffer(srcBuffer);
        if (dstBuffer instanceof GPUBuffer) {
          this.gpuBuffers.add(dstBuffer);
          // Return WeightBuffer for matmul weights, raw GPUBuffer for norms/other
          if (isMatmulWeight) {
            const layout: WeightLayout = location.layout === 'column' ? 'column' : 'row';
            return createWeightBuffer(dstBuffer, 'f32', layout, location.shape, name);
          }
          return applyBufferLayout(dstBuffer, location);
        }
        return dstBuffer;
      }

      // CPU path - need to slice for typed array alignment
      const bf16 = new Uint16Array(shardView.slice().buffer);
      const f32 = new Float32Array(bf16.length);
      const tmp = new ArrayBuffer(4);
      const u32View = new Uint32Array(tmp);
      const f32View = new Float32Array(tmp);
      for (let i = 0; i < bf16.length; i++) {
        u32View[0] = bf16[i] << 16;
        f32[i] = f32View[0];
      }
      return f32;
    }

    // Handle F32/F16 data
    if (toGPU) {
      const device = getDevice();
      const buffer = acquireBuffer(location.size, undefined, name);
      device!.queue.writeBuffer(buffer, 0, shardView as GPUAllowSharedBufferSource);
      this.gpuBuffers.add(buffer);

      // Determine dtype and layout for WeightBuffer
      const dtype: WeightDtype = location.dtype === 'F16' ? 'f16' : 'f32';
      const layout: WeightLayout = location.layout === 'column' ? 'column' : 'row';
      const isMatmulWeight = shouldDequantizeToF16(name);

      // Return WeightBuffer for matmul weights, raw GPUBuffer for norms/other
      if (isMatmulWeight) {
        return createWeightBuffer(buffer, dtype, layout, location.shape, name);
      }

      // Non-matmul weights (e.g., RMSNorm) are consumed as F32 in kernels.
      // If stored as F16, upcast to F32 to avoid interpreting f16 bits as f32.
      if (dtype === 'f16') {
        const numElements = location.shape.reduce((a, b) => a * b, 1);
        const inputTensor = createTensor(buffer, 'f16', [numElements], `${name}_f16`);
        const f32Tensor = await castF16ToF32(inputTensor);
        releaseBuffer(buffer);
        this.gpuBuffers.add(f32Tensor.buffer);
        return applyBufferLayout(f32Tensor.buffer, location);
      }

      // Note: norm weights don't need dtype tracking (kernel handles F32 internally)
      return applyBufferLayout(buffer, location);
    } else {
      // CPU path - need to slice for typed array alignment
      if (location.dtype === 'F16') {
        const f16 = new Uint16Array(shardView.slice().buffer);
        const f32 = new Float32Array(f16.length);
        for (let i = 0; i < f16.length; i++) {
          f32[i] = f16ToF32(f16[i]);
        }
        return f32;
      }
      return new Float32Array(shardView.slice().buffer);
    }
  }

  /**
   * Check if model requires (1 + weight) offset for RMSNorm weights
   *
   * NOTE: GGUF files do NOT have the offset baked in - they store raw weights.
   * The +1 offset is applied at load time based on the manifest's config flag.
   */
  private _needsNormWeightOffset(): boolean {
    if (!this.manifest) {
      debugTrace.loader(' _needsNormWeightOffset: no manifest');
      return false;
    }

    const config = (this.manifest.config || {}) as ModelConfig;
    const arch = config.architectures?.[0] || (this.manifest.architecture as string) || '';
    const modelType = config.model_type || '';

    // Both Gemma 2 and Gemma 3 use (1 + weight) formula for RMSNorm
    const isGemma2 = /gemma.*2|gemma2/i.test(arch) || /gemma.*2|gemma2/i.test(modelType);
    const isGemma3 = /gemma.*3|gemma3/i.test(arch) || /gemma.*3|gemma3/i.test(modelType);
    const isGemmaFamily = isGemma2 || isGemma3;

    // Check explicit manifest flag first, then fall back to architecture detection
    const explicitFlag = (config as { rms_norm_weight_offset?: boolean }).rms_norm_weight_offset;
    const needsOffset = explicitFlag ?? isGemmaFamily;

    if (needsOffset && !this._normOffsetLogged) {
      this._normOffsetLogged = true;
      const family = isGemma2 ? 'Gemma 2' : 'Gemma 3';
      debugTrace.loader(` Applying +1 norm weight offset for ${family} layer norms`);
    }

    return needsOffset;
  }

  /**
   * Apply +1 offset to norm weights for Gemma 3+ models
   *
   * IMPORTANT: actualNumElements must be provided to avoid reading garbage padding
   * from the buffer pool's power-of-2 bucketing.
   */
  private async _applyNormWeightOffset(
    tensor: GPUBuffer | Float32Array,
    actualNumElements?: number,
    bufferDtype: 'f16' | 'f32' | 'bf16' = 'f32'
  ): Promise<GPUBuffer | Float32Array> {
    const device = getDevice();
    if (!device) {
      log.warn('Loader', ' No GPU device for norm offset');
      return tensor;
    }

    if (tensor instanceof GPUBuffer) {
      // Use provided dtype to determine element size (norm weights default to f32)
      const isF16 = bufferDtype === 'f16' || bufferDtype === 'bf16';
      const bytesPerElement = isF16 ? 2 : 4;

      // Use actual element count if provided, otherwise infer from buffer size
      const numElements = actualNumElements ?? Math.floor(tensor.size / bytesPerElement);
      const dataSize = numElements * bytesPerElement;

      // Ensure we don't read past the buffer
      const readSize = Math.min(dataSize, tensor.size);

      const stagingBuffer = device.createBuffer({
        size: readSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(tensor, 0, stagingBuffer, 0, readSize);
      device.queue.submit([encoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const rawData = stagingBuffer.getMappedRange().slice(0);
      stagingBuffer.unmap();
      stagingBuffer.destroy();

      // Convert to F32 for offset calculation
      let data: Float32Array;
      if (isF16) {
        const u16Data = new Uint16Array(rawData);
        data = new Float32Array(u16Data.length);
        if (bufferDtype === 'bf16') {
          const tmp = new ArrayBuffer(4);
          const u32View = new Uint32Array(tmp);
          const f32View = new Float32Array(tmp);
          for (let i = 0; i < u16Data.length; i++) {
            u32View[0] = u16Data[i] << 16;
            data[i] = f32View[0];
          }
        } else {
          for (let i = 0; i < u16Data.length; i++) {
            data[i] = f16ToF32(u16Data[i]);
          }
        }
      } else {
        data = new Float32Array(rawData);
      }

      const offsetData = new Float32Array(numElements);
      for (let i = 0; i < numElements; i++) {
        offsetData[i] = 1.0 + data[i];
      }

      // Debug: log first norm weight transformation (once per model load)
      if (!this._normOffsetDebugLogged) {
        this._normOffsetDebugLogged = true;
        const beforeMin = Math.min(...Array.from(data.slice(0, Math.min(256, numElements))));
        const beforeMax = Math.max(...Array.from(data.slice(0, Math.min(256, numElements))));
        const afterMin = Math.min(...Array.from(offsetData.slice(0, Math.min(256, numElements))));
        const afterMax = Math.max(...Array.from(offsetData.slice(0, Math.min(256, numElements))));
        debugTrace.loader(`Norm +1 offset: before=[${beforeMin.toFixed(3)}, ${beforeMax.toFixed(3)}] after=[${afterMin.toFixed(3)}, ${afterMax.toFixed(3)}]`);
      }

      releaseBuffer(tensor);
      const newBuffer = acquireBuffer(offsetData.byteLength, undefined, 'norm_offset');
      device.queue.writeBuffer(newBuffer, 0, offsetData);
      return newBuffer;
    }

    if (tensor instanceof Float32Array) {
      const numElements = actualNumElements ?? tensor.length;
      const offsetData = new Float32Array(numElements);
      for (let i = 0; i < numElements; i++) {
        offsetData[i] = 1.0 + tensor[i];
      }
      // Always upload to GPU to prevent double-offset in pipeline
      // Pipeline's getNormWeightBuffer returns GPUBuffer as-is, skipping offset
      const newBuffer = acquireBuffer(offsetData.byteLength, undefined, 'norm_offset');
      device.queue.writeBuffer(newBuffer, 0, offsetData);
      return newBuffer;
    }

    log.warn('Loader', ' Unknown tensor type for norm offset');
    return tensor;
  }

  /**
   * Load embedding weights
   */
  private async _loadEmbeddings(_onProgress: ((progress: LoadProgress) => void) | null): Promise<void> {
    const embeddingNames = [
      'language_model.model.embed_tokens.weight',
      'model.embed_tokens.weight',
      'embed_tokens.weight',
      'token_embd.weight',
      'wte.weight',
      'transformer.wte.weight',
    ];

    const maybeDowncastEmbeddings = async (name: string, loc?: TensorLocation): Promise<void> => {
      const caps = getKernelCapabilities();
      if (!caps.hasF16 || this.keepF32Weights) return;

      const current = this.embeddings;
      if (!current || current instanceof Float32Array || isCpuWeightBuffer(current)) {
        return;
      }

      const dtype = isWeightBuffer(current)
        ? current.dtype
        : (loc?.dtype === 'F16' ? 'f16' : 'f32');
      if (dtype !== 'f32') {
        return;
      }

      const buffer = isWeightBuffer(current) ? current.buffer : current;
      const elems = buffer.size / 4;
      const inputTensor = createTensor(buffer, 'f32', [elems], 'embed_f32');
      const f16Tensor = await castF32ToF16(inputTensor);

      const layout = isWeightBuffer(current)
        ? current.layout
        : (loc ? this._resolveWeightLayout(loc, name) : 'row');
      const shape = isWeightBuffer(current)
        ? Array.from(current.shape)
        : (loc?.shape ?? [elems]);
      const newWeightBuffer = createWeightBuffer(f16Tensor.buffer, 'f16', layout, shape, name);

      releaseBuffer(buffer);
      this.embeddings = newWeightBuffer;
      this.gpuBuffers.add(f16Tensor.buffer);
    };

    for (const name of embeddingNames) {
      const loc = this.tensorLocations.get(name);
      const shouldStream = loc ? this._shouldStreamLargeWeight(name, loc, 'Embedding') : false;
      const tensor = await this._loadTensor(name, !shouldStream, true);
      if (shouldStream && tensor && !(tensor instanceof Float32Array)) {
        throw new Error(`[Loader] Embedding "${name}" too large for GPU and cannot be loaded on CPU (dtype=${loc?.dtype ?? 'unknown'}).`);
      }
      if (tensor && (tensor instanceof GPUBuffer || isWeightBuffer(tensor) || tensor instanceof Float32Array)) {
        // For GGUF tied embeddings (used as lm_head), need column layout
        // GGUF stores embeddings as [hidden_size, vocab_size] (column-major)
        // When used as lm_head, matmul needs transposeB=false
        log.info('Loader', `Embeddings tensor loaded: name=${name}, hasShape=${!!loc?.shape}, shape=${loc?.shape ? `[${loc.shape.join(',')}]` : 'none'}, isWeightBuffer=${isWeightBuffer(tensor)}`);

        // WeightBuffer already has layout set correctly from _loadTensor
        if (isWeightBuffer(tensor)) {
          this.embeddings = tensor;
          await maybeDowncastEmbeddings(name, loc);
          break;
        }

        if (tensor instanceof Float32Array && loc?.shape && shouldStream) {
          const layout = this._resolveWeightLayout(loc, name);
          const dtype: WeightDtype = loc.dtype === 'F16' ? 'f16' : 'f32';
          this.embeddings = createCpuWeightBuffer(tensor, dtype, layout, loc.shape, name);
          log.warn('Loader', `Embeddings stored on CPU for chunked gather (layout=${layout})`);
          break;
        }

        // If tensor is a raw GPUBuffer (not WeightBuffer), wrap with dtype/layout metadata.
        if (tensor instanceof GPUBuffer && loc?.shape && loc.shape.length === 2) {
          const layout = this._resolveWeightLayout(loc, name);
          const dtype: WeightDtype = loc.dtype === 'F16' ? 'f16' : 'f32';
          this.embeddings = createWeightBuffer(tensor, dtype, layout, loc.shape, name);
          log.info('Loader', `Wrapped embeddings as WeightBuffer (layout=${layout}, dtype=${dtype})`);
          await maybeDowncastEmbeddings(name, loc);
          break;
        }
        this.embeddings = tensor;
        await maybeDowncastEmbeddings(name, loc);
        break;
      }
    }

    if (!this.embeddings) {
      log.warn('Loader', ' Embeddings not found');
    }
  }

  /**
   * Load a single layer's weights
   */
  private async _loadLayer(
    layerIdx: number,
    _onProgress: ((progress: LoadProgress) => void) | null
  ): Promise<void> {
    const prefixes = [
      `language_model.model.layers.${layerIdx}`,
      `model.layers.${layerIdx}`,
      `layers.${layerIdx}`,
      `blk.${layerIdx}`,
    ];

    const weights: LayerWeights = {
      inputNorm: null,
      qProj: null,
      kProj: null,
      vProj: null,
      oProj: null,
      qNorm: null,
      kNorm: null,
      postAttentionNorm: null,
      preFeedforwardNorm: null,
      postFeedforwardNorm: null,
      postNorm: null,
      postAttnNorm: null,
      ffnGate: null,
      ffnUp: null,
      ffnDown: null,
      ffnGateUp: null,
    };

    const tryLoad = async (suffixes: string[]): Promise<GPUBuffer | WeightBuffer | Float32Array | null> => {
      for (const prefix of prefixes) {
        for (const suffix of suffixes) {
          const tensor = await this._loadTensor(`${prefix}.${suffix}`, true, true);
          if (tensor && (tensor instanceof GPUBuffer || tensor instanceof Float32Array || isWeightBuffer(tensor))) {
            return tensor;
          }
        }
      }
      return null;
    };

    const tryLoadNorm = async (suffixes: string[]): Promise<GPUBuffer | Float32Array | null> => {
      // Find tensor location to get actual shape (needed to avoid reading garbage from buffer pool padding)
      let actualNumElements: number | undefined;
      for (const prefix of prefixes) {
        for (const suffix of suffixes) {
          const name = `${prefix}.${suffix}`;
          const location = this.tensorLocations.get(name);
          if (location) {
            // Norm weights are 1D tensors with shape [hiddenSize]
            actualNumElements = location.shape.reduce((a, b) => a * b, 1);
            break;
          }
        }
        if (actualNumElements) break;
      }

      const tensor = await tryLoad(suffixes);
      if (!tensor) return null;

      // Norm weights are never WeightBuffer (they're f32 and not matmul weights)
      // Cast is safe because _loadTensor only returns WeightBuffer for matmul weights
      const normTensor = tensor as GPUBuffer | Float32Array;

      if (this._needsNormWeightOffset()) {
        return this._applyNormWeightOffset(normTensor, actualNumElements);
      }
      return normTensor;
    };

    // Load attention weights in parallel
    const [inputNorm, qProj, kProj, vProj, oProj, qNorm, kNorm, postAttentionNorm, preFeedforwardNorm, postFeedforwardNorm] = await Promise.all([
      tryLoadNorm(['input_layernorm.weight', 'attn_norm.weight']),
      tryLoad(['self_attn.q_proj.weight', 'attention.wq.weight', 'attn_q.weight']),
      tryLoad(['self_attn.k_proj.weight', 'attention.wk.weight', 'attn_k.weight']),
      tryLoad(['self_attn.v_proj.weight', 'attention.wv.weight', 'attn_v.weight']),
      tryLoad(['self_attn.o_proj.weight', 'attention.wo.weight', 'attn_output.weight']),
      // Gemma 3: q_norm and k_norm use Gemma3RMSNorm with (1+weight) formula
      // Same as layer norms - all Gemma 3 norms use (1+weight)
      // See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py
      tryLoadNorm(['self_attn.q_norm.weight', 'attn_q_norm.weight']),
      tryLoadNorm(['self_attn.k_norm.weight', 'attn_k_norm.weight']),
      tryLoadNorm(['post_attention_layernorm.weight', 'post_attention_norm.weight', 'ffn_norm.weight']),
      tryLoadNorm(['pre_feedforward_layernorm.weight']),
      tryLoadNorm(['post_feedforward_layernorm.weight', 'post_ffw_norm.weight']),
    ]);

    weights.inputNorm = inputNorm;
    weights.qProj = qProj;
    weights.kProj = kProj;
    weights.vProj = vProj;
    weights.oProj = oProj;
    weights.qNorm = qNorm;
    weights.kNorm = kNorm;

    // Log q_norm/k_norm loading status for layer 0 only
    if (layerIdx === 0) {
      const hasOffset = this._needsNormWeightOffset();
      debugTrace.loader(`Layer 0 norm weights: qNorm=${qNorm ? 'found' : 'null'}, kNorm=${kNorm ? 'found' : 'null'}, offset=${hasOffset ? '+1 applied' : 'none'}`);
    }
    weights.postAttentionNorm = postAttentionNorm;
    weights.preFeedforwardNorm = preFeedforwardNorm;
    weights.postFeedforwardNorm = postFeedforwardNorm;
    weights.postNorm = weights.postAttentionNorm || weights.preFeedforwardNorm;
    weights.postAttnNorm = weights.postNorm;

    if (!this.isMoE || !this._isExpertLayer(layerIdx)) {
      // Load FFN weights in parallel
      const [ffnGateUp, ffnGate, ffnUp, ffnDown] = await Promise.all([
        tryLoad(['mlp.gate_up_proj.weight', 'ffn_gate_up.weight', 'feed_forward.w1_w3.weight']),
        tryLoad(['mlp.gate_proj.weight', 'feed_forward.w1.weight', 'ffn_gate.weight']),
        tryLoad(['mlp.up_proj.weight', 'feed_forward.w3.weight', 'ffn_up.weight']),
        tryLoad(['mlp.down_proj.weight', 'feed_forward.w2.weight', 'ffn_down.weight']),
      ]);

      if (ffnGateUp) {
        // Fused path: no separate gate/up weights
        weights.ffnGateUp = ffnGateUp;
        weights.ffnGate = null;
        weights.ffnUp = null;
        debugTrace.loader(`Layer ${layerIdx}: Using fused gate_up_proj for 2-pass FFN`);
      } else {
        // Separate path: use gate and up individually (3-pass FFN)
        weights.ffnGate = ffnGate;
        weights.ffnUp = ffnUp;
      }

      weights.ffnDown = ffnDown;

      // Set aliases for pipeline compatibility
      weights.gate = weights.ffnGate;
      weights.up = weights.ffnUp;
      weights.down = weights.ffnDown;
      weights.gateUp = weights.ffnGateUp;
    }

    if (this.isMoE && this._isExpertLayer(layerIdx)) {
      const [routerWeight, routerBias] = await Promise.all([
        tryLoad(['mlp.router.weight', 'block_sparse_moe.gate.weight']),
        tryLoad(['mlp.router.bias']),
      ]);
      // Router weights are not matmul weights, so they're GPUBuffer | Float32Array (not WeightBuffer)
      weights.routerWeight = routerWeight as GPUBuffer | Float32Array | null;
      weights.routerBias = routerBias as GPUBuffer | Float32Array | null;
    }

    // Attention sinks are small tensors (not matmul weights), so they're GPUBuffer | Float32Array
    weights.attentionSinks = await tryLoad(['self_attn.sinks']) as GPUBuffer | Float32Array | null;

    this.layers.set(layerIdx, weights);

    // Downcast matmul weights to f16 when supported
    const caps = getKernelCapabilities();
    if (caps.hasF16) {
      // Note: buffer-dtypes.js removed - use getLayout from weight-buffer.js
      const { getLayout: getWBLayout, isWeightBuffer: isWB, getWeightDtype: getWBDtype } = await import('../gpu/weight-buffer.js');
      const matmulKeys: (keyof LayerWeights)[] = ['qProj', 'kProj', 'vProj', 'oProj', 'ffnGate', 'ffnUp', 'ffnDown', 'ffnGateUp'];
      for (const key of matmulKeys) {
        const buf = weights[key];

        // Handle WeightBuffer
        if (isWB(buf)) {
          const wbDtype = getWBDtype(buf);
          if (wbDtype === 'f32') {
            if (this.keepF32Weights) {
              debugTrace.loader(`Layer ${layerIdx} keeping ${key} in f32 (keepF32Weights=true)`);
              continue;
            }
            const elems = buf.buffer.size / 4;
            const wasColumnMajor = getWBLayout(buf) === 'column';
            debugTrace.loader(`Layer ${layerIdx} downcasting WeightBuffer ${key}: bufSize=${buf.buffer.size}, elems=${elems}, columnMajor=${wasColumnMajor}`);
            try {
              const inputTensor = createTensor(buf.buffer, 'f32', [elems], `${key}_f32`);
              const f16Tensor = await castF32ToF16(inputTensor);
              // Create new WeightBuffer with f16 dtype, preserving layout
              const newWeightBuffer = createWeightBuffer(f16Tensor.buffer, 'f16', wasColumnMajor ? 'column' : 'row', buf.shape as number[], buf.label);
              debugTrace.loader(`Layer ${layerIdx} ${key} downcast result: f16Size=${f16Tensor.buffer.size}`);
              releaseBuffer(buf.buffer);
              (weights as unknown as Record<string, GPUBuffer | WeightBuffer | Float32Array | null>)[key] = newWeightBuffer;
              this.gpuBuffers.add(f16Tensor.buffer);
            } catch (e) {
              log.warn('Loader', `Failed to downcast ${key} to f16:`, (e as Error).message);
            }
          }
          continue;
        }

        // Handle raw GPUBuffer (legacy path - WeakMap tracking removed, assume f32)
        if (buf instanceof GPUBuffer) {
          const dtype = getWeightDtype(buf) || 'f32';
          if (dtype === 'f32') {
            if (this.keepF32Weights) {
              debugTrace.loader(`Layer ${layerIdx} keeping ${key} in f32 (keepF32Weights=true)`);
              continue;
            }
            const elems = buf.size / 4;
            // Preserve column-major layout through the f32→f16 downcast (raw GPUBuffer defaults to row)
            const wasColumnMajor = getWBLayout(buf) === 'column';
            debugTrace.loader(`Layer ${layerIdx} downcasting ${key}: bufSize=${buf.size}, elems=${elems}, expectedF16=${elems * 2}, columnMajor=${wasColumnMajor}`);
            try {
              const inputTensor = createTensor(buf, 'f32', [elems], `${key}_f32`);
              const f16Tensor = await castF32ToF16(inputTensor);
              // Create WeightBuffer with f16 dtype, preserving layout
              const loc = this.tensorLocations.get(key);
              const shape = loc?.shape ?? [elems];
              const newWeightBuffer = createWeightBuffer(f16Tensor.buffer, 'f16', wasColumnMajor ? 'column' : 'row', shape, key);
              debugTrace.loader(`Layer ${layerIdx} ${key} downcast result: f16Size=${f16Tensor.buffer.size}`);
              releaseBuffer(buf);
              (weights as unknown as Record<string, GPUBuffer | WeightBuffer | Float32Array | null>)[key] = newWeightBuffer;
              this.gpuBuffers.add(f16Tensor.buffer);
            } catch (e) {
              log.warn('Loader', `Failed to downcast ${key} to f16:`, (e as Error).message);
            }
          }
        }
      }
    }
  }

  /**
   * Check if layer uses MoE
   */
  private _isExpertLayer(_layerIdx: number): boolean {
    return this.isMoE;
  }

  /**
   * Pre-load specific shards for an expert (lazy loading support)
   */
  private async _preloadShardsForExpert(layerIdx: number, expertIdx: number): Promise<void> {
    // Get required shards from manifest mapping
    const shardIndices = getShardsForExpert(layerIdx, expertIdx);
    if (shardIndices.length === 0) {
      // No mapping available, fall back to loading all shards on demand
      return;
    }

    // Pre-load only the shards needed for this expert
    for (const shardIndex of shardIndices) {
      if (!this.shardCache.has(shardIndex)) {
        await this._loadShard(shardIndex);
      }
    }
  }

  /**
   * Prefetch experts for next layer (overlap loading with compute)
   * Call this after router selects experts for current layer
   * @param nextLayerIdx Layer index to prefetch for
   * @param expertIndices Expert indices likely to be used
   */
  prefetchExperts(nextLayerIdx: number, expertIndices: number[]): void {
    const numLayers = (this.manifest?.config as Record<string, unknown>)?.num_hidden_layers as number ?? 0;
    if (!this.isMoE || nextLayerIdx >= numLayers) {
      return;
    }

    // Fire-and-forget: load shards in background
    // This overlaps shard loading with current layer's compute
    const promises = expertIndices.map(async (expertIdx) => {
      // Check if already cached
      if (this.expertCache?.has(nextLayerIdx, expertIdx)) {
        return;
      }
      // Pre-load the shards (not the full expert tensor upload)
      await this._preloadShardsForExpert(nextLayerIdx, expertIdx);
    });

    // Don't await - let it run in background
    Promise.all(promises).catch((e) => {
      log.warn('Loader', ' Expert prefetch error:', e);
    });
  }

  /**
   * Get likely experts for next layer based on current layer's routing
   * Simple heuristic: same experts tend to be selected across layers
   */
  predictNextLayerExperts(currentExperts: number[]): number[] {
    // For now, just predict same experts will be used
    // More sophisticated: track expert correlation across layers
    return currentExperts;
  }

  /**
   * Load expert weights on demand (lazy loading from OPFS)
   */
  async loadExpert(layerIdx: number, expertIdx: number): Promise<ExpertWeights> {
    // Check LRU cache first
    if (this.expertCache) {
      const cached = this.expertCache.get(layerIdx, expertIdx);
      if (cached) {
        return cached;
      }
    }

    // Fall back to simple map for non-cached experts (GPT-OSS packed weights)
    const key = `layer_${layerIdx}_expert_${expertIdx}`;
    if (this.experts.has(key)) {
      return this.experts.get(key)!;
    }

    debugTrace.loader(`Loading expert ${expertIdx} for layer ${layerIdx}`);

    // Pre-load only the shards containing this expert's tensors
    await this._preloadShardsForExpert(layerIdx, expertIdx);

    // Get tensor names from manifest if available (for logging/debugging)
    const tensorNames = getTensorsForExpert(layerIdx, expertIdx);
    if (tensorNames.length > 0) {
      debugTrace.loader(`Expert ${layerIdx}_${expertIdx} tensors: ${tensorNames.length}`);
    }

    const prefix = `layers.${layerIdx}.block_sparse_moe.experts.${expertIdx}`;
    const altPrefix = `model.layers.${layerIdx}.block_sparse_moe.experts.${expertIdx}`;

    let weights: ExpertWeights = {
      gate: (await this._loadTensor(`${prefix}.w1.weight`) ||
            await this._loadTensor(`${altPrefix}.w1.weight`)) as GPUBuffer | Float32Array | null,
      up: (await this._loadTensor(`${prefix}.w3.weight`) ||
          await this._loadTensor(`${altPrefix}.w3.weight`)) as GPUBuffer | Float32Array | null,
      down: (await this._loadTensor(`${prefix}.w2.weight`) ||
            await this._loadTensor(`${altPrefix}.w2.weight`)) as GPUBuffer | Float32Array | null,
    };

    // Try GPT-OSS naming if Mixtral naming not found
    if (!weights.gate && !weights.up && !weights.down) {
      const gptOssPrefix = `model.layers.${layerIdx}.mlp.experts`;
      const packedKey = `layer_${layerIdx}_gptoss_packed`;
      let packed = this.experts.get(packedKey);

      if (!packed) {
        const config = this.manifest?.config as ModelConfig | undefined;
        const numExpertsFromConfig = config?.num_local_experts || config?.num_experts || 32;

        packed = {
          isGptOss: true,
          numExperts: numExpertsFromConfig,
          gateUpBlocks: await this._loadTensor(`${gptOssPrefix}.gate_up_proj_blocks`) as GPUBuffer | null,
          gateUpScales: await this._loadTensor(`${gptOssPrefix}.gate_up_proj_scales`) as GPUBuffer | null,
          gateUpBias: await this._loadTensor(`${gptOssPrefix}.gate_up_proj_bias`) as GPUBuffer | null,
          downBlocks: await this._loadTensor(`${gptOssPrefix}.down_proj_blocks`) as GPUBuffer | null,
          downScales: await this._loadTensor(`${gptOssPrefix}.down_proj_scales`) as GPUBuffer | null,
          downBias: await this._loadTensor(`${gptOssPrefix}.down_proj_bias`) as GPUBuffer | null,
        };

        this.experts.set(packedKey, packed);
      }

      weights = {
        isGptOss: true,
        expertIdx,
        numExperts: packed.numExperts,
        gateUpBlocks: packed.gateUpBlocks,
        gateUpScales: packed.gateUpScales,
        gateUpBias: packed.gateUpBias,
        downBlocks: packed.downBlocks,
        downScales: packed.downScales,
        downBias: packed.downBias,
      };
    }

    // Downcast Mixtral-style F32 weights to F16
    if (!weights.isGptOss) {
      const caps = getKernelCapabilities();
      if (caps.hasF16) {
        // Note: buffer-dtypes.js removed - use getLayout from weight-buffer.js
        const { isWeightBuffer: isWB, getLayout: getWBLayout, getWeightDtype: getWBDtype } = await import('../gpu/weight-buffer.js');
        for (const k of ['gate', 'up', 'down'] as const) {
          const buf = weights[k];

          // Handle WeightBuffer
          if (isWB(buf)) {
            const wbDtype = getWBDtype(buf);
            if (wbDtype === 'f32') {
              const elems = buf.buffer.size / 4;
              const wasColumnMajor = getWBLayout(buf) === 'column';
              try {
                const inputTensor = createTensor(buf.buffer, 'f32', [elems], `expert_${k}_f32`);
                const f16Tensor = await castF32ToF16(inputTensor);
                const newWeightBuffer = createWeightBuffer(f16Tensor.buffer, 'f16', wasColumnMajor ? 'column' : 'row', buf.shape as number[], buf.label);
                releaseBuffer(buf.buffer);
                weights[k] = newWeightBuffer;
                this.gpuBuffers.add(f16Tensor.buffer);
              } catch (e) {
                log.warn('Loader', `Failed to downcast expert ${k} to f16:`, (e as Error).message);
              }
            }
            continue;
          }

          // Handle raw GPUBuffer (legacy path - WeakMap tracking removed, assume f32)
          if (buf instanceof GPUBuffer) {
            const dtype = getWeightDtype(buf) || 'f32';
            if (dtype === 'f32') {
              const elems = buf.size / 4;
              // Preserve column-major layout through the f32→f16 downcast (raw GPUBuffer defaults to row)
              const wasColumnMajor = getWBLayout(buf) === 'column';
              try {
                const inputTensor = createTensor(buf, 'f32', [elems], `expert_${k}_f32`);
                const f16Tensor = await castF32ToF16(inputTensor);
                // Create WeightBuffer with f16 dtype, preserving layout
                const newWeightBuffer = createWeightBuffer(f16Tensor.buffer, 'f16', wasColumnMajor ? 'column' : 'row', [elems], `expert_${k}`);
                releaseBuffer(buf);
                weights[k] = newWeightBuffer;
                this.gpuBuffers.add(f16Tensor.buffer);
              } catch (e) {
                log.warn('Loader', `Failed to downcast expert ${k} to f16:`, (e as Error).message);
              }
            }
          }
        }
      }
    }

    // Calculate expert size and store in LRU cache
    if (!weights.isGptOss && this.expertCache) {
      let sizeBytes = 0;
      const { isWeightBuffer: isWB2 } = await import('../gpu/weight-buffer.js');
      for (const k of ['gate', 'up', 'down'] as const) {
        const buf = weights[k];
        if (isWB2(buf)) {
          sizeBytes += buf.buffer.size;
        } else if (buf instanceof GPUBuffer) {
          sizeBytes += buf.size;
        }
      }
      // Use manifest-provided expert size if available, otherwise calculate
      const manifestBytes = getExpertBytes();
      if (manifestBytes > 0) {
        sizeBytes = manifestBytes;
      }
      this.expertCache.put(layerIdx, expertIdx, weights, sizeBytes);
    } else {
      // GPT-OSS packed weights use the simple map (shared across experts)
      this.experts.set(key, weights);
    }

    return weights;
  }

  /**
   * Load final layer norm and LM head
   */
  private async _loadFinalWeights(_onProgress: ((progress: LoadProgress) => void) | null): Promise<void> {
    // Try loading final norm with known names
    const finalNormNames = [
      'language_model.model.norm.weight',
      'model.norm.weight',
      'norm.weight',
      'output_norm.weight',
      'transformer.ln_f.weight',
    ];
    let finalNormElements: number | undefined;
    let finalNormDtype: 'f16' | 'f32' = 'f32';
    for (const name of finalNormNames) {
      const location = this.tensorLocations.get(name);
      if (location) {
        finalNormElements = location.shape.reduce((a, b) => a * b, 1);
        finalNormDtype = 'f32';
        this.finalNorm = await this._loadTensor(name, true, true) as GPUBuffer | Float32Array | null;
        break;
      }
    }

    if (this.finalNorm && this._needsNormWeightOffset()) {
      this.finalNorm = await this._applyNormWeightOffset(this.finalNorm, finalNormElements, finalNormDtype);
    }

    if (!this.finalNorm) {
      log.warn('Loader', ' Final norm not found');
    }

    const lmHeadNames = [
      'language_model.lm_head.weight',
      'lm_head.weight',
      'output.weight',
    ];
    let lmHeadName: string | null = null;
    let lmHeadLoc: TensorLocation | undefined;
    for (const name of lmHeadNames) {
      const loc = this.tensorLocations.get(name);
      if (!loc) continue;
      const shouldStream = this._shouldStreamLargeWeight(name, loc, 'LM head');
      const tensor = await this._loadTensor(name, !shouldStream, true);
      if (shouldStream && tensor && !(tensor instanceof Float32Array)) {
        throw new Error(`[Loader] LM head "${name}" too large for GPU and cannot be loaded on CPU (dtype=${loc.dtype}).`);
      }
      if (tensor && (tensor instanceof GPUBuffer || isWeightBuffer(tensor) || tensor instanceof Float32Array)) {
        lmHeadName = name;
        lmHeadLoc = loc;
        if (tensor instanceof Float32Array && shouldStream) {
          const layout = this._resolveWeightLayout(loc, name);
          const dtype: WeightDtype = loc.dtype === 'F16' ? 'f16' : 'f32';
          this.lmHead = createCpuWeightBuffer(tensor, dtype, layout, loc.shape, name);
          log.warn('Loader', `LM head stored on CPU for chunked matmul (layout=${layout})`);
        } else if (tensor instanceof GPUBuffer && loc.shape && loc.shape.length === 2) {
          const layout = this._resolveWeightLayout(loc, name);
          const dtype: WeightDtype = loc.dtype === 'F16' ? 'f16' : 'f32';
          this.lmHead = createWeightBuffer(tensor, dtype, layout, loc.shape, name);
          log.info('Loader', `Wrapped lm_head as WeightBuffer (layout=${layout}, dtype=${dtype})`);
        } else {
          this.lmHead = tensor as GPUBuffer | WeightBuffer | Float32Array;
        }
        break;
      }
    }

    if (!this.lmHead && this.embeddings) {
      debugTrace.loader(' Using tied embeddings as LM head');
      this.lmHead = this.embeddings;
    } else if (!this.lmHead) {
      log.warn('Loader', ' LM head not found');
    }

    // Downcast LM head to f16
    const caps = getKernelCapabilities();
    if (caps.hasF16 && !this.keepF32Weights && this.lmHead && !isCpuWeightBuffer(this.lmHead)) {
      const tiedToEmbeddings =
        this.lmHead === this.embeddings ||
        (isWeightBuffer(this.lmHead) && isWeightBuffer(this.embeddings) && this.lmHead.buffer === this.embeddings.buffer) ||
        (this.lmHead instanceof GPUBuffer && isWeightBuffer(this.embeddings) && this.lmHead === this.embeddings.buffer);

      if (!tiedToEmbeddings) {
        const dtype = isWeightBuffer(this.lmHead)
          ? this.lmHead.dtype
          : (lmHeadLoc?.dtype === 'F16' ? 'f16' : 'f32');
        if (dtype === 'f32') {
          try {
            const buffer = isWeightBuffer(this.lmHead) ? this.lmHead.buffer : this.lmHead;
            if (!(buffer instanceof GPUBuffer)) {
              return;
            }
            const elems = buffer.size / 4;
            const inputTensor = createTensor(buffer, 'f32', [elems], 'lmHead_f32');
            const f16Tensor = await castF32ToF16(inputTensor);
            const layout = isWeightBuffer(this.lmHead)
              ? this.lmHead.layout
              : (lmHeadLoc ? this._resolveWeightLayout(lmHeadLoc, lmHeadName ?? 'lm_head') : 'row');
            const shape = isWeightBuffer(this.lmHead)
              ? Array.from(this.lmHead.shape)
              : (lmHeadLoc?.shape ?? [elems]);
            const newWeightBuffer = createWeightBuffer(f16Tensor.buffer, 'f16', layout, shape, lmHeadName ?? 'lm_head');
            releaseBuffer(buffer);
            this.lmHead = newWeightBuffer;
            this.gpuBuffers.add(f16Tensor.buffer);
          } catch (e) {
            log.warn('Loader', `Failed to downcast lmHead to f16: ${(e as Error).message}`);
          }
        }
      }
    }
  }

  /**
   * Get layer weights
   */
  getLayerWeights(layerIdx: number): LayerWeights | null {
    return this.layers.get(layerIdx) || null;
  }

  /**
   * Get model configuration
   */
  getConfig(): ModelConfig {
    return (this.manifest?.config as ModelConfig) || {};
  }

  /**
   * Check if using unified memory (can run dense models efficiently)
   */
  canRunDense(): boolean {
    return this.isUnifiedMemory;
  }

  /**
   * Get loading statistics
   */
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

  /**
   * Get expert cache statistics
   */
  getExpertCacheStats(): CacheStats | null {
    return this.expertCache?.getStats() || null;
  }

  /**
   * Unload model and free resources
   */
  async unload(): Promise<void> {
    debugTrace.loader(' Unloading model...');

    // Stop memory logging if still running
    if (this._memoryLogInterval) {
      clearInterval(this._memoryLogInterval);
      this._memoryLogInterval = null;
    }

    for (const buffer of this.gpuBuffers) {
      releaseBuffer(buffer);
    }
    this.gpuBuffers.clear();

    // Clear expert cache
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

// Global loader instance
let globalLoader: DopplerLoader | null = null;

/**
 * Get global DopplerLoader instance
 */
export function getDopplerLoader(loadingConfig?: LoadingConfigSchema): DopplerLoader {
  if (!globalLoader) {
    globalLoader = new DopplerLoader(loadingConfig);
  } else if (loadingConfig) {
    globalLoader.setLoadingConfig(loadingConfig);
  }
  return globalLoader;
}

/**
 * Create new DopplerLoader instance
 */
export function createDopplerLoader(loadingConfig?: LoadingConfigSchema): DopplerLoader {
  return new DopplerLoader(loadingConfig);
}

export default DopplerLoader;
