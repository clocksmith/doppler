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
  computeHash,
} from '../storage/shard-manager.js';
import {
  parseManifest,
  getShardInfo,
  getShardCount,
  isMoE,
  getShardsForExpert,
  getTensorsForExpert,
  getExpertBytes,
  type RDRRManifest,
  type ShardInfo,
  type HashAlgorithm,
} from '../storage/rdrr-format.js';
import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../gpu/buffer-pool.js';
import { dequantize, dequantizeQ6K, castF32ToF16, runBF16ToF16 } from '../gpu/kernel-selector.js';
import { getBufferDtype, setBufferDtype, setBufferLayout, type BufferLayout } from '../gpu/buffer-dtypes.js';
import { getKernelHints, getKernelHintsSource } from '../gpu/kernel-hints.js';
import { ExpertCache, getExpertCache, type CacheStats } from './expert-cache.js';
import { type KernelHints } from '../storage/rdrr-format.js';

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Tensor location in loaded model
 */
export interface TensorLocation {
  shardIndex: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
  spans?: Array<{ shardIndex: number; offset: number; size: number }>;
  /** Weight storage layout: 'column' means pre-transposed for faster matmul */
  layout?: 'row' | 'column';
  /** Original shape before transpose (if layout is 'column') */
  originalShape?: number[];
}

/**
 * Loaded layer weights
 */
export interface LayerWeights {
  inputNorm: GPUBuffer | Float32Array | null;
  qProj: GPUBuffer | Float32Array | null;
  kProj: GPUBuffer | Float32Array | null;
  vProj: GPUBuffer | Float32Array | null;
  oProj: GPUBuffer | Float32Array | null;
  qNorm: GPUBuffer | Float32Array | null;
  kNorm: GPUBuffer | Float32Array | null;
  postAttentionNorm: GPUBuffer | Float32Array | null;
  preFeedforwardNorm: GPUBuffer | Float32Array | null;
  postFeedforwardNorm: GPUBuffer | Float32Array | null;
  postNorm: GPUBuffer | Float32Array | null;
  postAttnNorm: GPUBuffer | Float32Array | null;
  ffnGate: GPUBuffer | Float32Array | null;
  ffnUp: GPUBuffer | Float32Array | null;
  ffnDown: GPUBuffer | Float32Array | null;
  /** Fused gate+up projection [intermediateSize*2, hiddenSize] for 2-pass FFN */
  ffnGateUp?: GPUBuffer | Float32Array | null;
  // Aliases for pipeline compatibility
  gate?: GPUBuffer | Float32Array | null;
  up?: GPUBuffer | Float32Array | null;
  down?: GPUBuffer | Float32Array | null;
  /** Fused gate+up for pipeline compatibility */
  gateUp?: GPUBuffer | Float32Array | null;
  routerWeight?: GPUBuffer | Float32Array | null;
  routerBias?: GPUBuffer | Float32Array | null;
  attentionSinks?: GPUBuffer | Float32Array | null;
}

/**
 * Expert weights
 */
export interface ExpertWeights {
  gate?: GPUBuffer | Float32Array | null;
  up?: GPUBuffer | Float32Array | null;
  down?: GPUBuffer | Float32Array | null;
  isGptOss?: boolean;
  expertIdx?: number;
  numExperts?: number;
  gateUpBlocks?: GPUBuffer | null;
  gateUpScales?: GPUBuffer | null;
  gateUpBias?: GPUBuffer | null;
  downBlocks?: GPUBuffer | null;
  downScales?: GPUBuffer | null;
  downBias?: GPUBuffer | null;
}

/**
 * Loading progress information
 */
export interface LoadProgress {
  stage: 'manifest' | 'shards' | 'layers' | 'gpu_transfer' | 'complete';
  progress: number;
  /** Current layer index */
  layer?: number;
  /** Total layers */
  total?: number;
  /** Current shard index */
  shard?: number;
  /** Total shards */
  totalShards?: number;
  /** Bytes loaded so far */
  bytesLoaded?: number;
  /** Total bytes to load */
  totalBytes?: number;
  /** Loading speed in bytes per second */
  bytesPerSecond?: number;
  /** Human-readable message */
  message?: string;
}

/**
 * Loading options
 */
export interface LoadOptions {
  onProgress?: (progress: LoadProgress) => void;
  verifyHashes?: boolean;
}

/**
 * Custom shard loader options
 */
export interface CustomShardLoaderOptions {
  verify?: boolean;
}

/**
 * Custom shard loader function
 */
export type CustomShardLoader = (shardIndex: number) => Promise<Uint8Array>;

/**
 * Loader statistics
 */
export interface LoaderStats {
  modelId: string | null;
  isLoaded: boolean;
  isMoE: boolean;
  isUnifiedMemory: boolean;
  layersLoaded: number;
  expertsLoaded: number;
  gpuBuffers: number;
}

/**
 * GPU kernel capabilities
 */
interface KernelCapabilities {
  hasF16: boolean;
  hasSubgroups: boolean;
}

/**
 * Model config (flexible structure from manifest)
 */
interface ModelConfig {
  num_hidden_layers?: number;
  blockCount?: number;
  text_config?: { num_hidden_layers?: number };
  n_layer?: number;
  num_local_experts?: number;
  num_experts?: number;
  architectures?: string[];
  model_type?: string;
  [key: string]: unknown;
}

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
  embeddings: GPUBuffer | Float32Array | null = null;
  layers = new Map<number, LayerWeights>();
  experts = new Map<string, ExpertWeights>();
  lmHead: GPUBuffer | Float32Array | null = null;
  finalNorm: GPUBuffer | Float32Array | null = null;

  // Memory management
  heapManager: HeapManager | null = null;
  gpuBuffers = new Set<GPUBuffer>();

  // Expert cache for MoE models (LRU eviction)
  expertCache: ExpertCache | null = null;

  // Loading state
  loadedShards = new Set<number>();
  tensorLocations = new Map<string, TensorLocation>();

  // Shard cache (small LRU) to avoid repeated OPFS reads during load
  // Default is 2 for dense models, dynamically increased for MoE
  shardCache = new Map<number, ArrayBuffer>();
  maxShardCacheEntries = 2;  // Updated dynamically in _configureShardCache()

  // Custom shard loader (for Native Bridge support)
  customLoadShard: CustomShardLoader | null = null;
  verifyCustomShards = true;

  // Fused Q4_K matmul: skip dequantization for matmul weights, use fused kernel
  // Default is false; opt-in via manifest kernel hints.
  useFusedQ4K = false;

  // Internal tracking
  private _normOffsetLogged = false;

  constructor() {
    // All properties initialized above
  }

  /**
   * Set custom shard loader (e.g., for Native Bridge)
   */
  setCustomShardLoader(loadShardFn: CustomShardLoader, options: CustomShardLoaderOptions = {}): void {
    this.customLoadShard = loadShardFn;
    this.verifyCustomShards = options.verify !== false;
    console.log('[DopplerLoader] Custom shard loader configured');
  }

  /**
   * Load shard using custom loader or OPFS
   */
  private async _loadShard(shardIndex: number): Promise<ArrayBuffer> {
    if (this.shardCache.has(shardIndex)) {
      const cached = this.shardCache.get(shardIndex)!;
      // Refresh LRU order
      this.shardCache.delete(shardIndex);
      this.shardCache.set(shardIndex, cached);
      return cached;
    }

    if (this.customLoadShard) {
      let data: Uint8Array | ArrayBuffer = await this.customLoadShard(shardIndex);

      // Verify hash if enabled
      if (this.verifyCustomShards && this.manifest) {
        const shardInfo = this.manifest.shards?.[shardIndex];
        const expectedHash = shardInfo?.hash || shardInfo?.blake3;
        if (expectedHash) {
          const algorithm = this.manifest.hashAlgorithm || 'blake3';
          const computedHash = await computeHash(data, algorithm);
          if (computedHash !== expectedHash) {
            throw new Error(
              `Shard ${shardIndex} hash mismatch. Expected: ${expectedHash}, got: ${computedHash}`
            );
          }
        }
      }

      // Normalize to ArrayBuffer for downstream slicing
      let arrayBuffer: ArrayBuffer;
      if (data instanceof Uint8Array) {
        arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
      } else {
        arrayBuffer = data;
      }

      this.shardCache.set(shardIndex, arrayBuffer);
      if (this.shardCache.size > this.maxShardCacheEntries) {
        const oldestKey = this.shardCache.keys().next().value;
        if (oldestKey !== undefined) {
          this.shardCache.delete(oldestKey);
        }
      }

      return arrayBuffer;
    }

    const data = await loadShardFromOPFS(shardIndex);
    this.shardCache.set(shardIndex, data);
    if (this.shardCache.size > this.maxShardCacheEntries) {
      const oldestKey = this.shardCache.keys().next().value;
      if (oldestKey !== undefined) {
        this.shardCache.delete(oldestKey);
      }
    }
    return data;
  }

  /**
   * Initialize loader and detect capabilities
   */
  async init(): Promise<void> {
    console.log('[DopplerLoader] Initializing...');

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

    console.log('[DopplerLoader] Initialized:', {
      memory64: this.memoryCapabilities.hasMemory64,
      unified: this.isUnifiedMemory,
      f16: this.gpuCapabilities.hasF16,
      subgroups: this.gpuCapabilities.hasSubgroups,
    });
  }

  /**
   * Set manifest directly (for bridge/external loading)
   */
  setManifest(manifest: RDRRManifest): void {
    this.manifest = manifest;
    const config = manifest.config as ModelConfig | undefined;
    this.isMoE = manifest.moeConfig != null || (config?.num_local_experts ?? 0) > 1;
    this._configureShardCache();
    this._configureQ4KStrategy();
    console.log('[DopplerLoader] Manifest set externally');
  }

  private _configureQ4KStrategy(): void {
    const runtimeHints = getKernelHints();
    const manifestHints = this.manifest?.optimizations?.kernelHints as KernelHints | undefined;
    const hints = (runtimeHints ?? manifestHints) as KernelHints | undefined;
    const hintSource = getKernelHintsSource() ?? (manifestHints ? 'manifest' : 'unset');
    const q4kHint = hints?.q4kMatmul;
    const q4kLayout = (this.manifest?.config as { q4kLayout?: string } | undefined)?.q4kLayout;

    let useFused = q4kHint === 'fused_q4k';
    if (typeof window !== 'undefined' && (window as any).DOPPLER_DISABLE_FUSED_Q4K) {
      useFused = false;
    }
    if (q4kLayout === 'column_wise') {
      useFused = false;
    }

    this.useFusedQ4K = useFused;

    if (q4kHint || q4kLayout) {
      console.log('[DopplerLoader] Q4K strategy:', {
        fused: this.useFusedQ4K,
        q4kMatmul: q4kHint ?? 'unset',
        q4kLayout: q4kLayout ?? 'unset',
        hintSource,
      });
    }
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
    const preservedManifest = this.customLoadShard ? this.manifest : null;

    if (hasExistingModelState) {
      await this.unload();
    }

    // Restore preserved manifest after unload
    if (preservedManifest) {
      this.manifest = preservedManifest;
    }

    console.log(`[DopplerLoader] Loading model: ${modelId}`);
    this.modelId = modelId;

    // If using custom shard loader (bridge), manifest should be set externally
    if (!this.customLoadShard) {
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
    this._configureShardCache();

    // Enforce dense/MoE gating based on hardware
    if (!this.isMoE && !this.isUnifiedMemory) {
      console.warn(
        '[DopplerLoader] Dense model on discrete GPU - performance will be severely limited. ' +
        'Consider using an MoE model for better performance.'
      );
    }

    // Verify integrity if requested (only for OPFS path)
    if (verifyHashes && !this.customLoadShard) {
      const integrity = await verifyIntegrity();
      if (!integrity.valid) {
        throw new Error(
          `Model integrity check failed. ` +
          `Missing shards: ${integrity.missingShards.length}, ` +
          `Corrupt shards: ${integrity.corruptShards.length}`
        );
      }
    }

    // Build tensor location map from manifest
    this._buildTensorLocations();

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

    // Track shard loading with progress
    const originalLoadShard = this._loadShard.bind(this);
    this._loadShard = async (shardIndex: number): Promise<ArrayBuffer> => {
      const shardInfo = this.manifest?.shards?.[shardIndex];
      const shardSize = shardInfo?.size || 0;
      const data = await originalLoadShard(shardIndex);
      bytesLoaded += shardSize;
      shardsLoaded++;
      reportProgress('shards', 0.1 + (bytesLoaded / totalBytes) * 0.7);
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
    console.log(`[DopplerLoader] Loading ${numLayers} layers`);

    for (let l = 0; l < numLayers; l++) {
      await this._loadLayer(l, onProgress);

      if (onProgress) {
        const layerProgress = 0.1 + (bytesLoaded / totalBytes) * 0.7;
        const speed = getSpeed();
        onProgress({
          stage: 'layers',
          layer: l + 1,
          total: numLayers,
          progress: layerProgress,
          shard: shardsLoaded,
          totalShards,
          bytesLoaded,
          totalBytes,
          bytesPerSecond: speed,
          message: `Layer ${l + 1}/${numLayers} • ${formatBytes(bytesLoaded)} / ${formatBytes(totalBytes)} • ${formatBytes(speed)}/s`,
        });
      }
    }

    // Load final norm and LM head
    reportProgress('gpu_transfer', 0.85, 'Loading final weights...');
    await this._loadFinalWeights(onProgress);

    if (onProgress) {
      onProgress({ stage: 'complete', progress: 1.0 });
    }

    this.isLoaded = true;
    console.log(`[DopplerLoader] Model loaded: ${modelId}`);

    return (this.manifest.config as ModelConfig) || {};
  }

  /**
   * Build tensor location map from manifest
   */
  private _buildTensorLocations(): void {
    this.tensorLocations.clear();

    if (!this.manifest?.tensors) {
      console.warn('[DopplerLoader] No tensor locations in manifest');
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
    console.log(`[DopplerLoader] Tensor map: ${this.tensorLocations.size} tensors`);
  }

  /**
   * Apply layout metadata to a GPU buffer if the tensor has column-major storage.
   * This enables matmul to use transposeB=false for faster access.
   */
  private _applyBufferLayout(buffer: GPUBuffer, location: TensorLocation): GPUBuffer {
    if (location.layout === 'column') {
      setBufferLayout(buffer, 'column');
    }
    return buffer;
  }

  /**
   * Load a tensor by name
   */
  private async _loadTensor(
    name: string,
    toGPU = true,
    silent = false
  ): Promise<GPUBuffer | Float32Array | Uint8Array | null> {
    const location = this.tensorLocations.get(name);
    if (!location) {
      const altName = this._findAlternativeTensorName(name);
      if (altName) {
        return this._loadTensor(altName, toGPU, silent);
      }
      if (!silent) {
        console.warn(`[DopplerLoader] Tensor not found: ${name}`);
      }
      return null;
    }

    // Debug: Log tensor loading details for attention weights
    if (name.includes('attn_k') || name.includes('k_proj')) {
      console.log(`[DopplerLoader] Loading ${name}: shape=${JSON.stringify(location.shape)}, size=${location.size}, dtype=${location.dtype}, spans=${!!location.spans}`);
    }

    // Fast path for multi-shard tensors when uploading to GPU
    if (location.spans && toGPU) {
      console.log(`[DopplerLoader] Loading tensor "${name}" via spans path (${location.spans.length} spans, dtype=${location.dtype})`);
      const device = getDevice();
      if (!device) {
        console.warn('[DopplerLoader] GPU device not available; falling back to CPU assembly');
      } else {
        // Quantized tensors
        if (location.dtype === 'Q4_K_M' || location.dtype === 'Q4_K') {
          const caps = this.gpuCapabilities || getKernelCapabilities();
          const isMatmulWeight = this._shouldDequantizeToF16(name);
          const Q4K_K = 256;
          const Q4K_BLOCK_BYTES = 144;

          const isPackedQ4K =
            Array.isArray(location.shape) &&
            location.shape.length === 2 &&
            (() => {
              const [rows, cols] = location.shape;
              const expectedRowwise = rows * Math.ceil(cols / Q4K_K) * Q4K_BLOCK_BYTES;
              return location.size < expectedRowwise;
            })();

          // Fused Q4K path: keep raw quantized buffer for matmul weights
          // EXCLUDE embeddings - gather kernel doesn't support Q4K, only matmul does
          // Note: GGUF uses 'token_embd' (without second 'e'), HF uses 'embed_tokens'
          const isEmbedding = name.toLowerCase().includes('embd') ||
                              name.toLowerCase().includes('embed') ||
                              name.toLowerCase().includes('wte');
          if (this.useFusedQ4K && isMatmulWeight && !isEmbedding && caps?.hasSubgroups && !isPackedQ4K) {
            console.log(`[DopplerLoader] Loading Q4K weight: ${name} (size=${location.size})`);
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
            setBufferDtype(q4kBuffer, 'q4k');
            this.gpuBuffers.add(q4kBuffer);
            return q4kBuffer;
          }

          // Standard dequant path
          if (this.useFusedQ4K && isMatmulWeight && caps?.hasSubgroups && isPackedQ4K) {
            const [rows, cols] = location.shape;
            const expectedRowwise = rows * Math.ceil(cols / Q4K_K) * Q4K_BLOCK_BYTES;
            console.warn(
              `[DopplerLoader] Packed Q4K matmul weight (incompatible with fused matmul): ${name} ` +
              `shape=[${rows},${cols}] size=${location.size} expectedRowwise=${expectedRowwise}. ` +
              `Falling back to dequantized weights for correctness.`
            );
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
            const q4kHint = (this.manifest?.optimizations?.kernelHints as KernelHints | undefined)?.q4kMatmul;
            if (q4kHint === 'dequant_f32') {
              outputDtype = 'f32';
            } else if (q4kHint === 'dequant_f16' || !q4kHint) {
              outputDtype = caps?.hasF16 ? 'f16' : 'f32';
            } else {
              outputDtype = caps?.hasF16 ? 'f16' : 'f32';
            }
          }
          const dequantized = await dequantize(quantBuffer, numBlocks, { outputDtype });

          releaseBuffer(quantBuffer);
          this.gpuBuffers.add(dequantized);
          // GGUF stores ALL weights as [ne0, ne1] where ne1 rows of ne0 elements.
          // This is the TRANSPOSE of the weight matrix (e.g., [kv_dim, hidden] instead of [hidden, kv_dim]).
          // The shader with transposeB=true handles this: it reads B[n*K + k] and transposes to get B^T.
          // So we do NOT mark GGUF weights as column-major - let them use the default transposeB=true.
          return dequantized;
        }

        // Q6_K tensors (6-bit quantization)
        if (location.dtype === 'Q6_K') {
          console.log(`[DopplerLoader] Loading Q6_K tensor "${name}" via spans path (${location.spans.length} spans)`);
          const Q6K_BLOCK_BYTES = 210;
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
          console.log(`[DopplerLoader] Dequantizing Q6_K ${name}: size=${location.size}, numBlocks=${numBlocks}, expectedOutput=${numBlocks * 256 * 2} (f16)`);
          const dequantized = await dequantizeQ6K(quantBuffer, numBlocks, { outputDtype: 'f16' });
          console.log(`[DopplerLoader] Dequantized Q6_K ${name}: resultSize=${dequantized.size}`);

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
            console.log(`[DopplerLoader] DEBUG embed dequant first16: [${f32Samples.map(x => x.toFixed(4)).join(', ')}]`);
          }

          releaseBuffer(quantBuffer);
          this.gpuBuffers.add(dequantized);
          // GGUF stores ALL weights transposed - use default transposeB=true (no column layout)
          return dequantized;
        }

        // BF16 tensors
        if (location.dtype === 'BF16') {
          console.log(`[DopplerLoader] Loading BF16 tensor "${name}" with ${location.spans.length} spans, total size=${location.size}`);
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
            console.log(`[DopplerLoader] Wrote span ${span.shardIndex}: offset=${tensorOffset}, first4bytes=[${bytes[0]}, ${bytes[1]}, ${bytes[2]}, ${bytes[3]}]`);
          }

          const numElements = location.size / 2;
          const caps = this.gpuCapabilities || getKernelCapabilities();
          const isMatmulWeight = this._shouldDequantizeToF16(name);

          try {
            // Ensure all writeBuffer operations are flushed to GPU before conversion
            await device.queue.onSubmittedWorkDone();

            // For matmul weights with F16 support: BF16 → F16 (no intermediate F32 buffer)
            // This is critical for lm_head (262k vocab) which dominates decode time
            if (caps?.hasF16 && isMatmulWeight) {
              const f16Buffer = await runBF16ToF16(srcBuffer, numElements, name);
              releaseBuffer(srcBuffer);
              this.gpuBuffers.add(f16Buffer);
              console.log(`[DopplerLoader] BF16→F16 for matmul weight: ${name} (${numElements} elements, spans path)`);
              return this._applyBufferLayout(f16Buffer, location);
            }

            // Standard path: BF16 → F32
            console.log(`[DopplerLoader] Converting BF16→F32: ${numElements} elements`);
            const dstBuffer = await this._convertBF16ToF32GPU(srcBuffer, numElements, name);
            releaseBuffer(srcBuffer);
            if (dstBuffer instanceof GPUBuffer) {
              this.gpuBuffers.add(dstBuffer);
              return this._applyBufferLayout(dstBuffer, location);
            }
            return dstBuffer;
          } catch (err) {
            console.error(`[DopplerLoader] BF16 conversion failed:`, err);
            throw err;
          }
        }

        // Other dtypes
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
        return this._applyBufferLayout(buffer, location);
      }
    }

    // Load shard data into CPU memory (single-shard or CPU path)
    let shardData: ArrayBuffer;
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
      shardData = combined.buffer;
    } else {
      const fullShard = await this._loadShard(location.shardIndex);
      shardData = fullShard.slice(location.offset, location.offset + location.size);
    }

    // Handle quantized data
    if (location.dtype === 'Q4_K_M' || location.dtype === 'Q4_K') {
      if (toGPU) {
        const device = getDevice();
        const caps = this.gpuCapabilities || getKernelCapabilities();
        const isMatmulWeight = this._shouldDequantizeToF16(name);
        const Q4K_K = 256;
        const Q4K_BLOCK_BYTES = 144;

        const isPackedQ4K =
          Array.isArray(location.shape) &&
          location.shape.length === 2 &&
          (() => {
            const [rows, cols] = location.shape;
            const expectedRowwise = rows * Math.ceil(cols / Q4K_K) * Q4K_BLOCK_BYTES;
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
          console.log(`[DopplerLoader] Loading Q4K weight (single-shard): ${name} (size=${location.size})`);
          const q4kBuffer = acquireBuffer(location.size, undefined, `q4k_${name}`);
          device!.queue.writeBuffer(q4kBuffer, 0, new Uint8Array(shardData));
          setBufferDtype(q4kBuffer, 'q4k');
          this.gpuBuffers.add(q4kBuffer);
          return q4kBuffer;
        }

        if (this.useFusedQ4K && isMatmulWeight && caps?.hasSubgroups && isPackedQ4K) {
          const [rows, cols] = location.shape;
          const expectedRowwise = rows * Math.ceil(cols / Q4K_K) * Q4K_BLOCK_BYTES;
          console.warn(
            `[DopplerLoader] Packed Q4K matmul weight (incompatible with fused matmul): ${name} ` +
            `shape=[${rows},${cols}] size=${location.size} expectedRowwise=${expectedRowwise}. ` +
            `Falling back to dequantized weights for correctness.`
          );
        }

        // Standard dequant path: dequantize to f16 or f32
        const quantBuffer = acquireBuffer(location.size, undefined, `quant_${name}`);
        device!.queue.writeBuffer(quantBuffer, 0, new Uint8Array(shardData));

        const numBlocks = Math.ceil(location.size / 144);
        let outputDtype: 'f16' | 'f32' = 'f32';
        if (isMatmulWeight) {
          const q4kHint = (this.manifest?.optimizations?.kernelHints as KernelHints | undefined)?.q4kMatmul;
          if (q4kHint === 'dequant_f32') {
            outputDtype = 'f32';
          } else if (q4kHint === 'dequant_f16' || !q4kHint) {
            outputDtype = caps?.hasF16 ? 'f16' : 'f32';
          } else {
            outputDtype = caps?.hasF16 ? 'f16' : 'f32';
          }
        }
        console.log(`[DopplerLoader] Dequantizing ${name}: size=${location.size}, numBlocks=${numBlocks}, outputDtype=${outputDtype}, expectedOutput=${numBlocks * 256 * (outputDtype === 'f16' ? 2 : 4)}`);
        const dequantized = await dequantize(quantBuffer, numBlocks, { outputDtype });
        console.log(`[DopplerLoader] Dequantized ${name}: resultSize=${dequantized.size}`);

        releaseBuffer(quantBuffer);
        this.gpuBuffers.add(dequantized);
        // GGUF stores ALL weights transposed - use default transposeB=true (no column layout)
        return dequantized;
      }
      return new Uint8Array(shardData);
    }

    // Handle Q6_K data (single-shard path)
    if (location.dtype === 'Q6_K') {
      if (toGPU) {
        const device = getDevice();
        console.log(`[DopplerLoader] Loading Q6_K tensor "${name}" (single-shard), size=${location.size}`);
        const Q6K_BLOCK_BYTES = 210;
        const quantBuffer = acquireBuffer(location.size, undefined, `quant_${name}`);
        device!.queue.writeBuffer(quantBuffer, 0, new Uint8Array(shardData));

        const numBlocks = Math.floor(location.size / Q6K_BLOCK_BYTES);
        console.log(`[DopplerLoader] Dequantizing Q6_K ${name}: size=${location.size}, numBlocks=${numBlocks}, expectedOutput=${numBlocks * 256 * 2} (f16)`);
        const dequantized = await dequantizeQ6K(quantBuffer, numBlocks, { outputDtype: 'f16' });
        console.log(`[DopplerLoader] Dequantized Q6_K ${name}: resultSize=${dequantized.size}`);

        releaseBuffer(quantBuffer);
        this.gpuBuffers.add(dequantized);
        // GGUF stores ALL weights transposed - use default transposeB=true (no column layout)
        return dequantized;
      }
      return new Uint8Array(shardData);
    }

    // Handle BF16 data
    if (location.dtype === 'BF16') {
      if (toGPU) {
        const device = getDevice();
        const bf16Bytes = new Uint8Array(shardData);
        const srcBuffer = acquireBuffer(bf16Bytes.byteLength, undefined, `${name}_bf16`);
        device!.queue.writeBuffer(srcBuffer, 0, bf16Bytes);

        const numElements = bf16Bytes.byteLength / 2;
        const caps = this.gpuCapabilities || getKernelCapabilities();
        const isMatmulWeight = this._shouldDequantizeToF16(name);

        // For matmul weights with F16 support: BF16 → F16 (no intermediate F32 buffer)
        // This is critical for lm_head (262k vocab) which dominates decode time
        if (caps?.hasF16 && isMatmulWeight) {
          const f16Buffer = await runBF16ToF16(srcBuffer, numElements, name);
          releaseBuffer(srcBuffer);
          this.gpuBuffers.add(f16Buffer);
          console.log(`[DopplerLoader] BF16→F16 for matmul weight: ${name} (${numElements} elements)`);
          return this._applyBufferLayout(f16Buffer, location);
        }

        // Standard path: BF16 → F32
        const dstBuffer = await this._convertBF16ToF32GPU(srcBuffer, numElements, name);
        releaseBuffer(srcBuffer);
        if (dstBuffer instanceof GPUBuffer) {
          this.gpuBuffers.add(dstBuffer);
          return this._applyBufferLayout(dstBuffer, location);
        }
        return dstBuffer;
      }

      // CPU path
      const bf16 = new Uint16Array(shardData);
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
      device!.queue.writeBuffer(buffer, 0, new Uint8Array(shardData));
      this.gpuBuffers.add(buffer);
      return this._applyBufferLayout(buffer, location);
    } else {
      if (location.dtype === 'F16') {
        const f16 = new Uint16Array(shardData);
        const f32 = new Float32Array(f16.length);
        for (let i = 0; i < f16.length; i++) {
          f32[i] = this._f16ToF32(f16[i]);
        }
        return f32;
      }
      return new Float32Array(shardData);
    }
  }

  /**
   * Find alternative tensor name (handles different naming conventions)
   */
  private _findAlternativeTensorName(name: string): string | null {
    const prefixes = [
      'language_model.model.',
      'language_model.',
      'model.',
      '',
    ];

    for (const prefix of prefixes) {
      const prefixedName = prefix + name;
      if (prefixedName !== name && this.tensorLocations.has(prefixedName)) {
        return prefixedName;
      }
    }

    const patterns: [RegExp, string][] = [
      [/^layers\.(\d+)\./, 'model.layers.$1.'],
      [/^model\.layers\.(\d+)\./, 'layers.$1.'],
      [/\.weight$/, ''],
      [/$/, '.weight'],
      [/attention/, 'self_attn'],
      [/self_attn/, 'attention'],
      [/ffn/, 'mlp'],
      [/mlp/, 'ffn'],
    ];

    for (const [pattern, replacement] of patterns) {
      const altName = name.replace(pattern, replacement);
      if (altName !== name && this.tensorLocations.has(altName)) {
        return altName;
      }
      for (const prefix of prefixes) {
        const prefixedAlt = prefix + altName;
        if (this.tensorLocations.has(prefixedAlt)) {
          return prefixedAlt;
        }
      }
    }
    return null;
  }

  /**
   * Configure shard cache size based on model type.
   * For MoE models, cache enough shards for 2x top-k experts + 1 dense shard.
   * For dense models, keep the default (2 shards).
   */
  private _configureShardCache(): void {
    if (!this.manifest) return;

    const moe = this.manifest.moeConfig;
    if (moe && moe.numExpertsPerToken > 0) {
      // For MoE: cache 2x top-k experts (for current + next layer prefetch) + 1 dense shard
      const expertCacheSize = (moe.numExpertsPerToken * 2) + 1;
      // Cap at reasonable maximum (16 shards = ~1GB at 64MB/shard)
      this.maxShardCacheEntries = Math.min(16, Math.max(4, expertCacheSize));
      console.log(`[DopplerLoader] MoE shard cache: ${this.maxShardCacheEntries} entries (${moe.numExpertsPerToken} experts/token)`);
    } else {
      // Dense model - keep default
      this.maxShardCacheEntries = 2;
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
      console.log('[DopplerLoader] _needsNormWeightOffset: no manifest');
      return false;
    }

    const config = (this.manifest.config || {}) as ModelConfig;
    const arch = config.architectures?.[0] || (this.manifest.architecture as string) || '';
    const isGemma3 = /gemma3/i.test(arch) || /gemma3/i.test(config.model_type || '');

    // Check explicit manifest flag first, then fall back to architecture detection
    const explicitFlag = (config as { rms_norm_weight_offset?: boolean }).rms_norm_weight_offset;
    const needsOffset = explicitFlag ?? isGemma3;

    if (needsOffset && !this._normOffsetLogged) {
      this._normOffsetLogged = true;
      console.log('[DopplerLoader] Applying +1 norm weight offset for Gemma 3 layer norms');
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
    actualNumElements?: number
  ): Promise<GPUBuffer | Float32Array> {
    const device = getDevice();
    if (!device) {
      console.warn('[DopplerLoader] No GPU device for norm offset');
      return tensor;
    }

    if (tensor instanceof GPUBuffer) {
      // Use actual element count if provided, otherwise fall back to buffer size
      // Buffer pool rounds up to power of 2, so tensor.size may include garbage padding
      const numElements = actualNumElements ?? Math.floor(tensor.size / 4);
      const dataSize = numElements * 4;

      const stagingBuffer = device.createBuffer({
        size: dataSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(tensor, 0, stagingBuffer, 0, dataSize);
      device.queue.submit([encoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
      stagingBuffer.unmap();
      stagingBuffer.destroy();

      const offsetData = new Float32Array(numElements);
      for (let i = 0; i < numElements; i++) {
        offsetData[i] = 1.0 + data[i];
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

    console.warn('[DopplerLoader] Unknown tensor type for norm offset');
    return tensor;
  }

  /**
   * Convert F16 to F32
   */
  private _f16ToF32(h: number): number {
    const sign = (h >> 15) & 0x1;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;

    if (exp === 0) {
      if (mant === 0) return sign ? -0 : 0;
      const f = mant / 1024 * Math.pow(2, -14);
      return sign ? -f : f;
    }
    if (exp === 31) {
      return mant ? NaN : (sign ? -Infinity : Infinity);
    }

    const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
    return sign ? -f : f;
  }

  /**
   * Convert BF16 buffer to F32 on GPU
   */
  private async _convertBF16ToF32GPU(
    srcBuffer: GPUBuffer,
    numElements: number,
    name: string
  ): Promise<GPUBuffer> {
    console.log(`[_convertBF16ToF32GPU] Importing cast.js...`);
    const castModule = await import('../gpu/kernels/cast.js');
    console.log(`[_convertBF16ToF32GPU] castModule keys:`, Object.keys(castModule));
    const { runBF16ToF32 } = castModule;
    console.log(`[_convertBF16ToF32GPU] runBF16ToF32 type: ${typeof runBF16ToF32}`);
    const result = await runBF16ToF32(srcBuffer, numElements, name);
    console.log(`[_convertBF16ToF32GPU] runBF16ToF32 returned, result.size=${result?.size}`);

    // Debug: Verify conversion produced non-zero values
    if (name.includes('embed') && name.includes('embed_tokens')) {
      try {
        console.log(`[_convertBF16ToF32GPU] Checking embed buffer for non-zeros...`);
        const device = getDevice();
        const sampleSize = Math.min(1024, result.size);
        console.log(`[_convertBF16ToF32GPU] Creating staging buffer size=${sampleSize}`);
        const stagingBuffer = device!.createBuffer({
          size: sampleSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        console.log(`[_convertBF16ToF32GPU] Copying to staging buffer...`);
        const encoder = device!.createCommandEncoder();
        encoder.copyBufferToBuffer(result, 0, stagingBuffer, 0, sampleSize);
        device!.queue.submit([encoder.finish()]);
        console.log(`[_convertBF16ToF32GPU] Mapping staging buffer...`);
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        console.log(`[_convertBF16ToF32GPU] Reading data...`);
        const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        const nonZero = Array.from(data).filter(x => x !== 0);
        const nanCount = data.filter(x => !Number.isFinite(x)).length;
        console.log(`[BF16→F32 CHECK] nonZero=${nonZero.length}/${data.length}, nan=${nanCount}, sample=[${nonZero.slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
      } catch (err) {
        console.error(`[_convertBF16ToF32GPU] ERROR checking embed buffer:`, (err as Error).message);
      }
    }

    return result;
  }

  /**
   * Decide whether a quantized tensor should be dequantized directly to f16
   */
  private _shouldDequantizeToF16(name: string): boolean {
    const lower = name.toLowerCase();
    const matmulSuffixes = [
      // HuggingFace/SafeTensors naming
      'q_proj.weight',
      'k_proj.weight',
      'v_proj.weight',
      'o_proj.weight',
      'attention.wq.weight',
      'attention.wk.weight',
      'attention.wv.weight',
      'attention.wo.weight',
      'gate_proj.weight',
      'up_proj.weight',
      'down_proj.weight',
      'w1.weight',
      'w2.weight',
      'w3.weight',
      'lm_head.weight',
      'output.weight',
      // Weight-tied embedding/lm_head (Gemma, Llama, etc.)
      'embed_tokens.weight',
      'wte.weight',
      'token_embd.weight',
      // GGUF naming (blk.X.attn_q.weight, blk.X.ffn_gate.weight, etc.)
      'attn_q.weight',
      'attn_k.weight',
      'attn_v.weight',
      'attn_output.weight',
      'ffn_gate.weight',
      'ffn_up.weight',
      'ffn_down.weight',
      'ffn_gate_up.weight',
    ];

    return matmulSuffixes.some(suffix => lower.endsWith(suffix));
  }

  /**
   * Check if a weight is an embedding weight (needs column layout for LM head matmul).
   * GGUF stores all weights as [N,K] (transposed). For layer weights, we need transposeB=true
   * to compute A@W = A@W.T^T. But for embeddings used as LM head, we need transposeB=false
   * to compute hidden@E.T directly (the embedding IS already transposed in GGUF).
   */
  private _isEmbeddingWeight(name: string): boolean {
    const lower = name.toLowerCase();
    // Only match actual embedding and lm_head weights
    // Be careful NOT to match 'attn_output.weight' which ends with 'output.weight'
    const embeddingPatterns = [
      'embed_tokens.weight',
      'wte.weight',
      'token_embd.weight',
      'lm_head.weight',
    ];
    // Check exact suffix matches for embedding patterns
    if (embeddingPatterns.some(suffix => lower.endsWith(suffix))) {
      return true;
    }
    // For 'output.weight', only match if it's the top-level (no 'attn_' prefix)
    // This handles models that use 'output.weight' as the LM head name
    if (lower.endsWith('output.weight') && !lower.includes('attn_')) {
      return true;
    }
    return false;
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

    for (const name of embeddingNames) {
      const tensor = await this._loadTensor(name, true, true);
      if (tensor && (tensor instanceof GPUBuffer || tensor instanceof Float32Array)) {
        this.embeddings = tensor;
        break;
      }
    }

    if (!this.embeddings) {
      console.warn('[DopplerLoader] Embeddings not found');
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

    const tryLoad = async (suffixes: string[]): Promise<GPUBuffer | Float32Array | null> => {
      for (const prefix of prefixes) {
        for (const suffix of suffixes) {
          const tensor = await this._loadTensor(`${prefix}.${suffix}`, true, true);
          if (tensor && (tensor instanceof GPUBuffer || tensor instanceof Float32Array)) {
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

      if (this._needsNormWeightOffset()) {
        return this._applyNormWeightOffset(tensor, actualNumElements);
      }
      return tensor;
    };

    // Load attention weights in parallel
    const [inputNorm, qProj, kProj, vProj, oProj, qNorm, kNorm, postAttentionNorm, preFeedforwardNorm, postFeedforwardNorm] = await Promise.all([
      tryLoadNorm(['input_layernorm.weight', 'attn_norm.weight']),
      tryLoad(['self_attn.q_proj.weight', 'attention.wq.weight', 'attn_q.weight']),
      tryLoad(['self_attn.k_proj.weight', 'attention.wk.weight', 'attn_k.weight']),
      tryLoad(['self_attn.v_proj.weight', 'attention.wv.weight', 'attn_v.weight']),
      tryLoad(['self_attn.o_proj.weight', 'attention.wo.weight', 'attn_output.weight']),
      // Gemma 3: q_norm and k_norm use STANDARD RMSNorm (no +1 offset)
      // Only layer norms (input, post_attention, pre/post_feedforward) use the (1+weight) formula
      tryLoad(['self_attn.q_norm.weight', 'attn_q_norm.weight']),
      tryLoad(['self_attn.k_norm.weight', 'attn_k_norm.weight']),
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
        console.log(`[DopplerLoader] Layer ${layerIdx}: Using fused gate_up_proj for 2-pass FFN`);
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
      weights.routerWeight = routerWeight;
      weights.routerBias = routerBias;
    }

    weights.attentionSinks = await tryLoad(['self_attn.sinks']);

    this.layers.set(layerIdx, weights);

    // Downcast matmul weights to f16 when supported
    const caps = getKernelCapabilities();
    if (caps.hasF16) {
      const { isColumnMajorBuffer } = await import('../gpu/buffer-dtypes.js');
      const matmulKeys: (keyof LayerWeights)[] = ['qProj', 'kProj', 'vProj', 'oProj', 'ffnGate', 'ffnUp', 'ffnDown', 'ffnGateUp'];
      for (const key of matmulKeys) {
        const buf = weights[key];
        if (buf instanceof GPUBuffer) {
          const dtype = getBufferDtype(buf) || 'f32';
          if (dtype === 'f32') {
            const elems = buf.size / 4;
            // Preserve column-major layout through the f32→f16 downcast
            const wasColumnMajor = isColumnMajorBuffer(buf);
            console.log(`[DopplerLoader] Layer ${layerIdx} downcasting ${key}: bufSize=${buf.size}, elems=${elems}, expectedF16=${elems * 2}, columnMajor=${wasColumnMajor}`);
            try {
              const f16buf = await castF32ToF16(buf, elems);
              // Copy layout metadata to the new buffer
              if (wasColumnMajor) {
                setBufferLayout(f16buf, 'column');
              }
              console.log(`[DopplerLoader] Layer ${layerIdx} ${key} downcast result: f16Size=${f16buf.size}`);
              releaseBuffer(buf);
              (weights as unknown as Record<string, GPUBuffer | Float32Array | null>)[key] = f16buf;
              this.gpuBuffers.add(f16buf);
            } catch (e) {
              console.warn(`[DopplerLoader] Failed to downcast ${key} to f16:`, (e as Error).message);
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
      console.warn('[DopplerLoader] Expert prefetch error:', e);
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

    console.log(`[DopplerLoader] Loading expert ${expertIdx} for layer ${layerIdx}`);

    // Pre-load only the shards containing this expert's tensors
    await this._preloadShardsForExpert(layerIdx, expertIdx);

    // Get tensor names from manifest if available (for logging/debugging)
    const tensorNames = getTensorsForExpert(layerIdx, expertIdx);
    if (tensorNames.length > 0) {
      console.log(`[DopplerLoader] Expert ${layerIdx}_${expertIdx} tensors: ${tensorNames.length}`);
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
        const { isColumnMajorBuffer } = await import('../gpu/buffer-dtypes.js');
        for (const k of ['gate', 'up', 'down'] as const) {
          const buf = weights[k];
          if (buf instanceof GPUBuffer) {
            const dtype = getBufferDtype(buf) || 'f32';
            if (dtype === 'f32') {
              const elems = buf.size / 4;
              // Preserve column-major layout through the f32→f16 downcast
              const wasColumnMajor = isColumnMajorBuffer(buf);
              try {
                const f16buf = await castF32ToF16(buf, elems);
                // Copy layout metadata to the new buffer
                if (wasColumnMajor) {
                  setBufferLayout(f16buf, 'column');
                }
                releaseBuffer(buf);
                weights[k] = f16buf;
                this.gpuBuffers.add(f16buf);
              } catch (e) {
                console.warn(`[DopplerLoader] Failed to downcast expert ${k} to f16:`, (e as Error).message);
              }
            }
          }
        }
      }
    }

    // Calculate expert size and store in LRU cache
    if (!weights.isGptOss && this.expertCache) {
      let sizeBytes = 0;
      for (const k of ['gate', 'up', 'down'] as const) {
        const buf = weights[k];
        if (buf instanceof GPUBuffer) {
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
    for (const name of finalNormNames) {
      const location = this.tensorLocations.get(name);
      if (location) {
        finalNormElements = location.shape.reduce((a, b) => a * b, 1);
        this.finalNorm = await this._loadTensor(name, true, true) as GPUBuffer | Float32Array | null;
        break;
      }
    }

    if (this.finalNorm && this._needsNormWeightOffset()) {
      this.finalNorm = await this._applyNormWeightOffset(this.finalNorm, finalNormElements);
    }

    if (!this.finalNorm) {
      console.warn('[DopplerLoader] Final norm not found');
    }

    this.lmHead = (await this._loadTensor('language_model.lm_head.weight', true, true) ||
                  await this._loadTensor('lm_head.weight', true, true) ||
                  await this._loadTensor('output.weight', true, true)) as GPUBuffer | Float32Array | null;

    if (!this.lmHead && this.embeddings) {
      console.log('[DopplerLoader] Using tied embeddings as LM head');
      this.lmHead = this.embeddings;
    } else if (!this.lmHead) {
      console.warn('[DopplerLoader] LM head not found');
    }

    // Downcast LM head to f16
    const caps = getKernelCapabilities();
    if (caps.hasF16 && this.lmHead instanceof GPUBuffer && this.lmHead !== this.embeddings) {
      const dtype = getBufferDtype(this.lmHead) || 'f32';
      if (dtype === 'f32') {
        try {
          const elems = this.lmHead.size / 4;
          const f16buf = await castF32ToF16(this.lmHead, elems);
          releaseBuffer(this.lmHead);
          this.lmHead = f16buf;
          this.gpuBuffers.add(f16buf);
        } catch (e) {
          console.warn('[DopplerLoader] Failed to downcast lmHead to f16:', (e as Error).message);
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
    console.log('[DopplerLoader] Unloading model...');

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

    console.log('[DopplerLoader] Model unloaded');
  }
}

// Global loader instance
let globalLoader: DopplerLoader | null = null;

/**
 * Get global DopplerLoader instance
 */
export function getDopplerLoader(): DopplerLoader {
  if (!globalLoader) {
    globalLoader = new DopplerLoader();
  }
  return globalLoader;
}

/**
 * Create new DopplerLoader instance
 */
export function createDopplerLoader(): DopplerLoader {
  return new DopplerLoader();
}

export default DopplerLoader;
