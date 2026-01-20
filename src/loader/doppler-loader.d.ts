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

import type { MemoryCapabilities } from '../memory/capability.js';
import type { HeapManager } from '../memory/heap-manager.js';
import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { WeightBuffer, WeightLayout, CpuWeightBuffer } from '../gpu/weight-buffer.js';
import type { ExpertCache, CacheStats } from './expert-cache.js';
import type { ExpertWeights } from './weights.js';
import type { LoRAAdapter } from '../inference/pipeline/lora.js';
import type {
  TensorLocation,
  LayerWeights,
  LoadProgress,
  LoadOptions,
  CustomShardLoader,
  CustomShardLoaderOptions,
  LoaderStats,
  KernelCapabilities,
  Q4KConfig,
  ModelConfig,
} from './loader-types.js';
import type { ShardCache } from './shard-cache.js';
import type { LoadingConfigSchema } from '../config/schema/loading.schema.js';

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
export declare class DopplerLoader {
  // Capabilities
  memoryCapabilities: MemoryCapabilities | null;
  gpuCapabilities: KernelCapabilities | null;
  isUnifiedMemory: boolean;

  // Manifest and model info
  manifest: RDRRManifest | null;
  modelId: string | null;
  isMoE: boolean;

  // Loaded state
  isLoaded: boolean;
  embeddings: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;
  layers: Map<number, LayerWeights>;
  experts: Map<string, ExpertWeights>;
  lmHead: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;
  finalNorm: GPUBuffer | Float32Array | null;

  // Memory management
  heapManager: HeapManager | null;
  gpuBuffers: Set<GPUBuffer>;

  // Expert cache for MoE models (LRU eviction)
  expertCache: ExpertCache | null;

  // Loading state
  loadedShards: Set<number>;
  tensorLocations: Map<string, TensorLocation>;

  // Shard cache (LRU with request deduplication)
  shardCache: ShardCache;

  // Fused Q4_K matmul: skip dequantization for matmul weights, use fused kernel
  useFusedQ4K: boolean;

  // Q4K layout: 'row' = fused kernel compatible (fast), 'col' = dequant fallback
  q4kLayout: 'row' | 'col' | null;
  keepF32Weights: boolean;

  constructor(loadingConfig?: LoadingConfigSchema);

  setLoadingConfig(config: LoadingConfigSchema): void;

  setQ4KConfig(config: Q4KConfig): void;

  setCustomShardLoader(loadShardFn: CustomShardLoader, options?: CustomShardLoaderOptions): void;

  setTensorsJsonUrl(url: string | null): void;

  init(): Promise<void>;

  setManifest(manifest: RDRRManifest): void;

  loadLoRAWeights(manifest: RDRRManifest): Promise<LoRAAdapter>;

  load(modelId: string, options?: LoadOptions): Promise<ModelConfig>;

  prefetchExperts(nextLayerIdx: number, expertIndices: number[]): void;

  predictNextLayerExperts(currentExperts: number[]): number[];

  loadExpert(layerIdx: number, expertIdx: number): Promise<ExpertWeights>;

  getLayerWeights(layerIdx: number): LayerWeights | null;

  getConfig(): ModelConfig;

  canRunDense(): boolean;

  getStats(): LoaderStats;

  getExpertCacheStats(): CacheStats | null;

  unload(): Promise<void>;
}

export declare function getDopplerLoader(loadingConfig?: LoadingConfigSchema): DopplerLoader;

export declare function createDopplerLoader(loadingConfig?: LoadingConfigSchema): DopplerLoader;

export default DopplerLoader;
