/**
 * Doppler Config Schema
 *
 * Master configuration schema that composes all runtime configs together.
 * This provides a single unified interface for configuring the entire
 * Doppler inference engine.
 *
 * Individual configs remain importable for subsystems that only need
 * their specific domain. This master config is for:
 * - Serializing/restoring full engine state
 * - Configuration management UIs
 * - Debugging/logging full config state
 *
 * @module config/schema/doppler
 */

import type { ResolvedConfigSchema } from './preset.schema.js';
import type { PlatformSchema } from './platform.schema.js';
import type { DistributionConfigSchema } from './distribution.schema.js';
import type { StorageFullConfigSchema } from './storage.schema.js';
import type { LoadingConfigSchema } from './loading.schema.js';
import type { InferenceDefaultsConfigSchema } from './inference-defaults.schema.js';
import type { KVCacheConfigSchema } from './kvcache.schema.js';
import type { MoERuntimeConfigSchema } from './moe.schema.js';
import type { BufferPoolConfigSchema } from './buffer-pool.schema.js';
import type { GpuCacheConfigSchema } from './gpu-cache.schema.js';
import type { TunerConfigSchema } from './tuner.schema.js';
import type { MemoryLimitsConfigSchema } from './memory-limits.schema.js';
import type { DebugConfigSchema } from './debug.schema.js';
import type { BridgeConfigSchema } from './bridge.schema.js';

import { DEFAULT_DISTRIBUTION_CONFIG } from './distribution.schema.js';
import { DEFAULT_STORAGE_FULL_CONFIG } from './storage.schema.js';
import { DEFAULT_LOADING_CONFIG } from './loading.schema.js';
import { DEFAULT_INFERENCE_DEFAULTS_CONFIG } from './inference-defaults.schema.js';
import { DEFAULT_KVCACHE_CONFIG } from './kvcache.schema.js';
import { DEFAULT_MOE_RUNTIME_CONFIG } from './moe.schema.js';
import { DEFAULT_BUFFER_POOL_CONFIG } from './buffer-pool.schema.js';
import { DEFAULT_GPU_CACHE_CONFIG } from './gpu-cache.schema.js';
import { DEFAULT_TUNER_CONFIG } from './tuner.schema.js';
import { DEFAULT_MEMORY_LIMITS_CONFIG } from './memory-limits.schema.js';
import { DEFAULT_DEBUG_CONFIG } from './debug.schema.js';
import { DEFAULT_BRIDGE_CONFIG } from './bridge.schema.js';

// =============================================================================
// Runtime Config (all non-model-specific settings)
// =============================================================================

/**
 * Runtime configuration schema.
 *
 * Contains all configurable settings that are independent of the model.
 * These settings control engine behavior regardless of which model is loaded.
 */
export interface RuntimeConfigSchema {
  /** Network and download settings */
  distribution: DistributionConfigSchema;

  /** OPFS quota, VRAM estimation, alignment */
  storage: StorageFullConfigSchema;

  /** OPFS paths, shard cache, memory management */
  loading: LoadingConfigSchema;

  /** Batching, sampling, tokenizer defaults */
  inference: InferenceDefaultsConfigSchema;

  /** KV cache dtype and layout */
  kvcache: KVCacheConfigSchema;

  /** MoE routing and caching */
  moe: MoERuntimeConfigSchema;

  /** GPU buffer pool sizing */
  bufferPool: BufferPoolConfigSchema;

  /** Uniform cache limits */
  gpuCache: GpuCacheConfigSchema;

  /** Kernel autotuning settings */
  tuner: TunerConfigSchema;

  /** WASM heap and segment limits */
  memory: MemoryLimitsConfigSchema;

  /** Logging and tracing */
  debug: DebugConfigSchema;

  /** Native bridge settings (Tier 2) */
  bridge: BridgeConfigSchema;
}

/** Default runtime configuration */
export const DEFAULT_RUNTIME_CONFIG: RuntimeConfigSchema = {
  distribution: DEFAULT_DISTRIBUTION_CONFIG,
  storage: DEFAULT_STORAGE_FULL_CONFIG,
  loading: DEFAULT_LOADING_CONFIG,
  inference: DEFAULT_INFERENCE_DEFAULTS_CONFIG,
  kvcache: DEFAULT_KVCACHE_CONFIG,
  moe: DEFAULT_MOE_RUNTIME_CONFIG,
  bufferPool: DEFAULT_BUFFER_POOL_CONFIG,
  gpuCache: DEFAULT_GPU_CACHE_CONFIG,
  tuner: DEFAULT_TUNER_CONFIG,
  memory: DEFAULT_MEMORY_LIMITS_CONFIG,
  debug: DEFAULT_DEBUG_CONFIG,
  bridge: DEFAULT_BRIDGE_CONFIG,
};

// =============================================================================
// Master Doppler Config
// =============================================================================

/**
 * Master Doppler configuration schema.
 *
 * Combines model-specific configuration (resolved from preset + manifest)
 * with runtime configuration (engine settings) and platform overrides.
 *
 * Usage:
 * - `model`: Resolved config for the loaded model (architecture, layers, etc.)
 * - `runtime`: Engine settings (all domains from Phase 1-3 config extraction)
 * - `platform`: Optional platform-specific overrides (auto-detected if not set)
 *
 * @example
 * ```typescript
 * const config: DopplerConfigSchema = {
 *   model: resolvedModelConfig,
 *   runtime: {
 *     ...DEFAULT_RUNTIME_CONFIG,
 *     debug: { ...DEFAULT_DEBUG_CONFIG, logHistory: { maxEntries: 500 } },
 *   },
 * };
 * ```
 */
export interface DopplerConfigSchema {
  /** Model-specific configuration (from preset + manifest) */
  model?: ResolvedConfigSchema;

  /** Runtime configuration (engine settings) */
  runtime: RuntimeConfigSchema;

  /** Platform-specific overrides (auto-detected if not set) */
  platform?: Partial<PlatformSchema>;
}

export interface DopplerConfigOverrides extends Partial<Omit<DopplerConfigSchema, 'runtime'>> {
  runtime?: Partial<RuntimeConfigSchema>;
}

/** Default Doppler configuration (no model loaded) */
export const DEFAULT_DOPPLER_CONFIG: DopplerConfigSchema = {
  model: undefined,
  runtime: DEFAULT_RUNTIME_CONFIG,
  platform: undefined,
};

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create a Doppler configuration with optional overrides.
 *
 * Merges provided overrides with defaults, performing a deep merge
 * on nested objects.
 *
 * @param overrides - Partial configuration to merge with defaults
 * @returns Complete Doppler configuration
 *
 * @example
 * ```typescript
 * const config = createDopplerConfig({
 *   runtime: {
 *     debug: { logHistory: { maxEntries: 500 } },
 *   },
 * });
 * ```
 */
export function createDopplerConfig(
  overrides?: DopplerConfigOverrides
): DopplerConfigSchema {
  if (!overrides) {
    return { ...DEFAULT_DOPPLER_CONFIG };
  }

  return {
    model: overrides.model ?? DEFAULT_DOPPLER_CONFIG.model,
    runtime: overrides.runtime
      ? mergeRuntimeConfig(DEFAULT_RUNTIME_CONFIG, overrides.runtime)
      : { ...DEFAULT_RUNTIME_CONFIG },
    platform: overrides.platform,
  };
}

/**
 * Deep merge runtime config with overrides.
 *
 * Note: This merge may run twice in the CLI→browser flow:
 * 1. CLI: config-loader.ts deepMerge() processes user config with defaults
 * 2. Browser: setRuntimeConfig() calls this when config arrives via URL
 *
 * This is intentional - the double merge is idempotent for fully-specified
 * configs (CLI passes complete merged config), but allows browser-only
 * overrides to also work correctly.
 *
 * Uses object spread for most fields. Missing nested objects fall back to base.
 */
function mergeRuntimeConfig(
  base: RuntimeConfigSchema,
  overrides: Partial<RuntimeConfigSchema>
): RuntimeConfigSchema {
  return {
    distribution: { ...base.distribution, ...overrides.distribution },
    storage: overrides.storage
      ? {
          quota: { ...base.storage.quota, ...overrides.storage.quota },
          vramEstimation: { ...base.storage.vramEstimation, ...overrides.storage.vramEstimation },
          alignment: { ...base.storage.alignment, ...overrides.storage.alignment },
        }
      : { ...base.storage },
    loading: overrides.loading
      ? {
          shardCache: { ...base.loading.shardCache, ...overrides.loading.shardCache },
          memoryManagement: { ...base.loading.memoryManagement, ...overrides.loading.memoryManagement },
          opfsPath: { ...base.loading.opfsPath, ...overrides.loading.opfsPath },
          expertCache: { ...base.loading.expertCache, ...overrides.loading.expertCache },
        }
      : { ...base.loading },
    inference: overrides.inference
      ? {
          batching: { ...base.inference.batching, ...overrides.inference.batching },
          sampling: { ...base.inference.sampling, ...overrides.inference.sampling },
          compute: { ...base.inference.compute, ...overrides.inference.compute },
          tokenizer: { ...base.inference.tokenizer, ...overrides.inference.tokenizer },
          prompt: overrides.inference.prompt ?? base.inference.prompt,
          pipeline: overrides.inference.pipeline ?? base.inference.pipeline,
        }
      : { ...base.inference },
    kvcache: { ...base.kvcache, ...overrides.kvcache },
    moe: overrides.moe
      ? {
          routing: { ...base.moe.routing, ...overrides.moe.routing },
          cache: { ...base.moe.cache, ...overrides.moe.cache },
        }
      : { ...base.moe },
    bufferPool: overrides.bufferPool
      ? {
          bucket: { ...base.bufferPool.bucket, ...overrides.bufferPool.bucket },
          limits: { ...base.bufferPool.limits, ...overrides.bufferPool.limits },
          alignment: { ...base.bufferPool.alignment, ...overrides.bufferPool.alignment },
        }
      : { ...base.bufferPool },
    gpuCache: { ...base.gpuCache, ...overrides.gpuCache },
    tuner: { ...base.tuner, ...overrides.tuner },
    memory: overrides.memory
      ? {
          heapTesting: { ...base.memory.heapTesting, ...overrides.memory.heapTesting },
          segmentTesting: { ...base.memory.segmentTesting, ...overrides.memory.segmentTesting },
          addressSpace: { ...base.memory.addressSpace, ...overrides.memory.addressSpace },
          segmentAllocation: { ...base.memory.segmentAllocation, ...overrides.memory.segmentAllocation },
        }
      : { ...base.memory },
    debug: overrides.debug
      ? {
          logOutput: { ...base.debug.logOutput, ...overrides.debug.logOutput },
          logHistory: { ...base.debug.logHistory, ...overrides.debug.logHistory },
          logLevel: { ...base.debug.logLevel, ...overrides.debug.logLevel },
          trace: { ...base.debug.trace, ...overrides.debug.trace },
          pipeline: { ...base.debug.pipeline, ...overrides.debug.pipeline },
          probes: overrides.debug.probes ?? base.debug.probes,
        }
      : { ...base.debug },
    bridge: { ...base.bridge, ...overrides.bridge },
  };
}
