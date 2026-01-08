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
import type { HotSwapConfigSchema } from './hotswap.schema.js';
import type { BridgeConfigSchema } from './bridge.schema.js';

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

  /** Hot-swap security policy */
  hotSwap: HotSwapConfigSchema;

  /** Native bridge settings (Tier 2) */
  bridge: BridgeConfigSchema;
}

/** Default runtime configuration */
export declare const DEFAULT_RUNTIME_CONFIG: RuntimeConfigSchema;

/**
 * Master Doppler configuration schema.
 *
 * Combines model-specific configuration (resolved from preset + manifest)
 * with runtime configuration (engine settings) and platform overrides.
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
export declare const DEFAULT_DOPPLER_CONFIG: DopplerConfigSchema;

/**
 * Create a Doppler configuration with optional overrides.
 *
 * Merges provided overrides with defaults, performing a deep merge
 * on nested objects.
 */
export declare function createDopplerConfig(
  overrides?: DopplerConfigOverrides
): DopplerConfigSchema;
