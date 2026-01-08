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
import { DEFAULT_HOTSWAP_CONFIG } from './hotswap.schema.js';
import { DEFAULT_BRIDGE_CONFIG } from './bridge.schema.js';

// =============================================================================
// Runtime Config (all non-model-specific settings)
// =============================================================================

/** Default runtime configuration */
export const DEFAULT_RUNTIME_CONFIG = {
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
  hotSwap: DEFAULT_HOTSWAP_CONFIG,
  bridge: DEFAULT_BRIDGE_CONFIG,
};

// =============================================================================
// Master Doppler Config
// =============================================================================

/** Default Doppler configuration (no model loaded) */
export const DEFAULT_DOPPLER_CONFIG = {
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
 */
export function createDopplerConfig(
  overrides
) {
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
  base,
  overrides
) {
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
          largeWeights: { ...base.inference.largeWeights, ...overrides.inference.largeWeights },
          prompt: overrides.inference.prompt ?? base.inference.prompt,
          pipeline: overrides.inference.pipeline ?? base.inference.pipeline,
          kernelPath: overrides.inference.kernelPath ?? base.inference.kernelPath,
          chatTemplate: overrides.inference.chatTemplate
            ? { ...base.inference.chatTemplate, ...overrides.inference.chatTemplate }
            : base.inference.chatTemplate,
          // Model-specific inference overrides (merged with manifest.inference at load time)
          modelOverrides: overrides.inference.modelOverrides ?? base.inference.modelOverrides,
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
    hotSwap: overrides.hotSwap
      ? {
          ...base.hotSwap,
          ...overrides.hotSwap,
          trustedSigners: overrides.hotSwap.trustedSigners ?? base.hotSwap.trustedSigners,
        }
      : { ...base.hotSwap },
    bridge: { ...base.bridge, ...overrides.bridge },
  };
}
