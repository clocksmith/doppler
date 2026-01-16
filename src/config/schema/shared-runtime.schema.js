import { DEFAULT_DEBUG_CONFIG } from './debug.schema.js';
import { DEFAULT_KERNEL_THRESHOLDS } from './kernel-thresholds.schema.js';
import { DEFAULT_BUFFER_POOL_CONFIG } from './buffer-pool.schema.js';
import { DEFAULT_GPU_CACHE_CONFIG } from './gpu-cache.schema.js';
import { DEFAULT_MEMORY_LIMITS_CONFIG } from './memory-limits.schema.js';
import { DEFAULT_TUNER_CONFIG } from './tuner.schema.js';
import { DEFAULT_HOTSWAP_CONFIG } from './hotswap.schema.js';
import { DEFAULT_BRIDGE_CONFIG } from './bridge.schema.js';
import { DEFAULT_BENCHMARK_CONFIG } from './benchmark.schema.js';
import { DEFAULT_HARNESS_CONFIG } from './harness.schema.js';

// =============================================================================
// Kernel Registry Config
// =============================================================================

export const DEFAULT_KERNEL_REGISTRY_CONFIG = {
  url: null,
};

// =============================================================================
// Shared Runtime Config (cross-cutting for loading + inference)
// =============================================================================

export const DEFAULT_SHARED_RUNTIME_CONFIG = {
  debug: DEFAULT_DEBUG_CONFIG,
  benchmark: DEFAULT_BENCHMARK_CONFIG,
  harness: DEFAULT_HARNESS_CONFIG,
  platform: null,
  kernelRegistry: DEFAULT_KERNEL_REGISTRY_CONFIG,
  kernelThresholds: DEFAULT_KERNEL_THRESHOLDS,
  bufferPool: DEFAULT_BUFFER_POOL_CONFIG,
  gpuCache: DEFAULT_GPU_CACHE_CONFIG,
  memory: DEFAULT_MEMORY_LIMITS_CONFIG,
  tuner: DEFAULT_TUNER_CONFIG,
  hotSwap: DEFAULT_HOTSWAP_CONFIG,
  bridge: DEFAULT_BRIDGE_CONFIG,
};
