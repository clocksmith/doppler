/**
 * Shared Runtime Config Schema Definitions
 *
 * Cross-cutting configuration shared between model loading and inference.
 *
 * @module config/schema/shared-runtime
 */

import type { DebugConfigSchema } from './debug.schema.js';
import type { BenchmarkConfig } from './benchmark.schema.js';
import type { KernelThresholdsConfigSchema } from './kernel-thresholds.schema.js';
import type { BufferPoolConfigSchema } from './buffer-pool.schema.js';
import type { GpuCacheConfigSchema } from './gpu-cache.schema.js';
import type { MemoryLimitsConfigSchema } from './memory-limits.schema.js';
import type { TunerConfigSchema } from './tuner.schema.js';
import type { HotSwapConfigSchema } from './hotswap.schema.js';
import type { IntentBundleConfigSchema } from './intent-bundle.schema.js';
import type { BridgeConfigSchema } from './bridge.schema.js';
import type { PlatformSchema } from './platform.schema.js';
import type { HarnessConfigSchema } from './harness.schema.js';
import type { ToolingConfigSchema } from './tooling.schema.js';

/**
 * Kernel registry configuration (source/override).
 */
export interface KernelRegistryConfigSchema {
  /** Optional URL override for kernel registry JSON */
  url?: string | null;
}

/** Default kernel registry configuration */
export declare const DEFAULT_KERNEL_REGISTRY_CONFIG: KernelRegistryConfigSchema;

/**
 * Shared runtime configuration schema.
 */
export interface SharedRuntimeConfigSchema {
  /** Unified debug/log/trace configuration */
  debug: DebugConfigSchema;
  /** Benchmarking configuration defaults */
  benchmark: BenchmarkConfig;
  /** Harness configuration (CLI/test runner) */
  harness: HarnessConfigSchema;
  /** Tooling intent and diagnostics policy */
  tooling: ToolingConfigSchema;
  /** Optional platform override (auto-detect when null) */
  platform: Partial<PlatformSchema> | null;
  /** Kernel registry source config */
  kernelRegistry: KernelRegistryConfigSchema;
  /** Kernel selection thresholds */
  kernelThresholds: KernelThresholdsConfigSchema;
  /** GPU buffer pool sizing */
  bufferPool: BufferPoolConfigSchema;
  /** Uniform cache limits */
  gpuCache: GpuCacheConfigSchema;
  /** WASM heap and segment limits */
  memory: MemoryLimitsConfigSchema;
  /** Kernel autotuning settings */
  tuner: TunerConfigSchema;
  /** Hot-swap security policy */
  hotSwap: HotSwapConfigSchema;
  /** Intent bundle gating policy */
  intentBundle: IntentBundleConfigSchema;
  /** Native bridge settings (Tier 2) */
  bridge: BridgeConfigSchema;
}

/** Default shared runtime configuration */
export declare const DEFAULT_SHARED_RUNTIME_CONFIG: SharedRuntimeConfigSchema;
