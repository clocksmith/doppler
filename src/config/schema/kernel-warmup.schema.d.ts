/**
 * Kernel Warmup Config Schema
 *
 * Controls optional kernel prewarming.
 *
 * @module config/schema/kernel-warmup
 */

export interface KernelWarmupConfigSchema {
  /** Precompile pipelines for all supported kernel variants. */
  prewarm: boolean;
  /** Prewarm scheduling mode (parallel or sequential). */
  prewarmMode: 'parallel' | 'sequential';
}

/** Default kernel warmup configuration */
export declare const DEFAULT_KERNEL_WARMUP_CONFIG: KernelWarmupConfigSchema;
