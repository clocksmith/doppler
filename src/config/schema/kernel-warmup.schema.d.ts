/**
 * Kernel Warmup Config Schema
 *
 * Controls optional kernel prewarming and auto-tuning.
 *
 * @module config/schema/kernel-warmup
 */

export interface KernelWarmupConfigSchema {
  /** Precompile pipelines for all supported kernel variants. */
  prewarm: boolean;
  /** Prewarm scheduling mode (parallel or sequential). */
  prewarmMode: 'parallel' | 'sequential';
  /** Auto-tune kernel workgroup sizes for the active model config. */
  autoTune: boolean;
}

/** Default kernel warmup configuration */
export declare const DEFAULT_KERNEL_WARMUP_CONFIG: KernelWarmupConfigSchema;
