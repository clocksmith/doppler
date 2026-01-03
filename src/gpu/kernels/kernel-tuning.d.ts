/**
 * Kernel Tuning - Auto-tuning and kernel prewarming
 *
 * Provides utilities for tuning kernel workgroup sizes and
 * prewarming kernel pipelines for optimal performance.
 *
 * @module gpu/kernels/kernel-tuning
 */

// ============================================================================
// Workgroup Size Tuning
// ============================================================================

/**
 * Get tuned workgroup size for an operation
 */
export declare function getTunedWorkgroupSize(
  operation: string,
  inputSizes?: Record<string, number>
): Promise<[number, number, number]>;

// ============================================================================
// Auto-Tuning
// ============================================================================

/**
 * Run auto-tuning for all kernels with given model config
 */
export declare function autoTuneKernels(
  modelConfig?: Record<string, number>
): Promise<Record<string, any>>;

// ============================================================================
// Pipeline Prewarming
// ============================================================================

/**
 * Prewarm all supported kernel pipelines
 */
export declare function prewarmKernels(
  options?: { mode?: 'parallel' | 'sequential' }
): Promise<void>;
