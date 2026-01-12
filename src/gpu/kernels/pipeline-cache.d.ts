/**
 * Pipeline Cache - GPU pipeline creation and caching
 *
 * Handles creation and caching of compute pipelines, bind group layouts,
 * and pipeline layouts for kernel execution.
 *
 * @module gpu/kernels/pipeline-cache
 */

// ============================================================================
// Bind Group Layout
// ============================================================================

/**
 * Get or create a cached bind group layout.
 */
export declare function getOrCreateBindGroupLayout(
  label: string,
  entries: GPUBindGroupLayoutEntry[],
  deviceOverride?: GPUDevice | null
): GPUBindGroupLayout;

// ============================================================================
// Pipeline Layout
// ============================================================================

/**
 * Get or create a cached pipeline layout.
 */
export declare function getOrCreatePipelineLayout(
  label: string,
  bindGroupLayouts: GPUBindGroupLayout[],
  deviceOverride?: GPUDevice | null
): GPUPipelineLayout;

// ============================================================================
// Pipeline Creation
// ============================================================================

/**
 * Synchronously get a cached pipeline, or null if not cached.
 * Use this for fast path when you know the pipeline should be warm.
 */
export declare function getCachedPipeline(
  operation: string,
  variant: string,
  constants?: Record<string, number | boolean> | null
): GPUComputePipeline | null;

/**
 * Get a pipeline, using synchronous cache lookup when available.
 * Falls back to async compilation if not cached.
 * This is the preferred way to get pipelines in hot paths.
 */
export declare function getPipelineFast(
  operation: string,
  variant: string,
  bindGroupLayout?: GPUBindGroupLayout | null,
  constants?: Record<string, number | boolean> | null
): Promise<GPUComputePipeline>;

/**
 * Create a compute pipeline for a kernel
 */
export declare function createPipeline(
  operation: string,
  variant: string,
  bindGroupLayout?: GPUBindGroupLayout | null,
  constants?: Record<string, number | boolean> | null
): Promise<GPUComputePipeline>;

// ============================================================================
// Cache Management
// ============================================================================

/**
 * Clear the pipeline caches
 */
export declare function clearPipelineCaches(): void;

/**
 * Get pipeline cache statistics
 */
export declare function getPipelineCacheStats(): {
  pipelines: number;
  bindGroupLayouts: number;
  pipelineLayouts: number;
};
