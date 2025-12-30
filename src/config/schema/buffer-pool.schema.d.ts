/**
 * Buffer Pool Config Schema
 *
 * Configuration for GPU buffer pooling, including bucket sizing,
 * pool limits, and alignment settings for efficient buffer reuse.
 *
 * @module config/schema/buffer-pool
 */

/**
 * Configuration for GPU buffer pool bucket sizing.
 *
 * Controls how buffers are bucketed by size for efficient reuse.
 * Small buffers use power-of-2 rounding, large buffers use linear steps.
 */
export interface BufferPoolBucketConfigSchema {
  /** Minimum bucket size in bytes (buffers smaller than this round up to this) */
  minBucketSizeBytes: number;

  /** Threshold in bytes above which large buffer bucketing is used */
  largeBufferThresholdBytes: number;

  /** Step size in bytes for large buffer buckets (linear rounding) */
  largeBufferStepBytes: number;
}

/** Default buffer pool bucket configuration */
export declare const DEFAULT_BUFFER_POOL_BUCKET_CONFIG: BufferPoolBucketConfigSchema;

/**
 * Configuration for buffer pool size limits.
 *
 * Controls how many buffers can be pooled to prevent memory bloat.
 */
export interface BufferPoolLimitsConfigSchema {
  /** Maximum number of buffers per size/usage bucket */
  maxBuffersPerBucket: number;

  /** Maximum total number of buffers across all pools */
  maxTotalPooledBuffers: number;
}

/** Default buffer pool limits configuration */
export declare const DEFAULT_BUFFER_POOL_LIMITS_CONFIG: BufferPoolLimitsConfigSchema;

/**
 * Configuration for buffer alignment.
 *
 * Ensures buffers are aligned to WebGPU requirements.
 */
export interface BufferPoolAlignmentConfigSchema {
  /** Alignment boundary in bytes for buffer sizes */
  alignmentBytes: number;
}

/** Default buffer pool alignment configuration */
export declare const DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG: BufferPoolAlignmentConfigSchema;

/**
 * Configuration for global buffer budget enforcement.
 */
export interface BufferPoolBudgetConfigSchema {
  /** Hard byte budget for active + pooled buffers (0 = disabled) */
  maxTotalBytes: number;

  /** Ratio (0-1) that triggers proactive reclamation */
  highWatermarkRatio: number;

  /** Ratio (0-1) target after emergency reclaim */
  emergencyTrimTargetRatio: number;

  /** Throw when budget cannot be satisfied after reclaim */
  hardFailOnBudgetExceeded: boolean;
}

/** Default buffer pool budget configuration */
export declare const DEFAULT_BUFFER_POOL_BUDGET_CONFIG: BufferPoolBudgetConfigSchema;

/**
 * Complete buffer pool configuration schema.
 *
 * Combines bucket sizing, pool limits, and alignment settings.
 */
export interface BufferPoolConfigSchema {
  bucket: BufferPoolBucketConfigSchema;
  limits: BufferPoolLimitsConfigSchema;
  alignment: BufferPoolAlignmentConfigSchema;
  budget: BufferPoolBudgetConfigSchema;
}

/** Default buffer pool configuration */
export declare const DEFAULT_BUFFER_POOL_CONFIG: BufferPoolConfigSchema;
