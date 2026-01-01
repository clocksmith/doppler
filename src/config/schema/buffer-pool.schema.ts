/**
 * Buffer Pool Config Schema
 *
 * Configuration for GPU buffer pooling, including bucket sizing,
 * pool limits, and alignment settings for efficient buffer reuse.
 *
 * @module config/schema/buffer-pool
 */

// =============================================================================
// Buffer Pool Config
// =============================================================================

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
export const DEFAULT_BUFFER_POOL_BUCKET_CONFIG: BufferPoolBucketConfigSchema = {
  minBucketSizeBytes: 256, // 256 bytes
  largeBufferThresholdBytes: 32 * 1024 * 1024, // 32MB
  largeBufferStepBytes: 16 * 1024 * 1024, // 16MB
};

// =============================================================================
// Buffer Pool Limits Config
// =============================================================================

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
export const DEFAULT_BUFFER_POOL_LIMITS_CONFIG: BufferPoolLimitsConfigSchema = {
  maxBuffersPerBucket: 8,
  maxTotalPooledBuffers: 64,
};

// =============================================================================
// Buffer Pool Alignment Config
// =============================================================================

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
export const DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG: BufferPoolAlignmentConfigSchema = {
  alignmentBytes: 256, // WebGPU buffer alignment
};

// =============================================================================
// Complete Buffer Pool Config
// =============================================================================

/**
 * Complete buffer pool configuration schema.
 *
 * Combines bucket sizing, pool limits, and alignment settings.
 */
export interface BufferPoolConfigSchema {
  bucket: BufferPoolBucketConfigSchema;
  limits: BufferPoolLimitsConfigSchema;
  alignment: BufferPoolAlignmentConfigSchema;
}

/** Default buffer pool configuration */
export const DEFAULT_BUFFER_POOL_CONFIG: BufferPoolConfigSchema = {
  bucket: DEFAULT_BUFFER_POOL_BUCKET_CONFIG,
  limits: DEFAULT_BUFFER_POOL_LIMITS_CONFIG,
  alignment: DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG,
};
