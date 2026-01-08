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

/** Default buffer pool bucket configuration */
export const DEFAULT_BUFFER_POOL_BUCKET_CONFIG = {
  minBucketSizeBytes: 256, // 256 bytes
  largeBufferThresholdBytes: 32 * 1024 * 1024, // 32MB
  largeBufferStepBytes: 16 * 1024 * 1024, // 16MB
};

// =============================================================================
// Buffer Pool Limits Config
// =============================================================================

/** Default buffer pool limits configuration */
export const DEFAULT_BUFFER_POOL_LIMITS_CONFIG = {
  maxBuffersPerBucket: 8,
  maxTotalPooledBuffers: 64,
};

// =============================================================================
// Buffer Pool Alignment Config
// =============================================================================

/** Default buffer pool alignment configuration */
export const DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG = {
  alignmentBytes: 256, // WebGPU buffer alignment
};

// =============================================================================
// Complete Buffer Pool Config
// =============================================================================

/** Default buffer pool configuration */
export const DEFAULT_BUFFER_POOL_CONFIG = {
  bucket: DEFAULT_BUFFER_POOL_BUCKET_CONFIG,
  limits: DEFAULT_BUFFER_POOL_LIMITS_CONFIG,
  alignment: DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG,
};
