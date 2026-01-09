// =============================================================================
// Buffer Pool Config
// =============================================================================

export const DEFAULT_BUFFER_POOL_BUCKET_CONFIG = {
  minBucketSizeBytes: 256, // 256 bytes
  largeBufferThresholdBytes: 32 * 1024 * 1024, // 32MB
  largeBufferStepBytes: 16 * 1024 * 1024, // 16MB
};

// =============================================================================
// Buffer Pool Limits Config
// =============================================================================

export const DEFAULT_BUFFER_POOL_LIMITS_CONFIG = {
  maxBuffersPerBucket: 8,
  maxTotalPooledBuffers: 64,
};

// =============================================================================
// Buffer Pool Alignment Config
// =============================================================================

export const DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG = {
  alignmentBytes: 256, // WebGPU buffer alignment
};

// =============================================================================
// Complete Buffer Pool Config
// =============================================================================

export const DEFAULT_BUFFER_POOL_CONFIG = {
  bucket: DEFAULT_BUFFER_POOL_BUCKET_CONFIG,
  limits: DEFAULT_BUFFER_POOL_LIMITS_CONFIG,
  alignment: DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG,
};
