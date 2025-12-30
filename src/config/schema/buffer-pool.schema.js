import { MB } from './units.schema.js';

// =============================================================================
// Buffer Pool Config
// =============================================================================

export const DEFAULT_BUFFER_POOL_BUCKET_CONFIG = {
  minBucketSizeBytes: 256, // 256 bytes (WebGPU alignment)
  largeBufferThresholdBytes: 32 * MB,
  largeBufferStepBytes: 16 * MB,
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
// Buffer Pool Budget Config
// =============================================================================

export const DEFAULT_BUFFER_POOL_BUDGET_CONFIG = {
  maxTotalBytes: 0, // 0 disables hard budget
  highWatermarkRatio: 0.9,
  emergencyTrimTargetRatio: 0.75,
  hardFailOnBudgetExceeded: true,
};

// =============================================================================
// Complete Buffer Pool Config
// =============================================================================

export const DEFAULT_BUFFER_POOL_CONFIG = {
  bucket: DEFAULT_BUFFER_POOL_BUCKET_CONFIG,
  limits: DEFAULT_BUFFER_POOL_LIMITS_CONFIG,
  alignment: DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG,
  budget: DEFAULT_BUFFER_POOL_BUDGET_CONFIG,
};
