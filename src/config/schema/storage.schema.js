/**
 * Storage Config Schema
 *
 * Configuration for OPFS storage, quota management, and memory estimation.
 * These settings control how the system monitors disk space, estimates VRAM,
 * and aligns storage buffers.
 *
 * @module config/schema/storage
 */

// =============================================================================
// Quota Config
// =============================================================================

/** Default quota configuration */
export const DEFAULT_QUOTA_CONFIG = {
  lowSpaceThresholdBytes: 500 * 1024 * 1024, // 500MB
  criticalSpaceThresholdBytes: 100 * 1024 * 1024, // 100MB
  monitorIntervalMs: 30000, // 30 seconds
};

// =============================================================================
// VRAM Estimation Config
// =============================================================================

/** Default VRAM estimation configuration */
export const DEFAULT_VRAM_ESTIMATION_CONFIG = {
  unifiedMemoryRatio: 0.5, // 50% of system RAM
  fallbackVramBytes: 2 * 1024 * 1024 * 1024, // 2GB
  lowVramHeadroomBytes: 500 * 1024 * 1024, // 500MB
};

// =============================================================================
// Storage Alignment Config
// =============================================================================

/** Default storage alignment configuration */
export const DEFAULT_STORAGE_ALIGNMENT_CONFIG = {
  bufferAlignmentBytes: 4096, // 4KB alignment (typical page size)
};

// =============================================================================
// Complete Storage Config
// =============================================================================

/** Default storage configuration */
export const DEFAULT_STORAGE_FULL_CONFIG = {
  quota: DEFAULT_QUOTA_CONFIG,
  vramEstimation: DEFAULT_VRAM_ESTIMATION_CONFIG,
  alignment: DEFAULT_STORAGE_ALIGNMENT_CONFIG,
};
