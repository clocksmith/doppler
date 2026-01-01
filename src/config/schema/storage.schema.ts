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

/**
 * Configuration for OPFS quota monitoring.
 *
 * Controls thresholds for warning about low disk space and
 * the frequency of quota checks.
 */
export interface QuotaConfigSchema {
  /** Threshold in bytes below which space is considered low (warn user) */
  lowSpaceThresholdBytes: number;

  /** Threshold in bytes below which space is critical (block operations) */
  criticalSpaceThresholdBytes: number;

  /** Interval in milliseconds between quota monitoring checks */
  monitorIntervalMs: number;
}

/** Default quota configuration */
export const DEFAULT_QUOTA_CONFIG: QuotaConfigSchema = {
  lowSpaceThresholdBytes: 500 * 1024 * 1024, // 500MB
  criticalSpaceThresholdBytes: 100 * 1024 * 1024, // 100MB
  monitorIntervalMs: 30000, // 30 seconds
};

// =============================================================================
// VRAM Estimation Config
// =============================================================================

/**
 * Configuration for VRAM estimation on different platforms.
 *
 * Used to estimate available GPU memory when WebGPU doesn't provide
 * accurate limits (especially on unified memory systems like Apple Silicon).
 */
export interface VramEstimationConfigSchema {
  /** Ratio of system RAM to use for VRAM estimation on unified memory systems (0-1) */
  unifiedMemoryRatio: number;

  /** Fallback VRAM size in bytes when estimation is not possible */
  fallbackVramBytes: number;

  /** Headroom to leave when VRAM is low, in bytes */
  lowVramHeadroomBytes: number;
}

/** Default VRAM estimation configuration */
export const DEFAULT_VRAM_ESTIMATION_CONFIG: VramEstimationConfigSchema = {
  unifiedMemoryRatio: 0.5, // 50% of system RAM
  fallbackVramBytes: 2 * 1024 * 1024 * 1024, // 2GB
  lowVramHeadroomBytes: 500 * 1024 * 1024, // 500MB
};

// =============================================================================
// Storage Alignment Config
// =============================================================================

/**
 * Configuration for storage buffer alignment.
 *
 * Ensures buffers are aligned to optimal boundaries for GPU access.
 */
export interface StorageAlignmentConfigSchema {
  /** Alignment boundary in bytes for storage buffers */
  bufferAlignmentBytes: number;
}

/** Default storage alignment configuration */
export const DEFAULT_STORAGE_ALIGNMENT_CONFIG: StorageAlignmentConfigSchema = {
  bufferAlignmentBytes: 4096, // 4KB alignment (typical page size)
};

// =============================================================================
// Complete Storage Config
// =============================================================================

/**
 * Complete storage configuration schema.
 *
 * Combines quota monitoring, VRAM estimation, and alignment settings.
 */
export interface StorageFullConfigSchema {
  quota: QuotaConfigSchema;
  vramEstimation: VramEstimationConfigSchema;
  alignment: StorageAlignmentConfigSchema;
}

/** Default storage configuration */
export const DEFAULT_STORAGE_FULL_CONFIG: StorageFullConfigSchema = {
  quota: DEFAULT_QUOTA_CONFIG,
  vramEstimation: DEFAULT_VRAM_ESTIMATION_CONFIG,
  alignment: DEFAULT_STORAGE_ALIGNMENT_CONFIG,
};
