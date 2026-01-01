/**
 * Platform Config Schema Definitions
 *
 * Defines the structure for device-family platform configurations.
 * Platforms provide kernel overrides and tuning hints for specific GPU families.
 *
 * Note: GPU capabilities (hasSubgroups, hasF16, etc.) are detected at runtime,
 * not stored in config. Platforms assume those capabilities are available.
 *
 * @module config/schema/platform
 */

// =============================================================================
// Platform Detection
// =============================================================================

/**
 * Patterns for detecting which platform to use based on GPU adapter info.
 */
export interface PlatformDetectionSchema {
  /** Vendor string pattern (e.g., "apple", "nvidia", "amd") */
  vendor?: string;

  /** Architecture pattern (e.g., "gpu-family-apple-9") */
  architecture?: string;

  /** Device name pattern (e.g., "M3", "RTX 4090") */
  device?: string;

  /** WebGPU adapter description pattern */
  description?: string;
}

// =============================================================================
// Kernel Overrides
// =============================================================================

/**
 * Overrides for a specific kernel operation on this platform.
 */
export interface KernelOperationOverrideSchema {
  /** Preferred variant to use (if requirements are met) */
  preferred?: string;

  /** Variants to avoid on this platform (known issues, poor performance) */
  avoid?: string[];

  /** Workgroup size overrides per variant */
  workgroupOverrides?: Record<string, [number, number, number]>;

  /** WGSL override constants per variant */
  wgslOverrides?: Record<string, Record<string, number>>;
}

/**
 * All kernel overrides for this platform.
 */
export type KernelOverridesSchema = Record<string, KernelOperationOverrideSchema>;

// =============================================================================
// Memory Hints
// =============================================================================

/**
 * Memory-related hints for this platform.
 */
export interface MemoryHintsSchema {
  /** Whether to prefer unified memory strategies (Apple Silicon) */
  preferUnifiedMemory?: boolean;

  /** Optimal buffer alignment in bytes */
  bufferAlignment?: number;

  /** Optimal shard size for this platform */
  optimalShardSize?: number;

  /** Maximum recommended buffer pool size */
  maxBufferPoolSize?: number;
}

// =============================================================================
// Platform Config
// =============================================================================

/**
 * Complete platform configuration for a GPU device family.
 */
export interface PlatformSchema {
  /** Unique identifier (e.g., "apple-m3", "nvidia-rtx40") */
  id: string;

  /** Human-readable name */
  name: string;

  /** Detection patterns to match this platform */
  detection: PlatformDetectionSchema;

  /** Kernel-specific overrides */
  kernelOverrides?: KernelOverridesSchema;

  /** Memory hints */
  memoryHints?: MemoryHintsSchema;

  /** Whether this is a fallback/generic platform */
  isGeneric?: boolean;

  /** Notes about this platform */
  notes?: string;
}

// =============================================================================
// Runtime Capabilities (Detected, not configured)
// =============================================================================

/**
 * GPU capabilities detected at runtime.
 * These are NOT stored in config files - they come from WebGPU adapter.
 */
export interface RuntimeCapabilities {
  /** Whether shader-f16 feature is available */
  hasF16: boolean;

  /** Whether subgroups feature is available */
  hasSubgroups: boolean;

  /** Subgroup size (if subgroups available) */
  subgroupSize?: number;

  /** Maximum workgroup size per dimension */
  maxWorkgroupSize: number;

  /** Maximum compute workgroup storage (shared memory) in bytes */
  maxSharedMemory: number;

  /** Maximum storage buffer binding size */
  maxStorageBufferBindingSize: number;

  /** Maximum buffer size */
  maxBufferSize: number;
}

// =============================================================================
// Resolved Platform Config
// =============================================================================

/**
 * Platform config with runtime capabilities merged in.
 * This is what selection.js receives.
 */
export interface ResolvedPlatformConfig {
  /** Platform configuration */
  platform: PlatformSchema;

  /** Runtime-detected GPU capabilities */
  capabilities: RuntimeCapabilities;
}

// =============================================================================
// Platform Registry
// =============================================================================

/**
 * Collection of all available platform configs.
 */
export interface PlatformRegistrySchema {
  /** Schema version */
  version: string;

  /** All platform configs by ID */
  platforms: Record<string, PlatformSchema>;

  /** ID of the generic/fallback platform */
  fallbackId: string;
}
