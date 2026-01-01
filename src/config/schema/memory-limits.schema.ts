/**
 * Memory Limits Config Schema
 *
 * Configuration for memory capability detection and heap management.
 * These settings control heap test sizes, segment allocation, and
 * fallback limits for both Memory64 and segmented heap strategies.
 *
 * @module config/schema/memory-limits
 */

// =============================================================================
// Unit Constants
// =============================================================================

const MB = 1024 * 1024;
const GB = 1024 * MB;

// =============================================================================
// Heap Testing Config
// =============================================================================

/**
 * Configuration for heap size testing.
 *
 * Controls the sizes probed when detecting maximum WASM heap size
 * and the fallback when all probes fail.
 */
export interface HeapTestingConfigSchema {
  /** Sizes to test for maximum heap, in descending order (bytes) */
  heapTestSizes: number[];

  /** Fallback maximum heap size when all probes fail (bytes) */
  fallbackMaxHeapBytes: number;
}

/** Default heap testing configuration */
export const DEFAULT_HEAP_TESTING_CONFIG: HeapTestingConfigSchema = {
  heapTestSizes: [16 * GB, 8 * GB, 4 * GB, 2 * GB, 1 * GB],
  fallbackMaxHeapBytes: 1 * GB,
};

// =============================================================================
// Segment Testing Config
// =============================================================================

/**
 * Configuration for segment size testing.
 *
 * Controls the sizes probed when detecting maximum ArrayBuffer size
 * for segmented heap mode.
 */
export interface SegmentTestingConfigSchema {
  /** Sizes to test for maximum segment, in descending order (bytes) */
  segmentTestSizes: number[];

  /** Safe segment size to use as default (bytes) */
  safeSegmentSizeBytes: number;
}

/** Default segment testing configuration */
export const DEFAULT_SEGMENT_TESTING_CONFIG: SegmentTestingConfigSchema = {
  segmentTestSizes: [1 * GB, 512 * MB, 256 * MB, 128 * MB],
  safeSegmentSizeBytes: 256 * MB,
};

// =============================================================================
// Address Space Config
// =============================================================================

/**
 * Configuration for virtual address space.
 *
 * Controls the target address space size used to calculate
 * recommended segment count.
 */
export interface AddressSpaceConfigSchema {
  /** Target virtual address space size (bytes) */
  targetAddressSpaceBytes: number;
}

/** Default address space configuration */
export const DEFAULT_ADDRESS_SPACE_CONFIG: AddressSpaceConfigSchema = {
  targetAddressSpaceBytes: 8 * GB,
};

// =============================================================================
// Segment Allocation Config
// =============================================================================

/**
 * Configuration for segment allocation.
 *
 * Controls fallback behavior when segment allocation fails.
 */
export interface SegmentAllocationConfigSchema {
  /** Fallback segment size for Memory64 init failure (bytes) */
  fallbackSegmentSizeBytes: number;

  /** Fallback sizes to try when segment allocation fails, in descending order (bytes) */
  segmentFallbackSizes: number[];
}

/** Default segment allocation configuration */
export const DEFAULT_SEGMENT_ALLOCATION_CONFIG: SegmentAllocationConfigSchema = {
  fallbackSegmentSizeBytes: 4 * GB,
  segmentFallbackSizes: [512 * MB, 256 * MB, 128 * MB],
};

// =============================================================================
// Complete Memory Limits Config
// =============================================================================

/**
 * Complete memory limits configuration schema.
 *
 * Combines heap testing, segment testing, address space,
 * and segment allocation settings.
 */
export interface MemoryLimitsConfigSchema {
  heapTesting: HeapTestingConfigSchema;
  segmentTesting: SegmentTestingConfigSchema;
  addressSpace: AddressSpaceConfigSchema;
  segmentAllocation: SegmentAllocationConfigSchema;
}

/** Default memory limits configuration */
export const DEFAULT_MEMORY_LIMITS_CONFIG: MemoryLimitsConfigSchema = {
  heapTesting: DEFAULT_HEAP_TESTING_CONFIG,
  segmentTesting: DEFAULT_SEGMENT_TESTING_CONFIG,
  addressSpace: DEFAULT_ADDRESS_SPACE_CONFIG,
  segmentAllocation: DEFAULT_SEGMENT_ALLOCATION_CONFIG,
};
