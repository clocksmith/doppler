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

/** Default heap testing configuration */
export const DEFAULT_HEAP_TESTING_CONFIG = {
  heapTestSizes: [16 * GB, 8 * GB, 4 * GB, 2 * GB, 1 * GB],
  fallbackMaxHeapBytes: 1 * GB,
};

// =============================================================================
// Segment Testing Config
// =============================================================================

/** Default segment testing configuration */
export const DEFAULT_SEGMENT_TESTING_CONFIG = {
  segmentTestSizes: [1 * GB, 512 * MB, 256 * MB, 128 * MB],
  safeSegmentSizeBytes: 256 * MB,
};

// =============================================================================
// Address Space Config
// =============================================================================

/** Default address space configuration */
export const DEFAULT_ADDRESS_SPACE_CONFIG = {
  targetAddressSpaceBytes: 8 * GB,
};

// =============================================================================
// Segment Allocation Config
// =============================================================================

/** Default segment allocation configuration */
export const DEFAULT_SEGMENT_ALLOCATION_CONFIG = {
  fallbackSegmentSizeBytes: 4 * GB,
  segmentFallbackSizes: [512 * MB, 256 * MB, 128 * MB],
};

// =============================================================================
// Complete Memory Limits Config
// =============================================================================

/** Default memory limits configuration */
export const DEFAULT_MEMORY_LIMITS_CONFIG = {
  heapTesting: DEFAULT_HEAP_TESTING_CONFIG,
  segmentTesting: DEFAULT_SEGMENT_TESTING_CONFIG,
  addressSpace: DEFAULT_ADDRESS_SPACE_CONFIG,
  segmentAllocation: DEFAULT_SEGMENT_ALLOCATION_CONFIG,
};
