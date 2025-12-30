import { MB, GB } from './units.schema.js';

// =============================================================================
// Heap Testing Config
// =============================================================================

export const DEFAULT_HEAP_TESTING_CONFIG = {
  heapTestSizes: [16 * GB, 8 * GB, 4 * GB, 2 * GB, 1 * GB],
  fallbackMaxHeapBytes: 1 * GB,
};

// =============================================================================
// Segment Testing Config
// =============================================================================

export const DEFAULT_SEGMENT_TESTING_CONFIG = {
  segmentTestSizes: [1 * GB, 512 * MB, 256 * MB, 128 * MB],
  safeSegmentSizeBytes: 256 * MB,
};

// =============================================================================
// Address Space Config
// =============================================================================

export const DEFAULT_ADDRESS_SPACE_CONFIG = {
  targetAddressSpaceBytes: 8 * GB,
};

// =============================================================================
// Segment Allocation Config
// =============================================================================

export const DEFAULT_SEGMENT_ALLOCATION_CONFIG = {
  fallbackSegmentSizeBytes: 4 * GB,
  segmentFallbackSizes: [512 * MB, 256 * MB, 128 * MB],
};

// =============================================================================
// Emulated Storage Config
// =============================================================================

export const DEFAULT_EMULATED_STORAGE_CONFIG = {
  vramBudgetBytes: 2 * GB,
  ramBudgetBytes: 8 * GB,
};

// =============================================================================
// Complete Memory Limits Config
// =============================================================================

export const DEFAULT_MEMORY_LIMITS_CONFIG = {
  heapTesting: DEFAULT_HEAP_TESTING_CONFIG,
  segmentTesting: DEFAULT_SEGMENT_TESTING_CONFIG,
  addressSpace: DEFAULT_ADDRESS_SPACE_CONFIG,
  segmentAllocation: DEFAULT_SEGMENT_ALLOCATION_CONFIG,
  emulatedStorage: DEFAULT_EMULATED_STORAGE_CONFIG,
};
