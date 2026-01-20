/**
 * Memory Limits Config Schema
 *
 * Configuration for memory capability detection and heap management.
 * These settings control heap test sizes, segment allocation, and
 * fallback limits for both Memory64 and segmented heap strategies.
 *
 * @module config/schema/memory-limits
 */

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
export declare const DEFAULT_HEAP_TESTING_CONFIG: HeapTestingConfigSchema;

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
export declare const DEFAULT_SEGMENT_TESTING_CONFIG: SegmentTestingConfigSchema;

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
export declare const DEFAULT_ADDRESS_SPACE_CONFIG: AddressSpaceConfigSchema;

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
export declare const DEFAULT_SEGMENT_ALLOCATION_CONFIG: SegmentAllocationConfigSchema;

/**
 * Configuration for emulated storage budgets.
 *
 * Controls default VRAM and RAM budgets for the EmulatedVramStore
 * tiered storage system.
 */
export interface EmulatedStorageConfigSchema {
  /** Default VRAM budget (bytes) */
  vramBudgetBytes: number;

  /** Default RAM budget (bytes) */
  ramBudgetBytes: number;
}

/** Default emulated storage configuration */
export declare const DEFAULT_EMULATED_STORAGE_CONFIG: EmulatedStorageConfigSchema;

/**
 * Complete memory limits configuration schema.
 *
 * Combines heap testing, segment testing, address space,
 * segment allocation, and emulated storage settings.
 */
export interface MemoryLimitsConfigSchema {
  heapTesting: HeapTestingConfigSchema;
  segmentTesting: SegmentTestingConfigSchema;
  addressSpace: AddressSpaceConfigSchema;
  segmentAllocation: SegmentAllocationConfigSchema;
  emulatedStorage: EmulatedStorageConfigSchema;
}

/** Default memory limits configuration */
export declare const DEFAULT_MEMORY_LIMITS_CONFIG: MemoryLimitsConfigSchema;
