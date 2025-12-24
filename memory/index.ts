/**
 * DOPPLER Memory Module
 * Agent-A | Domain: memory/
 *
 * @module memory
 */

// Re-export from capability
export {
  getMemoryCapabilities,
} from './capability.js';

export type {
  MemoryStrategy,
  SegmentedLimits,
  MemoryCapabilities,
} from './capability.js';

// Re-export from unified-detect
export {
  detectUnifiedMemory,
} from './unified-detect.js';

export type {
  AppleSiliconInfo,
  AMDUnifiedInfo,
  LimitIndicators,
  UnifiedMemoryInfo,
} from './unified-detect.js';

// Re-export from address-table
export {
  AddressTable,
  ADDRESS_TABLE_CONSTANTS,
} from './address-table.js';

export type {
  DecodedAddress,
  AddressRangeChunk,
  AddressTableConstants,
} from './address-table.js';

// Re-export from heap-manager
export {
  HeapManager,
  getHeapManager,
} from './heap-manager.js';

export type {
  MemorySegment,
  AllocationResult,
  HeapStats,
} from './heap-manager.js';
