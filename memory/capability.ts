/**
 * Memory Capability Detection
 * Agent-A | Domain: memory/
 *
 * Detects browser memory capabilities:
 * - Memory64 (WASM large heap support)
 * - Unified memory (Apple/AMD Strix)
 * - Maximum heap sizes
 *
 * @module memory/capability
 */

import { detectUnifiedMemory, type UnifiedMemoryInfo } from './unified-detect.js';

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Memory strategy type
 */
export type MemoryStrategy = 'MEMORY64' | 'SEGMENTED';

/**
 * Segmented heap limits
 */
export interface SegmentedLimits {
  maxSegmentSize: number;
  recommendedSegments: number;
}

/**
 * Memory capabilities result
 */
export interface MemoryCapabilities {
  /** Browser supports WASM Memory64 */
  hasMemory64: boolean;
  /** GPU shares system RAM (Apple/Strix) */
  isUnifiedMemory: boolean;
  /** Details from unified-detect */
  unifiedMemoryInfo: UnifiedMemoryInfo;
  /** Max single heap size (Memory64 only) */
  maxHeapSize: number | null;
  /** Segment limits (non-Memory64) */
  segmentedLimits: SegmentedLimits | null;
  /** Recommended heap strategy */
  strategy: MemoryStrategy;
}

// ============================================================================
// Detection Functions
// ============================================================================

/**
 * Memory64 feature detection via WASM binary probe
 * Tests if browser supports 64-bit memory addressing
 */
async function probeMemory64(): Promise<boolean> {
  // Minimal WASM module declaring memory64
  // (module (memory i64 1))
  const memory64Binary = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, // WASM magic
    0x01, 0x00, 0x00, 0x00, // Version 1
    0x05, 0x04, 0x01, // Memory section, 1 entry
    0x04, 0x01, 0x00, // memory64 flag (0x04), min 1 page, no max
  ]);

  try {
    await WebAssembly.compile(memory64Binary);
    return true;
  } catch {
    return false;
  }
}

/**
 * Estimate maximum usable heap size
 * Tests allocation limits without OOM
 */
async function probeMaxHeapSize(): Promise<number> {
  const GB = 1024 * 1024 * 1024;
  const testSizes = [16 * GB, 8 * GB, 4 * GB, 2 * GB, 1 * GB];

  for (const size of testSizes) {
    try {
      // Try to create a WASM memory of this size
      const pages = Math.ceil(size / 65536); // 64KB pages
      new WebAssembly.Memory({ initial: 1, maximum: pages });
      return size;
    } catch {
      continue;
    }
  }

  return 1 * GB; // Fallback to 1GB
}

/**
 * Probe segmented heap limits (for non-Memory64 browsers)
 * Returns max size per ArrayBuffer and recommended segment count
 */
function probeSegmentedLimits(): SegmentedLimits {
  const MB = 1024 * 1024;
  const GB = 1024 * MB;

  // Test actual allocation limits - browsers often can't allocate large ArrayBuffers
  // Start with smaller sizes that are more likely to succeed
  const testSizes = [1 * GB, 512 * MB, 256 * MB, 128 * MB];

  let maxSegmentSize = 256 * MB; // Safe default

  for (const size of testSizes) {
    try {
      // Actually try to allocate to see if it works
      const testBuffer = new ArrayBuffer(size);
      if (testBuffer.byteLength === size) {
        maxSegmentSize = size;
        break; // Use the largest working size
      }
    } catch {
      continue;
    }
  }

  return {
    maxSegmentSize,
    recommendedSegments: Math.ceil((8 * GB) / maxSegmentSize), // Target ~8GB address space
  };
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Main capability detection - call this at init
 */
export async function getMemoryCapabilities(): Promise<MemoryCapabilities> {
  const hasMemory64 = await probeMemory64();
  const unifiedMemory = await detectUnifiedMemory();
  const maxHeapSize = hasMemory64 ? await probeMaxHeapSize() : null;
  const segmentedLimits = !hasMemory64 ? probeSegmentedLimits() : null;

  const strategy: MemoryStrategy = hasMemory64 ? 'MEMORY64' : 'SEGMENTED';

  return {
    hasMemory64,
    isUnifiedMemory: unifiedMemory.isUnified,
    unifiedMemoryInfo: unifiedMemory,
    maxHeapSize,
    segmentedLimits,
    strategy,
  };
}
