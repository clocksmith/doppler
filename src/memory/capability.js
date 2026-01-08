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

import { detectUnifiedMemory } from './unified-detect.js';
import { getRuntimeConfig } from '../config/runtime.js';

// ============================================================================
// Detection Functions
// ============================================================================

/**
 * Memory64 feature detection via WASM binary probe
 * Tests if browser supports 64-bit memory addressing
 * @returns {Promise<boolean>}
 */
async function probeMemory64() {
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
 * @returns {Promise<number>}
 */
async function probeMaxHeapSize() {
  const { heapTestSizes, fallbackMaxHeapBytes } = getRuntimeConfig().memory.heapTesting;

  for (const size of heapTestSizes) {
    try {
      // Try to create a WASM memory of this size
      const pages = Math.ceil(size / 65536); // 64KB pages
      new WebAssembly.Memory({ initial: 1, maximum: pages });
      return size;
    } catch {
      continue;
    }
  }

  return fallbackMaxHeapBytes;
}

/**
 * Probe segmented heap limits (for non-Memory64 browsers)
 * Returns max size per ArrayBuffer and recommended segment count
 * @returns {import('./capability.js').SegmentedLimits}
 */
function probeSegmentedLimits() {
  const { segmentTestSizes, safeSegmentSizeBytes } = getRuntimeConfig().memory.segmentTesting;
  const { targetAddressSpaceBytes } = getRuntimeConfig().memory.addressSpace;

  let maxSegmentSize = safeSegmentSizeBytes; // Safe default

  for (const size of segmentTestSizes) {
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
    recommendedSegments: Math.ceil(targetAddressSpaceBytes / maxSegmentSize),
  };
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Main capability detection - call this at init
 * @returns {Promise<import('./capability.js').MemoryCapabilities>}
 */
export async function getMemoryCapabilities() {
  const hasMemory64 = await probeMemory64();
  const unifiedMemory = await detectUnifiedMemory();
  const maxHeapSize = hasMemory64 ? await probeMaxHeapSize() : null;
  const segmentedLimits = !hasMemory64 ? probeSegmentedLimits() : null;

  /** @type {import('./capability.js').MemoryStrategy} */
  const strategy = hasMemory64 ? 'MEMORY64' : 'SEGMENTED';

  return {
    hasMemory64,
    isUnifiedMemory: unifiedMemory.isUnified,
    unifiedMemoryInfo: unifiedMemory,
    maxHeapSize,
    segmentedLimits,
    strategy,
  };
}
