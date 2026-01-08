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

import { UnifiedMemoryInfo } from './unified-detect.js';

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

/**
 * Main capability detection - call this at init
 */
export function getMemoryCapabilities(): Promise<MemoryCapabilities>;
