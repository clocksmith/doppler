/**
 * Unified Memory Detection
 * Agent-A | Domain: memory/
 *
 * Detects if system has unified memory (CPU/GPU share RAM):
 * - Apple Silicon (M1/M2/M3/M4/M5)
 * - AMD Strix Halo (Ryzen AI Max)
 * - Other APUs with large shared memory
 *
 * @module memory/unified-detect
 */

/**
 * Apple Silicon detection result
 */
export interface AppleSiliconInfo {
  isApple: boolean;
  mSeriesGen?: number | null;
  vendor?: string;
  device?: string;
  description?: string;
}

/**
 * AMD unified memory detection result
 */
export interface AMDUnifiedInfo {
  isAMDUnified: boolean;
  isStrix?: boolean;
  vendor?: string;
  device?: string;
  description?: string;
}

/**
 * WebGPU buffer limit indicators
 */
export interface LimitIndicators {
  largeBuffers: boolean;
  maxBufferSize?: number;
  maxStorageBufferBindingSize?: number;
}

/**
 * Unified memory detection result
 */
export interface UnifiedMemoryInfo {
  isUnified: boolean;
  apple?: AppleSiliconInfo;
  amd?: AMDUnifiedInfo;
  limits?: LimitIndicators;
  estimatedMemoryGB?: number | null;
  reason: string;
}

/**
 * Main unified memory detection
 */
export function detectUnifiedMemory(): Promise<UnifiedMemoryInfo>;
