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
 * Unified memory detection result
 */
export interface UnifiedMemoryInfo {
  isUnified: boolean;
  apple?: {
    isApple: boolean;
    mSeriesGen?: number | null;
    vendor?: string;
    device?: string;
    description?: string;
  };
  amd?: {
    isAMDUnified: boolean;
    isStrix?: boolean;
    vendor?: string;
    device?: string;
    description?: string;
  };
  limits?: {
    largeBuffers: boolean;
    maxBufferSize?: number;
    maxStorageBufferBindingSize?: number;
  };
  estimatedMemoryGB?: number | null;
  reason: string;
}

/**
 * Main unified memory detection
 */
export function detectUnifiedMemory(): Promise<UnifiedMemoryInfo>;
