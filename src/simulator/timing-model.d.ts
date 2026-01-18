/**
 * Timing Model for NVIDIA Superchip Simulation
 *
 * Provides delay injection to simulate realistic compute and memory
 * timing for GH200/GB200 superchips based on theoretical peak performance.
 *
 * @module simulator/timing-model
 */

import type { EmulationConfigSchema, EmulatedGPUSpec } from '../config/schema/emulation.schema.js';

// =============================================================================
// Timing Results
// =============================================================================

/**
 * Result of a timing calculation
 */
export interface TimingResult {
  /** Calculated time in milliseconds */
  timeMs: number;
  /** Breakdown of timing components */
  breakdown?: TimingBreakdown;
}

/**
 * Breakdown of timing for complex operations
 */
export interface TimingBreakdown {
  /** Compute time in ms */
  computeMs: number;
  /** Memory access time in ms */
  memoryMs: number;
  /** Communication time in ms */
  communicationMs: number;
}

// =============================================================================
// Timing Model
// =============================================================================

/**
 * Timing model for emulated NVIDIA hardware
 */
export declare class TimingModel {
  /** GPU specification for timing calculations */
  readonly gpuSpec: EmulatedGPUSpec;

  /** Timing mode */
  readonly mode: 'functional' | 'timed' | 'hybrid';

  /** Timing scaling factors */
  readonly scaling: {
    computeScale: number;
    memoryScale: number;
    nvlinkScale: number;
  };

  constructor(config: EmulationConfigSchema);

  /**
   * Calculate time for matrix multiplication
   * @param M - Output rows
   * @param N - Output columns
   * @param K - Inner dimension
   * @param dtype - Data type ('f16', 'f32', 'fp8')
   * @param gpuCount - Number of GPUs (for tensor parallel)
   */
  computeMatmulTimeMs(M: number, N: number, K: number, dtype?: string, gpuCount?: number): TimingResult;

  /**
   * Calculate time for attention operation
   * @param seqLen - Sequence length
   * @param numHeads - Number of attention heads
   * @param headDim - Dimension per head
   * @param batchSize - Batch size
   */
  computeAttentionTimeMs(
    seqLen: number,
    numHeads: number,
    headDim: number,
    batchSize?: number
  ): TimingResult;

  /**
   * Calculate time for memory transfer
   * @param sizeBytes - Transfer size in bytes
   * @param bandwidthBytesPerSec - Bandwidth (defaults to HBM bandwidth)
   */
  computeMemoryTimeMs(sizeBytes: number, bandwidthBytesPerSec?: number): TimingResult;

  /**
   * Calculate time for NVLink transfer
   * @param sizeBytes - Transfer size in bytes
   * @param srcGpu - Source GPU index
   * @param dstGpu - Destination GPU index
   * @param nvlinkBandwidth - NVLink bandwidth (defaults to config)
   */
  computeNvlinkTimeMs(
    sizeBytes: number,
    srcGpu: number,
    dstGpu: number,
    nvlinkBandwidth?: number
  ): TimingResult;

  /**
   * Calculate time for all-reduce operation
   * @param sizeBytes - Buffer size in bytes
   * @param gpuCount - Number of GPUs participating
   */
  computeAllReduceTimeMs(sizeBytes: number, gpuCount: number): TimingResult;

  /**
   * Calculate time for all-gather operation
   * @param sizeBytes - Per-GPU buffer size in bytes
   * @param gpuCount - Number of GPUs participating
   */
  computeAllGatherTimeMs(sizeBytes: number, gpuCount: number): TimingResult;

  /**
   * Calculate time for reduce-scatter operation
   * @param sizeBytes - Total buffer size in bytes
   * @param gpuCount - Number of GPUs participating
   */
  computeReduceScatterTimeMs(sizeBytes: number, gpuCount: number): TimingResult;

  /**
   * Calculate time for embedding lookup
   * @param vocabSize - Vocabulary size
   * @param embeddingDim - Embedding dimension
   * @param tokenCount - Number of tokens to look up
   */
  computeEmbeddingTimeMs(vocabSize: number, embeddingDim: number, tokenCount: number): TimingResult;

  /**
   * Calculate time for RMSNorm
   * @param seqLen - Sequence length
   * @param hiddenDim - Hidden dimension
   */
  computeRmsNormTimeMs(seqLen: number, hiddenDim: number): TimingResult;

  /**
   * Inject delay based on timing calculation
   * Only delays in 'timed' mode, returns immediately in 'functional' mode
   * @param timeMs - Time to delay
   */
  injectDelay(timeMs: number): Promise<void>;

  /**
   * Get total injected delay so far
   */
  getTotalInjectedDelayMs(): number;

  /**
   * Reset timing statistics
   */
  reset(): void;
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a timing model from emulation config
 */
export declare function createTimingModel(config: EmulationConfigSchema): TimingModel;

/**
 * Calculate theoretical FLOPS for a GPU spec
 * @param spec - GPU specification
 * @param dtype - Data type
 */
export declare function calculateTheoreticalFlops(spec: EmulatedGPUSpec, dtype?: string): number;

/**
 * Convert FLOPS to time for a given operation count
 * @param flops - Operation count
 * @param theoreticalFlops - Peak FLOPS
 * @param efficiency - Utilization efficiency (0-1, default 0.7)
 */
export declare function flopsToTimeMs(
  flops: number,
  theoreticalFlops: number,
  efficiency?: number
): number;
