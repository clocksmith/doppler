/**
 * Logit Merge Kernel
 *
 * GPU primitive for merging logits from multiple models in multi-model inference.
 * This is a mechanism (how to merge), not policy (when/what to merge).
 * Reploid decides the weights and strategy; Doppler executes the merge.
 *
 * @module gpu/kernels/logit-merge
 */

/**
 * Merge strategy types.
 */
export type MergeStrategy = 'weighted' | 'max' | 'geometric';

/**
 * Configuration for logit merging.
 */
export interface LogitMergeConfig {
  /** Merge strategy */
  strategy: MergeStrategy;
  /** Weights for weighted merge (must sum to a positive value) */
  weights: number[];
  /** Temperature to apply after merge */
  temperature: number;
}

/**
 * Logit merge kernel executor.
 */
export declare class LogitMergeKernel {
  constructor();

  /**
   * Initialize the kernel pipelines.
   */
  init(): Promise<void>;

  /**
   * Merge two logit buffers on the GPU.
   *
   * @param logitsA - First logit buffer (vocabSize f32s)
   * @param logitsB - Second logit buffer (vocabSize f32s)
   * @param vocabSize - Vocabulary size
   * @param config - Merge configuration
   * @returns Merged logit buffer
   */
  merge(
    logitsA: GPUBuffer,
    logitsB: GPUBuffer,
    vocabSize: number,
    config: LogitMergeConfig
  ): Promise<GPUBuffer>;

  /**
   * Merge multiple logit buffers on the GPU.
   * Applies pairwise merging for more than 2 buffers.
   *
   * @param logitBuffers - Array of logit buffers
   * @param vocabSize - Vocabulary size
   * @param config - Merge configuration
   * @returns Merged logit buffer
   */
  mergeMultiple(
    logitBuffers: GPUBuffer[],
    vocabSize: number,
    config: LogitMergeConfig
  ): Promise<GPUBuffer>;
}

/**
 * Get the logit merge kernel instance.
 */
export declare function getLogitMergeKernel(): LogitMergeKernel;

/**
 * Merge two logit buffers using weighted averaging.
 * Convenience function that uses the singleton kernel.
 *
 * @param logitsA - First logit buffer
 * @param logitsB - Second logit buffer
 * @param vocabSize - Vocabulary size
 * @param weights - Merge weights
 * @param temperature - Temperature scaling
 * @returns Merged logit buffer
 */
export declare function mergeLogits(
  logitsA: GPUBuffer,
  logitsB: GPUBuffer,
  vocabSize: number,
  weights: number[],
  temperature: number
): Promise<GPUBuffer>;

/**
 * Merge multiple logit buffers.
 * Convenience function that uses the singleton kernel.
 *
 * @param logitBuffers - Array of logit buffers
 * @param vocabSize - Vocabulary size
 * @param weights - Merge weights
 * @param temperature - Temperature scaling
 * @returns Merged logit buffer
 */
export declare function mergeMultipleLogits(
  logitBuffers: GPUBuffer[],
  vocabSize: number,
  weights: number[],
  temperature: number
): Promise<GPUBuffer>;
