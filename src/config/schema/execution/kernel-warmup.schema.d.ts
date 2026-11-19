/**
 * Kernel Warmup Config Schema
 *
 * Controls optional kernel prewarming.
 *
 * @module config/schema/kernel-warmup
 */

export interface KernelWarmupConfigSchema {
  /** Precompile pipelines for all supported kernel variants. */
  prewarm: boolean;
  /** Prewarm scheduling mode (parallel or sequential). */
  prewarmMode: 'parallel' | 'sequential';
  /**
   * One greedy generation at the end of a text pipeline load, after the weights: the
   * first request then finds this model's pipelines compiled and, with
   * decodeLoop.wholeTokenProgram, a built program for its first decode step to verify.
   * Counted in load time (load phase "warmGeneration"); sequence state is reset after it.
   * null runs none.
   */
  warmGeneration: KernelWarmupGenerationSchema | null;
  /**
   * One prompt pass through a resident partition's own layers while the partition opens: its
   * pipelines are compiled and, on a routed-expert stage, the experts those tokens select are
   * loaded before the first request. Runs in a private attempt that is closed before readiness.
   * Counted in opening time. null runs none.
   */
  warmPartition: KernelWarmupPartitionSchema | null;
}

export interface KernelWarmupPartitionSchema {
  /** Token ids of the warm prompt. A stage that starts from a boundary activation runs zeros of this length. */
  tokenIds: number[];
  /** Single-token steps after the prompt pass (0 to 8). */
  decodeSteps: number;
}

export interface KernelWarmupGenerationSchema {
  /** Raw prompt text, tokenized without the chat template. */
  prompt: string;
  /** Tokens to generate (at least 2); a stop token may end the generation earlier. */
  maxTokens: number;
}

/** Default kernel warmup configuration */
export declare const DEFAULT_KERNEL_WARMUP_CONFIG: KernelWarmupConfigSchema;
