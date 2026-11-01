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
}

export interface KernelWarmupGenerationSchema {
  /** Raw prompt text, tokenized without the chat template. */
  prompt: string;
  /** Tokens to generate (at least 2); a stop token may end the generation earlier. */
  maxTokens: number;
}

/** Default kernel warmup configuration */
export declare const DEFAULT_KERNEL_WARMUP_CONFIG: KernelWarmupConfigSchema;
