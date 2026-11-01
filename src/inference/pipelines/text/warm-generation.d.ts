import type { KernelWarmupGenerationSchema } from '../../../config/schema/kernel-warmup.schema.js';

export interface WarmGenerationResult {
  generatedTokens: number;
  maxTokens: number;
  /** Why generation ended (`stats.stopReason`), e.g. a stop token before maxTokens. */
  stopReason: string | null;
  /** A whole-token program is built and kept for the next generation. */
  wholeTokenProgramReady: boolean;
}

export interface WarmGenerationPipeline {
  generator: { generateTokens(prompt: string, options: Record<string, unknown>): AsyncIterable<unknown> };
  stats: { stopReason?: string | null };
  wholeTokenProgram?: { program: { ready: boolean } } | null;
  reset(): void;
}

/**
 * Run the configured greedy warm generation on a loaded pipeline (raw prompt, no chat
 * template), then reset its sequence state. Throws on an invalid policy or a failed
 * generation; ending early on a stop token is reported, not an error.
 */
export declare function runWarmGeneration(
  pipeline: WarmGenerationPipeline,
  warmGeneration: KernelWarmupGenerationSchema
): Promise<WarmGenerationResult>;
