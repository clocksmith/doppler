/**
 * Pipeline Generation Logic
 *
 * Handles the token generation loop, batching, and decoding strategies.
 * Separated from main pipeline to isolate execution logic from state management.
 *
 * @module inference/pipelines/text/generator
 */

import type { CommandRecorder, ProfileTimings } from '../../../gpu/command-recorder.js';
import type { PipelineState } from './state.js';
import type { GenerateOptions, KVCacheSnapshot, LogitsStepResult, PrefillResult, PrefillEmbeddingResult, AdvanceEmbeddingResult, LayerContext } from './types.js';
import type { LogitsConfig, LogitsWeights } from './logits.js';
import type { WeightBufferConfig } from './weights.js';

export declare class PipelineGenerator {
  constructor(state: PipelineState);

  /**
   * Batching and readback cadence are controlled by runtime.inference.batching.
   */
  generate(prompt: string, options?: GenerateOptions): AsyncGenerator<string, void, void>;
  prefillKVOnly(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;
  prefillWithEmbedding(prompt: string, options?: GenerateOptions): Promise<PrefillEmbeddingResult>;
  prefillWithLogits(prompt: string, options?: GenerateOptions): Promise<PrefillResult>;
  decodeStepLogits(currentIds: number[], options?: GenerateOptions): Promise<LogitsStepResult>;
  advanceWithToken(tokenId: number, options?: GenerateOptions): Promise<void>;
  advanceWithTokenAndEmbedding(tokenId: number, options?: GenerateOptions): Promise<AdvanceEmbeddingResult>;
  generateWithPrefixKV(
    prefix: KVCacheSnapshot,
    prompt: string,
    options?: GenerateOptions
  ): AsyncGenerator<string, void, void>;
}
