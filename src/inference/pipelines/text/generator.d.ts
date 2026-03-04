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
import type { LogitsConfig, LogitsWeights } from './logits/index.js';
import type { WeightBufferConfig } from './weights.js';
import type { ChatMessage } from './chat-format.js';

export interface ChatRequestInput {
  messages: ChatMessage[];
}

export type PromptInput = string | ChatMessage[] | ChatRequestInput;

export declare class PipelineGenerator {
  constructor(state: PipelineState);

  /**
   * Batching and readback cadence are controlled by runtime.inference.batching.
   */
  generate(prompt: PromptInput, options?: GenerateOptions): AsyncGenerator<string, void, void>;
  prefillKVOnly(prompt: PromptInput, options?: GenerateOptions): Promise<KVCacheSnapshot>;
  prefillWithEmbedding(prompt: PromptInput, options?: GenerateOptions): Promise<PrefillEmbeddingResult>;
  prefillWithLogits(prompt: PromptInput, options?: GenerateOptions): Promise<PrefillResult>;
  decodeStepLogits(currentIds: number[], options?: GenerateOptions): Promise<LogitsStepResult>;
  advanceWithToken(tokenId: number, options?: GenerateOptions): Promise<void>;
  advanceWithTokenAndEmbedding(tokenId: number, options?: GenerateOptions): Promise<AdvanceEmbeddingResult>;
  generateWithPrefixKV(
    prefix: KVCacheSnapshot,
    prompt: PromptInput,
    options?: GenerateOptions
  ): AsyncGenerator<string, void, void>;
}
