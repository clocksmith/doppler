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
import type { GenerateOptions, KVCacheSnapshot, LogitsStepResult, PrefillResult, PrefillEmbeddingResult, AdvanceEmbeddingResult, LayerContext, WorkloadPhaseTiming } from './types.js';
import type { LogitsConfig, LogitsWeights } from './logits/index.js';
import type { WeightBufferConfig } from './weights.js';
import type { ChatMessage } from './chat-format.js';

export interface ChatRequestInput {
  messages: ChatMessage[];
}

export type PromptInput = string | ChatMessage[] | ChatRequestInput;

export interface DiffusionGemmaCanvasLogitsInput {
  canvas: number[] | Int32Array | Uint32Array;
  selfConditioningLogits?: Float32Array | number[] | DiffusionGemmaGpuLogitsState | null;
}

export interface DiffusionGemmaGpuLogitsState {
  logitsBuffer: GPUBuffer;
  logitsDtype: 'f32';
  vocabSize: number;
  canvasLength: number;
  temperature: number;
  releaseOnUse?: boolean;
  release?: () => void;
}

export interface DiffusionGemmaCanvasStepInput extends DiffusionGemmaCanvasLogitsInput {
  temperature: number;
}

export interface DiffusionGemmaCanvasStepResult {
  argmaxCanvas: Int32Array;
  entropies: Float32Array;
  selfConditioningLogits: DiffusionGemmaGpuLogitsState;
}

export declare function resolvePrefillChunkSubmitMode(
  runtimeConfig: unknown,
  modelConfig: unknown
): 'sync' | 'async';

export declare class PipelineGenerator {
  constructor(state: PipelineState);

  /**
   * Batching and readback cadence are controlled by runtime.inference.batching.
   */
  generate(prompt: PromptInput, options?: GenerateOptions): AsyncGenerator<string, void, void>;
  generateTokens(prompt: PromptInput, options?: GenerateOptions): AsyncGenerator<number, void, void>;
  generateTokenIds(
    prompt: PromptInput,
    options?: GenerateOptions
  ): Promise<{ tokenIds: number[]; stats: import('./types.js').PipelineStats }>;
  resetGenerationState(): void;
  resetToSeqLen(seqLen: number): void;
  prefillKVOnly(prompt: PromptInput, options?: GenerateOptions): Promise<KVCacheSnapshot>;
  computeDiffusionGemmaCanvasLogits(
    args: DiffusionGemmaCanvasLogitsInput,
    options?: GenerateOptions & { __internalGenerate?: boolean }
  ): Promise<Float32Array>;
  computeDiffusionGemmaCanvasStep(
    args: DiffusionGemmaCanvasStepInput,
    options?: GenerateOptions & { __internalGenerate?: boolean }
  ): Promise<DiffusionGemmaCanvasStepResult>;
  prefillWithEmbedding(prompt: PromptInput, options?: GenerateOptions): Promise<PrefillEmbeddingResult>;
  prefillWithLogits(prompt: PromptInput, options?: GenerateOptions): Promise<PrefillResult>;
  prefillWithTokenLogits(prompt: PromptInput, tokenIds: readonly number[], options?: GenerateOptions): Promise<{
    seqLen: number;
    tokens: number[];
    tokenIds: number[];
    logits: Float32Array;
    logitsByTokenId: Record<number, number>;
    phase?: WorkloadPhaseTiming | null;
  }>;
  prefillWithTokenLogitsFromKV(prefix: KVCacheSnapshot, prompt: PromptInput, tokenIds: readonly number[], options?: GenerateOptions): Promise<{
    seqLen: number;
    prefixTokens: number[];
    tokens: number[];
    tokenIds: number[];
    logits: Float32Array;
    logitsByTokenId: Record<number, number>;
    phase?: WorkloadPhaseTiming | null;
  }>;
  decodeStepLogits(currentIds: number[], options?: GenerateOptions): Promise<LogitsStepResult>;
  advanceWithToken(tokenId: number, options?: GenerateOptions): Promise<void>;
  advanceWithTokenAndEmbedding(tokenId: number, options?: GenerateOptions): Promise<AdvanceEmbeddingResult>;
  generateWithPrefixKV(
    prefix: KVCacheSnapshot,
    prompt: PromptInput,
    options?: GenerateOptions
  ): AsyncGenerator<string, void, void>;
}
