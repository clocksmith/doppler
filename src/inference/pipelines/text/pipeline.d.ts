/**
 * text.d.ts - Main Text Inference Pipeline (Thin Orchestrator)
 *
 * This module orchestrates inference by delegating to specialized modules:
 * - state.js: Holds model configuration, weights, and runtime state
 * - generator.js: Handles token generation loops and decoding
 * - init.js: Initialization, weight loading, KV cache, RoPE
 *
 * The pipeline maintains state and coordinates the flow from input tokens to generated output.
 *
 * @module inference/pipelines/text
 */

import { PipelineState } from './state.js';
import { PipelineGenerator } from './generator.js';
import type { Manifest } from './config.js';
import type { WeightLoadResult, PipelineContexts } from './init.js';
import type { GenerateOptions, KVCacheSnapshot, LogitsStepResult, PrefillResult, PrefillEmbeddingResult, SequenceEncodeOptions, SequenceEncodeResult, AdvanceEmbeddingResult, LayerWeights, ExpertWeights, RouterWeights, GenerationResult, PipelineStats, BatchingStats, WorkloadPhaseTiming, NativeLoRAPrefillOptions, NativeLoRAPrefillResult } from './types.js';
import type { ChatMessage } from './chat-format.js';
import type { LoRAAdapter } from './lora.js';
import { getBufferPool as getGlobalBufferPool } from '../../../memory/buffer-pool.js';
import type { EmulationStats } from '../../../config/schema/index.js';
import type {
  DiffusionGemmaCanvasLogitsInput,
  DiffusionGemmaCanvasStepInput,
  DiffusionGemmaCanvasStepResult,
} from './generator.js';

// Re-export types for external use
export type { GenerateOptions, KVCacheSnapshot, LogitsStepResult, PrefillResult, PrefillEmbeddingResult, SequenceEncodeOptions, SequenceEncodeResult, AdvanceEmbeddingResult, LayerWeights, ExpertWeights, RouterWeights, GenerationResult, PipelineStats, BatchingStats, NativeLoRAPrefillOptions, NativeLoRAPrefillResult };
export type { PipelineContexts };

export interface ChatRequestInput {
  messages: ChatMessage[];
}

export type PromptInput = string | ChatMessage[] | ChatRequestInput;

export declare function buildConservativeMultimodalGenerationOptions(
  options?: GenerateOptions
): GenerateOptions;

export declare class AbortError extends Error {
  code: 'ABORT_ERR';
  constructor(message?: string);
}

export declare function isAbortError(err: unknown): boolean;

// ============================================================================
// Main Inference Pipeline Class
// ============================================================================

export declare class InferencePipeline extends PipelineState {
  private generator;

  // Progress callback
  private _onProgress;
  private _preloadedWeights;

  constructor();

  // ==========================================================================
  // Initialization
  // ==========================================================================

  initialize(contexts?: PipelineContexts): Promise<void>;

  loadModel(manifest: Manifest): Promise<void>;

  private _loadWeights(): Promise<void>;
  private _ensureVisionWeightsLoaded(): Promise<void>;
  private _ensureAudioWeightsLoaded(): Promise<void>;

  setPreloadedWeights(weights: WeightLoadResult): void;

  private _initRoPE(): Promise<void>;

  private _resolveLayerPipeline(): void;

  // ==========================================================================
  // Generation Delegates
  // ==========================================================================

  generate(prompt: PromptInput, options?: GenerateOptions): AsyncGenerator<string, void, void>;
  generateTokens(prompt: PromptInput, options?: GenerateOptions): AsyncGenerator<number, void, void>;
  generateTokenIds(
    prompt: PromptInput,
    options?: GenerateOptions
  ): Promise<{ tokenIds: number[]; stats: PipelineStats }>;

  resetToSeqLen(seqLen: number): void;
  resetGenerationState(): void;

  decodeStepLogits(currentIds: number[], options?: GenerateOptions): Promise<LogitsStepResult>;
  prefillWithToken(prompt: PromptInput, options: GenerateOptions, tokenContract: { padTokenId: number | null }): Promise<import('./generator/token-selection.js').SelectedTokenResult>;
  decodeStepWithToken(currentIds: number[], options: GenerateOptions, tokenContract: { padTokenId: number | null }): Promise<import('./generator/token-selection.js').SelectedTokenResult>;

  advanceWithToken(tokenId: number, options?: GenerateOptions): Promise<void>;

  advanceWithTokenAndEmbedding(tokenId: number, options?: GenerateOptions): Promise<AdvanceEmbeddingResult>;

  prefillKVOnly(prompt: PromptInput, options?: GenerateOptions): Promise<KVCacheSnapshot>;

  prefillForLoRATraining(
    inputIds: readonly number[] | Int32Array | Uint32Array,
    options: NativeLoRAPrefillOptions
  ): Promise<NativeLoRAPrefillResult>;

  computeDiffusionGemmaCanvasLogits(
    args: DiffusionGemmaCanvasLogitsInput,
    options?: GenerateOptions & { __internalGenerate?: boolean }
  ): Promise<Float32Array>;

  computeDiffusionGemmaCanvasStep(
    args: DiffusionGemmaCanvasStepInput,
    options?: GenerateOptions & { __internalGenerate?: boolean }
  ): Promise<DiffusionGemmaCanvasStepResult>;

  prefillWithEmbedding(prompt: PromptInput, options?: GenerateOptions): Promise<PrefillEmbeddingResult>;

  embed(prompt: string, options?: GenerateOptions): Promise<{
    embedding: Float32Array;
    tokens: number[];
    seqLen: number;
    embeddingMode: string;
    phase?: WorkloadPhaseTiming | null;
  }>;

  embedBatch(prompts: string[], options?: GenerateOptions): Promise<Array<{
    embedding: Float32Array;
    tokens: number[];
    seqLen: number;
    embeddingMode: string;
  }>>;

  encodeSequence(sequence: string, options?: SequenceEncodeOptions): Promise<SequenceEncodeResult>;

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

  applyKVCacheSnapshot(snapshot: KVCacheSnapshot): void;

  generateWithPrefixKV(
    prefix: KVCacheSnapshot,
    prompt: PromptInput,
    options?: GenerateOptions
  ): AsyncGenerator<string, void, void>;

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  getStats(): PipelineStats;

  getBatchingStats(): BatchingStats;

  getMemoryStats(): {
    used: number;
    pool?: { currentBytesAllocated?: number; peakBytesAllocated?: number; activeBuffers?: number; pooledBuffers?: number };
    kvCache?: {
      allocated?: number;
      used?: number;
      seqLen?: number;
      maxSeqLen?: number;
      layout?: string | null;
      kvDtype?: string | null;
      counters?: Record<string, unknown> | null;
    };
    emulation?: EmulationStats;
  };

  getKVCacheStats(): { seqLen: number; maxSeqLen: number } | null;

  getBufferPool(): ReturnType<typeof getGlobalBufferPool> | null;

  unload(): Promise<void>;

  /** Rejects changes during generation or after unloading; retains revocation checks. */
  setLoRAAdapter(adapter: LoRAAdapter | null): void;

  getActiveLoRA(): LoRAAdapter | null;

  reset(): void;

  releaseGPUResources(): void;
}

// ============================================================================
// Factory Function
// ============================================================================

export declare class EmbeddingPipeline extends InferencePipeline {
  generate(prompt: PromptInput, options?: GenerateOptions): AsyncGenerator<string, void, void>;
}

export { InferencePipeline as Pipeline };
