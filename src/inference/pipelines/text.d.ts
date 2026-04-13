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

import { PipelineState } from './text/state.js';
import { PipelineGenerator } from './text/generator.js';
import type { Manifest } from './text/config.js';
import type { WeightLoadResult, PipelineContexts } from './text/init.js';
import type { GenerateOptions, KVCacheSnapshot, LogitsStepResult, PrefillResult, PrefillEmbeddingResult, AdvanceEmbeddingResult, LayerWeights, ExpertWeights, RouterWeights, GenerationResult, PipelineStats, BatchingStats } from './text/types.js';
import type { ChatMessage } from './text/chat-format.js';
import type { LoRAAdapter } from './text/lora.js';
import type { DiffusionPipeline } from './experimental/diffusion/pipeline.js';
import type { EnergyPipeline } from './experimental/energy/pipeline.js';
import type { StructuredJsonHeadPipeline } from './structured/json-head-pipeline.js';
import type { EnergyRowHeadPipeline } from './energy-head/row-head-pipeline.js';
import { getBufferPool as getGlobalBufferPool } from '../../memory/buffer-pool.js';
import type { EmulationStats } from '../../config/schema/index.js';

// Re-export types for external use
export type { GenerateOptions, KVCacheSnapshot, LogitsStepResult, PrefillResult, PrefillEmbeddingResult, AdvanceEmbeddingResult, LayerWeights, ExpertWeights, RouterWeights, GenerationResult, PipelineStats, BatchingStats };
export type { PipelineContexts };

export interface ChatRequestInput {
  messages: ChatMessage[];
}

export type PromptInput = string | ChatMessage[] | ChatRequestInput;

export declare function buildConservativeMultimodalGenerationOptions(
  options?: GenerateOptions
): GenerateOptions;

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

  decodeStepLogits(currentIds: number[], options?: GenerateOptions): Promise<LogitsStepResult>;

  advanceWithToken(tokenId: number, options?: GenerateOptions): Promise<void>;

  advanceWithTokenAndEmbedding(tokenId: number, options?: GenerateOptions): Promise<AdvanceEmbeddingResult>;

  prefillKVOnly(prompt: PromptInput, options?: GenerateOptions): Promise<KVCacheSnapshot>;

  prefillWithEmbedding(prompt: PromptInput, options?: GenerateOptions): Promise<PrefillEmbeddingResult>;

  embed(prompt: string, options?: GenerateOptions): Promise<{
    embedding: Float32Array;
    tokens: number[];
    seqLen: number;
    embeddingMode: string;
  }>;

  embedBatch(prompts: string[], options?: GenerateOptions): Promise<Array<{
    embedding: Float32Array;
    tokens: number[];
    seqLen: number;
    embeddingMode: string;
  }>>;

  prefillWithLogits(prompt: PromptInput, options?: GenerateOptions): Promise<PrefillResult>;

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
    kvCache?: { allocated?: number; used?: number; seqLen?: number; maxSeqLen?: number };
    emulation?: EmulationStats;
  };

  getKVCacheStats(): { seqLen: number; maxSeqLen: number } | null;

  getBufferPool(): ReturnType<typeof getGlobalBufferPool> | null;

  unload(): Promise<void>;

  setLoRAAdapter(adapter: LoRAAdapter | null): void;

  getActiveLoRA(): LoRAAdapter | null;

  reset(): void;

  releaseGPUResources(): void;
}

// ============================================================================
// Factory Function
// ============================================================================

export declare function createPipeline(
  manifest: Manifest,
  contexts?: PipelineContexts
): Promise<
  InferencePipeline |
  EmbeddingPipeline |
  DiffusionPipeline |
  EnergyPipeline |
  StructuredJsonHeadPipeline |
  EnergyRowHeadPipeline
>;

export declare class EmbeddingPipeline extends InferencePipeline {
  generate(prompt: PromptInput, options?: GenerateOptions): AsyncGenerator<string, void, void>;
}

export { InferencePipeline as Pipeline };
