/**
 * pipeline.ts - Main Inference Pipeline (Thin Orchestrator)
 *
 * This module orchestrates inference by delegating to specialized modules:
 * - state.ts: Holds model configuration, weights, and runtime state
 * - generator.ts: Handles token generation loops and decoding
 * - init.ts: Initialization, weight loading, KV cache, RoPE
 *
 * The pipeline maintains state and coordinates the flow from input tokens to generated output.
 *
 * @module inference/pipeline
 */

import { PipelineState } from './pipeline/state.js';
import { PipelineGenerator } from './pipeline/generator.js';
import type { Manifest } from './pipeline/config.js';
import type { WeightLoadResult, PipelineContexts } from './pipeline/init.js';
import type { GenerateOptions, KVCacheSnapshot, LayerWeights, ExpertWeights, RouterWeights, GenerationResult, PipelineStats, BatchingStats } from './pipeline/types.js';
import type { LoRAAdapter } from './pipeline/lora.js';
import { getBufferPool as getGlobalBufferPool } from '../gpu/buffer-pool.js';

// Re-export types for external use
export type { GenerateOptions, KVCacheSnapshot, LayerWeights, ExpertWeights, RouterWeights, GenerationResult, PipelineStats, BatchingStats };
export type { PipelineContexts };

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

  setPreloadedWeights(weights: WeightLoadResult): void;

  private _initRoPE(): Promise<void>;

  private _resolveLayerPipeline(): void;

  // ==========================================================================
  // Generation Delegates
  // ==========================================================================

  generate(prompt: string, options?: GenerateOptions): AsyncGenerator<string, void, void>;

  prefillKVOnly(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;

  applyKVCacheSnapshot(snapshot: KVCacheSnapshot): void;

  generateWithPrefixKV(
    prefix: KVCacheSnapshot,
    prompt: string,
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

export declare function createPipeline(manifest: Manifest, contexts?: PipelineContexts): Promise<InferencePipeline>;

export { InferencePipeline as Pipeline };
