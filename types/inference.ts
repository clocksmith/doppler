/**
 * Inference Pipeline Types
 */

import type { ModelConfig } from './model.js';

/** Inference configuration */
export interface InferenceConfig {
  /** Maximum tokens to generate */
  maxTokens: number;
  /** Sampling temperature (0 = greedy) */
  temperature: number;
  /** Top-k sampling (0 = disabled) */
  topK: number;
  /** Top-p (nucleus) sampling */
  topP: number;
  /** Repetition penalty */
  repetitionPenalty: number;
  /** Stop sequences */
  stopSequences?: string[];
  /** Stop token IDs */
  stopTokenIds?: number[];
  /** Enable streaming */
  stream: boolean;
}

/** Default inference configuration values */
export const DEFAULT_INFERENCE_CONFIG: InferenceConfig = {
  maxTokens: 256,
  temperature: 0.7,
  topK: 40,
  topP: 0.9,
  repetitionPenalty: 1.0,
  stream: true,
};

/** Sampling configuration */
export interface SamplingConfig {
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  minP?: number;
}

/** Sampling result */
export interface SamplingResult {
  tokenId: number;
  logprob: number;
  topLogprobs?: Array<{ tokenId: number; logprob: number }>;
}

/** Generation result */
export interface GenerationResult {
  text: string;
  tokens: number[];
  finishReason: FinishReason;
  usage: UsageStats;
}

/** Reason generation finished */
export type FinishReason = 'stop' | 'length' | 'eos' | 'error';

/** Token usage statistics */
export interface UsageStats {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

/** Performance statistics */
export interface PerfStats {
  prefillTimeMs: number;
  decodeTimeMs: number;
  totalTimeMs: number;
  tokensPerSecond: number;
  prefillTokensPerSecond: number;
  timeToFirstToken: number;
}

/** KV Cache configuration */
export interface KVCacheConfig {
  maxSeqLen: number;
  numLayers: number;
  numKVHeads: number;
  headDim: number;
  dtype: 'f16' | 'f32';
  slidingWindow?: number;
}

/** KV Cache interface */
export interface KVCache {
  readonly config: KVCacheConfig;
  readonly seqLen: number;

  /** Get K buffer for a layer */
  getK(layer: number): GPUBuffer;

  /** Get V buffer for a layer */
  getV(layer: number): GPUBuffer;

  /** Advance sequence position */
  advance(numTokens: number): void;

  /** Reset cache */
  reset(): void;

  /** Get memory usage in bytes */
  getMemoryUsage(): number;
}

/** MoE expert routing plan */
export interface ExpertPlan {
  /** Expert indices for each token [batchSize * topK] */
  expertIndices: Uint32Array;
  /** Routing weights for each token [batchSize * topK] */
  weights: Float32Array;
  /** Number of tokens routed to each expert */
  expertCounts: Uint32Array;
  /** Token indices sorted by expert */
  sortedTokenIndices: Uint32Array;
}

/** MoE router configuration */
export interface RouterConfig {
  numExperts: number;
  numExpertsPerToken: number;
  routerJitter?: number;
  capacityFactor?: number;
}

/** Tokenizer backend interface */
export interface TokenizerBackend {
  /** Encode text to token IDs */
  encode(text: string): number[];

  /** Decode token IDs to text */
  decode(tokens: number[]): string;

  /** Decode a single token */
  decodeToken(tokenId: number): string;

  /** Get vocabulary size */
  vocabSize(): number;

  /** Get special token IDs */
  specialTokens(): SpecialTokens;
}

/** Special token IDs */
export interface SpecialTokens {
  bos?: number;
  eos?: number;
  pad?: number;
  unk?: number;
  mask?: number;
}

/** Pipeline initialization options */
export interface PipelineOptions {
  /** Model configuration */
  config: ModelConfig;
  /** GPU device */
  device: GPUDevice;
  /** Storage context for loading shards */
  storage: StorageContext;
  /** Runtime options */
  runtime?: RuntimeOptions;
  /** Progress callback */
  onProgress?: ProgressCallback;
}

/** Storage context for shard loading */
export interface StorageContext {
  loadShard(index: number): Promise<Uint8Array>;
}

/** Runtime options */
export interface RuntimeOptions {
  debug?: boolean;
  profile?: boolean;
  kvCacheSize?: number;
  attentionKernel?: 'auto' | 'tiled_large' | 'tiled_small' | 'streaming';
}

/** Progress callback */
export type ProgressCallback = (
  phase: ProgressPhase,
  progress: number,
  detail?: string
) => void;

/** Progress phases */
export type ProgressPhase =
  | 'init'
  | 'download'
  | 'load'
  | 'warmup'
  | 'prefill'
  | 'decode';

/** Generation event (for streaming) */
export type GenerationEvent =
  | { type: 'token'; token: string; tokenId: number }
  | { type: 'done'; result: GenerationResult }
  | { type: 'error'; error: Error };
