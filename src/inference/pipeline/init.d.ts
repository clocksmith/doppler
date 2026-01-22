/**
 * Pipeline initialization - model loading, tokenizer setup, KV cache, RoPE.
 *
 * This module handles all initialization tasks for the inference pipeline:
 * - Loading model manifest and parsing configuration
 * - Initializing tokenizer
 * - Setting up KV cache (standard or sliding window)
 * - Computing RoPE frequency buffers (linear or YARN scaling)
 * - Loading model weights via DopplerLoader
 * - Setting up MoE router if applicable
 *
 * @module inference/pipeline/init
 */

import type { ParsedModelConfig, Manifest } from './config.js';
import type { KernelCapabilities } from '../../gpu/device.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../gpu/weight-buffer.js';
import { KVCache, SlidingWindowKVCache, TieredKVCache } from '../kv-cache.js';
import { Tokenizer, type ModelManifest as TokenizerManifest } from '../tokenizer.js';
import { MoERouter } from '../moe-router.js';
import { SpeculativeDecoder } from '../speculative.js';
import type { LayerWeights, RouterWeights } from './types.js';
import type {
  KVCacheConfigSchema,
  RuntimeConfigSchema,
  LoadingConfigSchema,
  KernelPathRef,
  MoERoutingConfigSchema,
  SpeculativeConfigSchema,
} from '../../config/schema/index.js';

/**
 * External contexts that can be injected into the pipeline.
 */
export interface PipelineContexts {
  /** GPU context (device, capabilities) */
  gpu?: { device?: GPUDevice; capabilities?: KernelCapabilities };
  /** Memory context for allocation */
  memory?: Record<string, unknown>;
  /** Storage context for custom shard loading */
  storage?: {
    loadShard?: (index: number) => Promise<ArrayBuffer | Uint8Array>;
  };
  /** Base URL for loading model files */
  baseUrl?: string;
  /** Runtime configuration overrides */
  runtime?: {
    /** Kernel path for explicit kernel dispatch ordering */
    kernelPath?: KernelPathRef;
  };
  /** Full runtime config (merged with defaults) */
  runtimeConfig?: Partial<RuntimeConfigSchema> | RuntimeConfigSchema;
  /** Progress callback for weight loading */
  onProgress?: (progress: { percent: number; message?: string }) => void;
}

/**
 * RoPE configuration.
 */
export interface RoPEConfig {
  headDim: number;
  maxSeqLen: number;
  ropeTheta: number;
  ropeLocalTheta?: number | null;
  ropeScale?: number;
  ropeScalingType?: string;
  ropeScaling?: {
    factor?: number;
    beta_fast?: number;
    beta_slow?: number;
    original_max_position_embeddings?: number;
  };
}

/**
 * RoPE frequency buffers.
 * Note: All buffers in a single RoPEBuffers instance will be the same type
 * (either all GPUBuffer or all Float32Array), never mixed.
 */
export interface RoPEBuffers {
  cos: GPUBuffer | Float32Array;
  sin: GPUBuffer | Float32Array;
  localCos?: GPUBuffer | Float32Array;
  localSin?: GPUBuffer | Float32Array;
}

/**
 * Type guard to check if RoPE buffers are GPU buffers.
 */
export function isGPURoPEBuffers(buffers: RoPEBuffers): buffers is {
  cos: GPUBuffer;
  sin: GPUBuffer;
  localCos?: GPUBuffer;
  localSin?: GPUBuffer;
};

/**
 * KV cache configuration.
 */
export interface KVCacheConfig {
  numLayers: number;
  numHeads: number;
  headDim: number;
  maxSeqLen: number;
  useGPU: boolean;
  layout: 'contiguous' | 'paged' | 'tiered';
  kvDtype: 'f16' | 'f32';
  pageSize?: number;
  slidingWindow?: number;
}

/**
 * Initialize RoPE (Rotary Position Embedding) frequency buffers.
 */
export function initRoPEFrequencies(
  config: RoPEConfig,
  useGPU: boolean
): Promise<RoPEBuffers>;

/**
 * Create and configure KV cache based on model configuration.
 */
export function createKVCache(
  modelConfig: ParsedModelConfig,
  useGPU: boolean,
  debug?: boolean,
  runtimeConfig?: KVCacheConfigSchema
): KVCache | SlidingWindowKVCache | TieredKVCache;

/**
 * Options for tokenizer initialization.
 */
export interface InitTokenizerOptions {
  /** Base URL for loading tokenizer files */
  baseUrl?: string;
  /** Preset tokenizer config as fallback hints (manifest takes precedence) */
  presetTokenizer?: {
    bosToken?: string;
    eosTokens?: string[];
    padToken?: string;
    addBosToken?: boolean;
    addEosToken?: boolean;
    hfModel?: string;
    allowArchFallback?: boolean;
  };
}

/**
 * Initialize tokenizer from manifest.
 */
export function initTokenizer(
  manifest: Manifest & TokenizerManifest,
  options?: InitTokenizerOptions
): Promise<Tokenizer>;

/**
 * Weight loading result.
 */
export interface WeightLoadResult {
  layerWeights: Map<string, LayerWeights>;
  embeddings: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;
  lmHead: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;
  finalNorm: GPUBuffer | Float32Array | null;
  layerRouterWeights: Map<number, RouterWeights>;
}

/** Options for loadWeights */
export interface LoadWeightsOptions {
  storageContext?: { loadShard?: (shardIdx: number) => Promise<ArrayBuffer | Uint8Array> };
  loadingConfig?: LoadingConfigSchema;
  onProgress?: (info: { stage: string; progress: number }) => void;
  verifyHashes?: boolean;
  baseUrl?: string;
}

/**
 * Load model weights via DopplerLoader.
 */
export function loadWeights(
  manifest: Manifest,
  modelConfig: ParsedModelConfig,
  options?: LoadWeightsOptions
): Promise<WeightLoadResult>;

/**
 * Apply Gemma chat template to a prompt.
 */
export function applyGemmaChatTemplate(prompt: string): string;

/**
 * Apply Llama 3 chat template to a prompt.
 */
export function applyLlama3ChatTemplate(prompt: string): string;

/**
 * Apply GPT-OSS chat template to a prompt.
 */
export function applyGptOssChatTemplate(prompt: string): string;

/**
 * Apply ChatML template to a prompt.
 */
export function applyChatMLTemplate(prompt: string): string;

/**
 * Apply Qwen chat template to a prompt.
 */
export function applyQwenChatTemplate(prompt: string): string;

/**
 * Apply chat template based on template type from config.
 */
export function applyChatTemplate(prompt: string, templateType: string | null | undefined): string;

/**
 * Check if a token is a stop token.
 */
export function isStopToken(
  token: number,
  stopTokenIds: number[],
  eosTokenId?: number
): boolean;

/**
 * Initialize MoE router if model uses Mixture of Experts.
 */
export function initMoERouter(
  modelConfig: ParsedModelConfig,
  moeRoutingConfig: MoERoutingConfigSchema,
  layerWeights: Map<string, LayerWeights>
): MoERouter | null;

/**
 * Initialize speculative decoder if draft model is available.
 */
export function initSpeculativeDecoder(
  manifest: Manifest,
  speculativeConfig: SpeculativeConfigSchema
): SpeculativeDecoder | null;

/**
 * Fuse Q/K/V projection weights into a single QKV weight for optimized inference.
 */
export function fuseQKVWeights(
  layerWeights: Map<string, LayerWeights>,
  modelConfig: ParsedModelConfig
): void;

/**
 * Initialize NVIDIA superchip emulation if enabled in runtime config.
 *
 * Creates an EmulationContext with virtual GPUs, CPUs, and interconnect
 * simulation for testing distributed inference patterns.
 *
 * @param runtimeConfig - Runtime configuration with emulation settings
 * @returns EmulationContext if enabled and supported, null otherwise
 */
export function initEmulation(
  runtimeConfig: RuntimeConfigSchema
): Promise<import('../../simulator/index.js').EmulationContext | null>;

/**
 * Destroy emulation context and clean up resources.
 *
 * @param emulation - Emulation context to destroy
 */
export function destroyEmulation(
  emulation: import('../../simulator/index.js').EmulationContext | null
): Promise<void>;
