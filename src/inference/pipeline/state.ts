/**
 * Pipeline State
 *
 * Holds the state of the inference pipeline:
 * - Model configuration and weights
 * - Runtime state (tokenizer, KV cache, etc.)
 * - Statistics
 *
 * @module inference/pipeline/state
 */

import { Tokenizer } from '../tokenizer.js';
import { KVCache, SlidingWindowKVCache } from '../kv-cache.js';
import { MoERouter } from '../moe-router.js';
import { SpeculativeDecoder } from '../speculative.js';
import { DecodeBufferManager } from '../decode-buffers.js';
import type { Manifest, ParsedModelConfig } from './config.js';
import type { LayerWeights, ExpertWeights, RouterWeights } from './types.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../gpu/weight-buffer.js';
import type { DopplerLoader } from '../../loader/doppler-loader.js';
import type { CompiledLayerPipeline } from './layer-plan.js';
import type { LoRAAdapter } from './lora.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import type { RuntimeConfigSchema } from '../../config/schema/index.js';
import type { PipelineStats, BatchingStats } from './types.js';

import type { WeightDebugFlags } from './weights.js';
import type { LogitsDebugFlags } from './logits.js';
import type { KernelPathRef, KernelPathSchema } from '../../config/schema/index.js';
import type { KernelPathSource } from '../../config/kernel-path-loader.js';

export class PipelineState {
  // Components
  tokenizer: Tokenizer | null = null;
  kvCache: KVCache | SlidingWindowKVCache | null = null;
  moeRouter: MoERouter | null = null;
  speculativeDecoder: SpeculativeDecoder | null = null;
  decodeBuffers: DecodeBufferManager | null = null;

  // Debug flags (combined for both layer and logits)
  debugFlags: WeightDebugFlags & LogitsDebugFlags = {};
  decodeStepCount = 0;
  runtimeKernelPath: KernelPathRef | null = null;
  resolvedKernelPath: KernelPathSchema | null = null;
  kernelPathSource: KernelPathSource = 'none';
  disableRecordedLogits = false;
  disableFusedDecode = false;

  // Model state
  manifest: Manifest | null = null;
  modelConfig: ParsedModelConfig | null = null;
  weights: Map<string, LayerWeights | GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null> = new Map();
  expertWeights: Map<string, ExpertWeights> = new Map();

  // Runtime state
  isLoaded = false;
  isGenerating = false;
  currentSeqLen = 0;
  runtimeConfig: RuntimeConfigSchema = getRuntimeConfig();

  // DopplerLoader instance
  dopplerLoader: DopplerLoader | null = null;

  // GPU context
  gpuContext: { device?: GPUDevice } | null = null;
  useGPU = false;

  // Memory and storage contexts
  memoryContext: Record<string, unknown> | null = null;
  storageContext: { loadShard?: (index: number) => Promise<ArrayBuffer | Uint8Array> } | null = null;

  // Stats
  stats: PipelineStats = {
    prefillTimeMs: 0,
    decodeTimeMs: 0,
    prefillTokens: 0,
    decodeTokens: 0,
    memoryUsageBytes: 0,
    // Add missing properties expected by pipeline.ts if strict
    // pipeline.ts had: tokensGenerated, totalTimeMs, gpuTimePrefillMs, gpuTimeDecodeMs
    // types.ts has: prefillTimeMs, decodeTimeMs, prefillTokens, decodeTokens, memoryUsageBytes
    // I need to reconcile. I'll check pipeline.ts usage.
  } as any; 
  // TODO: pipeline.ts defines PipelineStats differently than types.ts?
  // I will check pipeline.ts definition in next step. For now I use `any` cast or just define what was in pipeline.ts
  // pipeline.ts had:
  // tokensGenerated: number; totalTimeMs: number; prefillTimeMs: number; decodeTimeMs: number; gpuTimePrefillMs?: number; gpuTimeDecodeMs?: number;

  batchingStats: BatchingStats = {
    batchedForwardCalls: 0,
    unbatchedForwardCalls: 0,
    totalBatchedTimeMs: 0,
    totalUnbatchedTimeMs: 0,
    gpuSubmissions: 0,
  };

  // Base URL for loading assets
  baseUrl: string | null = null;

  // RoPE frequency buffers (global for full_attention layers)
  ropeFreqsCos: Float32Array | GPUBuffer | null = null;
  ropeFreqsSin: Float32Array | GPUBuffer | null = null;
  // Local RoPE frequencies for sliding_attention layers (Gemma 3: 10K theta vs 1M global)
  ropeLocalCos: Float32Array | GPUBuffer | null = null;
  ropeLocalSin: Float32Array | GPUBuffer | null = null;

  // Debug
  debug = false;
  // Optional layer pipeline plan (JSON-configured)
  layerPipelinePlan: CompiledLayerPipeline | null = null;

  // Tied embeddings
  useTiedEmbeddings = false;
  embeddingVocabSize: number | null = null;
  embeddingTranspose = false;  // True for GGUF [H,V] layout

  // MoE router weights per layer
  layerRouterWeights: Map<number, RouterWeights> | null = null;

  // LoRA adapter (optional)
  lora: LoRAAdapter | null = null;
}
