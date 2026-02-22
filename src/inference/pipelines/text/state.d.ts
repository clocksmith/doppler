/**
 * Pipeline State
 *
 * Holds the state of the inference pipeline:
 * - Model configuration and weights
 * - Runtime state (tokenizer, KV cache, etc.)
 * - Statistics
 *
 * @module inference/pipelines/text/state
 */

import { Tokenizer } from '../../tokenizer.js';
import { KVCache, SlidingWindowKVCache, TieredKVCache, BasisDecomposedPagedCache } from '../../kv-cache.js';
import { MoERouter } from '../../moe-router.js';
import { SpeculativeDecoder } from '../../speculative.js';
import { DecodeBufferManager } from '../../decode-buffers.js';
import { DecodeRing } from '../../decode-ring.js';
import type { Manifest, ParsedModelConfig } from './config.js';
import type { LayerWeights, ExpertWeights, RouterWeights, PipelineStats, BatchingStats } from './types.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import type { DopplerLoader } from '../../../loader/doppler-loader.js';
import type { CompiledLayerPipeline } from './layer-plan.js';
import type { ExecutionPlanState } from './execution-plan.js';
import type { LoRAAdapter } from './lora.js';
import type { RuntimeConfigSchema, KernelPathSchema } from '../../../config/schema/index.js';
import type { WeightDebugFlags } from './weights.js';
import type { LogitsDebugFlags } from './logits.js';
import type { KernelPathSource } from '../../../config/kernel-path-loader.js';
import type { EmulationContext } from '/proto/simulator/index.js';

export class PipelineState {
  // Components
  tokenizer: Tokenizer | null;
  kvCache: KVCache | SlidingWindowKVCache | TieredKVCache | BasisDecomposedPagedCache | null;
  moeRouter: MoERouter | null;
  speculativeDecoder: SpeculativeDecoder | null;
  decodeBuffers: DecodeBufferManager | null;
  decodeRing: DecodeRing | null;

  // Emulation context (null when emulation is disabled)
  emulation: EmulationContext | null;

  // Debug flags (combined for both layer and logits)
  debugFlags: WeightDebugFlags & LogitsDebugFlags;
  decodeStepCount: number;
  resolvedKernelPath: KernelPathSchema | null;
  kernelPathSource: KernelPathSource;
  executionPlanState: ExecutionPlanState | null;
  disableRecordedLogits: boolean;
  disableFusedDecode: boolean;

  // Model state
  manifest: Manifest | null;
  modelConfig: ParsedModelConfig | null;
  weights: Map<string, LayerWeights | GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null>;
  expertWeights: Map<string, ExpertWeights>;

  // Runtime state
  isLoaded: boolean;
  isGenerating: boolean;
  currentSeqLen: number;
  currentTokenIds: number[] | null;
  runtimeConfig: RuntimeConfigSchema;

  // DopplerLoader instance
  dopplerLoader: DopplerLoader | null;

  // GPU context
  gpuContext: { device?: GPUDevice } | null;
  useGPU: boolean;

  // Memory and storage contexts
  memoryContext: Record<string, unknown> | null;
  storageContext: { loadShard?: (index: number) => Promise<ArrayBuffer | Uint8Array> } | null;

  // Stats
  stats: PipelineStats;
  batchingStats: BatchingStats;

  // Base URL for loading assets
  baseUrl: string | null;

  // RoPE frequency buffers (global for full_attention layers)
  ropeFreqsCos: Float32Array | GPUBuffer | null;
  ropeFreqsSin: Float32Array | GPUBuffer | null;
  // Local RoPE frequencies for sliding_attention layers (Gemma 3: 10K theta vs 1M global)
  ropeLocalCos: Float32Array | GPUBuffer | null;
  ropeLocalSin: Float32Array | GPUBuffer | null;

  // Debug
  debug: boolean;
  // Optional layer pipeline plan (JSON-configured)
  layerPipelinePlan: CompiledLayerPipeline | null;

  // Tied embeddings
  useTiedEmbeddings: boolean;
  embeddingVocabSize: number | null;
  embeddingTranspose: boolean;

  // MoE router weights per layer
  layerRouterWeights: Map<number, RouterWeights> | null;

  // LoRA adapter (optional)
  lora: LoRAAdapter | null;

  constructor();
}
