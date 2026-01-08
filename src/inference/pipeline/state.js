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

import { getRuntimeConfig } from '../../config/runtime.js';

export class PipelineState {
  constructor() {
    // Components
    /** @type {import('../tokenizer.js').Tokenizer | null} */
    this.tokenizer = null;
    /** @type {import('../kv-cache.js').KVCache | import('../kv-cache.js').SlidingWindowKVCache | null} */
    this.kvCache = null;
    /** @type {import('../moe-router.js').MoERouter | null} */
    this.moeRouter = null;
    /** @type {import('../speculative.js').SpeculativeDecoder | null} */
    this.speculativeDecoder = null;
    /** @type {import('../decode-buffers.js').DecodeBufferManager | null} */
    this.decodeBuffers = null;

    // Debug flags (combined for both layer and logits)
    /** @type {import('./weights.js').WeightDebugFlags & import('./logits.js').LogitsDebugFlags} */
    this.debugFlags = {};
    /** @type {number} */
    this.decodeStepCount = 0;
    /** @type {import('../../config/schema/index.js').KernelPathRef | null} */
    this.runtimeKernelPath = null;
    /** @type {import('../../config/schema/index.js').KernelPathSchema | null} */
    this.resolvedKernelPath = null;
    /** @type {import('../../config/kernel-path-loader.js').KernelPathSource} */
    this.kernelPathSource = 'none';
    /** @type {boolean} */
    this.disableRecordedLogits = false;
    /** @type {boolean} */
    this.disableFusedDecode = false;

    // Model state
    /** @type {import('./config.js').Manifest | null} */
    this.manifest = null;
    /** @type {import('./config.js').ParsedModelConfig | null} */
    this.modelConfig = null;
    /** @type {Map<string, import('./types.js').LayerWeights | GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer | import('../../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | null>} */
    this.weights = new Map();
    /** @type {Map<string, import('./types.js').ExpertWeights>} */
    this.expertWeights = new Map();

    // Runtime state
    /** @type {boolean} */
    this.isLoaded = false;
    /** @type {boolean} */
    this.isGenerating = false;
    /** @type {number} */
    this.currentSeqLen = 0;
    /** @type {import('../../config/schema/index.js').RuntimeConfigSchema} */
    this.runtimeConfig = getRuntimeConfig();

    // DopplerLoader instance
    /** @type {import('../../loader/doppler-loader.js').DopplerLoader | null} */
    this.dopplerLoader = null;

    // GPU context
    /** @type {{ device?: GPUDevice } | null} */
    this.gpuContext = null;
    /** @type {boolean} */
    this.useGPU = false;

    // Memory and storage contexts
    /** @type {Record<string, unknown> | null} */
    this.memoryContext = null;
    /** @type {{ loadShard?: (index: number) => Promise<ArrayBuffer | Uint8Array> } | null} */
    this.storageContext = null;

    // Stats
    /** @type {import('./types.js').PipelineStats} */
    this.stats = {
      prefillTimeMs: 0,
      decodeTimeMs: 0,
      prefillTokens: 0,
      decodeTokens: 0,
      memoryUsageBytes: 0,
      tokensGenerated: 0,
      totalTimeMs: 0,
    };

    /** @type {import('./types.js').BatchingStats} */
    this.batchingStats = {
      batchedForwardCalls: 0,
      unbatchedForwardCalls: 0,
      totalBatchedTimeMs: 0,
      totalUnbatchedTimeMs: 0,
      gpuSubmissions: 0,
    };

    // Base URL for loading assets
    /** @type {string | null} */
    this.baseUrl = null;

    // RoPE frequency buffers (global for full_attention layers)
    /** @type {Float32Array | GPUBuffer | null} */
    this.ropeFreqsCos = null;
    /** @type {Float32Array | GPUBuffer | null} */
    this.ropeFreqsSin = null;
    // Local RoPE frequencies for sliding_attention layers (Gemma 3: 10K theta vs 1M global)
    /** @type {Float32Array | GPUBuffer | null} */
    this.ropeLocalCos = null;
    /** @type {Float32Array | GPUBuffer | null} */
    this.ropeLocalSin = null;

    // Debug
    /** @type {boolean} */
    this.debug = false;
    // Optional layer pipeline plan (JSON-configured)
    /** @type {import('./layer-plan.js').CompiledLayerPipeline | null} */
    this.layerPipelinePlan = null;

    // Tied embeddings
    /** @type {boolean} */
    this.useTiedEmbeddings = false;
    /** @type {number | null} */
    this.embeddingVocabSize = null;
    /** @type {boolean} */
    this.embeddingTranspose = false;

    // MoE router weights per layer
    /** @type {Map<number, import('./types.js').RouterWeights> | null} */
    this.layerRouterWeights = null;

    // LoRA adapter (optional)
    /** @type {import('./lora.js').LoRAAdapter | null} */
    this.lora = null;
  }
}
