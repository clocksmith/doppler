

import { getRuntimeConfig } from '../../../config/runtime.js';

export class PipelineState {
  constructor() {
    // Components

    this.tokenizer = null;

    this.kvCache = null;

    this.moeRouter = null;

    this.speculativeDecoder = null;

    this.decodeBuffers = null;

    this.decodeRing = null;

    this.finitenessBuffer = null;

    // Emulation context (null when emulation is disabled)
    // @type {import('/proto/simulator/index.js').EmulationContext|null}

    this.emulation = null;

    // Debug flags (combined for both layer and logits)

    this.debugFlags = {};

    this.decodeStepCount = 0;

    this.resolvedKernelPath = null;

    this.kernelPathSource = 'none';

    this.disableRecordedLogits = false;

    this.disableFusedDecode = false;

    // Model state

    this.manifest = null;

    this.modelConfig = null;

    this.weights = new Map();

    this.expertWeights = new Map();

    // Runtime state

    this.isLoaded = false;

    this.isGenerating = false;

    this.currentSeqLen = 0;

    this.currentTokenIds = null;

    this.runtimeConfig = getRuntimeConfig();

    // DopplerLoader instance

    this.dopplerLoader = null;

    // GPU context

    this.gpuContext = null;

    this.useGPU = false;

    // Memory and storage contexts

    this.memoryContext = null;

    this.storageContext = null;

    // Stats

    this.stats = {
      prefillTimeMs: 0,
      decodeTimeMs: 0,
      ttftMs: 0,
      prefillTokens: 0,
      decodeTokens: 0,
      memoryUsageBytes: 0,
      tokensGenerated: 0,
      totalTimeMs: 0,
      decodeRecordMs: 0,
      decodeSubmitWaitMs: 0,
      decodeReadbackWaitMs: 0,
      decodeProfileSteps: [],
      attentionInputs: [],
    };


    this.batchingStats = {
      batchedForwardCalls: 0,
      unbatchedForwardCalls: 0,
      totalBatchedTimeMs: 0,
      totalUnbatchedTimeMs: 0,
      gpuSubmissions: 0,
    };

    // Base URL for loading assets

    this.baseUrl = null;

    // RoPE frequency buffers (global for full_attention layers)

    this.ropeFreqsCos = null;

    this.ropeFreqsSin = null;
    // Local RoPE frequencies for sliding_attention layers (different theta than global)

    this.ropeLocalCos = null;

    this.ropeLocalSin = null;

    // Debug

    this.debug = false;
    // Optional layer pipeline plan (JSON-configured)

    this.layerPipelinePlan = null;

    // Tied embeddings

    this.useTiedEmbeddings = false;

    this.embeddingVocabSize = null;

    this.embeddingTranspose = false;

    // MoE router weights per layer

    this.layerRouterWeights = null;

    // LoRA adapter (optional)

    this.lora = null;
  }
}
