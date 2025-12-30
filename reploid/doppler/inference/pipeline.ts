/**
 * pipeline.ts - Main Inference Pipeline (Thin Orchestrator)
 *
 * This module orchestrates inference by delegating to specialized modules:
 * - init.ts: Initialization, weight loading, KV cache, RoPE
 * - embed.ts: Token embedding with optional Gemma scaling
 * - layer.ts: Transformer layer processing (attention + FFN)
 * - logits.ts: Final layer norm and LM head projection
 * - sampling.ts: Token sampling strategies
 * - config.ts: Model configuration parsing
 *
 * The pipeline maintains state (weights, caches, tokenizer) and coordinates
 * the flow from input tokens to generated output.
 *
 * @module inference/pipeline
 */

import { MoERouter } from './moe-router.js';
import { SpeculativeDecoder } from './speculative.js';
import { KVCache, SlidingWindowKVCache } from './kv-cache.js';
import { Tokenizer } from './tokenizer.js';
import { getDevice, setTrackSubmits } from '../gpu/device.js';
import { releaseBuffer, readBuffer } from '../gpu/buffer-pool.js';
import { runArgmax, runGPUSample, recordArgmax, recordGPUSample, isGPUSamplingAvailable } from '../gpu/kernels/sample.js';
import { recordCheckStop } from '../gpu/kernels/check-stop.js';
import { recordGather } from '../gpu/kernels/gather.js';
import { recordScale } from '../gpu/kernels/scale.js';
import { markWarmed as markKernelCacheWarmed } from '../gpu/kernel-selection-cache.js';
import { resetSubmitStats, logSubmitStats, getSubmitStats } from '../gpu/submit-tracker.js';
import { createCommandRecorder, createProfilingRecorder, CommandRecorder, type ProfileTimings } from '../gpu/command-recorder.js';
import { setKernelHints, clearKernelHints } from '../gpu/kernel-hints.js';
import type { KernelHints } from '../storage/rdrr-format.js';
import { allowReadback } from '../gpu/perf-guards.js';
import { log, setGPUDevice } from '../debug/index.js';

// Pipeline sub-modules
import { sample, applyRepetitionPenalty, logitsSanity, getTopK, type SamplingOptions } from './pipeline/sampling.js';
import { parseModelConfig, type ParsedModelConfig, type Manifest } from './pipeline/config.js';
import {
  normalizeAttentionKernel,
  initRoPEFrequencies,
  createKVCache,
  initTokenizer,
  loadWeights,
  type WeightLoadResult,
  applyGemmaChatTemplate,
  applyLlama3ChatTemplate,
  isStopToken,
  initMoERouter,
  initSpeculativeDecoder,
  type PipelineContexts,
} from './pipeline/init.js';
import { embed } from './pipeline/embed.js';
import { processLayer, type LayerContext } from './pipeline/layer.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU, extractLastPositionLogits, type LogitsConfig, type LogitsWeights } from './pipeline/logits.js';
import { createWeightBufferHelpers, type WeightBufferConfig, type WeightDebugFlags } from './pipeline/weights.js';
import type { LoRAAdapter } from './pipeline/lora.js';
import type { ExpertLoader } from './pipeline/moe-impl.js';
import type { LayerWeights, ExpertWeights, RouterWeights, GenerationResult } from './pipeline/types.js';
import type { DopplerLoader, LoadProgress } from '../loader/doppler-loader.js';
import { verbose as loaderVerbose } from '../loader/log.js';
import type { LogitsDebugFlags } from './pipeline/logits.js';
import { getDopplerLoader } from '../loader/doppler-loader.js';

// Re-export types for external use
export type { LayerWeights, ExpertWeights, RouterWeights, GenerationResult };
export { PipelineContexts };

// ============================================================================
// TypeScript Interfaces
// ============================================================================

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stopSequences?: string[];
  useSpeculative?: boolean;
  onToken?: ((tokenId: number, text: string) => void) | null;
  useChatTemplate?: boolean;
  decode?: (tokens: number[]) => string;
  debug?: boolean;
  /** Specific layers to debug (enables batching with selective checkpoints).
   *  If set, CommandRecorder stays enabled but flushes at these layers.
   *  Example: [0, 12, 25] debugs first, middle, and last layers only. */
  debugLayers?: number[];
  signal?: AbortSignal;
  /** Enable GPU timestamp profiling for kernel-level timing.
   *  Requires 'timestamp-query' WebGPU feature. Results logged to console. */
  profile?: boolean;
  /** Log benchmark stats (TTFT, prefill time, decode speed) after generation.
   *  Default: false */
  benchmark?: boolean;
  /** Explicitly disable GPU command batching for debugging.
   *  When true, each GPU operation is submitted individually.
   *  Default: false */
  disableBatching?: boolean;

  // Batch generation options
  /** Number of tokens to generate per GPU submission batch.
   *  Default: 1 (single-token mode for backward compatibility)
   *  Higher values reduce GPU sync overhead but delay token streaming. */
  batchSize?: number;
  /** Callback invoked after each batch completes.
   *  Receives array of {id, text} pairs for the batch. */
  onBatch?: ((tokens: Array<{ id: number; text: string }>) => void) | null;
  /** Stop condition checking mode for batched generation.
   *  - 'batch': Check stop conditions after entire batch (faster, may overshoot by up to batchSize-1)
   *  - 'per-token': Check after each token using GPU kernel (accurate, default)
   *  Default: 'per-token' */
  stopCheckMode?: 'batch' | 'per-token';
}

export interface LayerConfig {
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  numExperts?: number;
  topK?: number;
}

export interface PipelineStats {
  tokensGenerated: number;
  totalTimeMs: number;
  prefillTimeMs: number;
  decodeTimeMs: number;
  gpuTimePrefillMs?: number;
  gpuTimeDecodeMs?: number;
}

export interface BatchingStats {
  batchedForwardCalls: number;
  unbatchedForwardCalls: number;
  totalBatchedTimeMs: number;
  totalUnbatchedTimeMs: number;
}

export interface KVCacheSnapshot {
  cache: KVCache;
  seqLen: number;
  tokens: number[];
}

function sumProfileTimings(timings: ProfileTimings | null | undefined): number | null {
  if (!timings || Object.keys(timings).length === 0) return null;
  let total = 0;
  for (const value of Object.values(timings)) {
    if (Number.isFinite(value)) {
      total += value;
    }
  }
  return total;
}

// ============================================================================
// Main Inference Pipeline Class
// ============================================================================

export class InferencePipeline {
  // Components
  tokenizer: Tokenizer | null = null;
  kvCache: KVCache | SlidingWindowKVCache | null = null;
  moeRouter: MoERouter | null = null;
  speculativeDecoder: SpeculativeDecoder | null = null;

  // Model state
  manifest: Manifest | null = null;
  modelConfig: ParsedModelConfig | null = null;
  weights: Map<string, LayerWeights | GPUBuffer | Float32Array | null> = new Map();
  expertWeights: Map<string, ExpertWeights> = new Map();

  // Runtime state
  isLoaded = false;
  isGenerating = false;
  currentSeqLen = 0;

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
    tokensGenerated: 0,
    totalTimeMs: 0,
    prefillTimeMs: 0,
    decodeTimeMs: 0,
    gpuTimePrefillMs: undefined,
    gpuTimeDecodeMs: undefined,
  };
  batchingStats: BatchingStats = { batchedForwardCalls: 0, unbatchedForwardCalls: 0, totalBatchedTimeMs: 0, totalUnbatchedTimeMs: 0 };

  // Base URL for loading assets
  baseUrl: string | null = null;

  // RoPE frequency buffers (global for full_attention layers)
  ropeFreqsCos: Float32Array | GPUBuffer | null = null;
  ropeFreqsSin: Float32Array | GPUBuffer | null = null;
  // Local RoPE frequencies for sliding_attention layers (Gemma 3: 10K theta vs 1M global)
  ropeLocalCos: Float32Array | GPUBuffer | null = null;
  ropeLocalSin: Float32Array | GPUBuffer | null = null;

  // Attention kernel override
  attentionKernelOverride: 'tiled_large' | 'tiled_small' | 'streaming' | null = null;
  manifestAttentionKernelDefault: 'tiled_large' | 'tiled_small' | 'streaming' | null = null;

  // Debug
  debug = false;

  // Tied embeddings
  useTiedEmbeddings = false;
  embeddingVocabSize: number | null = null;

  // MoE router weights per layer
  layerRouterWeights: Map<number, RouterWeights> | null = null;

  // LoRA adapter (optional)
  loraAdapter: LoRAAdapter | null = null;

  // Debug flags (combined for both layer and logits)
  private _debugFlags: WeightDebugFlags & LogitsDebugFlags = {};
  private _decodeStepCount = 0;
  private _runtimeKernelHints: KernelHints | null = null;

  // Progress callback
  private _onProgress: ((progress: { percent: number; message?: string; stage?: string; layer?: number; total?: number }) => void) | null = null;
  private _preloadedWeights: WeightLoadResult | null = null;

  constructor() {}

  // ==========================================================================
  // Initialization
  // ==========================================================================

  async initialize(contexts: PipelineContexts = {}): Promise<void> {
    if (contexts.gpu?.device) {
      this.gpuContext = { device: contexts.gpu.device };
      this.useGPU = true;
    }
    if (contexts.memory) this.memoryContext = contexts.memory;
    if (contexts.storage) this.storageContext = contexts.storage;
    if (contexts.baseUrl) this.baseUrl = contexts.baseUrl;

    if (contexts.runtime?.attentionKernel) {
      this.attentionKernelOverride = normalizeAttentionKernel(contexts.runtime.attentionKernel);
    }
    if (contexts.runtime?.debug) this.debug = true;
    if (contexts.runtime?.kernelHints) {
      this._runtimeKernelHints = contexts.runtime.kernelHints;
    }
    if (contexts.onProgress) this._onProgress = contexts.onProgress;

    const device = getDevice();
    if (device) setGPUDevice(device);

    log.debug('Pipeline', 'Initialized', { useGPU: this.useGPU, debug: this.debug });
  }

  async loadModel(manifest: Manifest): Promise<void> {
    this.manifest = manifest;
    this.modelConfig = parseModelConfig(manifest);

    if (manifest.optimizations?.debug || manifest.runtime?.debug) this.debug = true;

    // Set kernel hints from manifest (if present), otherwise apply defaults.
    const kernelHints = manifest.optimizations?.kernelHints as KernelHints | undefined;
    if (kernelHints) {
      setKernelHints(kernelHints, 'manifest');
      console.log('[Pipeline] Kernel hints loaded from manifest:', kernelHints);
    } else {
      const quant = String(manifest.quantization || '').toLowerCase();
      const defaultHints: KernelHints = {
        computePrecision: 'auto',
        f16Matmul: 'gemv_subgroup',
        attentionPrefill: 'tiled_large',
        attentionDecode: 'streaming',
      };
      if (quant.includes('q4')) {
        defaultHints.q4kMatmul = 'dequant_f16';
      }
      setKernelHints(defaultHints, 'manifest');
      console.log('[Pipeline] Kernel hints defaulted for manifest:', defaultHints);
    }

    if (this._runtimeKernelHints) {
      setKernelHints(this._runtimeKernelHints, 'runtime');
      console.log('[Pipeline] Kernel hints overridden at runtime:', this._runtimeKernelHints);
    }

    const manifestKernel = manifest.optimizations?.attentionKernel || manifest.attentionKernel || manifest.runtime?.attentionKernel;
    this.manifestAttentionKernelDefault = normalizeAttentionKernel(manifestKernel);
    if (!this.attentionKernelOverride && this.manifestAttentionKernelDefault) {
      this.attentionKernelOverride = this.manifestAttentionKernelDefault;
    }

    console.log('[Pipeline] Model config:', {
      numLayers: this.modelConfig.numLayers,
      hiddenSize: this.modelConfig.hiddenSize,
      vocabSize: this.modelConfig.vocabSize,
      numHeads: this.modelConfig.numHeads,
      numKVHeads: this.modelConfig.numKVHeads,
      headDim: this.modelConfig.headDim,
      useMoE: this.modelConfig.useMoE,
    });

    // Initialize tokenizer
    this.tokenizer = await initTokenizer(manifest, this.baseUrl ?? undefined);
    const tokenizerVocabSize = this.tokenizer.getVocabSize();
    if (Number.isFinite(tokenizerVocabSize) && tokenizerVocabSize > 0) {
      if (tokenizerVocabSize !== this.modelConfig.vocabSize) {
        // Don't override - use model's vocab size for embedding compatibility
        console.log(`[Pipeline] Tokenizer vocabSize=${tokenizerVocabSize} differs from model=${this.modelConfig.vocabSize}, using model size`);
      }
    }

    // Initialize KV cache
    this.kvCache = createKVCache(this.modelConfig, this.useGPU, this.debug);

    // Initialize MoE router if needed
    if (this.modelConfig.useMoE) {
      this.moeRouter = new MoERouter({
        numExperts: this.modelConfig.numExperts,
        topK: this.modelConfig.moeTopK || 2,
        hiddenSize: this.modelConfig.hiddenSize,
        normalizeWeights: true,
      });
    }

    // Initialize speculative decoder if draft model
    if (manifest.draftModel) {
      this.speculativeDecoder = initSpeculativeDecoder(manifest);
    }

    // Load weights
    await this._loadWeights();

    // Initialize RoPE frequencies
    await this._initRoPE();

    this.isLoaded = true;
    console.log('[Pipeline] Model loaded successfully');
  }

  private async _loadWeights(): Promise<void> {
    const result = this._preloadedWeights || await loadWeights(
      this.manifest!,
      this.modelConfig!,
      {
        storageContext: this.storageContext ?? undefined,
        onProgress: (info: { stage: string; progress: number; message?: string; layer?: number; total?: number; shard?: number; totalShards?: number }) => {
          // Shard and layer progress are logged by the loader with source info (RAM/OPFS/network)
          // Only log other stages here to avoid duplicate logs
          if (info.stage !== 'layers' && info.stage !== 'shards') {
            loaderVerbose(`${info.stage}: ${Math.round(info.progress * 100)}%${info.message ? ` - ${info.message}` : ''}`);
          }
          // Forward to UI callback if set
          if (this._onProgress) {
            this._onProgress({
              percent: info.progress * 100,
              message: info.message,
              stage: info.stage,
              layer: info.layer,
              total: info.total,
            });
          }
        },
        // Skip hash verification - already verified during download
        verifyHashes: false,
      }
    );

    // Store weights in map
    result.layerWeights.forEach((w, k) => this.weights.set(k, w));
    this.weights.set('embed', result.embeddings);
    this.weights.set('lm_head', result.lmHead);
    this.weights.set('final_norm', result.finalNorm);

    this.useTiedEmbeddings = result.useTiedEmbeddings;
    this.embeddingVocabSize = result.embeddingVocabSize;
    this.layerRouterWeights = result.layerRouterWeights;

    // Store DopplerLoader reference for expert loading
    this.dopplerLoader = getDopplerLoader();

    // Initialize MoE router with weights
    if (this.modelConfig!.useMoE && this.moeRouter) {
      this.moeRouter = initMoERouter(this.modelConfig!, result.layerWeights);
    }
  }

  setPreloadedWeights(weights: WeightLoadResult): void {
    this._preloadedWeights = weights;
  }

  private async _initRoPE(): Promise<void> {
    const config = this.modelConfig!;
    const ropeBuffers = await initRoPEFrequencies({
      headDim: config.headDim,
      maxSeqLen: config.maxSeqLen || 4096,
      ropeTheta: config.ropeTheta,
      ropeLocalTheta: config.ropeLocalTheta,  // Gemma 3: 10K for local layers
      ropeScale: config.ropeScale,
      ropeScalingType: config.ropeScalingType,
      ropeScaling: config.ropeScaling,
    }, this.useGPU);
    this.ropeFreqsCos = ropeBuffers.cos;
    this.ropeFreqsSin = ropeBuffers.sin;
    this.ropeLocalCos = ropeBuffers.localCos ?? null;
    this.ropeLocalSin = ropeBuffers.localSin ?? null;
  }

  // ==========================================================================
  // Generation
  // ==========================================================================

  async *generate(prompt: string, options: GenerateOptions = {}): AsyncGenerator<string, void, void> {
    if (!this.isLoaded) throw new Error('Model not loaded');
    if (this.isGenerating) throw new Error('Generation already in progress');

    this.isGenerating = true;
    this._decodeStepCount = 0;
    this.stats.gpuTimePrefillMs = undefined;
    this.stats.gpuTimeDecodeMs = undefined;
    const startTime = performance.now();

    const opts = {
      maxTokens: options.maxTokens ?? 512,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      repetitionPenalty: options.repetitionPenalty ?? 1.1,
      stopSequences: options.stopSequences ?? [],
      useSpeculative: options.useSpeculative ?? false,
      useChatTemplate: options.useChatTemplate ?? false,
      debug: options.debug ?? this.debug,
      debugLayers: options.debugLayers,  // Selective layer debugging
      profile: options.profile ?? false,  // GPU timestamp profiling
      benchmark: options.benchmark ?? false,  // Benchmark stats logging
      disableBatching: options.disableBatching ?? false,  // Explicit batching control
      // Batch generation options
      batchSize: options.batchSize ?? 1,
      stopCheckMode: options.stopCheckMode ?? 'per-token' as const,
    };

    try {
      // Apply chat template if requested
      let processedPrompt = prompt;
      if (opts.useChatTemplate) {
        if (this.modelConfig!.isGemma3) {
          processedPrompt = applyGemmaChatTemplate(prompt);
          if (opts.debug) console.log('[Pipeline] Applied Gemma chat template');
        } else if (this.modelConfig!.isLlama3Instruct) {
          processedPrompt = applyLlama3ChatTemplate(prompt);
          if (opts.debug) console.log('[Pipeline] Applied Llama 3 chat template');
        }
      }

      // Tokenize
      const inputIds = this.tokenizer!.encode(processedPrompt);
      const generatedIds = [...inputIds];

      if (opts.debug) {
        console.log(`[Pipeline] Input: ${inputIds.length} tokens`);
      }

      // Prefill
      const prefillStart = performance.now();
      const prefillLogits = await this._prefill(inputIds, opts);
      this.stats.prefillTimeMs = performance.now() - prefillStart;

      // Debug: show input tokens
      if (opts.debug) {
        const inputTokenTexts = inputIds
          .map(id => `${id}="${this.tokenizer?.decode?.([id]) || '?'}"`)
          .join(', ');
        console.log(`[Pipeline] Input tokens (${inputIds.length}): ${inputTokenTexts}`);
      }

      // Apply repetition penalty and sample first token
      applyRepetitionPenalty(prefillLogits, generatedIds, opts.repetitionPenalty);

      // Debug: check logits after repetition penalty
      if (opts.debug) {
        const topAfterPenalty = getTopK(prefillLogits, 5, (tokens) => this.tokenizer?.decode?.(tokens) || '?');
        console.log(`[Pipeline] After rep penalty top-5: ${topAfterPenalty.map(t => `"${t.text}"(${(t.prob * 100).toFixed(1)}%)`).join(', ')}`);
      }

      const firstToken = sample(prefillLogits, {
        temperature: opts.temperature,
        topP: opts.topP,
        topK: opts.topK,
      });

      if (opts.debug) {
        console.log(`[Pipeline] First token sampled: id=${firstToken} text="${this.tokenizer?.decode?.([firstToken]) || '?'}"`);
      }

      generatedIds.push(firstToken);

      // Yield first token
      const firstText = this.tokenizer!.decode([firstToken], true, false);
      yield firstText;
      if (options.onToken) options.onToken(firstToken, firstText);

      // Check stop conditions
      const stopTokenIds = this.modelConfig!.stopTokenIds || [];
      const eosToken = this.tokenizer!.getSpecialTokens?.()?.eos;
      let tokensGenerated = 1;

      // Mark kernel cache as warmed after first prefill
      markKernelCacheWarmed();

      // Decode loop
      const decodeStart = performance.now();
      const useBatchPath = opts.batchSize > 1 && this.useGPU && isGPUSamplingAvailable();

      if (opts.debug && useBatchPath) {
        console.log(`[Pipeline] Using batch decode path with batchSize=${opts.batchSize}, stopCheckMode=${opts.stopCheckMode}`);
      }

      while (tokensGenerated < opts.maxTokens) {
        if (options.signal?.aborted) break;

        if (useBatchPath) {
          // Batch path: generate multiple tokens per GPU submit
          const remaining = opts.maxTokens - tokensGenerated;
          const thisBatchSize = Math.min(opts.batchSize, remaining);
          const lastToken = generatedIds[generatedIds.length - 1];

          try {
            const batchResult = await this._generateNTokensGPU(lastToken, thisBatchSize, generatedIds, opts);

            // Process batch results - yield and callback for each token
            const batchTokens: Array<{ id: number; text: string }> = [];
            for (const tokenId of batchResult.tokens) {
              generatedIds.push(tokenId);
              tokensGenerated++;

              const tokenText = this.tokenizer!.decode([tokenId], true, false);
              yield tokenText;
              if (options.onToken) options.onToken(tokenId, tokenText);
              batchTokens.push({ id: tokenId, text: tokenText });
            }

            // Call batch callback
            if (options.onBatch) options.onBatch(batchTokens);

            // Check if we hit a stop condition
            if (batchResult.actualCount < thisBatchSize) {
              break;  // Early stop detected
            }

            // Check stop sequences after batch
            if (opts.stopSequences.length > 0) {
              const fullText = this.tokenizer!.decode(generatedIds.slice(inputIds.length), false);
              if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
            }
          } catch (error) {
            // Fallback to single-token path on batch error
            console.warn('[Pipeline] Batch decode failed, falling back to single-token:', error);
            const nextToken = await this._decodeStep(generatedIds, opts);
            generatedIds.push(nextToken);
            tokensGenerated++;

            const tokenText = this.tokenizer!.decode([nextToken], true, false);
            yield tokenText;
            if (options.onToken) options.onToken(nextToken, tokenText);

            if (isStopToken(nextToken, stopTokenIds, eosToken)) break;
          }
        } else {
          // Single-token path (existing behavior)
          const nextToken = await this._decodeStep(generatedIds, opts);
          generatedIds.push(nextToken);
          tokensGenerated++;

          const tokenText = this.tokenizer!.decode([nextToken], true, false);
          yield tokenText;
          if (options.onToken) options.onToken(nextToken, tokenText);

          // Check stop
          if (isStopToken(nextToken, stopTokenIds, eosToken)) break;

          // Check stop sequences
          if (opts.stopSequences.length > 0) {
            const fullText = this.tokenizer!.decode(generatedIds.slice(inputIds.length), false);
            if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
          }
        }
      }

      this.stats.decodeTimeMs = performance.now() - decodeStart;
      this.stats.tokensGenerated = tokensGenerated;
      this.stats.totalTimeMs = performance.now() - startTime;

      if (opts.debug) {
        console.log(`[Pipeline] Generated ${tokensGenerated} tokens in ${this.stats.totalTimeMs.toFixed(0)}ms`);
      }

      // Log benchmark stats when enabled
      if (opts.benchmark) {
        const ttft = this.stats.prefillTimeMs;
        const decodeTokens = tokensGenerated - 1; // First token comes from prefill
        const decodeSpeed = decodeTokens > 0 ? (decodeTokens / this.stats.decodeTimeMs * 1000) : 0;
        console.log(`[Benchmark] TTFT: ${ttft.toFixed(0)}ms | Prefill: ${this.stats.prefillTimeMs.toFixed(0)}ms | Decode: ${this.stats.decodeTimeMs.toFixed(0)}ms (${decodeTokens} tokens @ ${decodeSpeed.toFixed(1)} tok/s)`);
      }
    } finally {
      this.isGenerating = false;
    }
  }

  async prefillKVOnly(prompt: string, options: GenerateOptions = {}): Promise<KVCacheSnapshot> {
    if (!this.isLoaded) throw new Error('Model not loaded');
    this.stats.gpuTimePrefillMs = undefined;

    const opts = {
      useChatTemplate: options.useChatTemplate ?? false,
      debug: options.debug ?? this.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? false,
    };

    let processedPrompt = prompt;
    if (opts.useChatTemplate) {
      if (this.modelConfig!.isGemma3) {
        processedPrompt = applyGemmaChatTemplate(prompt);
      } else if (this.modelConfig!.isLlama3Instruct) {
        processedPrompt = applyLlama3ChatTemplate(prompt);
      }
    }

    const inputIds = this.tokenizer!.encode(processedPrompt);
    if (opts.debug) {
      console.log(`[Pipeline] PrefillKVOnly: ${inputIds.length} tokens`);
    }

    await this._prefill(inputIds, opts);

    const snapshot = this.kvCache?.clone();
    if (!snapshot) {
      throw new Error('KV cache unavailable after prefill');
    }

    return {
      cache: snapshot,
      seqLen: this.currentSeqLen,
      tokens: inputIds,
    };
  }

  applyKVCacheSnapshot(snapshot: KVCacheSnapshot): void {
    this.kvCache = snapshot.cache.clone();
    if (this.useGPU && this.kvCache) {
      const device = getDevice();
      if (device) {
        this.kvCache.setGPUContext({ device });
      }
    }
    this.currentSeqLen = snapshot.seqLen;
  }

  async *generateWithPrefixKV(
    prefix: KVCacheSnapshot,
    prompt: string,
    options: GenerateOptions = {}
  ): AsyncGenerator<string, void, void> {
    if (!this.isLoaded) throw new Error('Model not loaded');
    if (this.isGenerating) throw new Error('Generation already in progress');

    this.applyKVCacheSnapshot(prefix);

    this.isGenerating = true;
    this._decodeStepCount = 0;
    this.stats.gpuTimePrefillMs = undefined;
    this.stats.gpuTimeDecodeMs = undefined;
    const startTime = performance.now();

    const opts = {
      maxTokens: options.maxTokens ?? 512,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      repetitionPenalty: options.repetitionPenalty ?? 1.1,
      stopSequences: options.stopSequences ?? [],
      useSpeculative: options.useSpeculative ?? false,
      useChatTemplate: options.useChatTemplate ?? false,
      debug: options.debug ?? this.debug,
      debugLayers: options.debugLayers,
      profile: options.profile ?? false,
      benchmark: options.benchmark ?? false,
      disableBatching: options.disableBatching ?? false,
      batchSize: options.batchSize ?? 1,
      stopCheckMode: options.stopCheckMode ?? 'per-token' as const,
    };

    try {
      let processedPrompt = prompt;
      if (opts.useChatTemplate) {
        if (this.modelConfig!.isGemma3) {
          processedPrompt = applyGemmaChatTemplate(prompt);
        } else if (this.modelConfig!.isLlama3Instruct) {
          processedPrompt = applyLlama3ChatTemplate(prompt);
        }
      }

      const inputIds = this.tokenizer!.encode(processedPrompt);
      const generatedIds = [...prefix.tokens, ...inputIds];
      const promptTokenCount = generatedIds.length;

      const prefillStart = performance.now();
      const prefillLogits = await this._prefill(inputIds, opts);
      this.stats.prefillTimeMs = performance.now() - prefillStart;

      applyRepetitionPenalty(prefillLogits, generatedIds, opts.repetitionPenalty);
      const firstToken = sample(prefillLogits, {
        temperature: opts.temperature,
        topP: opts.topP,
        topK: opts.topK,
      });

      generatedIds.push(firstToken);

      const firstText = this.tokenizer!.decode([firstToken], true, false);
      yield firstText;
      if (options.onToken) options.onToken(firstToken, firstText);

      const stopTokenIds = this.modelConfig!.stopTokenIds || [];
      const eosToken = this.tokenizer!.getSpecialTokens?.()?.eos;
      let tokensGenerated = 1;

      markKernelCacheWarmed();

      const decodeStart = performance.now();
      const useBatchPath = opts.batchSize > 1 && this.useGPU && isGPUSamplingAvailable();

      while (tokensGenerated < opts.maxTokens) {
        if (options.signal?.aborted) break;

        if (useBatchPath) {
          const remaining = opts.maxTokens - tokensGenerated;
          const thisBatchSize = Math.min(opts.batchSize, remaining);
          const lastToken = generatedIds[generatedIds.length - 1];

          try {
            const batchResult = await this._generateNTokensGPU(lastToken, thisBatchSize, generatedIds, opts);
            const batchTokens: Array<{ id: number; text: string }> = [];
            for (const tokenId of batchResult.tokens) {
              generatedIds.push(tokenId);
              tokensGenerated++;
              const tokenText = this.tokenizer!.decode([tokenId], true, false);
              yield tokenText;
              if (options.onToken) options.onToken(tokenId, tokenText);
              batchTokens.push({ id: tokenId, text: tokenText });
            }
            if (options.onBatch) options.onBatch(batchTokens);
            if (batchResult.actualCount < thisBatchSize) break;
            if (opts.stopSequences.length > 0) {
              const fullText = this.tokenizer!.decode(generatedIds.slice(promptTokenCount), false);
              if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
            }
          } catch (error) {
            console.warn('[Pipeline] Batch decode failed, falling back to single-token:', error);
            const nextToken = await this._decodeStep(generatedIds, opts);
            generatedIds.push(nextToken);
            tokensGenerated++;
            const tokenText = this.tokenizer!.decode([nextToken], true, false);
            yield tokenText;
            if (options.onToken) options.onToken(nextToken, tokenText);
            if (isStopToken(nextToken, stopTokenIds, eosToken)) break;
          }
        } else {
          const nextToken = await this._decodeStep(generatedIds, opts);
          generatedIds.push(nextToken);
          tokensGenerated++;
          const tokenText = this.tokenizer!.decode([nextToken], true, false);
          yield tokenText;
          if (options.onToken) options.onToken(nextToken, tokenText);
          if (isStopToken(nextToken, stopTokenIds, eosToken)) break;
          if (opts.stopSequences.length > 0) {
            const fullText = this.tokenizer!.decode(generatedIds.slice(promptTokenCount), false);
            if (opts.stopSequences.some(seq => fullText.endsWith(seq))) break;
          }
        }
      }

      this.stats.decodeTimeMs = performance.now() - decodeStart;
      this.stats.tokensGenerated = tokensGenerated;
      this.stats.totalTimeMs = performance.now() - startTime;
    } finally {
      this.isGenerating = false;
    }
  }

  // ==========================================================================
  // Prefill and Decode
  // ==========================================================================

  private async _prefill(inputIds: number[], opts: GenerateOptions): Promise<Float32Array> {
    const numTokens = inputIds.length;
    const config = this.modelConfig!;
    const startPos = this.currentSeqLen;
    this.stats.gpuTimePrefillMs = undefined;

    // Embed tokens
    const embedBufferRaw = this.weights.get('embed');
    if (!(embedBufferRaw instanceof GPUBuffer)) {
      throw new Error('Embed buffer not found or not a GPUBuffer');
    }
    const embedBuffer = embedBufferRaw;
    if (opts.debug) {
      console.log(`[Pipeline] Embed buffer: type=${embedBuffer?.constructor?.name}, size=${embedBuffer?.size ?? 'N/A'}`);
    }

    // Create CommandRecorder for batched GPU operations
    // This reduces GPU submits from 260+ per forward pass to 1
    const device = getDevice();
    // Disable CommandRecorder in debug mode to allow per-step debug readbacks (same as decode).
    const useCheckpoints = opts.debugLayers && opts.debugLayers.length > 0;
    const disableBatching = opts.disableBatching === true || opts.debug === true;
    const createRecorder = (label: string) => {
      if (!device || disableBatching) return undefined;
      return opts.profile ? createProfilingRecorder(label) : createCommandRecorder(label);
    };
    const recorder = createRecorder('prefill');
    const context = this._buildLayerContext(recorder);
    let gpuTimePrefillMs = 0;
    let hasGpuTimePrefill = false;
    const recordProfile = async (rec: CommandRecorder | undefined) => {
      if (!opts.profile || !rec?.isProfilingEnabled()) return;
      const timings = await rec.resolveProfileTimings();
      const total = sumProfileTimings(timings);
      if (total !== null) {
        gpuTimePrefillMs += total;
        hasGpuTimePrefill = true;
      }
    };

    // Enable submit tracking for benchmarking
    const benchmarkSubmits = opts.debug;
    if (benchmarkSubmits) {
      setTrackSubmits(true);
      resetSubmitStats();
    }

    let hiddenStates = await embed(inputIds, embedBuffer, {
      hiddenSize: config.hiddenSize,
      vocabSize: config.vocabSize,
      scaleEmbeddings: config.scaleEmbeddings,
      debug: opts.debug,
      recorder,
    });

    // Debug: check hidden states after embedding
    // IMPORTANT: Must flush recorder before reading buffer (GPU operations are batched)
    if (opts.debug && hiddenStates instanceof GPUBuffer) {
      if (recorder) {
        await recorder.submitAndWait();
        await recordProfile(recorder);
      }
      const sample = await readBuffer(hiddenStates, Math.min(512, hiddenStates.size));
      const f32 = new Float32Array(sample);
      const nanCount = f32.filter(x => !Number.isFinite(x)).length;
      const maxAbs = Math.max(...Array.from(f32).map(x => Math.abs(x)));
      const first8 = Array.from(f32).slice(0, 8).map(x => x.toFixed(4)).join(', ');
      console.log(`[Pipeline] After embed: buffer.label=${hiddenStates.label}, buffer.size=${hiddenStates.size}, maxAbs=${maxAbs.toFixed(4)}`);
      console.log(`[Pipeline] After embed first8=[${first8}], nan=${nanCount}/${f32.length}`);
    }

    // Process all layers
    if (opts.debug) {
      console.log(`[Pipeline] LAYER_LOOP_START: numLayers=${config.numLayers}, useGPU=${context.useGPU}`);
    }
    // Track current recorder (undefined in debug mode to enable per-layer readbacks)
    let currentRecorder = recorder;
    for (let l = 0; l < config.numLayers; l++) {
      // Update context recorder in case it changed at checkpoint
      context.recorder = currentRecorder;

      const prevStates = hiddenStates;
      hiddenStates = await processLayer(l, hiddenStates, numTokens, true, context) as GPUBuffer;

      // Check if this layer is a debug checkpoint
      const isCheckpoint = useCheckpoints && opts.debugLayers?.includes(l);

      // Flush recorder at checkpoint to enable GPU readback
      if (isCheckpoint && currentRecorder) {
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
        currentRecorder = undefined;  // Clear so debug readback works
      }

      // Debug: trace last-token hidden state through layers (position-sensitive issues)
      // Runs when: (1) full debug mode without recorder, OR (2) at checkpoint layers
      const shouldDebug = opts.debug && hiddenStates instanceof GPUBuffer && (!recorder || isCheckpoint);
      if (shouldDebug && !currentRecorder) {
        const device = getDevice();
        if (device) {
          if (allowReadback(`pipeline.prefill.layer-${l}`)) {
            try {
              // Read the full last-token vector to match reference maxAbs (HF hooks use full hidden_size).
              const sampleSize = config.hiddenSize * 4;
              const staging = device.createBuffer({
                size: sampleSize,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
              });
              const enc = device.createCommandEncoder();
              const lastTokenOffset = (numTokens - 1) * config.hiddenSize * 4;
              enc.copyBufferToBuffer(hiddenStates, lastTokenOffset, staging, 0, sampleSize);
              device.queue.submit([enc.finish()]);
              await staging.mapAsync(GPUMapMode.READ);
              const data = new Float32Array(staging.getMappedRange().slice(0));
              staging.unmap();
              staging.destroy();
              let min = Infinity;
              let max = -Infinity;
              let maxAbs = 0;
              for (const v of data) {
                if (!Number.isFinite(v)) continue;
                if (v < min) min = v;
                if (v > max) max = v;
                const av = Math.abs(v);
                if (av > maxAbs) maxAbs = av;
              }
              const sample = Array.from(data).slice(0, 3).map(x => x.toFixed(3)).join(', ');
              console.log(`[Pipeline] LAYER_${l}_LAST[pos=${numTokens - 1}]: min=${min.toFixed(3)}, max=${max.toFixed(3)}, maxAbs=${maxAbs.toFixed(2)}, sample=[${sample}]`);
            } catch (e) {
              console.log(`[Pipeline] LAYER_${l}_LAST: error reading buffer: ${e}`);
            }
          }
        }
      }

      // Recreate recorder after checkpoint to continue batching for remaining layers
      if (isCheckpoint && useCheckpoints && l < config.numLayers - 1) {
        currentRecorder = createRecorder('prefill-cont');
      }

      if (prevStates instanceof GPUBuffer && prevStates !== hiddenStates) {
        // When using recorder, track for deferred cleanup after submit
        // (releasing now would allow pool reuse before recorded ops execute)
        if (currentRecorder) {
          currentRecorder.trackTemporaryBuffer(prevStates);
        } else {
          releaseBuffer(prevStates);
        }
      }
    }

    // Log submit stats after layer loop
    if (benchmarkSubmits) {
      logSubmitStats(`Prefill (${numTokens} tokens, ${config.numLayers} layers)`);
      setTrackSubmits(false);
    }

    // Debug: check final hidden states before logits (at LAST token position)
    if (opts.debug) {
      console.log(`[Pipeline] LAYER_LOOP_DONE, hiddenStates type=${hiddenStates?.constructor?.name}`);
      if (hiddenStates instanceof GPUBuffer && allowReadback('pipeline.prefill.final-hidden')) {
        const device = getDevice();
        // Read from LAST token position (where logits will be computed from)
        const lastTokenOffset = (numTokens - 1) * config.hiddenSize * 4;  // F32
        // Read the full last-token vector for accurate stats.
        const sampleSize = config.hiddenSize * 4;
        const staging = device.createBuffer({
          size: sampleSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(hiddenStates, lastTokenOffset, staging, 0, sampleSize);
        device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        const nanCount = Array.from(data).filter(x => !Number.isFinite(x)).length;
        const nonZero = Array.from(data).filter(x => Number.isFinite(x) && x !== 0).slice(0, 5);
        console.log(`[Pipeline] FINAL_HIDDEN[pos=${numTokens - 1}]: nan=${nanCount}/${data.length}, sample=[${nonZero.map(x => x.toFixed(4)).join(', ')}]`);
      }
    }

    if (hasGpuTimePrefill) {
      this.stats.gpuTimePrefillMs = gpuTimePrefillMs;
    }

    // Compute logits (record into prefill recorder when available to avoid extra submits)
    let logits: Float32Array;
    let logitsVocabSize = config.vocabSize;
    if (currentRecorder) {
      const recorded = await recordLogitsGPU(
        currentRecorder,
        hiddenStates as GPUBuffer,
        numTokens,
        this._getLogitsWeights(),
        this._getLogitsConfig()
      );
      logitsVocabSize = recorded.vocabSize;
      if (hiddenStates instanceof GPUBuffer) {
        currentRecorder.trackTemporaryBuffer(hiddenStates);
      }

      await currentRecorder.submitAndWait();
      await recordProfile(currentRecorder);

      const logitsData = await readBuffer(recorded.logitsBuffer, numTokens * logitsVocabSize * 4);
      releaseBuffer(recorded.logitsBuffer);
      logits = new Float32Array(logitsData);
    } else {
      logits = await computeLogits(
        hiddenStates,
        numTokens,
        this._getLogitsWeights(),
        this._getLogitsConfig(),
        this.useGPU,
        this._debugFlags
      );

      if (hiddenStates instanceof GPUBuffer) releaseBuffer(hiddenStates);
    }

    this.currentSeqLen = startPos + numTokens;

    // Extract last position logits
    const lastLogits = extractLastPositionLogits(logits, numTokens, logitsVocabSize);

    // Log prefill logits for debug
    if (opts.debug) {
      logitsSanity(lastLogits, 'Prefill', (tokens) => this.tokenizer?.decode?.(tokens) || '?');
    }

    // Debug: check KV cache state after prefill
    if (opts.debug) {
      if (this.kvCache?.hasGPUCache?.()) {
        console.log(`[Pipeline] KV cache active after prefill: seqLen=${this.kvCache.layers?.[0]?.seqLen ?? '?'}`);
      } else {
        console.log(`[Pipeline] WARNING: KV cache NOT active after prefill! hasGPUCache=${this.kvCache?.hasGPUCache?.()}`);
      }
    }

    return lastLogits;
  }

  private async _decodeStep(currentIds: number[], opts: GenerateOptions): Promise<number> {
    const lastToken = currentIds[currentIds.length - 1];
    const numTokens = 1;
    const config = this.modelConfig!;

    this._decodeStepCount++;
    const isDebugStep = opts.debug && this._decodeStepCount <= 5;
    if (isDebugStep) {
      const tokenText = this.tokenizer?.decode?.([lastToken]) || '?';
      console.log(`[Decode][${this._decodeStepCount}] token="${tokenText}" pos=${this.currentSeqLen}`);
    }

    // Create CommandRecorder for batched GPU operations
    const device = getDevice();
    // Disable CommandRecorder in debug mode to allow per-step debug readbacks.
    // Use profiling recorder when profiling is enabled.
    let recorder: CommandRecorder | undefined;
    if (device && !opts.debug) {
      recorder = opts.profile
        ? createProfilingRecorder('decode')
        : createCommandRecorder('decode');
    }
    const context = this._buildLayerContext(recorder);

    // Embed single token
    const embedBufferRaw = this.weights.get('embed');
    if (!(embedBufferRaw instanceof GPUBuffer)) {
      throw new Error('Embed buffer not found or not a GPUBuffer');
    }
    let hiddenStates = await embed([lastToken], embedBufferRaw, {
      hiddenSize: config.hiddenSize,
      vocabSize: config.vocabSize,
      scaleEmbeddings: config.scaleEmbeddings,
      recorder,
    });

    // Debug: check embedding output for decode step 1
    if (opts.debug && this._decodeStepCount === 1 && hiddenStates instanceof GPUBuffer) {
      const embedData = await readBuffer(hiddenStates);
      const embedArr = new Float32Array(embedData);
      const sample = embedArr.slice(0, 5);
      const maxAbs = Math.max(...embedArr.map(Math.abs));
      const nonZero = embedArr.filter(x => Math.abs(x) > 1e-10).length;
      console.log(`[Decode][1] Embed check: maxAbs=${maxAbs.toFixed(2)}, nonZero=${nonZero}/${embedArr.length}, sample=[${Array.from(sample).map(v => v.toFixed(3)).join(', ')}]`);
    }

    // Enable submit tracking for first decode step benchmarking
    const benchmarkSubmits = this._decodeStepCount <= 3 && opts.debug;
    if (benchmarkSubmits) {
      setTrackSubmits(true);
      resetSubmitStats();
    }

    // Debug: check KV cache status for decode
    const hasGPUCache = context.kvCache?.hasGPUCache?.() ?? false;
    if (opts.debug && this._decodeStepCount === 1) {
      console.log(`[Decode] KV cache check: hasGPUCache=${hasGPUCache}, currentSeqLen=${context.currentSeqLen}`);
    }

    // Process all layers
    for (let l = 0; l < config.numLayers; l++) {
      const prevStates = hiddenStates;
      hiddenStates = await processLayer(l, hiddenStates, numTokens, false, context) as GPUBuffer;
      if (prevStates instanceof GPUBuffer && prevStates !== hiddenStates) {
        // When using recorder, track for deferred cleanup after submit
        // (releasing now would allow pool reuse before recorded ops execute)
        if (recorder) {
          recorder.trackTemporaryBuffer(prevStates);
        } else {
          releaseBuffer(prevStates);
        }
      }
    }

    // FUSED DECODE PATH: Record layers + logits + sampling in single command buffer
    // This reduces GPU syncs from 7+ per token to 2, enabling ~3-4x speedup
    const useGPUSampling = this.useGPU && isGPUSamplingAvailable();
    const useFusedDecode = recorder && useGPUSampling && hiddenStates instanceof GPUBuffer;

    if (useFusedDecode) {
      // Continue recording logits into same command buffer (no submit yet)
      const { logitsBuffer, vocabSize } = await recordLogitsGPU(
        recorder,
        hiddenStates,
        numTokens,
        this._getLogitsWeights(),
        this._getLogitsConfig(),
      );

      // Continue recording sampling into same command buffer (no submit yet)
      // Use argmax for greedy (temperature < 0.01) or top-k sampling for temperature >= 0.01
      const sampleOutputBuffer = opts.temperature < 0.01
        ? await recordArgmax(recorder, logitsBuffer, vocabSize)
        : await recordGPUSample(recorder, logitsBuffer, vocabSize, {
            temperature: opts.temperature,
            topK: opts.topK,
          });

      // Track buffers for cleanup after submit
      recorder.trackTemporaryBuffer(hiddenStates);
      recorder.trackTemporaryBuffer(logitsBuffer);

      // NOW submit everything at once (layers + logits + sampling)
      recorder.submit();

      // Single sync point: copy sample result to staging buffer and read
      if (!allowReadback('pipeline.decode.sample')) {
        throw new Error('[Pipeline] GPU readback disabled for sampling');
      }

      const stagingBuffer = device.createBuffer({
        label: 'sample_staging',
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const copyEncoder = device.createCommandEncoder({ label: 'sample_copy' });
      copyEncoder.copyBufferToBuffer(sampleOutputBuffer, 0, stagingBuffer, 0, 4);
      device.queue.submit([copyEncoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const nextToken = new Uint32Array(stagingBuffer.getMappedRange())[0];
      stagingBuffer.unmap();
      stagingBuffer.destroy();

      releaseBuffer(sampleOutputBuffer);

      // Log submit stats
      if (benchmarkSubmits) {
        logSubmitStats(`Decode step ${this._decodeStepCount} (${config.numLayers} layers, fused)`);
        setTrackSubmits(false);
      }

      // Resolve and log profiling timings (use warn so it's not silenced by benchmark mode)
      if (opts.profile && recorder.isProfilingEnabled()) {
        const timings = await recorder.resolveProfileTimings();
        const total = sumProfileTimings(timings);
        if (total !== null) {
          this.stats.gpuTimeDecodeMs = (this.stats.gpuTimeDecodeMs ?? 0) + total;
        }
        if (timings) {
          console.warn(`[Profile] Decode step ${this._decodeStepCount}:`);
          console.warn(CommandRecorder.formatProfileReport(timings));
        }
      }

      this.currentSeqLen++;
      return nextToken;
    }

    // FALLBACK PATH: Submit layers first, then do logits + sampling separately
    // Used when: debug mode or no recorder
    if (recorder) {
      await recorder.submitAndWait();

      // Resolve and log profiling timings for layers (logits not included in this path)
      // Use warn so it's not silenced by benchmark mode
      if (opts.profile && recorder.isProfilingEnabled()) {
        const timings = await recorder.resolveProfileTimings();
        const total = sumProfileTimings(timings);
        if (total !== null) {
          this.stats.gpuTimeDecodeMs = (this.stats.gpuTimeDecodeMs ?? 0) + total;
        }
        if (timings) {
          console.warn(`[Profile] Decode step ${this._decodeStepCount} (layers only):`);
          console.warn(CommandRecorder.formatProfileReport(timings));
        }
      }
    }

    // Log submit stats after decode layer loop
    if (benchmarkSubmits) {
      logSubmitStats(`Decode step ${this._decodeStepCount} (${config.numLayers} layers)`);
      setTrackSubmits(false);
    }

    // Debug: check hidden states after layer processing (decode step 1 only in debug mode)
    if (opts.debug && this._decodeStepCount === 1 && hiddenStates instanceof GPUBuffer) {
      const debugDevice = getDevice();
      if (debugDevice) {
        if (allowReadback('pipeline.decode.debug-hidden')) {
          const sampleSize = Math.min(512, hiddenStates.size);
          const staging = debugDevice.createBuffer({
            size: sampleSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          });
          const enc = debugDevice.createCommandEncoder();
          enc.copyBufferToBuffer(hiddenStates, 0, staging, 0, sampleSize);
          debugDevice.queue.submit([enc.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const data = new Float32Array(staging.getMappedRange().slice(0));
          staging.unmap();
          staging.destroy();
          const nanCount = Array.from(data).filter(x => !Number.isFinite(x)).length;
          const nonZero = Array.from(data).filter(x => Number.isFinite(x) && x !== 0).slice(0, 5);
          console.log(`[Decode][1] HIDDEN_AFTER_LAYERS: nan=${nanCount}/${data.length}, nonZero=${nonZero.length}, sample=[${nonZero.map(x => x.toFixed(4)).join(', ')}]`);
        }
      }
    }

    // GPU sampling path (non-fused, for temperature > 0 or fallback)
    if (useGPUSampling) {
      const logitsResult = await computeLogitsGPU(
        hiddenStates,
        numTokens,
        this._getLogitsWeights(),
        this._getLogitsConfig(),
        this._debugFlags
      );

      if (hiddenStates instanceof GPUBuffer) releaseBuffer(hiddenStates);

      if (logitsResult) {
        const { logitsBuffer, vocabSize } = logitsResult;

        const nextToken = opts.temperature < 0.01
          ? await runArgmax(logitsBuffer, vocabSize)
          : await runGPUSample(logitsBuffer, vocabSize, {
              temperature: opts.temperature,
              topK: opts.topK,
            });

        releaseBuffer(logitsBuffer);
        this.currentSeqLen++;
        return nextToken;
      }
    }

    // CPU path: read back logits, sample on CPU
    const logits = await computeLogits(
      hiddenStates,
      numTokens,
      this._getLogitsWeights(),
      this._getLogitsConfig(),
      this.useGPU,
      this._debugFlags
    );

    if (hiddenStates instanceof GPUBuffer) releaseBuffer(hiddenStates);

    // Log top-5 for debug
    if (isDebugStep) {
      logitsSanity(logits, `Decode[${this._decodeStepCount}]`, opts.decode);
    }

    // Apply penalty and sample
    applyRepetitionPenalty(logits, currentIds, opts.repetitionPenalty);
    const nextToken = sample(logits, {
      temperature: opts.temperature,
      topP: opts.topP,
      topK: opts.topK,
    });

    this.currentSeqLen++;
    return nextToken;
  }

  /**
   * GPU-Only Decode Loop: Generate N tokens in a single GPU submit.
   * Records all N decode iterations in one command buffer to eliminate CPU roundtrips.
   *
   * Returns: { tokens: number[], actualCount: number }
   * actualCount may be < N if EOS or max_tokens is hit.
   */
  private async _generateNTokensGPU(
    startToken: number,
    N: number,
    currentIds: number[],
    opts: GenerateOptions
  ): Promise<{ tokens: number[], actualCount: number }> {
    const device = getDevice();
    const config = this.modelConfig!;
    const recorder = opts.profile
      ? createProfilingRecorder('batch_decode')
      : createCommandRecorder('batch_decode');

    const stopCheckMode = opts.stopCheckMode ?? 'per-token';

    // Create stop flags buffer: N flags (one per iteration)
    const stopBufferSize = stopCheckMode === 'per-token' ? N * 4 : 0;
    const stopBuffer = stopCheckMode === 'per-token'
      ? device.createBuffer({
        size: stopBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      })
      : null;

    // Get stop token config
    const stopTokenIds = config.stopTokenIds || [];
    const eosToken = this.tokenizer?.getSpecialTokens?.()?.eos;
    const eosTokenId = eosToken ?? stopTokenIds[0] ?? 1;  // fallback to 1 (common EOS)
    const maxTokens = opts.maxTokens || 1024;

    // Create single-token buffer for each iteration
    // We'll create N+1 token buffers (one for start token + N for each iteration's output)
    const tokenBuffers: GPUBuffer[] = [];
    for (let i = 0; i <= N; i++) {
      const buf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      tokenBuffers.push(buf);
    }

    // Write start token to first buffer
    device.queue.writeBuffer(tokenBuffers[0], 0, new Uint32Array([startToken]));

    // Build context for layer processing
    const context = this._buildLayerContext(recorder);
    const embedBufferRaw = this.weights.get('embed');
    if (!(embedBufferRaw instanceof GPUBuffer)) {
      throw new Error('Embed buffer not found or not a GPUBuffer');
    }
    const embedBuffer = embedBufferRaw;

    // Record N decode iterations
    for (let i = 0; i < N; i++) {
      const currentPos = this.currentSeqLen + i;

      // 1. Embed: Read tokenBuffers[i] → hidden states
      let hiddenStates = await recordGather(
        recorder,
        tokenBuffers[i],  // indices buffer (contains 1 token)
        embedBuffer,
        1,  // numTokens = 1
        config.hiddenSize,
        config.vocabSize
      );

      // Apply Gemma embedding scaling if needed: scale by sqrt(hidden_size)
      if (config.scaleEmbeddings) {
        const scaleFactor = Math.sqrt(config.hiddenSize);
        const prevHidden = hiddenStates;
        hiddenStates = await recordScale(recorder, hiddenStates, scaleFactor, {
          count: config.hiddenSize,  // 1 token * hiddenSize
        });
        if (prevHidden !== hiddenStates) {
          recorder.trackTemporaryBuffer(prevHidden);
        }
      }

      // 2. Process all layers
      for (let l = 0; l < config.numLayers; l++) {
        const prevStates = hiddenStates;
        hiddenStates = await processLayer(l, hiddenStates, 1, false, context) as GPUBuffer;
        if (prevStates !== hiddenStates) {
          recorder.trackTemporaryBuffer(prevStates);
        }
      }

      // 3. Compute logits
      const { logitsBuffer, vocabSize } = await recordLogitsGPU(
        recorder,
        hiddenStates,
        1,  // numTokens = 1
        this._getLogitsWeights(),
        this._getLogitsConfig()
      );
      recorder.trackTemporaryBuffer(hiddenStates);

      // 4. Sample next token → write to tokenBuffers[i+1]
      // Use temperature-based sampling if temperature > 0, otherwise argmax
      const temperature = opts.temperature ?? 0.7;
      const topK = opts.topK ?? 40;
      const sampledTokenBuffer = temperature < 0.01
        ? await recordArgmax(recorder, logitsBuffer, vocabSize)
        : await recordGPUSample(recorder, logitsBuffer, vocabSize, { temperature, topK });
      recorder.trackTemporaryBuffer(logitsBuffer);

      // Copy sampled token to tokenBuffers[i+1] for next iteration
      const encoder = recorder.getEncoder();
      encoder.copyBufferToBuffer(sampledTokenBuffer, 0, tokenBuffers[i + 1], 0, 4);

      // 5. Check stop condition (only in 'per-token' mode)
      if (stopCheckMode === 'per-token') {
        const stopFlagBuffer = recordCheckStop(recorder, {
          sampledTokenBuffer: tokenBuffers[i + 1],
          eosTokenId,
          maxTokens,
          currentPos: currentPos + 1,
        });

        // Copy stop flag to main stopBuffer
        encoder.copyBufferToBuffer(stopFlagBuffer, 0, stopBuffer!, i * 4, 4);
        recorder.trackTemporaryBuffer(stopFlagBuffer);
      }

      recorder.trackTemporaryBuffer(sampledTokenBuffer);
    }

    // Submit all N iterations at once
    recorder.submit();

    // Readback tokens and stop flags
    if (!allowReadback('pipeline.decode.multi-token')) {
      throw new Error('[Pipeline] GPU readback disabled for multi-token decode');
    }

    const copyEncoder = device.createCommandEncoder();

    // Only copy stop buffer if we recorded stop flags (per-token mode)
    let stopStagingBuffer: GPUBuffer | null = null;
    if (stopCheckMode === 'per-token' && stopBuffer) {
      stopStagingBuffer = device.createBuffer({
        size: stopBufferSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      copyEncoder.copyBufferToBuffer(stopBuffer, 0, stopStagingBuffer, 0, stopBufferSize);
    }

    // Copy each token buffer to staging
    const tokenStagingBuffers: GPUBuffer[] = [];
    for (let i = 1; i <= N; i++) {  // Skip tokenBuffers[0] (start token)
      const staging = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      copyEncoder.copyBufferToBuffer(tokenBuffers[i], 0, staging, 0, 4);
      tokenStagingBuffers.push(staging);
    }

    device.queue.submit([copyEncoder.finish()]);

    // Map all buffers
    const mapPromises = tokenStagingBuffers.map(b => b.mapAsync(GPUMapMode.READ));
    if (stopStagingBuffer) {
      mapPromises.push(stopStagingBuffer.mapAsync(GPUMapMode.READ));
    }
    await Promise.all(mapPromises);

    // Read tokens
    const tokens: number[] = [];
    for (const staging of tokenStagingBuffers) {
      const tokenId = new Uint32Array(staging.getMappedRange())[0];
      tokens.push(tokenId);
    }

    // Find first stop based on mode
    let actualCount = N;
    if (stopCheckMode === 'per-token' && stopStagingBuffer) {
      // Use GPU-computed stop flags
      const stopFlags = new Uint32Array(stopStagingBuffer.getMappedRange().slice(0));
      for (let i = 0; i < N; i++) {
        if (stopFlags[i] === 1) {
          actualCount = i + 1;
          break;
        }
      }
      stopStagingBuffer.unmap();
    } else {
      // Batch mode: check stop tokens on CPU after readback
      for (let i = 0; i < N; i++) {
        if (isStopToken(tokens[i], stopTokenIds, eosToken)) {
          actualCount = i + 1;
          break;
        }
      }
    }

    tokenStagingBuffers.forEach(b => b.unmap());

    // Trim to actual count
    const generatedTokens = tokens.slice(0, actualCount);

    // Cleanup
    tokenBuffers.forEach(b => b.destroy());
    stopBuffer?.destroy();
    tokenStagingBuffers.forEach(b => b.destroy());
    if (stopStagingBuffer) stopStagingBuffer.destroy();

    if (opts.profile && recorder.isProfilingEnabled()) {
      const timings = await recorder.resolveProfileTimings();
      const total = sumProfileTimings(timings);
      if (total !== null) {
        this.stats.gpuTimeDecodeMs = (this.stats.gpuTimeDecodeMs ?? 0) + total;
      }
    }

    this.currentSeqLen += actualCount;

    return { tokens: generatedTokens, actualCount };
  }

  // ==========================================================================
  // Context and Config Builders
  // ==========================================================================

  private _buildLayerContext(recorder?: CommandRecorder): LayerContext {
    const config = this.modelConfig!;
    const { getWeightBuffer, getNormWeightBuffer } = createWeightBufferHelpers(
      this._getWeightBufferConfig(),
      this._debugFlags
    );

    return {
      config,
      weights: this.weights,
      kvCache: this.kvCache!,
      currentSeqLen: this.currentSeqLen,
      useGPU: this.useGPU,
      debug: this.debug,
      ropeFreqsCos: this.ropeFreqsCos,
      ropeFreqsSin: this.ropeFreqsSin,
      ropeLocalCos: this.ropeLocalCos,  // Gemma 3: Local RoPE for sliding_attention layers
      ropeLocalSin: this.ropeLocalSin,
      attentionKernelOverride: this.attentionKernelOverride,
      weightConfig: this._getWeightBufferConfig(),
      debugFlags: this._debugFlags,
      expertWeights: this.expertWeights,
      expertLoader: this.dopplerLoader as ExpertLoader | null,
      moeRouter: this.moeRouter,
      layerRouterWeights: this.layerRouterWeights as Map<number, { weight: GPUBuffer | Float32Array; bias: GPUBuffer | Float32Array | null }> | undefined,
      recorder,
      lora: this.loraAdapter,
    };
  }

  private _getWeightBufferConfig(): WeightBufferConfig {
    return {
      rmsNormWeightOffset: this.modelConfig!.rmsNormWeightOffset,
    };
  }

  private _getLogitsWeights(): LogitsWeights {
    const finalNorm = this.weights.get('final_norm');
    const lmHead = this.weights.get('lm_head');
    if (!finalNorm || !(finalNorm instanceof GPUBuffer || finalNorm instanceof Float32Array)) {
      throw new Error('Final norm not found or invalid type');
    }
    if (!lmHead || !(lmHead instanceof GPUBuffer || lmHead instanceof Float32Array)) {
      throw new Error('LM head not found or invalid type');
    }
    return { finalNorm, lmHead };
  }

  private _getLogitsConfig(): LogitsConfig {
    const config = this.modelConfig!;
    return {
      hiddenSize: config.hiddenSize,
      vocabSize: config.vocabSize,
      rmsNormEps: config.rmsNormEps,
      useTiedEmbeddings: this.useTiedEmbeddings,
      embeddingVocabSize: this.embeddingVocabSize,
    };
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  getStats(): PipelineStats {
    return { ...this.stats };
  }

  getBatchingStats(): BatchingStats {
    return { ...this.batchingStats };
  }

  async unload(): Promise<void> {
    this.kvCache?.clear();
    this.weights.clear();
    this.expertWeights.clear();
    this.loraAdapter = null;
    this.isLoaded = false;
    this.currentSeqLen = 0;
    console.log('[Pipeline] Unloaded');
  }

  setLoRAAdapter(adapter: LoRAAdapter | null): void {
    this.loraAdapter = adapter;
  }

  getActiveLoRA(): LoRAAdapter | null {
    return this.loraAdapter;
  }

  reset(): void {
    this.kvCache?.clear();
    this.currentSeqLen = 0;
    this._decodeStepCount = 0;
    this._debugFlags = {};
    this.stats = {
      tokensGenerated: 0,
      totalTimeMs: 0,
      prefillTimeMs: 0,
      decodeTimeMs: 0,
      gpuTimePrefillMs: undefined,
      gpuTimeDecodeMs: undefined,
    };
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export async function createPipeline(manifest: Manifest, contexts: PipelineContexts = {}): Promise<InferencePipeline> {
  const pipeline = new InferencePipeline();
  await pipeline.initialize(contexts);
  await pipeline.loadModel(manifest);
  return pipeline;
}

// Backwards compatibility alias
export { InferencePipeline as Pipeline };
