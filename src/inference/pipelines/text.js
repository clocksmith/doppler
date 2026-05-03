
import { getDevice, initDevice, getKernelCapabilities } from '../../gpu/device.js';
import { getBufferPool as getGlobalBufferPool, readBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { log } from '../../debug/index.js';
import { configurePerfGuards } from '../../gpu/perf-guards.js';
import { MoERouter } from '../moe-router.js';
import { DecodeBufferManager } from '../decode-buffers.js';
import { DecodeRing } from '../decode-ring.js';
import { applyPipelineContexts, restorePipelineContexts } from './context.js';
import { createInitializedPipeline } from './factory.js';

// Pipeline sub-modules
import { PipelineState } from './text/state.js';
import { PipelineGenerator } from './text/generator.js';
import { parseModelConfig } from './text/config.js';
import {
  initRoPEFrequencies,
  createKVCache,
  loadWeights,
  initMoERouter,
  initSpeculativeDecoder,
  fuseQKVWeights,
  initEmulation,
  destroyEmulation,
} from './text/init.js';
import { formatChatMessages } from './text/chat-format.js';
import {
  runKernelWarmup,
  applyModelBatchingRuntimeDefaults,
  resolveKernelPathState,
  initTokenizerFromManifest,
  assertManifestComputeLaneBinding,
} from './text/model-load.js';
import { resolvePerLayerInputsSession } from './text/generator-helpers.js';
import { getKernelPathActivationDtype } from '../../config/kernel-path-loader.js';
import { applyPipelineDebugConfig } from './text/debug-utils.js';
import { resolveLayerPipeline } from './text/layer-plan.js';
import { compileExecutionPlanState, resolveActiveExecutionPlan } from './text/execution-plan.js';
import { assertDtypeConsistency } from './text/dtype-contract.js';
import { applyExecutionV1RuntimeConfig, hasExecutionV1 } from './text/execution-v1.js';
import { getPlatform } from '../../config/platforms/loader.js';
import {
  createLinearAttentionRuntime,
  hasLinearAttentionLayers,
  resetLinearAttentionRuntime,
  restoreLinearAttentionRuntime,
} from './text/linear-attention.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { registerPipeline, getPipelineFactory } from './registry.js';
import { selectRuleValue } from '../../rules/rule-registry.js';

// AbortSignal contract: every public inference primitive on this pipeline
// accepts `options.signal` (or `args.signal`). When the signal aborts the
// call rejects with `AbortError` so callers (the dream typing-clock
// controller, batch eval scripts, etc.) can cancel superseded work without
// waiting for the model to finish. Internal checkpoints fire between
// dispatches; mid-dispatch cancellation is not provided because WebGPU
// command-buffer cancellation is not exposed by the spec.
export class AbortError extends Error {
  constructor(message = 'Doppler: aborted') {
    super(message);
    this.name = 'AbortError';
    this.code = 'ABORT_ERR';
  }
}

export function isAbortError(err) {
  return !!err && (err.name === 'AbortError' || err.code === 'ABORT_ERR' || err.code === 20);
}

function assertNotAborted(signal) {
  if (signal && signal.aborted) {
    throw new AbortError(typeof signal.reason === 'string' ? signal.reason : 'Doppler: aborted');
  }
}
import { initConvLayerState } from './text/ops.js';
import { destroyPleBufferCache, destroyPleRuntimeCache } from './text/per-layer-inputs.js';

function destroyMoERouter(router) {
  if (router && typeof router.destroy === 'function') {
    router.destroy();
  }
}

function createPipelineLoadPhaseError(error, phase, context = {}) {
  const message = error?.message || String(error);
  const wrapped = new Error(
    `Pipeline load phase "${phase}" failed: ${message}`,
    error instanceof Error ? { cause: error } : undefined
  );
  wrapped.name = error?.name || 'Error';
  if (error?.code !== undefined) {
    wrapped.code = error.code;
  }
  wrapped.details = {
    ...(error?.details && typeof error.details === 'object' ? error.details : {}),
    pipelineLoadPhase: phase,
    ...context,
  };
  return wrapped;
}

async function withPipelineLoadPhase(phase, context, run) {
  try {
    return await run();
  } catch (error) {
    if (error?.details?.pipelineLoadPhase) {
      throw error;
    }
    throw createPipelineLoadPhaseError(error, phase, context);
  }
}

function resolveSingleSpecialTokenId(tokenizer, tokenText, label) {
  const rawTokenIds = tokenizer?.encode?.(tokenText);
  const tokenIds = Array.isArray(rawTokenIds)
    ? rawTokenIds
    : (ArrayBuffer.isView(rawTokenIds) ? Array.from(rawTokenIds) : null);
  if (!Array.isArray(tokenIds) || tokenIds.length !== 1) {
    throw new Error(
      `[Pipeline] transcribeImage: tokenizer must encode ${label} "${tokenText}" as exactly one token.`
    );
  }
  const tokenId = Number(tokenIds[0]);
  if (!Number.isFinite(tokenId) || Math.floor(tokenId) !== tokenId || tokenId < 0) {
    throw new Error(
      `[Pipeline] transcribeImage: tokenizer returned invalid ${label} token id "${tokenIds[0]}".`
    );
  }
  return tokenId;
}

function expandImagePlaceholderTokenIds(tokenIds, imageTokenId, numImageTokens, options = {}) {
  const normalizedTokenIds = Array.isArray(tokenIds)
    ? Int32Array.from(tokenIds)
    : (ArrayBuffer.isView(tokenIds) ? Int32Array.from(tokenIds) : null);
  if (!(normalizedTokenIds instanceof Int32Array)) {
    throw new Error(
      '[Pipeline] transcribeImage: tokenizer.encode() must return an array or typed array of token IDs.'
    );
  }
  if (!Number.isFinite(numImageTokens) || Math.floor(numImageTokens) !== numImageTokens || numImageTokens < 1) {
    throw new Error(
      `[Pipeline] transcribeImage: image token span must be a positive integer, got ${numImageTokens}.`
    );
  }

  let placeholderIndex = -1;
  let placeholderCount = 0;
  for (let i = 0; i < normalizedTokenIds.length; i++) {
    if (normalizedTokenIds[i] !== imageTokenId) continue;
    if (placeholderIndex < 0) {
      placeholderIndex = i;
    }
    placeholderCount++;
  }

  if (placeholderCount !== 1) {
    throw new Error(
      `[Pipeline] transcribeImage: expected exactly one image_token_id (${imageTokenId}) placeholder ` +
      `from the chat template, got ${placeholderCount}.`
    );
  }

  const boiTokenId = Number.isInteger(options.boiTokenId) ? options.boiTokenId : null;
  const eoiTokenId = Number.isInteger(options.eoiTokenId) ? options.eoiTokenId : null;
  const prefixExtra = boiTokenId == null ? 0 : 1;
  const suffixExtra = eoiTokenId == null ? 0 : 1;
  const expandedLength = normalizedTokenIds.length - 1 + prefixExtra + numImageTokens + suffixExtra;
  const expanded = new Int32Array(expandedLength);
  expanded.set(normalizedTokenIds.subarray(0, placeholderIndex), 0);
  let writeOffset = placeholderIndex;
  if (boiTokenId != null) {
    expanded[writeOffset++] = boiTokenId;
  }
  expanded.fill(imageTokenId, writeOffset, writeOffset + numImageTokens);
  const imageStartOffset = writeOffset;
  writeOffset += numImageTokens;
  if (eoiTokenId != null) {
    expanded[writeOffset++] = eoiTokenId;
  }
  expanded.set(
    normalizedTokenIds.subarray(placeholderIndex + 1),
    writeOffset
  );

  return {
    inputIds: expanded,
    imageStartOffset,
  };
}

export function buildConservativeMultimodalGenerationOptions(options = {}) {
  return {
    ...options,
    disableCommandBatching: true,
    disableMultiTokenDecode: true,
    stopCheckMode: 'per-token',
  };
}

// ============================================================================
// Main Inference Pipeline Class
// ============================================================================

export class InferencePipeline extends PipelineState {

  generator;

  // Progress callback

  _onProgress = null;


  _preloadedWeights = null;
  runtimeOverrides = null;

  constructor() {
    super();
    this.generator = new PipelineGenerator(this);
    this.decodeBuffers = new DecodeBufferManager();
    this.decodeRing = new DecodeRing();
    this.linearAttentionRuntime = createLinearAttentionRuntime();
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================


  async initialize(contexts = {}) {
    const { runtimeConfig, sharedDebug } = applyPipelineContexts(this, contexts, {
      assignGpuContext: true,
      assignUseGPU: true,
      assignMemoryContext: true,
      assignStorageContext: true,
    });
    this.runtimeConfig = runtimeConfig;
    this.runtimeOverrides = contexts.runtimeConfig == null
      ? null
      : (typeof structuredClone === 'function'
        ? structuredClone(contexts.runtimeConfig)
        : JSON.parse(JSON.stringify(contexts.runtimeConfig)));
    applyPipelineDebugConfig(sharedDebug?.pipeline);
    configurePerfGuards(sharedDebug?.perfGuards);

    if (!this.gpuContext?.device && typeof globalThis.navigator !== 'undefined' && globalThis.navigator?.gpu) {
      const device = await initDevice();
      if (!device || typeof device !== 'object' || typeof device.createBuffer !== 'function' || !device.queue) {
        throw new Error(
          'GPU device initialization returned an invalid device object. ' +
          'Expected an object with queue and createBuffer. Check WebGPU adapter availability.'
        );
      }
      this.gpuContext = { device };
      this.useGPU = true;
    }

    this.emulation = await initEmulation(this.runtimeConfig);

    this.debug = sharedDebug?.pipeline?.enabled === true;
    log.debug('Pipeline', 'Initialized', { useGPU: this.useGPU, debug: this.debug });
  }


  async loadModel(manifest) {
    const loadStart = performance.now();
    this.manifest = manifest;
    this.decodeRing?.release();
    if (this.sampleReadbackBuffer) {
      this.sampleReadbackBuffer.destroy();
      this.sampleReadbackBuffer = null;
    }
    this.linearAttentionRuntime = resetLinearAttentionRuntime(this.linearAttentionRuntime);
    destroyMoERouter(this.moeRouter);
    this.moeRouter = null;

    // ========================================================================
    // Config Resolution Passes
    //
    // The following passes mutate this.runtimeConfig in a fixed order.
    // Each pass is allowed to read the full runtimeConfig but must only
    // mutate its own documented subset. Reordering passes may change
    // resolved values.
    //
    // Phase 1 — applyExecutionV1RuntimeConfig
    //   Reads: manifest.inference.execution, kernelCapabilities, platform
    //   Mutates: runtimeConfig.inference (kernelPath, pipeline, compute,
    //            session via runtimeInferencePatch)
    //
    // Phase 2 — parseModelConfig + applyModelBatchingRuntimeDefaults
    //   Reads: manifest.architecture, runtimeConfig.inference.modelOverrides
    //   Mutates: runtimeConfig.inference.batching,
    //            runtimeConfig.inference.generation
    //
    // Phase 3 — resolveKernelPathState
    //   Reads: manifest, modelConfig.kernelPath, runtimeConfig.inference.kernelPath
    //   Mutates: runtimeConfig.inference.compute.activationDtype,
    //            runtimeConfig.inference.session.kvcache.kvDtype,
    //            runtimeConfig.inference.session.compute.defaults.outputDtype
    //
    // Phase 4 — _resolveLayerPipeline
    //   Reads: runtimeConfig.inference.pipeline, modelConfig.layerPipeline,
    //          executionV1State.runtimeInferencePatch.pipeline
    //   Mutates: this.layerPipelinePlan (does not mutate runtimeConfig)
    // ========================================================================

    let configResolutionPhase = 0;

    // Phase 1: execution-v1 runtime config
    configResolutionPhase = 1;
    log.debug('Pipeline', `Config resolution phase ${configResolutionPhase}: applyExecutionV1RuntimeConfig`);
    if (hasExecutionV1(manifest.inference)) {
      let capabilities = null;
      let platform = null;
      try {
        capabilities = getKernelCapabilities();
      } catch {
        // Device not yet initialized — transforms will be skipped
      }
      try {
        platform = getPlatform();
      } catch {
        // Platform not yet initialized — use null fallback
      }

      const executionV1Runtime = applyExecutionV1RuntimeConfig({
        runtimeConfig: this.runtimeConfig,
        runtimeOverrides: this.runtimeOverrides,
        manifest,
        modelId: manifest.modelId ?? 'model',
        numLayers: Number(manifest.architecture?.numLayers ?? 0),
        capabilities,
        platform,
      });
      if (executionV1Runtime.executionV1State) {
        this.runtimeConfig = executionV1Runtime.runtimeConfig;
        this.executionV1State = executionV1Runtime.executionV1State;
        const transformInfo = this.executionV1State.appliedTransforms?.length > 0
          ? `, transforms=[${this.executionV1State.appliedTransforms.join(', ')}]`
          : '';
        const fallbackInfo = this.executionV1State.fallbackKernelPath
          ? ', fallbackKernelPath=yes'
          : '';
        const laneIntegrity = this.executionV1State.laneIntegrity;
        const laneInfo = laneIntegrity?.status === 'transformed'
          ? `, laneIntegrity=transformed(declared=${laneIntegrity.declared.activationDtype}/${laneIntegrity.declared.kvDtype},` +
            `executed=${laneIntegrity.executed.activationDtype}/${laneIntegrity.executed.kvDtype})`
          : '';
        log.info(
          'Pipeline',
          `Execution v1 enabled (steps=${this.executionV1State.resolvedSteps.all.length}, ` +
          `kernelPathInline=${this.executionV1State.runtimeInferencePatch.kernelPath ? 'yes' : 'no'}, ` +
          `pipelineInline=${this.executionV1State.runtimeInferencePatch.pipeline ? 'yes' : 'no'}` +
          `${transformInfo}${laneInfo}${fallbackInfo})`
        );
      }
    }

    // Phase 2: model config + batching defaults
    configResolutionPhase = 2;
    log.debug('Pipeline', `Config resolution phase ${configResolutionPhase}: parseModelConfig + applyModelBatchingRuntimeDefaults`);
    const modelOverrides = (this.runtimeConfig.inference.modelOverrides);
    this.modelConfig = parseModelConfig(manifest, modelOverrides);
    this.runtimeConfig = applyModelBatchingRuntimeDefaults(
      this.runtimeConfig,
      manifest,
      this.modelConfig,
      this.runtimeOverrides
    );
    this.useTiedEmbeddings = this.modelConfig.useTiedEmbeddings;
    this.embeddingVocabSize = this.modelConfig.embeddingVocabSize;
    this.embeddingTranspose = this.modelConfig.embeddingTranspose;

    // Vision capability detection — gated by manifest fields
    const imageTokenId = manifest.image_token_id;
    const hasVisionQuant = manifest.quantizationInfo?.vision != null;
    if (Number.isInteger(imageTokenId) && imageTokenId > 0 && hasVisionQuant) {
      this.visionCapable = true;
      this.imageTokenId = imageTokenId;
      this.visionConfig = this.modelConfig.visionConfig;
      if (!this.visionConfig) {
        throw new Error(
          `Manifest declares image_token_id=${imageTokenId} and quantizationInfo.vision ` +
          'but no vision_config was resolved. Check conversion config.'
        );
      }
      log.info('Pipeline', `Vision capable: imageTokenId=${imageTokenId}`);
    } else {
      this.visionCapable = false;
    }

    // Audio capability detection — gated by manifest fields
    const audioTokenId = manifest.audio_token_id;
    const hasAudioQuant = manifest.quantizationInfo?.audio != null;
    if (Number.isInteger(audioTokenId) && audioTokenId > 0 && hasAudioQuant) {
      this.audioCapable = true;
      this.audioTokenId = audioTokenId;
      this.audioConfig = this.modelConfig.audioConfig;
      if (!this.audioConfig) {
        throw new Error(
          `Manifest declares audio_token_id=${audioTokenId} and quantizationInfo.audio ` +
          'but no audio_config was resolved. Check conversion config.'
        );
      }
      log.info('Pipeline', `Audio capable: audioTokenId=${audioTokenId}`);
    } else {
      this.audioCapable = false;
    }

    await runKernelWarmup({
      useGPU: this.useGPU,
      kernelWarmup: this.runtimeConfig.shared?.kernelWarmup,
      modelConfig: this.modelConfig,
    });

    // Phase 3: kernel path resolution + dtype contract
    configResolutionPhase = 3;
    log.debug('Pipeline', `Config resolution phase ${configResolutionPhase}: resolveKernelPathState`);
    const kernelPathState = resolveKernelPathState({
      manifest,
      runtimeConfig: this.runtimeConfig,
      runtimeOverrides: this.runtimeOverrides,
      modelConfig: this.modelConfig,
    });
    this.resolvedKernelPath = kernelPathState.resolvedKernelPath;
    this.kernelPathSource = kernelPathState.kernelPathSource;
    this.runtimeConfig = kernelPathState.runtimeConfig;

    // Phase 4: layer pipeline resolution
    configResolutionPhase = 4;
    log.debug('Pipeline', `Config resolution phase ${configResolutionPhase}: _resolveLayerPipeline`);
    this._resolveLayerPipeline();
    log.debug('Pipeline', `Config resolution complete (${configResolutionPhase} phases)`);

    const cfg = this.modelConfig;
    const moeStr = cfg.useMoE ? `, MoE(${cfg.numExperts}x${cfg.moeTopK})` : '';
    const kernelInfo = this.resolvedKernelPath ? `kernelPath=${this.resolvedKernelPath.id}` : 'kernelPath=none';
    log.info('Pipeline', `${cfg.numLayers}L/${cfg.hiddenSize}H/${cfg.numHeads}heads (${cfg.headDim}dim)${moeStr}, ${kernelInfo}`);

    this.tokenizer = await withPipelineLoadPhase(
      'tokenizer',
      { modelId: manifest.modelId ?? null },
      () => initTokenizerFromManifest(
        manifest,
        this.baseUrl,
        this.storageContext
      )
    );
    const tokenizerVocabSize = this.tokenizer.getVocabSize();
    if (Number.isFinite(tokenizerVocabSize) && tokenizerVocabSize > 0) {
      if (tokenizerVocabSize !== this.modelConfig.vocabSize) {
        log.info('Pipeline', `Tokenizer vocabSize=${tokenizerVocabSize} differs from model=${this.modelConfig.vocabSize}, using model size`);
      }
    }

    // Manifest quantizationInfo.compute is the binding lane identity.
    assertManifestComputeLaneBinding({ manifest, runtimeConfig: this.runtimeConfig });

    // Initialize KV cache
    if (this.modelConfig.decodeStrategy === 'replay_prefill') {
      this.kvCache = null;
      log.warn(
        'Pipeline',
        'Replay-prefill decode enabled for this model. Incremental KV-cache decode is disabled ' +
        'because the model config did not resolve explicit layerTypes for mixed-geometry/shared-KV decode.'
      );
    } else {
      this.kvCache = createKVCache(this.modelConfig, this.useGPU, this.debug, this.runtimeConfig.inference);
    }
    this.executionPlanState = compileExecutionPlanState({
      runtimeConfig: this.runtimeConfig,
      resolvedKernelPath: this.resolvedKernelPath,
      kernelPathSource: this.kernelPathSource,
      fallbackKernelPath: this.executionV1State?.fallbackKernelPath ?? null,
    });
    const activeExecutionPlan = resolveActiveExecutionPlan(this);
    log.info(
      'Pipeline',
      `Execution plan: active=${activeExecutionPlan.id}, dtype=${activeExecutionPlan.activationDtype}, ` +
      `kernelPath=${activeExecutionPlan.kernelPathId ?? 'none'}`
    );

    // Issue 1: Validate dtype consistency across all three resolution paths
    // (execution plan, runtimeConfig.inference.compute, and layer context).
    // The layer context is not yet built at this point, so pass null for it.
    // This logs a warning if the execution plan and runtimeConfig disagree.
    assertDtypeConsistency(this.executionPlanState, this.runtimeConfig, null);

    const kpActivation = getKernelPathActivationDtype(this.resolvedKernelPath);
    if (kpActivation && kpActivation !== activeExecutionPlan.activationDtype) {
      throw new Error(
        `Dtype contract violation: execution plan activationDtype="${activeExecutionPlan.activationDtype}" ` +
        `but kernel path "${this.resolvedKernelPath.id}" declares activationDtype="${kpActivation}".`
      );
    }

    // Initialize MoE router if needed
    if (this.modelConfig.useMoE) {
      this.moeRouter = new MoERouter({
        numExperts: this.modelConfig.numExperts,
        topK: this.modelConfig.moeTopK,
        hiddenSize: this.modelConfig.hiddenSize,
        normalizeWeights: this.runtimeConfig.inference.moe.routing.normalizeWeights,
      });
    }

    // Initialize speculative decoder
    if (manifest.draftModel) {
      this.speculativeDecoder = initSpeculativeDecoder(
        manifest,
        this.runtimeConfig.inference.speculative
      );
    }

    // Load weights
    await withPipelineLoadPhase(
      'loadWeights',
      { modelId: manifest.modelId ?? null },
      () => this._loadWeights()
    );

    // Initialize RoPE frequencies
    await this._initRoPE();

    // Initialize conv layer states for gated short conv layers (LFM2)
    await this._initConvLayerStates();

    this.isLoaded = true;
    const loadMs = performance.now() - loadStart;
    this.stats.modelLoadMs = loadMs;
    log.info('Pipeline', `Model loaded successfully (${loadMs.toFixed(0)}ms)`);
  }


  async _loadWeights() {
    const result = this._preloadedWeights || await loadWeights(
      (this.manifest),
      (this.modelConfig),
      {
        storageContext: this.storageContext ?? undefined,
        loadingConfig: this.runtimeConfig.loading,
        baseUrl: this.baseUrl ?? undefined,
        resolvedKernelPath: this.resolvedKernelPath,
        kernelPathSource: this.kernelPathSource,
        keepF32Weights: this.runtimeConfig.inference.compute.keepF32Weights === true,
        loaderDebug: this.runtimeConfig?.shared?.debug?.loader ?? null,
        perLayerInputSession: resolvePerLayerInputsSession(
          this.modelConfig.perLayerInputsSession ?? null,
          this.runtimeConfig?.inference?.session?.perLayerInputs ?? null
        ),
        onProgress: (info) => {
          if (info.stage !== 'layers' && info.stage !== 'shards') {
            log.verbose('Loader', `${info.stage}: ${Math.round(info.progress * 100)}%${info.message ? ` - ${info.message}` : ''}`);
          }
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
      }
    );

    result.layerWeights.forEach((w, k) => this.weights.set(k, w));
    this.weights.set('embed', result.embeddings);
    this.weights.set('lm_head', result.lmHead);
    this.weights.set('final_norm', result.finalNorm);
    this.weights.set('per_layer_inputs', result.perLayerInputWeights);
    this.embeddingPostprocessor = result.embeddingPostprocessor;

    this.layerRouterWeights = result.layerRouterWeights;

    this.dopplerLoader = getDopplerLoader(this.runtimeConfig.loading);

    if ((this.modelConfig).useMoE && this.moeRouter) {
      this.moeRouter = initMoERouter(
        (this.modelConfig),
        this.runtimeConfig.inference.moe.routing,
        result.layerWeights
      );
    }

    if (this.useGPU && this.modelConfig) {
      fuseQKVWeights(result.layerWeights, this.modelConfig, this.resolvedKernelPath);
    }

    if (this.useGPU && this.modelConfig) {
      const activeExecutionPlan = resolveActiveExecutionPlan(this);
      try {
        this.decodeBuffers?.ensureBuffers({
          hiddenSize: this.modelConfig.hiddenSize,
          intermediateSize: this.modelConfig.maxIntermediateSize,
          activationDtype: activeExecutionPlan.activationDtype,
          enablePingPong: true,
        });

        const device = getDevice();
        if (device) {
          this.finitenessBuffer = device.createBuffer({
            label: 'finiteness_status',
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
        }
      } catch (bufferError) {
        this.decodeBuffers?.release();
        if (this.finitenessBuffer) {
          this.finitenessBuffer.destroy();
          this.finitenessBuffer = null;
        }
        throw bufferError;
      }
    }
  }


  setPreloadedWeights(weights) {
    this._preloadedWeights = weights;
  }


  async _initRoPE() {
    const config = (this.modelConfig);
    const maxSeqLen = config.maxSeqLen;
    const ropeBuffers = await initRoPEFrequencies({
      headDim: config.globalHeadDim ?? config.headDim,
      localHeadDim: config.headDim,
      rotaryDim: config.ropeRotaryDim,
      ropeLocalRotaryDim: config.ropeLocalRotaryDim,
      ropeFrequencyBaseDim: config.ropeFrequencyBaseDim,
      ropeLocalFrequencyBaseDim: config.ropeLocalFrequencyBaseDim,
      maxSeqLen,
      ropeTheta: config.ropeTheta,
      ropeLocalTheta: config.ropeLocalTheta,
      mropeInterleaved: config.mropeInterleaved,
      mropeSection: config.mropeSection,
      partialRotaryFactor: config.partialRotaryFactor,
      ropeLocalPartialRotaryFactor: config.ropeLocalPartialRotaryFactor,
      ropeScale: config.ropeScale,
      ropeLocalScale: config.ropeLocalScale,
      ropeScalingType: config.ropeScalingType,
      ropeLocalScalingType: config.ropeLocalScalingType,
      ropeScaling: config.ropeScaling,
      ropeLocalScaling: config.ropeLocalScaling,
    }, this.useGPU);
    this.ropeFreqsCos = ropeBuffers.cos;
    this.ropeFreqsSin = ropeBuffers.sin;
    this.ropeLocalCos = ropeBuffers.localCos ?? null;
    this.ropeLocalSin = ropeBuffers.localSin ?? null;
  }


  async _initConvLayerStates() {
    const config = this.modelConfig;
    if (!config?.layerTypes) return;
    const { getDevice } = await import('../../gpu/device.js');
    const device = getDevice();
    if (!device) return;

    const hiddenSize = config.hiddenSize;
    const convStates = new Map();

    for (let i = 0; i < config.layerTypes.length; i++) {
      const lt = String(config.layerTypes[i] ?? '').toLowerCase();
      if (lt !== 'conv' && lt !== 'convolution') continue;

      const layerWeights = this.weights.get(`layer_${i}`);
      if (!layerWeights) continue;
      const convKernel = layerWeights?.convKernel;
      if (!convKernel) continue;

      const convState = {};
      try {
        await initConvLayerState(
          convState,
          convKernel,
          layerWeights.convInProj ?? null,
          hiddenSize,
          `L${i}.conv`,
          i
        );
        if (!convState.convWeightGPU || !convState.convStateGPU) {
          continue;
        }
        convStates.set(i, convState);
      } catch (e) {
        log.warn('Pipeline', `Conv layer ${i} state init failed: ${e.message}`);
      }
    }

    if (convStates.size > 0) {
      this.convLayerStates = convStates;
      log.info('Pipeline', `Initialized ${convStates.size} conv layer states (kernelSize=${convStates.values().next().value?.kernelSize})`);
    }
  }


  async _loadVisionWeights() {
    const loader = this.dopplerLoader ?? getDopplerLoader(this.runtimeConfig.loading);
    const vc = this.visionConfig;
    const depth = vc.depth;

    const loadRequiredTensor = async (name, toGPU = true) => {
      const tensor = await loader.loadTensor(name, toGPU, true);
      if (!tensor) {
        throw new Error(`Vision tensor "${name}" is missing from the converted artifact.`);
      }
      return tensor;
    };
    const loadScalar = async (name) => {
      const tensor = await loadRequiredTensor(name, false);
      if (tensor instanceof Float32Array) {
        if (tensor.length !== 1) {
          throw new Error(`Vision scalar "${name}" must be a single-element tensor, got length=${tensor.length}.`);
        }
        return tensor[0];
      }
      if (ArrayBuffer.isView(tensor) && tensor.length === 1) {
        return Number(tensor[0]);
      }
      if (typeof tensor === 'number') {
        return tensor;
      }
      throw new Error(
        `Vision scalar "${name}" must decode to a single numeric value, ` +
        `got ${tensor?.constructor?.name ?? typeof tensor} length=${tensor?.length ?? 'N/A'}.`
      );
    };
    const loadClipRange = async (prefix) => ({
      inputMin: await loadScalar(`${prefix}.input_min`),
      inputMax: await loadScalar(`${prefix}.input_max`),
      outputMin: await loadScalar(`${prefix}.output_min`),
      outputMax: await loadScalar(`${prefix}.output_max`),
    });

    if (vc.visionArchitecture === 'gemma4') {
      const visionWeights = {
        textHiddenSize: this.modelConfig.hiddenSize,
        patchInputProj: await loadRequiredTensor('model.vision_tower.patch_embedder.input_proj.weight'),
        patchPositionEmbeddingTable: await loadRequiredTensor('model.vision_tower.patch_embedder.position_embedding_table', false),
        projector: await loadRequiredTensor('model.embed_vision.embedding_projection.weight'),
        layers: [],
      };

      if (!(visionWeights.patchPositionEmbeddingTable instanceof Float32Array)) {
        throw new Error(
          'Gemma 4 vision position_embedding_table must decode to Float32Array on CPU. ' +
          'Re-convert the artifact if this tensor was quantized incorrectly.'
        );
      }

      for (let i = 0; i < depth; i++) {
        const prefix = `model.vision_tower.encoder.layers.${i}`;
        const attnPrefix = `${prefix}.self_attn`;
        const mlpPrefix = `${prefix}.mlp`;
        visionWeights.layers.push({
          inputLayerNorm: await loadRequiredTensor(`${prefix}.input_layernorm.weight`),
          postAttentionLayerNorm: await loadRequiredTensor(`${prefix}.post_attention_layernorm.weight`),
          preFeedforwardLayerNorm: await loadRequiredTensor(`${prefix}.pre_feedforward_layernorm.weight`),
          postFeedforwardLayerNorm: await loadRequiredTensor(`${prefix}.post_feedforward_layernorm.weight`),
          qNorm: await loadRequiredTensor(`${attnPrefix}.q_norm.weight`),
          kNorm: await loadRequiredTensor(`${attnPrefix}.k_norm.weight`),
          qProj: await loadRequiredTensor(`${attnPrefix}.q_proj.linear.weight`),
          kProj: await loadRequiredTensor(`${attnPrefix}.k_proj.linear.weight`),
          vProj: await loadRequiredTensor(`${attnPrefix}.v_proj.linear.weight`),
          oProj: await loadRequiredTensor(`${attnPrefix}.o_proj.linear.weight`),
          qProjClip: await loadClipRange(`${attnPrefix}.q_proj`),
          kProjClip: await loadClipRange(`${attnPrefix}.k_proj`),
          vProjClip: await loadClipRange(`${attnPrefix}.v_proj`),
          oProjClip: await loadClipRange(`${attnPrefix}.o_proj`),
          gateProj: await loadRequiredTensor(`${mlpPrefix}.gate_proj.linear.weight`),
          upProj: await loadRequiredTensor(`${mlpPrefix}.up_proj.linear.weight`),
          downProj: await loadRequiredTensor(`${mlpPrefix}.down_proj.linear.weight`),
          gateProjClip: await loadClipRange(`${mlpPrefix}.gate_proj`),
          upProjClip: await loadClipRange(`${mlpPrefix}.up_proj`),
          downProjClip: await loadClipRange(`${mlpPrefix}.down_proj`),
        });
      }

      this.visionWeights = visionWeights;
      log.info('Pipeline', `Vision weights loaded (${depth} Gemma 4 encoder layers)`);
      return;
    }

    const visionWeights = {};

    // Patch embedding weights
    const patchProjName = 'visual.patch_embed.proj.weight';
    const patchProjBiasName = 'visual.patch_embed.proj.bias';
    visionWeights.patchProjWeight = await loader.loadTensor(patchProjName, true, true);
    visionWeights.patchProjBias = await loader.loadTensor(patchProjBiasName, true, true);

    // Vision encoder layer weights
    visionWeights.layers = [];
    for (let i = 0; i < depth; i++) {
      const prefix = `visual.blocks.${i}`;
      const layerW = {
        norm1Weight: await loader.loadTensor(`${prefix}.norm1.weight`, true, true),
        norm2Weight: await loader.loadTensor(`${prefix}.norm2.weight`, true, true),
        qkvWeight: await loader.loadTensor(`${prefix}.attn.qkv.weight`, true, true),
        qkvBias: await loader.loadTensor(`${prefix}.attn.qkv.bias`, true, true),
        projWeight: await loader.loadTensor(`${prefix}.attn.proj.weight`, true, true),
        projBias: await loader.loadTensor(`${prefix}.attn.proj.bias`, true, true),
        fc1Weight: await loader.loadTensor(`${prefix}.mlp.fc1.weight`, true, true),
        fc1Bias: await loader.loadTensor(`${prefix}.mlp.fc1.bias`, true, true),
        fc2Weight: await loader.loadTensor(`${prefix}.mlp.fc2.weight`, true, true),
        fc2Bias: await loader.loadTensor(`${prefix}.mlp.fc2.bias`, true, true),
      };
      visionWeights.layers.push(layerW);
    }

    // Spatial merge projection
    visionWeights.mergerLnWeight = await loader.loadTensor('visual.merger.ln_q.weight', true, true);
    visionWeights.mergerMlp0Weight = await loader.loadTensor('visual.merger.mlp.0.weight', true, true);
    visionWeights.mergerMlp0Bias = await loader.loadTensor('visual.merger.mlp.0.bias', true, true);
    visionWeights.mergerMlp2Weight = await loader.loadTensor('visual.merger.mlp.2.weight', true, true);
    visionWeights.mergerMlp2Bias = await loader.loadTensor('visual.merger.mlp.2.bias', true, true);

    this.visionWeights = visionWeights;
    log.info('Pipeline', `Vision weights loaded (${depth} encoder layers)`);
  }

  async _ensureVisionWeightsLoaded() {
    if (!this.visionCapable) {
      throw new Error(
        'Pipeline does not support vision weights (no image_token_id in manifest).'
      );
    }
    if (this.visionWeights) {
      return;
    }
    log.info('Pipeline', 'Loading vision weights on demand');
    await this._loadVisionWeights();
  }


  async _loadAudioWeights() {
    const loader = this.dopplerLoader ?? getDopplerLoader(this.runtimeConfig.loading);
    const ac = this.audioConfig;
    const depth = ac.depth;

    const loadRequiredTensor = async (name, toGPU = true) => {
      const tensor = await loader.loadTensor(name, toGPU, true);
      if (!tensor) {
        throw new Error(`Audio tensor "${name}" is missing from the converted artifact.`);
      }
      return tensor;
    };
    const loadScalar = async (name) => {
      const tensor = await loadRequiredTensor(name, false);
      if (tensor instanceof Float32Array) {
        if (tensor.length !== 1) {
          throw new Error(`Audio scalar "${name}" must be a single-element tensor, got length=${tensor.length}.`);
        }
        return tensor[0];
      }
      if (ArrayBuffer.isView(tensor) && tensor.length === 1) {
        return Number(tensor[0]);
      }
      if (typeof tensor === 'number') {
        return tensor;
      }
      throw new Error(
        `Audio scalar "${name}" must decode to a single numeric value, ` +
        `got ${tensor?.constructor?.name ?? typeof tensor} length=${tensor?.length ?? 'N/A'}.`
      );
    };
    const loadClipRange = async (prefix) => ({
      inputMin: await loadScalar(`${prefix}.input_min`),
      inputMax: await loadScalar(`${prefix}.input_max`),
      outputMin: await loadScalar(`${prefix}.output_min`),
      outputMax: await loadScalar(`${prefix}.output_max`),
    });

    const audioWeights = {
      // Subsampling
      subsampleConv0Weight: await loadRequiredTensor('model.audio_tower.subsample_conv_projection.layer0.conv.weight'),
      subsampleNorm0Weight: await loadRequiredTensor('model.audio_tower.subsample_conv_projection.layer0.norm.weight'),
      subsampleConv1Weight: await loadRequiredTensor('model.audio_tower.subsample_conv_projection.layer1.conv.weight'),
      subsampleNorm1Weight: await loadRequiredTensor('model.audio_tower.subsample_conv_projection.layer1.norm.weight'),
      subsampleInputProjWeight: await loadRequiredTensor('model.audio_tower.subsample_conv_projection.input_proj_linear.weight'),
      // Output
      outputProjWeight: await loadRequiredTensor('model.audio_tower.output_proj.weight'),
      outputProjBias: await loadRequiredTensor('model.audio_tower.output_proj.bias'),
      audioEmbeddingProjWeight: await loadRequiredTensor('model.embed_audio.embedding_projection.weight'),
      layers: [],
    };

    for (let i = 0; i < depth; i++) {
      const prefix = `model.audio_tower.layers.${i}`;
      const layer = {
        // Feed-forward 1 (Macaron half-step)
        feedForward1: {
          preLayerNorm: await loadRequiredTensor(`${prefix}.feed_forward1.pre_layer_norm.weight`),
          ffwLayer1Weight: await loadRequiredTensor(`${prefix}.feed_forward1.ffw_layer_1.linear.weight`),
          ffwLayer1Clip: await loadClipRange(`${prefix}.feed_forward1.ffw_layer_1`),
          ffwLayer2Weight: await loadRequiredTensor(`${prefix}.feed_forward1.ffw_layer_2.linear.weight`),
          ffwLayer2Clip: await loadClipRange(`${prefix}.feed_forward1.ffw_layer_2`),
          postLayerNorm: await loadRequiredTensor(`${prefix}.feed_forward1.post_layer_norm.weight`),
        },
        // Self-attention
        normPreAttn: await loadRequiredTensor(`${prefix}.norm_pre_attn.weight`),
        qProj: await loadRequiredTensor(`${prefix}.self_attn.q_proj.linear.weight`),
        qProjClip: await loadClipRange(`${prefix}.self_attn.q_proj`),
        kProj: await loadRequiredTensor(`${prefix}.self_attn.k_proj.linear.weight`),
        kProjClip: await loadClipRange(`${prefix}.self_attn.k_proj`),
        vProj: await loadRequiredTensor(`${prefix}.self_attn.v_proj.linear.weight`),
        vProjClip: await loadClipRange(`${prefix}.self_attn.v_proj`),
        perDimScale: await loadRequiredTensor(`${prefix}.self_attn.per_dim_scale`),
        relativeKProj: await loadRequiredTensor(`${prefix}.self_attn.relative_k_proj.weight`),
        postProj: await loadRequiredTensor(`${prefix}.self_attn.post.linear.weight`),
        postProjClip: await loadClipRange(`${prefix}.self_attn.post`),
        normPostAttn: await loadRequiredTensor(`${prefix}.norm_post_attn.weight`),
        // Convolution module (LConv1D)
        lconvPreLayerNorm: await loadRequiredTensor(`${prefix}.lconv1d.pre_layer_norm.weight`),
        lconvLinearStartWeight: await loadRequiredTensor(`${prefix}.lconv1d.linear_start.linear.weight`),
        lconvLinearStartClip: await loadClipRange(`${prefix}.lconv1d.linear_start`),
        lconvDepthwiseWeight: await loadRequiredTensor(`${prefix}.lconv1d.depthwise_conv1d.weight`),
        lconvConvNorm: await loadRequiredTensor(`${prefix}.lconv1d.conv_norm.weight`),
        lconvLinearEndWeight: await loadRequiredTensor(`${prefix}.lconv1d.linear_end.linear.weight`),
        lconvLinearEndClip: await loadClipRange(`${prefix}.lconv1d.linear_end`),
        // Feed-forward 2 (Macaron half-step)
        feedForward2: {
          preLayerNorm: await loadRequiredTensor(`${prefix}.feed_forward2.pre_layer_norm.weight`),
          ffwLayer1Weight: await loadRequiredTensor(`${prefix}.feed_forward2.ffw_layer_1.linear.weight`),
          ffwLayer1Clip: await loadClipRange(`${prefix}.feed_forward2.ffw_layer_1`),
          ffwLayer2Weight: await loadRequiredTensor(`${prefix}.feed_forward2.ffw_layer_2.linear.weight`),
          ffwLayer2Clip: await loadClipRange(`${prefix}.feed_forward2.ffw_layer_2`),
          postLayerNorm: await loadRequiredTensor(`${prefix}.feed_forward2.post_layer_norm.weight`),
        },
        // Final layer norm
        normOut: await loadRequiredTensor(`${prefix}.norm_out.weight`),
      };
      audioWeights.layers.push(layer);
    }

    this.audioWeights = audioWeights;
    log.info('Pipeline', `Audio weights loaded (${depth} conformer layers)`);
  }

  async _ensureAudioWeightsLoaded() {
    if (!this.audioCapable) {
      throw new Error(
        'Pipeline does not support audio weights (no audio_token_id in manifest).'
      );
    }
    if (this.audioWeights) {
      return;
    }
    log.info('Pipeline', 'Loading audio weights on demand');
    await this._loadAudioWeights();
  }


  // ==========================================================================
  // Vision: transcribeImage
  // ==========================================================================

  /**
   * Transcribe text from an image using the vision encoder and text decoder.
   *
   * @param {object} params
   * @param {Uint8Array|Float32Array} params.imageBytes  Raw image pixel data
   * @param {number}                  params.width       Image width
   * @param {number}                  params.height      Image height
   * @param {string}                  [params.prompt]    Custom transcription prompt
   * @param {number}                  [params.maxTokens] Max tokens to generate
   * @param {number}                  [params.softTokenBudget] Per-request soft token budget (Gemma 4 tiers: 70/140/280/560/1120)
   * @returns {Promise<{ text: string, tokens: number[] }>}
   */
  async transcribeImage({ imageBytes, width, height, prompt, maxTokens, softTokenBudget, signal }) {
    assertNotAborted(signal);
    if (!this.visionCapable) {
      throw new Error(
        'Pipeline does not support image transcription (no image_token_id in manifest).'
      );
    }
    await this._ensureVisionWeightsLoaded();
    assertNotAborted(signal);

    this.reset();

    // Lazy-load vision module (avoids GPU kernel dependency for text-only pipelines)
    const { encodeImage } = await import('./vision/index.js');

    // Step 1: Encode image through vision pipeline
    const encodeResult = await encodeImage({
      pixels: imageBytes,
      width,
      height,
      visionConfig: this.visionConfig,
      weights: this.visionWeights,
      softTokenBudget,
    });

    // Step 2: Build the multimodal prompt from the model's chat template and
    // expand the single <|image|> placeholder into the exact visual-token span.
    const requestedPrompt = prompt ?? 'Describe the image in one short sentence.';
    const imageTokenId = this.visionConfig?.imageTokenId ?? this.modelConfig?.imageTokenId;
    if (imageTokenId == null) {
      throw new Error(
        'Pipeline missing image_token_id. Re-convert the model with image token metadata.'
      );
    }
    if (this.visionConfig?.visionArchitecture !== 'gemma4') {
      throw new Error(
        `[Pipeline] transcribeImage: unsupported vision architecture "${this.visionConfig?.visionArchitecture ?? 'unknown'}". ` +
        'This runtime path currently requires Gemma 4 multimodal prompt expansion.'
      );
    }
    const templateType = this.modelConfig?.chatTemplateType ?? 'gemma4';
    const chatOptions = this.modelConfig?.chatTemplateThinking === true ? { thinking: true } : undefined;
    const multimodalPrompt = formatChatMessages([
      {
        role: 'user',
        content: [
          { type: 'image' },
          { type: 'text', text: requestedPrompt },
        ],
      },
    ], templateType, chatOptions);
    const promptTokenIds = this.tokenizer.encode(multimodalPrompt);
    const imageTokenSpanLength = encodeResult.numTokens;
    const effectiveBudget = softTokenBudget ?? this.visionConfig?.defaultOutputLength;
    const maxImageTokenSpanLength = Number(effectiveBudget);
    if (!Number.isFinite(maxImageTokenSpanLength) || maxImageTokenSpanLength < 1 || Math.floor(maxImageTokenSpanLength) !== maxImageTokenSpanLength) {
      throw new Error(
        `[Pipeline] transcribeImage: invalid soft token budget ${effectiveBudget}. ` +
        'Expected a positive integer from the resolved vision config or softTokenBudget parameter.'
      );
    }
    if (imageTokenSpanLength > maxImageTokenSpanLength) {
      throw new Error(
        `[Pipeline] transcribeImage: encoded Gemma 4 image produced ${imageTokenSpanLength} soft tokens, ` +
        `which exceeds the effective soft token budget=${maxImageTokenSpanLength}.`
      );
    }
    const boiTokenId = resolveSingleSpecialTokenId(this.tokenizer, '<|image>', 'Gemma 4 BOI token');
    const eoiTokenId = resolveSingleSpecialTokenId(this.tokenizer, '<image|>', 'Gemma 4 EOI token');
    const { inputIds: fullTokenIds, imageStartOffset } = expandImagePlaceholderTokenIds(
      promptTokenIds,
      imageTokenId,
      imageTokenSpanLength,
      { boiTokenId, eoiTokenId }
    );
    const padTokenId = this.tokenizer?.getSpecialTokens?.()?.pad;
    if (!Number.isFinite(padTokenId) || Math.floor(padTokenId) !== padTokenId || padTokenId < 0) {
      throw new Error(
        `[Pipeline] transcribeImage: Gemma 4 multimodal prefill requires a tokenizer pad token ID, got ${padTokenId}.`
      );
    }

    // Step 3: Generate with embedding override at the image token offset.
    const tokens = [];
    const maxGen = maxTokens ?? 512;
    const stopTokenIds = this.modelConfig.stopTokenIds;

    try {
      const generation = await this.generator.generateTokenIds('', buildConservativeMultimodalGenerationOptions({
        inputIds: fullTokenIds,
        embeddingOverrides: {
          prefixLength: encodeResult.numTokens,
          offset: imageStartOffset,
          embeddings: encodeResult.features,
        },
        __internalEmbeddingInputSpan: {
          offset: imageStartOffset,
          length: encodeResult.numTokens,
          tokenId: padTokenId,
        },
        __internalMultimodalBidirectionalSpan: {
          offset: imageStartOffset,
          length: encodeResult.numTokens,
        },
        maxTokens: maxGen,
        temperature: 0,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1,
      }));
      for (const token of generation.tokenIds ?? []) {
        if (Array.isArray(stopTokenIds) && stopTokenIds.includes(token)) break;
        tokens.push(token);
      }
    } finally {
      if (encodeResult.features) {
        releaseBuffer(encodeResult.features);
      }
    }

    const text = this.tokenizer.decode(tokens);
    return { text, tokens };
  }


  // ==========================================================================
  // Video: transcribeVideo
  // ==========================================================================

  /**
   * Transcribe text from video using the vision encoder (per-frame) and text decoder.
   *
   * Video is processed as sampled frames through the existing vision encoder.
   * Each frame is encoded independently and the visual tokens are concatenated.
   *
   * @param {object} params
   * @param {Array<{ pixels: Uint8Array|Float32Array, width: number, height: number }>} params.frames  Decoded video frames
   * @param {string}  [params.prompt]    Custom transcription prompt
   * @param {number}  [params.maxTokens] Max tokens to generate
   * @param {number}  [params.maxFrames=8] Maximum frames to sample
   * @param {number}  [params.perFrameSoftTokenBudget] Soft token budget per frame
   * @returns {Promise<{ text: string, tokens: number[] }>}
   */
  async transcribeVideo({ frames, prompt, maxTokens, maxFrames, perFrameSoftTokenBudget, signal }) {
    assertNotAborted(signal);
    if (!this.visionCapable) {
      throw new Error(
        'Pipeline does not support video transcription (no image_token_id in manifest for vision encoder).'
      );
    }
    await this._ensureVisionWeightsLoaded();

    this.reset();

    // Lazy-load video module
    const { encodeVideo } = await import('./video/index.js');

    // Step 1: Encode video frames through vision pipeline
    const encodeResult = await encodeVideo({
      frames,
      visionConfig: this.visionConfig,
      weights: this.visionWeights,
      maxFrames: maxFrames ?? 8,
      perFrameSoftTokenBudget,
    });

    // Step 2: Build the multimodal prompt with <|video|> placeholder
    const requestedPrompt = prompt ?? 'Describe the video in one short sentence.';
    const videoTokenId = this.tokenizer?.model?.tokenToId?.('<|video_token|>')
      ?? this.tokenizer?.encode?.('<|video|>')?.find?.((id) => id !== undefined)
      ?? null;
    // Fall back to image token ID for video placeholder expansion
    const placeholderTokenId = videoTokenId ?? this.visionConfig?.imageTokenId ?? this.imageTokenId;
    if (placeholderTokenId == null) {
      throw new Error(
        'Pipeline missing video/image token ID for video placeholder expansion.'
      );
    }

    const templateType = this.modelConfig?.chatTemplateType ?? 'gemma4';
    const chatOptions = this.modelConfig?.chatTemplateThinking === true ? { thinking: true } : undefined;
    const multimodalPrompt = formatChatMessages([
      {
        role: 'user',
        content: [
          { type: 'video' },
          { type: 'text', text: requestedPrompt },
        ],
      },
    ], templateType, chatOptions);
    const promptTokenIds = this.tokenizer.encode(multimodalPrompt);
    const videoTokenSpanLength = encodeResult.numTokens;

    // Resolve BOV/EOV tokens (reuse image BOI/EOI if video-specific ones don't exist)
    const bovTokenId = resolveSingleSpecialTokenId(this.tokenizer, '<|video|>', 'Gemma 4 BOV token');
    const eovTokenId = resolveSingleSpecialTokenId(this.tokenizer, '<video|>', 'Gemma 4 EOV token');

    const { inputIds: fullTokenIds, imageStartOffset: videoStartOffset } = expandImagePlaceholderTokenIds(
      promptTokenIds,
      placeholderTokenId,
      videoTokenSpanLength,
      { boiTokenId: bovTokenId, eoiTokenId: eovTokenId }
    );

    const padTokenId = this.tokenizer?.getSpecialTokens?.()?.pad;
    if (!Number.isFinite(padTokenId) || Math.floor(padTokenId) !== padTokenId || padTokenId < 0) {
      throw new Error(
        `[Pipeline] transcribeVideo: Gemma 4 multimodal prefill requires a tokenizer pad token ID, got ${padTokenId}.`
      );
    }

    // Step 3: Generate with embedding override at the video token offset
    const tokens = [];
    const maxGen = maxTokens ?? 512;
    const stopTokenIds = this.modelConfig.stopTokenIds;

    try {
      const generation = await this.generator.generateTokenIds('', buildConservativeMultimodalGenerationOptions({
        inputIds: fullTokenIds,
        embeddingOverrides: {
          prefixLength: encodeResult.numTokens,
          offset: videoStartOffset,
          embeddings: encodeResult.features,
        },
        __internalEmbeddingInputSpan: {
          offset: videoStartOffset,
          length: encodeResult.numTokens,
          tokenId: padTokenId,
        },
        __internalMultimodalBidirectionalSpan: {
          offset: videoStartOffset,
          length: encodeResult.numTokens,
        },
        maxTokens: maxGen,
        temperature: 0,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1,
      }));
      for (const token of generation.tokenIds ?? []) {
        if (Array.isArray(stopTokenIds) && stopTokenIds.includes(token)) break;
        tokens.push(token);
      }
    } finally {
      if (encodeResult.features) {
        releaseBuffer(encodeResult.features);
      }
    }

    const text = this.tokenizer.decode(tokens);
    return { text, tokens };
  }


  // ==========================================================================
  // Audio: transcribeAudio
  // ==========================================================================

  /**
   * Transcribe text from audio using the audio encoder and text decoder.
   *
   * @param {object} params
   * @param {Float32Array}             params.audio     Raw audio PCM (mono, 16kHz)
   * @param {string}                   [params.prompt]  Custom transcription prompt
   * @param {number}                   [params.maxTokens] Max tokens to generate
   * @returns {Promise<{ text: string, tokens: number[] }>}
   */
  async transcribeAudio({ audio, prompt, maxTokens, signal }) {
    assertNotAborted(signal);
    if (!this.audioCapable) {
      throw new Error(
        'Pipeline does not support audio transcription (no audio_token_id in manifest).'
      );
    }
    await this._ensureAudioWeightsLoaded();

    this.reset();

    // Lazy-load audio modules
    const { extractLogMelSpectrogram } = await import('./audio/mel.js');
    const { encodeAudio } = await import('./audio/index.js');

    // Step 1: Extract mel spectrogram
    const { features: melFeatures, numFrames, nMels } = extractLogMelSpectrogram(audio);

    // Step 2: Encode audio through conformer pipeline
    const encodeResult = await encodeAudio({
      melFeatures,
      numFrames,
      nMels,
      audioConfig: this.audioConfig,
      weights: this.audioWeights,
    });

    // Step 3: Build the multimodal prompt with <|audio|> placeholder
    const requestedPrompt = prompt ?? 'Transcribe the audio.';
    const audioTokenId = this.audioConfig?.audioTokenId ?? this.audioTokenId;
    if (audioTokenId == null) {
      throw new Error(
        'Pipeline missing audio_token_id. Re-convert the model with audio token metadata.'
      );
    }
    const templateType = this.modelConfig?.chatTemplateType ?? 'gemma4';
    const chatOptions = this.modelConfig?.chatTemplateThinking === true ? { thinking: true } : undefined;
    const multimodalPrompt = formatChatMessages([
      {
        role: 'user',
        content: [
          { type: 'audio' },
          { type: 'text', text: requestedPrompt },
        ],
      },
    ], templateType, chatOptions);
    const promptTokenIds = this.tokenizer.encode(multimodalPrompt);
    const audioTokenSpanLength = encodeResult.numTokens;

    // Resolve BOA/EOA tokens
    const boaTokenId = resolveSingleSpecialTokenId(this.tokenizer, '<|audio|>', 'Gemma 4 BOA token');
    const eoaTokenId = resolveSingleSpecialTokenId(this.tokenizer, '<audio|>', 'Gemma 4 EOA token');

    // Expand single audio placeholder token into the full audio token span
    const { inputIds: fullTokenIds, imageStartOffset: audioStartOffset } = expandImagePlaceholderTokenIds(
      promptTokenIds,
      audioTokenId,
      audioTokenSpanLength,
      { boiTokenId: boaTokenId, eoiTokenId: eoaTokenId }
    );

    const padTokenId = this.tokenizer?.getSpecialTokens?.()?.pad;
    if (!Number.isFinite(padTokenId) || Math.floor(padTokenId) !== padTokenId || padTokenId < 0) {
      throw new Error(
        `[Pipeline] transcribeAudio: Gemma 4 multimodal prefill requires a tokenizer pad token ID, got ${padTokenId}.`
      );
    }

    // Step 4: Generate with embedding override at the audio token offset
    const tokens = [];
    const maxGen = maxTokens ?? 512;
    const stopTokenIds = this.modelConfig.stopTokenIds;

    try {
      const generation = await this.generator.generateTokenIds('', buildConservativeMultimodalGenerationOptions({
        inputIds: fullTokenIds,
        embeddingOverrides: {
          prefixLength: encodeResult.numTokens,
          offset: audioStartOffset,
          embeddings: encodeResult.features,
        },
        __internalEmbeddingInputSpan: {
          offset: audioStartOffset,
          length: encodeResult.numTokens,
          tokenId: padTokenId,
        },
        __internalMultimodalBidirectionalSpan: {
          offset: audioStartOffset,
          length: encodeResult.numTokens,
        },
        maxTokens: maxGen,
        temperature: 0,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1,
      }));
      for (const token of generation.tokenIds ?? []) {
        if (Array.isArray(stopTokenIds) && stopTokenIds.includes(token)) break;
        tokens.push(token);
      }
    } finally {
      if (encodeResult.features) {
        releaseBuffer(encodeResult.features);
      }
    }

    const text = this.tokenizer.decode(tokens);
    return { text, tokens };
  }


  // ==========================================================================
  // Capability Detection
  // ==========================================================================

  get capabilities() {
    const caps = ['generation'];
    if (typeof this.prefillWithEmbedding === 'function') caps.push('embedding');
    if (this.visionCapable) caps.push('multimodal');
    if (this.audioCapable) caps.push('audio');
    if (this.visionCapable) caps.push('video');
    return Object.freeze(caps);
  }


  // Layer pipeline precedence (lowest to highest):
  //   1. execution-v1-produced pipeline (via runtimeInferencePatch.pipeline)
  //   2. model config pipeline (manifest inference.pipeline)
  //   3. runtime config pipeline (runtime.inference.pipeline)
  // If runtime overrides an execution-v1-produced pipeline, a warning is logged
  // because the execution graph's pipeline was designed for the resolved kernel
  // path and capability set.
  _resolveLayerPipeline() {
    if (!this.modelConfig) return;
    const runtimePlan = this.runtimeConfig.inference.pipeline ?? null;
    const modelPlan = this.modelConfig.layerPipeline ?? null;

    // Detect when runtime config would override an execution-v1-produced pipeline
    const runtimeHasSteps = runtimePlan?.steps && runtimePlan.steps.length > 0;
    const executionV1ProducedPipeline = this.executionV1State?.runtimeInferencePatch?.pipeline != null;
    if (runtimeHasSteps && executionV1ProducedPipeline) {
      log.warn(
        'Pipeline',
        'Runtime config pipeline overrides execution-v1-produced pipeline. ' +
        'The execution graph designed this pipeline for the resolved kernel path and capability set. ' +
        'Verify that the runtime override is intentional.'
      );
    }
    if (runtimeHasSteps && !executionV1ProducedPipeline && modelPlan?.steps?.length > 0) {
      log.debug(
        'Pipeline',
        'Runtime config pipeline overrides model config pipeline.'
      );
    }

    this.layerPipelinePlan = resolveLayerPipeline(modelPlan, runtimePlan, this.modelConfig.numLayers);
    if (this.layerPipelinePlan) {
      log.info(
        'Pipeline',
        `Layer pipeline plan enabled (source=${this.layerPipelinePlan.source}, steps=${this.layerPipelinePlan.steps.length}, overrides=${this.layerPipelinePlan.overrides.length})`
      );
    }
  }

  // ==========================================================================
  // Generation Delegates
  // ==========================================================================


  generate(prompt, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.generate(prompt, options);
  }

  generateTokens(prompt, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.generateTokens(prompt, options);
  }

  generateTokenIds(prompt, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.generateTokenIds(prompt, options);
  }

  decodeStepLogits(currentIds, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.decodeStepLogits(currentIds, options);
  }

  advanceWithToken(tokenId, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.advanceWithToken(tokenId, options);
  }

  advanceWithTokenAndEmbedding(tokenId, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.advanceWithTokenAndEmbedding(tokenId, options);
  }


  prefillKVOnly(prompt, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.prefillKVOnly(prompt, options);
  }

  prefillWithEmbedding(prompt, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.prefillWithEmbedding(prompt, options);
  }

  async embed(prompt, options = {}) {
    assertNotAborted(options?.signal);
    const result = await this.prefillWithEmbedding(prompt, options);
    assertNotAborted(options?.signal);
    return {
      embedding: result.embedding,
      tokens: result.tokens,
      seqLen: result.seqLen,
      embeddingMode: result.embeddingMode,
    };
  }

  async embedBatch(prompts, options = {}) {
    if (!Array.isArray(prompts)) {
      throw new Error('embedBatch expects an array of prompts');
    }
    assertNotAborted(options?.signal);
    const batchOptions = { ...options, __skipStateSnapshot: true };
    const outputs = [];
    for (const prompt of prompts) {
      // Check between every prompt so a superseded revision drops the rest.
      assertNotAborted(options?.signal);
      outputs.push(await this.embed(prompt, batchOptions));
      this.resetForBatch();
    }
    return outputs;
  }

  // Run the vision encoder over a single image and return a mean-pooled
  // embedding in the model's text-hidden-size space. Decoder responsibility
  // (jpeg/png -> RGBA pixels) belongs to the caller; this method takes
  // already-decoded pixel data.
  async embedImage({ pixels, width, height, softTokenBudget, signal } = {}) {
    assertNotAborted(signal);
    if (!this.visionCapable) {
      throw new Error(
        'Pipeline does not support image embedding (no image_token_id in manifest).'
      );
    }
    if (pixels == null) {
      throw new Error('[Pipeline] embedImage: pixels are required.');
    }
    if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
      throw new Error('[Pipeline] embedImage: width and height must be positive integers.');
    }
    await this._ensureVisionWeightsLoaded();
    this.reset();

    const { encodeImage } = await import('./vision/index.js');
    const encodeResult = await encodeImage({
      pixels,
      width,
      height,
      visionConfig: this.visionConfig,
      weights: this.visionWeights,
      softTokenBudget,
    });

    const hiddenSize = this.modelConfig.hiddenSize;
    const numTokens = encodeResult.numTokens;
    if (!Number.isFinite(numTokens) || numTokens < 1) {
      releaseBuffer(encodeResult.features);
      throw new Error(`[Pipeline] embedImage: encoder produced ${numTokens} soft tokens; expected >= 1.`);
    }
    try {
      const bytes = await readBuffer(
        encodeResult.features,
        numTokens * hiddenSize * Float32Array.BYTES_PER_ELEMENT
      );
      const features = new Float32Array(bytes);
      const pooled = new Float32Array(hiddenSize);
      for (let t = 0; t < numTokens; t++) {
        const base = t * hiddenSize;
        for (let d = 0; d < hiddenSize; d++) {
          pooled[d] += features[base + d];
        }
      }
      const inv = 1 / numTokens;
      for (let d = 0; d < hiddenSize; d++) pooled[d] *= inv;
      return {
        embedding: pooled,
        embeddingDim: hiddenSize,
        numTokens,
        embeddingMode: 'mean',
      };
    } finally {
      releaseBuffer(encodeResult.features);
    }
  }

  // Run the audio encoder over a single PCM segment and return a mean-pooled
  // embedding in the model's audio-projection-output space (which equals the
  // text hidden size in Gemma 4). Decoder responsibility (webm/opus/wav ->
  // Float32 PCM at the model's expected sample rate) belongs to the caller.
  async embedAudio({ audio, signal } = {}) {
    assertNotAborted(signal);
    if (!this.audioCapable) {
      throw new Error(
        'Pipeline does not support audio embedding (no audio_token_id in manifest).'
      );
    }
    if (audio == null) {
      throw new Error('[Pipeline] embedAudio: audio is required.');
    }
    await this._ensureAudioWeightsLoaded();
    this.reset();

    const { extractLogMelSpectrogram } = await import('./audio/mel.js');
    const { encodeAudio } = await import('./audio/index.js');
    const { features: melFeatures, numFrames, nMels } = extractLogMelSpectrogram(audio);
    const encodeResult = await encodeAudio({
      melFeatures,
      numFrames,
      nMels,
      audioConfig: this.audioConfig,
      weights: this.audioWeights,
    });

    const hiddenSize = Number(this.audioConfig?.outputProjDims ?? this.modelConfig?.hiddenSize);
    const numTokens = encodeResult.numTokens;
    if (!Number.isFinite(hiddenSize) || hiddenSize < 1) {
      releaseBuffer(encodeResult.features);
      throw new Error('[Pipeline] embedAudio: audioConfig.outputProjDims is missing or invalid.');
    }
    if (!Number.isFinite(numTokens) || numTokens < 1) {
      releaseBuffer(encodeResult.features);
      throw new Error(`[Pipeline] embedAudio: encoder produced ${numTokens} tokens; expected >= 1.`);
    }
    try {
      const bytes = await readBuffer(
        encodeResult.features,
        numTokens * hiddenSize * Float32Array.BYTES_PER_ELEMENT
      );
      const features = new Float32Array(bytes);
      const pooled = new Float32Array(hiddenSize);
      for (let t = 0; t < numTokens; t++) {
        const base = t * hiddenSize;
        for (let d = 0; d < hiddenSize; d++) {
          pooled[d] += features[base + d];
        }
      }
      const inv = 1 / numTokens;
      for (let d = 0; d < hiddenSize; d++) pooled[d] *= inv;
      return {
        embedding: pooled,
        embeddingDim: hiddenSize,
        numTokens,
        embeddingMode: 'mean',
      };
    } finally {
      releaseBuffer(encodeResult.features);
    }
  }

  prefillWithLogits(prompt, options = {}) {
    return this.generator.prefillWithLogits(prompt, options);
  }


  applyKVCacheSnapshot(snapshot) {
    this.kvCache = snapshot.cache.clone();
    if (this.useGPU && this.kvCache) {
      const device = getDevice();
      if (device) {
        this.kvCache.setGPUContext({ device });
      }
    }
    if (
      hasLinearAttentionLayers(this.modelConfig?.layerTypes)
      && snapshot.linearAttention == null
    ) {
      throw new Error(
        'Snapshot is missing linear_attention recurrent state. ' +
        'Regenerate the snapshot with the current runtime.'
      );
    }
    this.linearAttentionRuntime = restoreLinearAttentionRuntime(
      this.linearAttentionRuntime,
      snapshot.linearAttention ?? null
    );
    this.currentSeqLen = snapshot.seqLen;
  }


  generateWithPrefixKV(prefix, prompt, options = {}) {
    return this.generator.generateWithPrefixKV(prefix, prompt, options);
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================


  getStats() {
    const stats = { ...this.stats };
    stats.batching ??= { ...this.batchingStats };
    if (this.executionPlanState) {
      const activeExecutionPlan = resolveActiveExecutionPlan(this);
      stats.executionPlan ??= {
        primary: this.executionPlanState?.primaryPlan
          ? {
            id: this.executionPlanState.primaryPlan.id,
            kernelPathId: this.executionPlanState.primaryPlan.kernelPathId ?? null,
            kernelPathSource: this.executionPlanState.primaryPlan.kernelPathSource ?? 'none',
            activationDtype: this.executionPlanState.primaryPlan.activationDtype,
            readbackInterval: this.executionPlanState.primaryPlan.readbackInterval ?? null,
            batchSize: this.executionPlanState.primaryPlan.defaultBatchSize,
            stopCheckMode: this.executionPlanState.primaryPlan.defaultStopCheckMode,
            disableCommandBatching: this.executionPlanState.primaryPlan.defaultDisableCommandBatching === true,
            ringTokens: this.executionPlanState.primaryPlan.ringTokens ?? null,
            ringStop: this.executionPlanState.primaryPlan.ringStop ?? null,
            ringStaging: this.executionPlanState.primaryPlan.ringStaging ?? null,
          }
          : null,
        fallback: this.executionPlanState?.fallbackPlan
          ? {
            id: this.executionPlanState.fallbackPlan.id,
            kernelPathId: this.executionPlanState.fallbackPlan.kernelPathId ?? null,
            kernelPathSource: this.executionPlanState.fallbackPlan.kernelPathSource ?? 'none',
            activationDtype: this.executionPlanState.fallbackPlan.activationDtype,
            readbackInterval: this.executionPlanState.fallbackPlan.readbackInterval ?? null,
            batchSize: this.executionPlanState.fallbackPlan.defaultBatchSize,
            stopCheckMode: this.executionPlanState.fallbackPlan.defaultStopCheckMode,
            disableCommandBatching: this.executionPlanState.fallbackPlan.defaultDisableCommandBatching === true,
            ringTokens: this.executionPlanState.fallbackPlan.ringTokens ?? null,
            ringStop: this.executionPlanState.fallbackPlan.ringStop ?? null,
            ringStaging: this.executionPlanState.fallbackPlan.ringStaging ?? null,
          }
          : null,
        activePlanIdAtStart: activeExecutionPlan.id,
        finalActivePlanId: this.executionPlanState.activePlanId ?? activeExecutionPlan.id,
        transitions: Array.isArray(this.stats.executionPlan?.transitions)
          ? [...this.stats.executionPlan.transitions]
          : [],
      };
      stats.kernelPathId ??= activeExecutionPlan.kernelPathId ?? this.resolvedKernelPath?.id ?? null;
      if (this.stats.operatorDiagnostics) {
        stats.operatorDiagnostics = this.stats.operatorDiagnostics;
      }
      stats.kernelPathSource ??= activeExecutionPlan.kernelPathSource ?? this.kernelPathSource ?? 'none';
    }
    const ringStats = this.decodeRing?.getStats();
    if (ringStats) {
      stats.decodeRing = ringStats;
    }
    return stats;
  }


  getBatchingStats() {
    return { ...this.batchingStats };
  }


  getMemoryStats() {

    const stats = { used: 0 };

    try {
      const poolStats = getGlobalBufferPool().getStats();
      stats.pool = poolStats;
      stats.used += poolStats.currentBytesAllocated || 0;
    } catch {
      // Buffer pool not initialized yet
    }

    if (this.kvCache) {
      const kvStats = this.kvCache.getMemoryStats();
      stats.kvCache = kvStats;
      stats.used += kvStats.allocated || 0;
    }

    if (this.emulation?.config?.statsEnabled) {
      stats.emulation = this.emulation.getStats();
    }

    return stats;
  }


  getKVCacheStats() {
    if (!this.kvCache) return null;
    const { seqLen, maxSeqLen } = this.kvCache.getMemoryStats();
    return { seqLen, maxSeqLen };
  }


  getBufferPool() {
    try {
      return getGlobalBufferPool();
    } catch {
      return null;
    }
  }


  async unload() {
    const storageContext = this.storageContext;
    this.storageContext = null;
    await destroyEmulation(this.emulation);
    this.emulation = null;
    this.decodeRing?.release();
    this.kvCache?.clear();
    destroyPleRuntimeCache(this.weights.get('per_layer_inputs'));
    destroyPleBufferCache(this.pleCache);
    this.pleCache = null;
    this.plePrefetchPending = null;
    this.weights.clear();
    this.expertWeights.clear();
    this.linearAttentionRuntime = resetLinearAttentionRuntime(this.linearAttentionRuntime);
    this.lora = null;
    destroyMoERouter(this.moeRouter);
    this.moeRouter = null;
    if (this.finitenessBuffer) {
      this.finitenessBuffer.destroy();
      this.finitenessBuffer = null;
    }
    if (this.sampleReadbackBuffer) {
      this.sampleReadbackBuffer.destroy();
      this.sampleReadbackBuffer = null;
    }
    if (typeof storageContext?.close === 'function') {
      await storageContext.close();
    }
    this.isLoaded = false;
    this.currentSeqLen = 0;
    restorePipelineContexts(this);
    log.info('Pipeline', 'Unloaded');
  }


  setLoRAAdapter(adapter) {
    this.lora = adapter;
  }


  getActiveLoRA() {
    return this.lora;
  }


  reset() {
    this.kvCache?.clear();
    this.linearAttentionRuntime = resetLinearAttentionRuntime(this.linearAttentionRuntime);
    this.currentSeqLen = 0;
    this.decodeStepCount = 0;
    this.debugFlags = {};
    this.decodeBuffers?.resetPingPong();
    this.decodeRing?.reset();
    // Reset stats
    this.stats.tokensGenerated = 0;
    this.stats.totalTimeMs = 0;
    this.stats.prefillTimeMs = 0;
    this.stats.decodeTimeMs = 0;
    this.stats.gpuTimePrefillMs = undefined;
    this.stats.gpuTimeDecodeMs = undefined;
    this.stats.prefillProfileSteps = [];
    this.stats.decodeProfileSteps = [];
    this.stats.executionPlan = null;
    this.stats.kernelPathId = null;
    this.stats.kernelPathSource = 'none';
    this.stats.attentionInputs = [];
  }

  /**
   * Lightweight reset for batch embedding: clears KV cache and sequence state
   * without resetting stats or debug flags (they are overwritten on each prefill).
   */
  resetForBatch() {
    this.kvCache?.clear();
    this.linearAttentionRuntime = resetLinearAttentionRuntime(this.linearAttentionRuntime);
    this.currentSeqLen = 0;
    this.decodeStepCount = 0;
    this.decodeBuffers?.resetPingPong();
    this.decodeRing?.reset();
  }


  releaseGPUResources() {
    this.decodeBuffers?.release();
    this.decodeRing?.release();
    destroyMoERouter(this.moeRouter);
    this.moeRouter = null;
    destroyPleRuntimeCache(this.weights.get('per_layer_inputs'));
    destroyPleBufferCache(this.pleCache);
    this.pleCache = null;
    this.plePrefetchPending = null;
    if (this.finitenessBuffer) {
      this.finitenessBuffer.destroy();
      this.finitenessBuffer = null;
    }
    if (this.sampleReadbackBuffer) {
      this.sampleReadbackBuffer.destroy();
      this.sampleReadbackBuffer = null;
    }
  }
}

// ============================================================================
// Factory Function
// ============================================================================


async function createTransformerPipeline(manifest, contexts = {}) {
  return createInitializedPipeline(InferencePipeline, manifest, contexts);
}

registerPipeline('transformer', createTransformerPipeline);
registerPipeline('gemma4', createTransformerPipeline);

export class EmbeddingPipeline extends InferencePipeline {
  async *generate() {
    throw new Error('Embedding pipeline does not support token generation. Use embed() or prefillWithEmbedding().');
  }
}

async function createEmbeddingPipeline(manifest, contexts = {}) {
  return createInitializedPipeline(EmbeddingPipeline, manifest, contexts);
}

registerPipeline('embedding', createEmbeddingPipeline);

function resolveLazyPipelineModules(modelType) {
  const modules = selectRuleValue('inference', 'config', 'pipelineModules', {
    modelType,
    modelTypeLower: String(modelType).toLowerCase(),
  });
  if (!Array.isArray(modules)) return [];
  return modules.filter((entry) => typeof entry === 'string' && entry.length > 0);
}

export async function createPipeline(manifest, contexts = {}) {
  const modelType = manifest?.modelType;
  if (typeof modelType !== 'string' || modelType.length === 0) {
    throw new Error('Manifest is missing modelType. Re-convert the model with modelType set.');
  }
  let factory = getPipelineFactory(modelType);

  if (!factory) {
    for (const modulePath of resolveLazyPipelineModules(modelType)) {
      await import(modulePath);
    }
    factory = getPipelineFactory(modelType);
  }

  if (!factory) {
    throw new Error(`No pipeline registered for modelType "${modelType}".`);
  }

  return factory(manifest, contexts);
}

export { InferencePipeline as Pipeline };
