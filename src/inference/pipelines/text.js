
import { getDevice, initDevice, getKernelCapabilities } from '../../gpu/device.js';
import { getBufferPool as getGlobalBufferPool } from '../../memory/buffer-pool.js';
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
import {
  runKernelWarmup,
  applyModelBatchingRuntimeDefaults,
  resolveKernelPathState,
  initTokenizerFromManifest,
} from './text/model-load.js';
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
import { initConvLayerState } from './text/ops.js';

function destroyMoERouter(router) {
  if (router && typeof router.destroy === 'function') {
    router.destroy();
  }
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
    this.runtimeOverrides = typeof structuredClone === 'function'
      ? structuredClone(runtimeConfig)
      : JSON.parse(JSON.stringify(runtimeConfig));
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
        log.info(
          'Pipeline',
          `Execution v1 enabled (steps=${this.executionV1State.resolvedSteps.all.length}, ` +
          `kernelPathInline=${this.executionV1State.runtimeInferencePatch.kernelPath ? 'yes' : 'no'}, ` +
          `pipelineInline=${this.executionV1State.runtimeInferencePatch.pipeline ? 'yes' : 'no'}` +
          `${transformInfo}${fallbackInfo})`
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
      this.modelConfig
    );
    this.useTiedEmbeddings = this.modelConfig.useTiedEmbeddings;
    this.embeddingVocabSize = this.modelConfig.embeddingVocabSize;
    this.embeddingTranspose = this.modelConfig.embeddingTranspose;

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

    this.tokenizer = await initTokenizerFromManifest(
      manifest,
      this.baseUrl,
      this.storageContext
    );
    const tokenizerVocabSize = this.tokenizer.getVocabSize();
    if (Number.isFinite(tokenizerVocabSize) && tokenizerVocabSize > 0) {
      if (tokenizerVocabSize !== this.modelConfig.vocabSize) {
        log.info('Pipeline', `Tokenizer vocabSize=${tokenizerVocabSize} differs from model=${this.modelConfig.vocabSize}, using model size`);
      }
    }

    // Check for execution-v1 kvDtype conflict with manifest quantization info
    if (this.executionV1State && this.resolvedKernelPath) {
      const manifestComputeHint = manifest?.quantizationInfo?.compute;
      const resolvedKvDtype = this.runtimeConfig.inference.session?.kvcache?.kvDtype;
      if (
        manifestComputeHint
        && resolvedKvDtype
        && String(manifestComputeHint).toLowerCase() !== String(resolvedKvDtype).toLowerCase()
      ) {
        log.warn(
          'Pipeline',
          `KV cache kvDtype from execution-v1 resolution (${resolvedKvDtype}) differs from ` +
          `manifest quantizationInfo.compute hint (${manifestComputeHint}). ` +
          `The kernel path dtype contract takes precedence.`
        );
      }
    }

    // Initialize KV cache
    this.kvCache = createKVCache(this.modelConfig, this.useGPU, this.debug, this.runtimeConfig.inference);
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
    await this._loadWeights();

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
      fuseQKVWeights(result.layerWeights, this.modelConfig);
    }

    if (this.useGPU && this.modelConfig) {
      const activeExecutionPlan = resolveActiveExecutionPlan(this);
      try {
        this.decodeBuffers?.ensureBuffers({
          hiddenSize: this.modelConfig.hiddenSize,
          intermediateSize: this.modelConfig.intermediateSize,
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
      headDim: config.headDim,
      rotaryDim: config.ropeRotaryDim,
      maxSeqLen,
      ropeTheta: config.ropeTheta,
      ropeLocalTheta: config.ropeLocalTheta,
      mropeInterleaved: config.mropeInterleaved,
      mropeSection: config.mropeSection,
      partialRotaryFactor: config.partialRotaryFactor,
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
    return this.generator.generate(prompt, options);
  }

  generateTokens(prompt, options = {}) {
    return this.generator.generateTokens(prompt, options);
  }

  generateTokenIds(prompt, options = {}) {
    return this.generator.generateTokenIds(prompt, options);
  }

  decodeStepLogits(currentIds, options = {}) {
    return this.generator.decodeStepLogits(currentIds, options);
  }

  advanceWithToken(tokenId, options = {}) {
    return this.generator.advanceWithToken(tokenId, options);
  }

  advanceWithTokenAndEmbedding(tokenId, options = {}) {
    return this.generator.advanceWithTokenAndEmbedding(tokenId, options);
  }


  prefillKVOnly(prompt, options = {}) {
    return this.generator.prefillKVOnly(prompt, options);
  }

  prefillWithEmbedding(prompt, options = {}) {
    return this.generator.prefillWithEmbedding(prompt, options);
  }

  async embed(prompt, options = {}) {
    const result = await this.prefillWithEmbedding(prompt, options);
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
    const outputs = [];
    for (const prompt of prompts) {
      outputs.push(await this.embed(prompt, options));
      this.reset();
    }
    return outputs;
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
    await destroyEmulation(this.emulation);
    this.emulation = null;
    this.decodeRing?.release();
    this.kvCache?.clear();
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
    this.stats.decodeProfileSteps = [];
    this.stats.executionPlan = null;
    this.stats.kernelPathId = null;
    this.stats.kernelPathSource = 'none';
    this.stats.attentionInputs = [];
  }


  releaseGPUResources() {
    this.decodeBuffers?.release();
    this.decodeRing?.release();
    destroyMoERouter(this.moeRouter);
    this.moeRouter = null;
    if (this.finitenessBuffer) {
      this.finitenessBuffer.destroy();
      this.finitenessBuffer = null;
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
