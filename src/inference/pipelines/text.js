
import { getDevice, initDevice, getKernelCapabilities } from '../../gpu/device.js';
import { getUniformCacheStats } from '../../gpu/uniform-cache.js';
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
import { resolvePerLayerInputsSession } from './text/generator/session-context.js';
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
import { createObservationContext } from '../observation-context.js';
import { createResolvedRuntimeSession } from './text/resolved-runtime-session.js';
import { assertBundledAdapterAuthorized } from '../../config/revocation-policy.js';
import { assertNotAborted } from './text/abort-contract.js';
export { AbortError, isAbortError } from './text/abort-contract.js';
import { destroyMoERouter } from './text/pipeline-load-timing.js';
export { buildConservativeMultimodalGenerationOptions } from './text/modality-token-contract.js';
import { initConvLayerState } from './text/ops.js';
import { destroyPleBufferCache, destroyPleRuntimeCache } from './text/per-layer-inputs.js';
import {
  initialize as initializeImpl,
  loadModel as loadModelImpl,
  _loadWeights as _loadWeightsImpl,
  _initRoPE as _initRoPEImpl,
  _initConvLayerStates as _initConvLayerStatesImpl,
  _loadVisionWeights as _loadVisionWeightsImpl,
  _ensureVisionWeightsLoaded as _ensureVisionWeightsLoadedImpl,
  _loadAudioWeights as _loadAudioWeightsImpl,
  _ensureAudioWeightsLoaded as _ensureAudioWeightsLoadedImpl,
} from './text/lifecycle.js';
import {
  transcribeImage as transcribeImageImpl,
  transcribeVideo as transcribeVideoImpl,
  transcribeAudio as transcribeAudioImpl,
  embed as embedImpl,
  embedBatch as embedBatchImpl,
  encodeSequence as encodeSequenceImpl,
  embedImage as embedImageImpl,
  embedAudio as embedAudioImpl,
} from './text/execution.js';

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
    return initializeImpl.call(this, contexts);
  }

  async loadModel(manifest) {
    return loadModelImpl.call(this, manifest);
  }

  async _loadWeights() {
    return _loadWeightsImpl.call(this);
  }

  setPreloadedWeights(weights) {
    this._preloadedWeights = weights;
  }

  async _initRoPE() {
    return _initRoPEImpl.call(this);
  }

  async _initConvLayerStates() {
    return _initConvLayerStatesImpl.call(this);
  }

  async _loadVisionWeights() {
    return _loadVisionWeightsImpl.call(this);
  }

  async _ensureVisionWeightsLoaded() {
    return _ensureVisionWeightsLoadedImpl.call(this);
  }

  async _loadAudioWeights() {
    return _loadAudioWeightsImpl.call(this);
  }

  async _ensureAudioWeightsLoaded() {
    return _ensureAudioWeightsLoadedImpl.call(this);
  }

  // ==========================================================================
  // Vision: transcribeImage
  // ==========================================================================

  async transcribeImage({ imageBytes, width, height, prompt, maxTokens, softTokenBudget, signal }) {
    return transcribeImageImpl.apply(this, arguments);
  }

  // ==========================================================================
  // Video: transcribeVideo
  // ==========================================================================

  async transcribeVideo({ frames, prompt, maxTokens, maxFrames, perFrameSoftTokenBudget, signal }) {
    return transcribeVideoImpl.apply(this, arguments);
  }

  // ==========================================================================
  // Audio: transcribeAudio
  // ==========================================================================

  async transcribeAudio({ audio, prompt, maxTokens, signal }) {
    return transcribeAudioImpl.apply(this, arguments);
  }

  // ==========================================================================
  // Capability Detection
  // ==========================================================================

  get capabilities() {
    const caps = ['generation'];
    if (typeof this.prefillWithEmbedding === 'function') caps.push('embedding');
    if (this.manifest?.inference?.supportsSequence === true) caps.push('sequence');
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

  resetToSeqLen(seqLen) {
    return this.generator.resetToSeqLen(seqLen);
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

  prefillForLoRATraining(inputIds, options) {
    return this.generator.prefillForLoRATraining(inputIds, options);
  }

  computeDiffusionGemmaCanvasLogits(args, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.computeDiffusionGemmaCanvasLogits(args, options);
  }

  computeDiffusionGemmaCanvasStep(args, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.computeDiffusionGemmaCanvasStep(args, options);
  }

  prefillWithEmbedding(prompt, options = {}) {
    assertNotAborted(options?.signal);
    return this.generator.prefillWithEmbedding(prompt, options);
  }

  async embed(prompt, options = {}) {
    return embedImpl.apply(this, arguments);
  }

  async embedBatch(prompts, options = {}) {
    return embedBatchImpl.apply(this, arguments);
  }

  async encodeSequence(sequence, options = {}) {
    return encodeSequenceImpl.apply(this, arguments);
  }

  // Run the vision encoder over a single image and return a mean-pooled
  // embedding in the model's text-hidden-size space. Decoder responsibility
  // (jpeg/png -> RGBA pixels) belongs to the caller; this method takes
  // already-decoded pixel data.
  async embedImage({ pixels, width, height, softTokenBudget, signal } = {}) {
    return embedImageImpl.apply(this, arguments);
  }

  // Run the audio encoder over a single PCM segment and return a mean-pooled
  // embedding in the model's audio-projection-output space (which equals the
  // text hidden size in Gemma 4). Decoder responsibility (webm/opus/wav ->
  // Float32 PCM at the model's expected sample rate) belongs to the caller.
  async embedAudio({ audio, signal } = {}) {
    return embedAudioImpl.apply(this, arguments);
  }

  prefillWithLogits(prompt, options = {}) {
    return this.generator.prefillWithLogits(prompt, options);
  }

  prefillWithTokenLogits(prompt, tokenIds, options = {}) {
    return this.generator.prefillWithTokenLogits(prompt, tokenIds, options);
  }

  prefillWithTokenLogitsFromKV(prefix, prompt, tokenIds, options = {}) {
    return this.generator.prefillWithTokenLogitsFromKV(prefix, prompt, tokenIds, options);
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
            readbackMode: this.executionPlanState.primaryPlan.readbackMode ?? null,
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
            readbackMode: this.executionPlanState.fallbackPlan.readbackMode ?? null,
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
    const uniformCacheStats = getUniformCacheStats();
    if (uniformCacheStats) {
      stats.uniformCache = uniformCacheStats;
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
    if (this.ownsDopplerLoader && this.dopplerLoader) {
      await this.dopplerLoader.unload();
    }
    this.dopplerLoader = null;
    this.ownsDopplerLoader = false;
    this.linearAttentionRuntime = resetLinearAttentionRuntime(this.linearAttentionRuntime);
    this.lora = null;
    this.revocationIdentity = null;
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
    assertBundledAdapterAuthorized(adapter);
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
