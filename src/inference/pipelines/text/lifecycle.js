import { getDevice, initDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { getUniformCacheStats } from '../../../gpu/uniform-cache.js';
import { getBufferPool as getGlobalBufferPool, readBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { log } from '../../../debug/index.js';
import { configurePerfGuards } from '../../../gpu/perf-guards.js';
import { MoERouter } from '../../moe-router.js';
import { DecodeBufferManager } from '../../decode-buffers.js';
import { DecodeRing } from '../../decode-ring.js';
import { applyPipelineContexts, restorePipelineContexts } from '../context.js';
import { createInitializedPipeline } from '../factory.js';
import { PipelineState } from './state.js';
import { PipelineGenerator } from './generator.js';
import { parseModelConfig } from './config.js';
import {
  initRoPEFrequencies,
  createKVCache,
  loadWeights,
  initMoERouter,
  initSpeculativeDecoder,
  fuseQKVWeights,
  initEmulation,
  destroyEmulation,
} from './init.js';
import { formatChatMessages } from './chat-format.js';
import {
  runKernelWarmup,
  applyModelBatchingRuntimeDefaults,
  resolveKernelPathState,
  initTokenizerFromManifest,
  assertManifestComputeLaneBinding,
} from './model-load.js';
import { resolvePerLayerInputsSession } from './generator/session-context.js';
import { getKernelPathActivationDtype } from '../../../config/kernel-path-loader.js';
import { applyPipelineDebugConfig } from './debug-utils.js';
import { resolveLayerPipeline } from './layer-plan.js';
import { compileExecutionPlanState, resolveActiveExecutionPlan } from './execution-plan.js';
import { assertDtypeConsistency } from './dtype-contract.js';
import { applyExecutionV1RuntimeConfig, hasExecutionV1 } from './execution-v1.js';
import { getPlatform } from '../../../config/platforms/loader.js';
import {
  createLinearAttentionRuntime,
  hasLinearAttentionLayers,
  resetLinearAttentionRuntime,
  restoreLinearAttentionRuntime,
} from './linear-attention.js';
import { getDopplerLoader } from '../../../loader/doppler-loader.js';
import { registerPipeline, getPipelineFactory } from '../registry.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { createObservationContext } from '../../observation-context.js';
import { createResolvedRuntimeSession } from './resolved-runtime-session.js';
import { assertBundledAdapterAuthorized } from '../../../config/revocation-policy.js';
import { initConvLayerState } from './ops.js';
import { destroyPleBufferCache, destroyPleRuntimeCache } from './per-layer-inputs.js';
import {
  createPipelineLoadTiming,
  destroyMoERouter,
  finishPipelineLoadTimingPhase,
  roundPipelineTimingMs,
  timedPipelineLoadPhase,
} from './pipeline-load-timing.js';
import {
  loadGlmOcrVisionWeights,
  loadRequiredVisionGpuTensor,
  loadVisionClipRange,
} from '../vision/weight-loading.js';

export async function initialize(contexts = {}) {
    const { runtimeConfig, sharedDebug } = applyPipelineContexts(this, contexts, {
      assignGpuContext: true,
      assignUseGPU: true,
      assignMemoryContext: true,
      assignStorageContext: true,
    });
    this.runtimeConfig = runtimeConfig;
    this.commandContext = contexts.commandContext ?? null;
    this.observationContext = createObservationContext({
      runtimeConfig,
      commandContext: this.commandContext,
    });
    this.dopplerLoader = contexts.loader || null;
    this.ownsDopplerLoader = contexts.ownsLoader === true;
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

export async function loadModel(manifest) {
    const loadStart = performance.now();
    const pipelineLoadTiming = createPipelineLoadTiming(manifest?.modelId ?? null);
    const resetStart = performance.now();
    this.stats.pipelineLoadTiming = pipelineLoadTiming;
    this.manifest = manifest;
    this.decodeRing?.release();
    if (this.sampleReadbackBuffer) {
      this.sampleReadbackBuffer.destroy();
      this.sampleReadbackBuffer = null;
    }
    this.linearAttentionRuntime = resetLinearAttentionRuntime(this.linearAttentionRuntime);
    destroyMoERouter(this.moeRouter);
    this.moeRouter = null;
    finishPipelineLoadTimingPhase(pipelineLoadTiming, 'reset', resetStart);

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
    const configResolutionStart = performance.now();

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
        useGPU: this.useGPU === true,
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
    finishPipelineLoadTimingPhase(pipelineLoadTiming, 'configResolution', configResolutionStart);

    await timedPipelineLoadPhase(
      pipelineLoadTiming,
      'kernelWarmup',
      { modelId: manifest.modelId ?? null },
      () => runKernelWarmup({
        useGPU: this.useGPU,
        kernelWarmup: this.runtimeConfig.shared?.kernelWarmup,
      })
    );

    const executionSetupStart = performance.now();
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
    finishPipelineLoadTimingPhase(pipelineLoadTiming, 'executionSetup', executionSetupStart);

    this.tokenizer = await timedPipelineLoadPhase(
      pipelineLoadTiming,
      'tokenizer',
      { modelId: manifest.modelId ?? null },
      () => initTokenizerFromManifest(
        manifest,
        this.baseUrl,
        this.storageContext
      )
    );
    pipelineLoadTiming.details.tokenizer = typeof this.tokenizer?.getLoadTiming === 'function'
      ? this.tokenizer.getLoadTiming()
      : null;
    const tokenizerVocabSize = this.tokenizer.getVocabSize();
    if (Number.isFinite(tokenizerVocabSize) && tokenizerVocabSize > 0) {
      if (tokenizerVocabSize !== this.modelConfig.vocabSize) {
        log.info('Pipeline', `Tokenizer vocabSize=${tokenizerVocabSize} differs from model=${this.modelConfig.vocabSize}, using model size`);
      }
    }

    const postTokenizerExecutionSetupStart = performance.now();
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
    this.resolvedRuntimeSession = createResolvedRuntimeSession({
      manifest,
      modelConfig: this.modelConfig,
      runtimeConfig: this.runtimeConfig,
      resolvedKernelPath: this.resolvedKernelPath,
      kernelPathSource: this.kernelPathSource,
      executionV1State: this.executionV1State,
      executionPlanState: this.executionPlanState,
    });

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
    finishPipelineLoadTimingPhase(
      pipelineLoadTiming,
      'executionSetup',
      postTokenizerExecutionSetupStart
    );

    // Load weights
    await timedPipelineLoadPhase(
      pipelineLoadTiming,
      'loadWeights',
      { modelId: manifest.modelId ?? null },
      () => this._loadWeights()
    );

    // Initialize RoPE frequencies
    await timedPipelineLoadPhase(
      pipelineLoadTiming,
      'rope',
      { modelId: manifest.modelId ?? null },
      () => this._initRoPE()
    );

    // Initialize conv layer states for gated short conv layers (LFM2)
    await timedPipelineLoadPhase(
      pipelineLoadTiming,
      'convStates',
      { modelId: manifest.modelId ?? null },
      () => this._initConvLayerStates()
    );

    this.isLoaded = true;
    const loadMs = performance.now() - loadStart;
    this.stats.modelLoadMs = loadMs;
    pipelineLoadTiming.totalMs = roundPipelineTimingMs(loadMs);
    pipelineLoadTiming.status = 'complete';
    log.info('Pipeline', `Model loaded successfully (${loadMs.toFixed(0)}ms)`);
  }

export async function _loadWeights() {
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
        loader: this.dopplerLoader ?? undefined,
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
    this.weights.set('lm_head_bias', result.lmHeadBias);
    if (result.lmHeadBias) {
      // Recorded/fused logits currently keep their result on GPU, while the
      // manifest-owned decoder bias is loaded as an exact CPU tensor. Route
      // biased heads through computeLogits until the execution graph declares
      // and records an explicit GPU bias-add step.
      this.disableRecordedLogits = true;
      this.disableFusedDecode = true;
    }
    this.weights.set('final_norm', result.finalNorm);
    this.weights.set('final_norm_bias', result.finalNormBias);
    if (result.finalNormBias) {
      this.disableRecordedLogits = true;
      this.disableFusedDecode = true;
    }
    this.weights.set('diffusion_gemma_self_conditioning', result.diffusionGemmaSelfConditioning);
    this.weights.set('per_layer_inputs', result.perLayerInputWeights);
    this.embeddingPostprocessor = result.embeddingPostprocessor;

    this.layerRouterWeights = result.layerRouterWeights;

    this.dopplerLoader = result.loader ?? getDopplerLoader(this.runtimeConfig.loading);
    this.stats.loadTiming = result.loadTiming ?? this.dopplerLoader?.getLoadTiming?.() ?? null;

    if ((this.modelConfig).useMoE && this.moeRouter) {
      this.moeRouter = initMoERouter(
        (this.modelConfig),
        this.runtimeConfig.inference.moe.routing,
        result.layerWeights
      );
    }

    if (this.useGPU && this.modelConfig) {
      const session = this.runtimeConfig?.inference?.session ?? null;
      fuseQKVWeights(result.layerWeights, this.modelConfig, this.resolvedKernelPath, {
        allowQ4K: session?.useFusedQKVSplitQKNorm === true
          || session?.useFusedQKVSplitQKNormRoPE === true,
      });
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

export async function _initRoPE() {
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

export async function _initConvLayerStates() {
    const config = this.modelConfig;
    if (!config?.layerTypes) return;
    const { getDevice } = await import('../../../gpu/device.js');
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

export async function _loadVisionWeights() {
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
    const loadClipRange = (prefix) => loadVisionClipRange(loadRequiredTensor, prefix);

    if (vc.visionArchitecture === 'glmocr') {
      this.visionWeights = await loadGlmOcrVisionWeights(loader, {
        textHiddenSize: this.modelConfig.hiddenSize,
        depth,
      });
      log.info('Pipeline', `Vision weights loaded (${depth} GLM-OCR encoder layers)`);
      return;
    }

    if (vc.visionArchitecture === 'gemma4') {
      const isEncoderFree = (depth === 0);
      const visionWeights = {
        textHiddenSize: this.modelConfig.hiddenSize,
        patchInputProj: await loadRequiredTensor('model.vision_tower.patch_embedder.input_proj.weight'),
        patchPositionEmbeddingTable: await loadRequiredVisionGpuTensor(loader,
          'model.vision_tower.patch_embedder.position_embedding_table'
        ),
        projector: isEncoderFree
          ? await loader.loadTensor('model.embed_vision.embedding_projection.weight', true, true)
          : await loadRequiredTensor('model.embed_vision.embedding_projection.weight'),
        layers: [],
      };

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

export async function _ensureVisionWeightsLoaded() {
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

export async function _loadAudioWeights() {
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

    const isEncoderFree = (depth === 0);
    const audioWeights = {
      // Subsampling
      subsampleConv0Weight: isEncoderFree ? null : await loadRequiredTensor('model.audio_tower.subsample_conv_projection.layer0.conv.weight'),
      subsampleNorm0Weight: isEncoderFree ? null : await loadRequiredTensor('model.audio_tower.subsample_conv_projection.layer0.norm.weight'),
      subsampleConv1Weight: isEncoderFree ? null : await loadRequiredTensor('model.audio_tower.subsample_conv_projection.layer1.conv.weight'),
      subsampleNorm1Weight: isEncoderFree ? null : await loadRequiredTensor('model.audio_tower.subsample_conv_projection.layer1.norm.weight'),
      subsampleInputProjWeight: isEncoderFree ? null : await loadRequiredTensor('model.audio_tower.subsample_conv_projection.input_proj_linear.weight'),
      // Output
      outputProjWeight: isEncoderFree ? null : await loadRequiredTensor('model.audio_tower.output_proj.weight'),
      outputProjBias: isEncoderFree ? null : await loadRequiredTensor('model.audio_tower.output_proj.bias'),
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

export async function _ensureAudioWeightsLoaded() {
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
