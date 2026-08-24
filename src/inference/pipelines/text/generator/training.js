import { getDevice, setTrackSubmits } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer, readBufferSlice, uploadData } from '../../../../memory/buffer-pool.js';
import { allowReadback } from '../../../../gpu/perf-guards.js';
import { log, trace, isTraceEnabled } from '../../../../debug/index.js';
import {
  CAPTURE_LEVELS,
  createDefaultCaptureConfig,
  validateCaptureConfig,
} from '../../../../debug/index.js';
import { validateCallTimeOptions } from '../../../../config/param-validator.js';
import { sample, applyRepetitionPenalty, logitsSanity, getTopK } from '../sampling.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU, extractLastPositionLogits, applySoftcapping } from '../logits/index.js';
import { OperatorEventEmitter } from '../operator-events.js';
import {
  getLogitsConfig,
  getLogitsWeights,
} from './logits-config.js';
import {
  assertTokenIdsInRange,
  assertTokenIdInRange,
  resolveStepOptions,
  resolveGenerateOptions,
  resolvePrefillOptions,
  resolvePrefillEmbeddingOptions,
  resolveAdvanceEmbeddingMode,
} from '../generator-runtime.js';
import { extractEmbeddingFromHiddenGPU } from '../embedding-extraction.js';
import { decodeReadback, getLogitsHealth } from '../debug-utils/index.js';
import {
  advanceDecodeStepCount,
  createTsirFixtureState,
  drainPendingTsirReads,
} from '../tsir-fixture-writer.js';
import {
  activateFallbackExecutionPlan,
  hasFallbackExecutionPlan,
  rebaseExecutionSessionPlan,
  resetActiveExecutionPlan,
  resolveMaxBatchDecodeTokens,
  resolvePrefillRecorderChunkLayers,
  resolveActiveExecutionPlan,
  setActiveExecutionPlan,
} from '../execution-plan.js';
import {
  cloneLinearAttentionRuntime,
  hasLinearAttentionLayers,
  resetLinearAttentionRuntime,
  restoreLinearAttentionRuntime,
} from '../linear-attention.js';
import { createTensor } from '../../../../gpu/tensor.js';
import {
  resolvePromptInput,
} from './prompt-input.js';
import {
  shouldDisableBatchDecodeAfterShortBatch,
  resolveHotVocabularyBatchDecodeAvailability,
  primePleDecodeRuntimeCache,
  recordPrefillProfileStep,
  resolveTokenText,
  usesReplayPrefillDecode,
  assertIncrementalDecodeSupport,
  summarizeExecutionPlan,
  shouldRetryWithFinitenessFallback,
  createUnhandledFinitenessPolicyError,
  resolveTargetPlanKVDtype,
  resolveCurrentKVCacheDtype,
  cloneRuntimeInferenceWithKVDtype,
} from '../generator-decode-policy.js';
import { FINITENESS_RESET_WORDS, borrowLinearWeight, borrowNormWeight, canUseChunkedSoftEmbeddingLogits, getExperimentalIntentBundleModule, normalizeCanvasTokenIds, normalizeSelectedLogitTokenIds, normalizeSelfConditioningLogits, normalizeSelfConditioningLogitsState, normalizeSelfConditioningSoftEmbeddingState, recordPrefillRecorderStats, releaseBorrowedWeight, resolveDiffusionGemmaSoftEmbeddingChunkRows, resolvePrefillChunkSubmitMode, resolveSuppressedSamplingTokenIds, traceActivationHealth } from './text.js';

export async function prefillKVOnly(prompt, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    assertIncrementalDecodeSupport(this._state, 'prefillKVOnly');
    this._resetDecodeRuntimeState();
    this._state.stats.gpuTimePrefillMs = undefined;
    this._state.stats.prefillProfileSteps = [];
    const opts = resolvePrefillOptions(this._state, options);
    const prefillStartSeqLen = this._state.currentSeqLen;
    const inputIds = this._resolvePromptOrInputIds(prompt, opts.useChatTemplate, 'prefillKVOnly', opts.inputIds);
    if (opts.debug) {
      log.debug('Pipeline', `PrefillKVOnly: ${inputIds.length} tokens`);
    }

    try {
      let prefillResult;
      try {
        prefillResult = await this._prefillToHidden(inputIds, opts);
      } catch (error) {
        if (this._shouldUseFinitenessFallback(error, 'prefillKVOnly')) {
          log.warn('Pipeline', `FinitenessGuard caught NaN/Inf during prefillKVOnly. Retrying with F32 precision.`);
          prefillResult = await this._retryWithPersistentFinitenessFallback(
            opts,
            'prefillKVOnly',
            1,
            () => this._prefillToHidden(inputIds, opts),
            prefillStartSeqLen
          );
        } else {
          throw error;
        }
      }

      const {
        numTokens,
        startPos,
        currentRecorder,
        recordProfile,
        currentHiddenBuffer,
      } = prefillResult;

      // Ensure prefill work completes before returning a usable snapshot.
      if (currentRecorder) {
        recordPrefillRecorderStats(this._state, currentRecorder);
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
      } else {
        const device = getDevice();
        if (device) {
          await device.queue.onSubmittedWorkDone();
        }
      }

      this._state.currentSeqLen = startPos + numTokens;
      releaseBuffer(currentHiddenBuffer);

      const snapshot = this._state.kvCache?.clone();
      if (!snapshot) {
        throw new Error('KV cache unavailable after prefill');
      }

      return {
        cache: snapshot,
        seqLen: this._state.currentSeqLen,
        tokens: inputIds,
        linearAttention: await cloneLinearAttentionRuntime(this._state.linearAttentionRuntime),
      };
    } finally {
      this._closeFinitenessFallbackWindow(opts);
    }
  }

export async function prefillForLoRATraining(inputIds, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');
    if (!Array.isArray(inputIds) && !(inputIds instanceof Int32Array) && !(inputIds instanceof Uint32Array)) {
      throw new Error('prefillForLoRATraining requires token IDs.');
    }
    const tokens = Array.from(inputIds, (value) => Number(value));
    if (tokens.length === 0) throw new Error('prefillForLoRATraining requires at least one token.');
    this._assertTokenIdsInRange(tokens, 'prefillForLoRATraining.inputIds');

    const config = this._state.modelConfig;
    const layerIdx = Number(options.layerIdx);
    if (!Number.isInteger(layerIdx) || layerIdx !== config.numLayers - 1) {
      throw new Error(`native_lora_target_not_supported: layerIdx must be the final layer (${config.numLayers - 1}).`);
    }
    if (options.module !== 'down_proj') {
      throw new Error('native_lora_target_not_supported: module must be down_proj.');
    }
    if (config.layerTypes?.[layerIdx] !== 'full_attention') {
      throw new Error('native_lora_target_not_supported: final layer must be full_attention.');
    }
    if (config.postFeedforwardNorm === true) {
      throw new Error('native_lora_target_not_supported: final-layer post-feedforward normalization is not supported.');
    }
    if (this._state.lora) {
      throw new Error('native_lora_base_capture_requires_inactive_adapter.');
    }

    this.resetToSeqLen(0);
    this._resetDecodeRuntimeState();
    const opts = resolvePrefillOptions(this._state, {
      ...options,
      inputIds: tokens,
      useChatTemplate: false,
    });
    let activationTensor = null;
    let hiddenTensor = null;
    let disposed = false;
    const releaseOwned = () => {
      if (disposed) return;
      disposed = true;
      if (activationTensor?.buffer) releaseBuffer(activationTensor.buffer);
      if (hiddenTensor?.buffer) releaseBuffer(hiddenTensor.buffer);
    };
    const capture = async ({ tensor, numTokens, hiddenSize, recorder }) => {
      if (activationTensor) {
        throw new Error('native_lora_training_capture_duplicated.');
      }
      const bytesPerElement = tensor.dtype === 'f16' ? 2 : 4;
      const byteLength = Math.ceil((numTokens * hiddenSize * bytesPerElement) / 4) * 4;
      const buffer = acquireBuffer(byteLength, undefined, `L${layerIdx}.native_lora_ffn_act`);
      if (recorder) {
        recorder.getEncoder().copyBufferToBuffer(tensor.buffer, 0, buffer, 0, byteLength);
      } else {
        const device = getDevice();
        const encoder = device.createCommandEncoder({ label: `L${layerIdx}.native_lora_capture` });
        encoder.copyBufferToBuffer(tensor.buffer, 0, buffer, 0, byteLength);
        device.queue.submit([encoder.finish()]);
      }
      activationTensor = createTensor(
        buffer,
        tensor.dtype,
        [numTokens, hiddenSize],
        `L${layerIdx}.native_lora_ffn_act`
      );
    };

    try {
      const prefill = await this._prefillToHidden(tokens, {
        ...opts,
        _trainingCapture: { layerIdx, stage: 'ffn_act', capture },
      });
      if (prefill.currentRecorder) {
        recordPrefillRecorderStats(this._state, prefill.currentRecorder);
        await prefill.currentRecorder.submitAndWait();
        await prefill.recordProfile(prefill.currentRecorder);
      } else {
        await getDevice().queue.onSubmittedWorkDone();
      }
      if (!activationTensor) {
        releaseBuffer(prefill.currentHiddenBuffer);
        throw new Error('native_lora_training_activation_not_captured.');
      }
      hiddenTensor = createTensor(
        prefill.currentHiddenBuffer,
        prefill.activationDtype,
        [prefill.numTokens, config.hiddenSize],
        `L${layerIdx}.native_lora_base_hidden`
      );
      this.resetToSeqLen(0);
      return {
        inputIds: tokens,
        layerIdx,
        module: 'down_proj',
        activation: activationTensor,
        baseHidden: hiddenTensor,
        dispose: releaseOwned,
      };
    } catch (error) {
      releaseOwned();
      this.resetToSeqLen(0);
      throw error;
    }
  }

export async function prefillWithEmbedding(prompt, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    assertIncrementalDecodeSupport(this._state, 'prefillWithEmbedding');
    const totalStart = performance.now();
    this._resetDecodeRuntimeState();
    this._state.stats.gpuTimePrefillMs = undefined;
    this._state.stats.prefillProfileSteps = [];
    const opts = resolvePrefillEmbeddingOptions(this._state, options);
    const diagnosticsEnabled = options?.diagnostics?.enabled === true
      || this._state.runtimeConfig?.shared?.harness?.mode === 'diagnose';
    const tsirFixtureCfg = this._state.runtimeConfig?.shared?.harness?.tsirFixture ?? null;
    if (diagnosticsEnabled || tsirFixtureCfg) {
      const captureConfig = {
        ...createDefaultCaptureConfig(),
        enabled: true,
        defaultLevel: CAPTURE_LEVELS.SLICE,
        ...(options?.diagnostics?.captureConfig ?? {}),
      };
      validateCaptureConfig(captureConfig);
      this._state.operatorDiagnostics = {
        enabled: diagnosticsEnabled === true,
        captureConfig,
        emitter: diagnosticsEnabled ? new OperatorEventEmitter({
          modelHash: this._state.manifest?.modelId ?? null,
          runtimeConfigHash: this._state.resolvedKernelPath?.id ?? null,
          executionPlanHash: opts.executionPlan?.id ?? null,
        }) : null,
        tsirFixture: createTsirFixtureState(tsirFixtureCfg),
      };
    }
    const prefillStartSeqLen = this._state.currentSeqLen;
    const inputStart = performance.now();
    const inputIds = this._resolvePromptOrInputIds(prompt, opts.useChatTemplate, 'prefillWithEmbedding', opts.inputIds);
    const inputMs = performance.now() - inputStart;
    if (opts.debug) {
      log.debug('Pipeline', `PrefillWithEmbedding: ${inputIds.length} tokens (mode=${opts.embeddingMode})`);
    }

    try {
      let prefillResult;
      const prefillStart = performance.now();
      try {
        prefillResult = await this._prefillToHidden(inputIds, {
          ...opts,
          _embeddingOnly: options.__skipStateSnapshot === true,
        });
      } catch (error) {
        if (shouldRetryWithFinitenessFallback(error)) {
          log.warn('Pipeline', `FinitenessGuard caught NaN/Inf during prefillWithEmbedding. Retrying with F32 precision.`);
          prefillResult = await this._retryWithPersistentFinitenessFallback(
            opts,
            'prefillWithEmbedding',
            1,
            () => this._prefillToHidden(inputIds, {
              ...opts,
              _embeddingOnly: options.__skipStateSnapshot === true,
            }),
            prefillStartSeqLen
          );
        } else {
          throw error;
        }
      }
      const prefillMs = performance.now() - prefillStart;

      const {
        numTokens,
        config,
        startPos,
        activationDtype,
        activationBytes,
        currentRecorder,
        recordProfile,
        currentHiddenBuffer,
      } = prefillResult;

      // Ensure prefill work completes before readback.
      let submitWaitMs = 0;
      const submitWaitStart = performance.now();
      if (currentRecorder) {
        recordPrefillRecorderStats(this._state, currentRecorder);
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
      } else {
        const device = getDevice();
        if (device) {
          await device.queue.onSubmittedWorkDone();
        }
      }
      submitWaitMs = performance.now() - submitWaitStart;

      if (!allowReadback('pipeline.prefill.embedding')) {
        throw new Error('GPU readback disabled; cannot return embedding');
      }

      let embedding;
      let tokenEmbeddings = null;
      let sequencePoolResult = null;
      let logits = null;
      let hiddenBytes = 0;
      let readbackMs = 0;
      let decodeHiddenMs = 0;
      let finalNormMs = 0;
      let extractMs = 0;
      let logitsMs = 0;
      try {
        if (options.__returnSequenceLogits === true) {
          const logitsStart = performance.now();
          logits = await computeLogits(
            currentHiddenBuffer,
            numTokens,
            getLogitsWeights(this._state),
            getLogitsConfig(this._state),
            this._state.useGPU,
            this._state.debugFlags,
            undefined,
            undefined,
            this._state.runtimeConfig.shared.debug.probes,
            { lastPositionOnly: false },
            this._state.operatorDiagnostics
          );
          logitsMs = performance.now() - logitsStart;
        }
        const hiddenSize = config.hiddenSize;
        hiddenBytes = numTokens * hiddenSize * activationBytes;
        const extractStart = performance.now();
        const extracted = await extractEmbeddingFromHiddenGPU({
          hiddenBuffer: currentHiddenBuffer,
          activationDtype,
          numTokens,
          hiddenSize,
          embeddingMode: opts.embeddingMode,
          finalNorm: this._state.weights.get('final_norm'),
          finalNormBias: this._state.weights.get('final_norm_bias') ?? null,
          config,
          embeddingPostprocessor: this._state.embeddingPostprocessor,
          returnTokenEmbeddings: options.__returnTokenEmbeddings === true,
          sequencePooling: options.__sequencePooling ?? null,
          tokenIds: inputIds,
        });
        embedding = extracted.embedding;
        tokenEmbeddings = extracted.tokenEmbeddings;
        sequencePoolResult = {
          pooledEmbedding: extracted.pooledSequenceEmbedding,
          tokenMask: extracted.tokenMask,
          includedTokenCount: extracted.includedTokenCount,
        };
        extractMs = performance.now() - extractStart;
      } finally {
        releaseBuffer(currentHiddenBuffer);
      }

      this._state.currentSeqLen = startPos + numTokens;
      const phase = {
        totalMs: performance.now() - totalStart,
        inputMs,
        prefillMs,
        submitWaitMs,
        readbackMs,
        decodeHiddenMs,
        finalNormMs,
        extractMs,
        logitsMs,
        hiddenBytes,
        tokens: numTokens,
        activationDtype,
        prefillRecordMs: this._state.stats.prefillRecordMs ?? null,
        prefillRecordOps: this._state.stats.prefillRecordOps ?? null,
        prefillRecordPasses: this._state.stats.prefillRecordPasses ?? null,
        prefillRecordOpLabels: this._state.stats.prefillRecordOpLabels ?? null,
        prefillSubmitWaitMs: this._state.stats.prefillSubmitWaitMs ?? null,
        gpuPrefillMs: this._state.stats.gpuTimePrefillMs ?? null,
      };

      // Batch embedding skips expensive KV cache clone and linear attention clone
      // since the caller will reset immediately after extracting the embedding.
      if (options.__skipStateSnapshot) {
        return {
          cache: null,
          seqLen: this._state.currentSeqLen,
          tokens: inputIds,
          embedding,
          tokenEmbeddings,
          pooledSequenceEmbedding: sequencePoolResult?.pooledEmbedding ?? null,
          tokenMask: sequencePoolResult?.tokenMask ?? null,
          includedTokenCount: sequencePoolResult?.includedTokenCount ?? 0,
          logits,
          embeddingMode: opts.embeddingMode,
          phase,
          linearAttention: null,
        };
      }

      const snapshot = this._state.kvCache?.clone();
      if (!snapshot) {
        throw new Error('KV cache unavailable after prefill');
      }

      return {
        cache: snapshot,
        seqLen: this._state.currentSeqLen,
        tokens: inputIds,
        embedding,
        tokenEmbeddings,
        pooledSequenceEmbedding: sequencePoolResult?.pooledEmbedding ?? null,
        tokenMask: sequencePoolResult?.tokenMask ?? null,
        includedTokenCount: sequencePoolResult?.includedTokenCount ?? 0,
        logits,
        embeddingMode: opts.embeddingMode,
        phase,
        linearAttention: await cloneLinearAttentionRuntime(this._state.linearAttentionRuntime),
      };
    } finally {
      this._closeFinitenessFallbackWindow(opts);
      this._state.stats.operatorDiagnostics = this._state.operatorDiagnostics?.emitter
        ? {
          enabled: true,
          timeline: this._state.operatorDiagnostics.emitter.getTimeline(),
          recordCount: this._state.operatorDiagnostics.emitter.length,
        }
        : null;
      this._state.operatorDiagnostics = null;
    }
  }

export async function prefillWithLogits(prompt, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    assertIncrementalDecodeSupport(this._state, 'prefillWithLogits');
    const totalStart = performance.now();
    this._resetDecodeRuntimeState();
    this._state.stats.gpuTimePrefillMs = undefined;
    this._state.stats.prefillProfileSteps = [];
    const opts = resolvePrefillOptions(this._state, options);
    const diagnosticsEnabled = options?.diagnostics?.enabled === true
      || this._state.runtimeConfig?.shared?.harness?.mode === 'diagnose';
    if (diagnosticsEnabled) {
      const captureConfig = {
        ...createDefaultCaptureConfig(),
        enabled: true,
        defaultLevel: CAPTURE_LEVELS.SLICE,
        ...(options?.diagnostics?.captureConfig ?? {}),
      };
      validateCaptureConfig(captureConfig);
      this._state.operatorDiagnostics = {
        enabled: true,
        captureConfig,
        emitter: new OperatorEventEmitter({
          modelHash: this._state.manifest?.modelId ?? null,
          runtimeConfigHash: this._state.resolvedKernelPath?.id ?? null,
          executionPlanHash: opts.executionPlan?.id ?? null,
        }),
        tsirFixture: null,
      };
    }
    try {
      const { inputIds, logits, phase: prefillPhase } = await this._prefillPromptToLogits(prompt, opts, 'prefillWithLogits');

      const snapshot = this._state.kvCache?.clone();
      if (!snapshot) {
        throw new Error('KV cache unavailable after prefill');
      }

      return {
        cache: snapshot,
        seqLen: this._state.currentSeqLen,
        tokens: inputIds,
        logits,
        phase: {
          ...(prefillPhase ?? {}),
          totalMs: performance.now() - totalStart,
        },
        linearAttention: await cloneLinearAttentionRuntime(this._state.linearAttentionRuntime),
      };
    } finally {
      this._closeFinitenessFallbackWindow(opts);
      this._state.stats.operatorDiagnostics = this._state.operatorDiagnostics?.emitter
        ? {
          enabled: true,
          timeline: this._state.operatorDiagnostics.emitter.getTimeline(),
          recordCount: this._state.operatorDiagnostics.emitter.length,
        }
        : null;
      this._state.operatorDiagnostics = null;
    }
  }

export async function prefillWithTokenLogits(prompt, tokenIds, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    assertIncrementalDecodeSupport(this._state, 'prefillWithTokenLogits');
    const totalStart = performance.now();
    const selectedTokenIds = normalizeSelectedLogitTokenIds(
      tokenIds,
      this._state.modelConfig.vocabSize,
      'prefillWithTokenLogits.tokenIds'
    );
    this._resetDecodeRuntimeState();
    this._state.stats.gpuTimePrefillMs = undefined;
    this._state.stats.prefillProfileSteps = [];
    const opts = resolvePrefillOptions(this._state, options);
    opts._selectedLogitTokenIds = selectedTokenIds;
    try {
      const { inputIds, logits, phase: prefillPhase } = await this._prefillPromptToLogits(prompt, opts, 'prefillWithTokenLogits');
      const logitsByTokenId = {};
      for (let index = 0; index < selectedTokenIds.length; index += 1) {
        logitsByTokenId[selectedTokenIds[index]] = logits[index];
      }
      return {
        seqLen: this._state.currentSeqLen,
        tokens: inputIds,
        tokenIds: selectedTokenIds,
        logits,
        logitsByTokenId,
        phase: {
          ...(prefillPhase ?? {}),
          totalMs: performance.now() - totalStart,
          selectedTokenCount: selectedTokenIds.length,
        },
      };
    } finally {
      this._closeFinitenessFallbackWindow(opts);
    }
  }

export async function prefillWithTokenLogitsFromKV(prefix, prompt, tokenIds, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    assertIncrementalDecodeSupport(this._state, 'prefillWithTokenLogitsFromKV');
    const totalStart = performance.now();
    if (!prefix || typeof prefix !== 'object' || !prefix.cache) {
      throw new Error('prefillWithTokenLogitsFromKV.prefix must be a KV cache snapshot.');
    }
    if (!Number.isInteger(prefix.seqLen) || prefix.seqLen < 0) {
      throw new Error('prefillWithTokenLogitsFromKV.prefix.seqLen must be a non-negative integer.');
    }
    if (!Array.isArray(prefix.tokens)) {
      throw new Error('prefillWithTokenLogitsFromKV.prefix.tokens must be an array.');
    }
    const selectedTokenIds = normalizeSelectedLogitTokenIds(
      tokenIds,
      this._state.modelConfig.vocabSize,
      'prefillWithTokenLogitsFromKV.tokenIds'
    );

    this._state.kvCache = prefix.cache.clone();
    if (this._state.useGPU && this._state.kvCache) {
      const device = getDevice();
      if (device) {
        this._state.kvCache.setGPUContext({ device });
      }
    }
    if (
      hasLinearAttentionLayers(this._state.modelConfig.layerTypes)
      && prefix.linearAttention == null
    ) {
      throw new Error(
        'Prefix snapshot is missing linear_attention recurrent state. ' +
        'Regenerate the prefix snapshot using the current runtime.'
      );
    }
    this._state.linearAttentionRuntime = restoreLinearAttentionRuntime(
      this._state.linearAttentionRuntime,
      prefix.linearAttention ?? null
    );
    this._state.currentSeqLen = prefix.seqLen;
    this._state.stats.gpuTimePrefillMs = undefined;
    this._state.stats.prefillProfileSteps = [];
    this._state.decodeRing?.reset();

    const opts = resolvePrefillOptions(this._state, options);
    opts._selectedLogitTokenIds = selectedTokenIds;
    try {
      const { inputIds, logits, phase: prefillPhase } = await this._prefillPromptToLogits(prompt, opts, 'prefillWithTokenLogitsFromKV');
      const logitsByTokenId = {};
      for (let index = 0; index < selectedTokenIds.length; index += 1) {
        logitsByTokenId[selectedTokenIds[index]] = logits[index];
      }
      return {
        seqLen: this._state.currentSeqLen,
        prefixTokens: prefix.tokens,
        tokens: inputIds,
        tokenIds: selectedTokenIds,
        logits,
        logitsByTokenId,
        phase: {
          ...(prefillPhase ?? {}),
          totalMs: performance.now() - totalStart,
          selectedTokenCount: selectedTokenIds.length,
          prefixTokens: prefix.tokens.length,
        },
      };
    } finally {
      this._closeFinitenessFallbackWindow(opts);
    }
  }

export async function* generateWithPrefixKV(prefix, prompt, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');
    assertIncrementalDecodeSupport(this._state, 'generateWithPrefixKV');

    validateCallTimeOptions(options);

    // Apply snapshot
    this._state.kvCache = prefix.cache.clone();
    if (this._state.useGPU && this._state.kvCache) {
      const device = getDevice();
      if (device) {
        this._state.kvCache.setGPUContext({ device });
      }
    }
    if (
      hasLinearAttentionLayers(this._state.modelConfig.layerTypes)
      && prefix.linearAttention == null
    ) {
      throw new Error(
        'Prefix snapshot is missing linear_attention recurrent state. ' +
        'Regenerate the prefix snapshot using the current runtime.'
      );
    }
    this._state.linearAttentionRuntime = restoreLinearAttentionRuntime(
      this._state.linearAttentionRuntime,
      prefix.linearAttention ?? null
    );
    this._state.currentSeqLen = prefix.seqLen;

    this._state.isGenerating = true;
    this._state.decodeStepCount = 0;
    resetActiveExecutionPlan(this._state);
    this._state.stats.gpuTimePrefillMs = undefined;
    this._state.stats.gpuTimeDecodeMs = undefined;
    this._state.stats.prefillProfileSteps = [];
    this._state.decodeRing?.reset();
    this._state.stats.decodeRecordMs = 0;
    this._state.stats.decodeRecordOps = 0;
    this._state.stats.decodeRecordPasses = 0;
    this._state.stats.decodeRecordOpLabels = {};
    this._state.stats.decodeSubmitWaitMs = 0;
    this._state.stats.decodeReadbackWaitMs = 0;
    this._state.stats.decodeReadbackMapWaitMs = 0;
    this._state.stats.decodeReadbackCleanupMs = 0;
    this._state.stats.decodeReadbackCopyMs = 0;
    this._state.stats.prefillRecordMs = 0;
    this._state.stats.prefillRecordOps = 0;
    this._state.stats.prefillRecordPasses = 0;
    this._state.stats.prefillRecordOpLabels = {};
    this._state.stats.prefillSubmitWaitMs = 0;
    this._state.stats.ttftMs = 0;
    const startTime = performance.now();

    const opts = resolveGenerateOptions(this._state, options);

    try {
      const processedPrompt = resolvePromptInput(this._state, prompt, opts.useChatTemplate, 'generateWithPrefixKV');

      const inputIds = this._state.tokenizer.encode(processedPrompt);
      this._assertTokenIdsInRange(inputIds, 'generateWithPrefixKV.encode');
      const generatedIds = [...prefix.tokens, ...inputIds];
      const promptTokenCount = generatedIds.length;
      this._state.stats.prefillTokens = inputIds.length;

      const prefillStart = performance.now();
      const prefillLogits = await this._prefill(inputIds, opts);
      this._state.stats.prefillTimeMs = performance.now() - prefillStart;

      applyRepetitionPenalty(prefillLogits, generatedIds, opts.repetitionPenalty);
      const padTokenId = this._state.tokenizer?.getSpecialTokens?.()?.pad;
      const firstToken = sample(prefillLogits, {
        temperature: opts.temperature,
        topP: opts.topP,
        topK: opts.topK,
        padTokenId,
        seed: opts.seed,
        suppressTokenIds: opts.suppressTokenIds,
      });

      const decodeRuntime = {
        stopTokenIds: this._state.modelConfig.stopTokenIds,
        eosToken: this._state.tokenizer.getSpecialTokens?.()?.eos,
        stopSequenceStart: promptTokenCount,
        decodeToken: (tokenId) => this._state.tokenizer.decode([tokenId], true, false),
        logBatchPath: false,
      };
      generatedIds.push(firstToken);
      this._state.stats.ttftMs = performance.now() - startTime;
      const firstText = resolveTokenText(
        this._state.tokenizer,
        [firstToken],
        `[${firstToken}]`,
        (tokens) => this._state.tokenizer?.decode?.(tokens, true, false),
        (tokens) => this._state.tokenizer?.decode?.(tokens, false, false)
      );
      yield firstText;
      if (options.onToken) options.onToken(firstToken, firstText);

      if (this._shouldStopAfterAppendedToken(generatedIds, firstToken, opts, decodeRuntime)) {
        this._recordStopReason('stop-token-or-sequence', firstToken);
        this._state.stats.decodeTimeMs = 0;
        this._state.stats.tokensGenerated = 1;
        this._state.stats.decodeTokens = 1;
      } else {
        yield* this._runDecodeLoop(generatedIds, opts, options, decodeRuntime);
      }
      this._state.stats.totalTimeMs = performance.now() - startTime;
    } finally {
      this._closeFinitenessFallbackWindow(opts);
      resetActiveExecutionPlan(this._state);
      this._state.isGenerating = false;
    }
  }
