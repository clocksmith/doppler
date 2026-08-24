

import { getDevice, setTrackSubmits } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer, readBufferSlice, uploadData } from '../../../memory/buffer-pool.js';
import { isGPUSamplingAvailable } from '../../../gpu/kernels/sample.js';
import { markWarmed as markKernelCacheWarmed } from '../../../gpu/kernel-selection-cache.js';
import { resetSubmitStats, logSubmitStats } from '../../../gpu/submit-tracker.js';
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from '../../../gpu/command-recorder.js';
import { allowReadback } from '../../../gpu/perf-guards.js';
import { log, trace, isTraceEnabled } from '../../../debug/index.js';
import {
  runMatmul,
  runRMSNorm,
  runGeLU,
  runResidualAdd,
  runScale,
  runSoftmax,
  runSoftEmbeddingSplitF16,
  runSoftEmbeddingLogitsF16,
} from '../../../gpu/kernel-selector.js';
import { runDiffusionGemmaCanvasStats } from '../../../gpu/kernels/diffusion-gemma-sampling.js';
import {
  CAPTURE_LEVELS,
  createDefaultCaptureConfig,
  validateCaptureConfig,
} from '../../../debug/index.js';
import { validateCallTimeOptions } from '../../../config/param-validator.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

// Pipeline sub-modules
import { sample, applyRepetitionPenalty, logitsSanity, getTopK } from './sampling.js';
import { createKVCache, isStopToken } from './init.js';
import { embed } from './embed.js';
import { processLayer } from './layer.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU, extractLastPositionLogits, applySoftcapping } from './logits/index.js';
import { OperatorEventEmitter } from './operator-events.js';
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer, getWeightDtype, getWeightMetadata, getLayout } from '../../../gpu/weight-buffer.js';
import {
  decodeStep,
  decodeStepLogits,
  advanceWithToken,
  generateNTokensGPU,
  shouldUseBatchDecode,
  sumProfileTimings,
  FinitenessError,
  advanceWithTokenAndEmbedding as runAdvanceWithTokenAndEmbedding,
} from './generator-steps.js';
import {
  buildLayerContext,
  debugCheckBuffer as debugCheckBufferHelper,
  resolvePerLayerInputsSession,
} from './generator/session-context.js';
import {
  getLogitsConfig,
  getLogitsWeights,
} from './generator/logits-config.js';
import { releaseSharedAttentionState } from './generator/attention-lifecycle.js';
import {
  assertTokenIdsInRange,
  assertTokenIdInRange,
  resolveStepOptions,
  resolveGenerateOptions,
  resolvePrefillOptions,
  resolvePrefillEmbeddingOptions,
  resolveAdvanceEmbeddingMode,
  getFinalNormWeights,
  extractEmbeddingFromHidden,
  extractTokenEmbeddingsFromHidden,
} from './generator-runtime.js';

import { resolveSamplingConfig } from './sampling-config.js';
import { decodeReadback, getLogitsHealth } from './debug-utils/index.js';
import { parseFinitenessStatusWords } from './finiteness-guard-status.js';
import { resolveDeferredRoundingWindowTokens } from './finiteness-policy.js';
import {
  advanceDecodeStepCount,
  createTsirFixtureState,
  drainPendingTsirReads,
} from './tsir-fixture-writer.js';
import {
  activateFallbackExecutionPlan,
  hasFallbackExecutionPlan,
  rebaseExecutionSessionPlan,
  resetActiveExecutionPlan,
  resolveMaxBatchDecodeTokens,
  resolvePrefillRecorderChunkLayers,
  resolveActiveExecutionPlan,
  setActiveExecutionPlan,
} from './execution-plan.js';
import {
  cloneLinearAttentionRuntime,
  hasLinearAttentionLayers,
  resetLinearAttentionRuntime,
  restoreLinearAttentionRuntime,
} from './linear-attention.js';
import {
  preparePerLayerInputs,
  createPleBufferCache,
  prefetchPerLayerRow,
  getPleHotVocabularyRuntime,
  hasGpuSplitPerLayerInputEmbeddings,
  hasRangeBackedPerLayerInputEmbeddings,
} from './per-layer-inputs.js';
import { createTensor } from '../../../gpu/tensor.js';
import { getQKNormOnesBuffer } from './attention/types.js';
import {
  getWeightBuffer as getPipelineWeightBuffer,
  getNormWeightBuffer as getPipelineNormWeightBuffer,
} from './weights.js';

// Extracted standalone helpers
import {
  resolvePromptInput,
} from './generator/prompt-input.js';
import {
  releasePerLayerInputBuffer,
  shouldDisablePrefillCommandBatching,
  resolveEffectivePrefillTokenChunkSize,
} from './generator/prefill-policy.js';
import {
  normalizePrefixEmbeddingOverride,
  resolvePrefillEmbeddingInputIds,
  resolvePrefillMultimodalBidirectionalSpan,
  applyPrefixEmbeddingOverride,
  resolvePrefixEmbeddingOverrideTransitionDeclaredBy,
} from './generator/prefix-embedding.js';
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
} from './generator-decode-policy.js';
import { FINITENESS_RESET_WORDS, borrowLinearWeight, borrowNormWeight, canUseChunkedSoftEmbeddingLogits, getExperimentalIntentBundleModule, normalizeCanvasTokenIds, normalizeSelectedLogitTokenIds, normalizeSelfConditioningLogits, normalizeSelfConditioningLogitsState, normalizeSelfConditioningSoftEmbeddingState, recordPrefillRecorderStats, releaseBorrowedWeight, resolveDiffusionGemmaSoftEmbeddingChunkRows, resolvePrefillChunkSubmitMode, resolveSuppressedSamplingTokenIds, traceActivationHealth } from './generator/text.js';
import {
  _createDiffusionGemmaSelfConditioningEmbeddings as _createDiffusionGemmaSelfConditioningEmbeddingsImpl,
  _createDiffusionGemmaSelfConditioningSoftEmbeddingState as _createDiffusionGemmaSelfConditioningSoftEmbeddingStateImpl,
  computeDiffusionGemmaCanvasLogits as computeDiffusionGemmaCanvasLogitsImpl,
  computeDiffusionGemmaCanvasStep as computeDiffusionGemmaCanvasStepImpl,
} from './generator/diffusion.js';
import {
  _replayPrefillDecodeLogits as _replayPrefillDecodeLogitsImpl,
  _shouldUseFinitenessFallback as _shouldUseFinitenessFallbackImpl,
  _recreateKVCacheForExecutionPlan as _recreateKVCacheForExecutionPlanImpl,
  _openFinitenessFallbackWindow as _openFinitenessFallbackWindowImpl,
  _closeFinitenessFallbackWindow as _closeFinitenessFallbackWindowImpl,
  _consumeFinitenessFallbackToken as _consumeFinitenessFallbackTokenImpl,
  _beginFinitenessFallback as _beginFinitenessFallbackImpl,
  _endFinitenessFallback as _endFinitenessFallbackImpl,
  _retryWithFinitenessFallback as _retryWithFinitenessFallbackImpl,
  _retryWithPersistentFinitenessFallback as _retryWithPersistentFinitenessFallbackImpl,
  _retryDecodeStepWithFinitenessWindow as _retryDecodeStepWithFinitenessWindowImpl,
} from './generator/recovery.js';
import {
  prefillKVOnly as prefillKVOnlyImpl,
  prefillForLoRATraining as prefillForLoRATrainingImpl,
  prefillWithEmbedding as prefillWithEmbeddingImpl,
  prefillWithLogits as prefillWithLogitsImpl,
  prefillWithTokenLogits as prefillWithTokenLogitsImpl,
  prefillWithTokenLogitsFromKV as prefillWithTokenLogitsFromKVImpl,
  generateWithPrefixKV as generateWithPrefixKVImpl,
} from './generator/training.js';
import {
  _generateTokensInternal as _generateTokensInternalImpl,
  generateTokenIds as generateTokenIdsImpl,
  _runDecodeLoop as _runDecodeLoopImpl,
  _decodeStep as _decodeStepImpl,
  decodeStepLogits as decodeStepLogitsImpl,
  advanceWithToken as advanceWithTokenImpl,
  advanceWithTokenAndEmbedding as advanceWithTokenAndEmbeddingImpl,
  _generateNTokensGPU as _generateNTokensGPUImpl,
} from './generator/decode-runtime.js';
import {
  _commitPrefillHiddenChunk as _commitPrefillHiddenChunkImpl,
  _prefillInputIdsToLogits as _prefillInputIdsToLogitsImpl,
  _prefillPromptToLogits as _prefillPromptToLogitsImpl,
  _prefillToHidden as _prefillToHiddenImpl,
  _prefill as _prefillImpl,
} from './generator/prefill-runtime.js';
export { resolvePrefillChunkSubmitMode } from './generator/text.js';

export class PipelineGenerator {

  _state;
  _finitenessFallbackWindow;

  _assertTokenIdsInRange(tokenIds, context = 'encode') {
    assertTokenIdsInRange(this._state, tokenIds, context);
  }

  _assertTokenIdInRange(tokenId, context = 'token') {
    assertTokenIdInRange(this._state, tokenId, context);
  }

  constructor(state) {
    this._state = state;
    this._finitenessFallbackWindow = null;
  }

  _resolveDeferredRoundingWindowTokens() {
    const activePlan = resolveActiveExecutionPlan(this._state);
    return activePlan?.deferredRoundingWindowTokens
      ?? resolveDeferredRoundingWindowTokens(this._state.runtimeConfig?.inference?.compute);
  }

  _getEffectiveActivationDtype() {
    return resolveActiveExecutionPlan(this._state).activationDtype;
  }

  _hasFinitenessFallbackWindow() {
    return this._finitenessFallbackWindow !== null;
  }

  _resetReplayPrefillRuntimeState() {
    this._state.kvCache?.clear?.();
    this._state.linearAttentionRuntime = resetLinearAttentionRuntime(this._state.linearAttentionRuntime);
    this._state.currentSeqLen = 0;
  }

  resetGenerationState() {
    if (this._state.isGenerating) {
      throw new Error('InferencePipeline.resetGenerationState: cannot reset while generation is in progress');
    }
    this._resetReplayPrefillRuntimeState();
    this._resetDecodeRuntimeState();
  }

  async _replayPrefillDecodeLogits(currentIds, opts) {
    return _replayPrefillDecodeLogitsImpl.apply(this, arguments);
  }

  _shouldUseFinitenessFallback(error, contextLabel) {
    return _shouldUseFinitenessFallbackImpl.apply(this, arguments);
  }

  _recreateKVCacheForExecutionPlan(plan, reasonLabel) {
    return _recreateKVCacheForExecutionPlanImpl.apply(this, arguments);
  }

  _openFinitenessFallbackWindow(opts, reasonLabel, tokenCount, rollbackSeqLen = undefined) {
    return _openFinitenessFallbackWindowImpl.apply(this, arguments);
  }

  _closeFinitenessFallbackWindow(opts) {
    return _closeFinitenessFallbackWindowImpl.apply(this, arguments);
  }

  _consumeFinitenessFallbackToken(opts) {
    return _consumeFinitenessFallbackTokenImpl.apply(this, arguments);
  }

  _resolveStepOptions(options = {}) {
    return resolveStepOptions(this._state, options);
  }

  _resetDecodeRuntimeState() {
    this._state.stats.prefillProfileSteps = [];
    this._state.stats.decodeMode = null;
    this._state.stats.batchGuardReason = null;
    this._state.stats.decodeProfileSteps = [];
    this._state.stats.ttftMs = 0;
    this._state.stats.decodeTimeMs = 0;
    this._state.stats.prefillRecordMs = 0;
    this._state.stats.prefillRecordOps = 0;
    this._state.stats.prefillRecordPasses = 0;
    this._state.stats.prefillRecordOpLabels = {};
    this._state.stats.prefillSubmitWaitMs = 0;
    this._state.stats.decodeRecordMs = 0;
    this._state.stats.decodeRecordOps = 0;
    this._state.stats.decodeRecordPasses = 0;
    this._state.stats.decodeRecordOpLabels = {};
    this._state.stats.decodeSubmitWaitMs = 0;
    this._state.stats.decodeReadbackWaitMs = 0;
    this._state.stats.decodeReadbackMapWaitMs = 0;
    this._state.stats.decodeReadbackCleanupMs = 0;
    this._state.stats.decodeReadbackCopyMs = 0;
    this._state.decodeStepCount = 0;
    this._state.disableRecordedLogits = false;
    this._state.disableFusedDecode = false;
    this._state.batchingStats = {
      batchedForwardCalls: 0,
      unbatchedForwardCalls: 0,
      totalBatchedTimeMs: 0,
      totalUnbatchedTimeMs: 0,
      gpuSubmissions: 0,
      requestedBatchTokens: 0,
      effectiveBatchTokens: 0,
      executedBatchTokens: 0,
      resolvedBatchTokens: 0,
      maxBatchTokenCap: null,
      batchClampCount: 0,
    };
    resetActiveExecutionPlan(this._state);
    this._state.decodeRing?.reset();
  }

  _getDecodeHelpers(debugCheckBuffer) {
    return {
      buildLayerContext: (recorder, isDecodeMode, debugLayers, executionPlan) =>
        buildLayerContext(this._state, recorder, isDecodeMode, debugLayers, debugCheckBuffer, executionPlan),
      getLogitsWeights: () => getLogitsWeights(this._state),
      getLogitsConfig: () => getLogitsConfig(this._state),
      releaseSharedAttentionState,
      debugCheckBuffer,
    };
  }

  async _getFinalNormWeights() {
    return getFinalNormWeights(this._state);
  }

  _extractEmbeddingFromHidden(hiddenStates, numTokens, hiddenSize, embeddingMode, finalNormWeights, config) {
    return extractEmbeddingFromHidden(
      hiddenStates,
      numTokens,
      hiddenSize,
      embeddingMode,
      finalNormWeights,
      config,
      this._state.embeddingPostprocessor,
      null,
      this._state.weights.get('final_norm_bias') ?? null
    );
  }

  _extractTokenEmbeddingsFromHidden(hiddenStates, numTokens, hiddenSize, finalNormWeights, config) {
    return extractTokenEmbeddingsFromHidden(
      hiddenStates,
      numTokens,
      hiddenSize,
      finalNormWeights,
      config,
      this._state.weights.get('final_norm_bias') ?? null
    );
  }

  _resolvePromptTokenIds(prompt, useChatTemplate, contextLabel) {
    const processedPrompt = resolvePromptInput(this._state, prompt, useChatTemplate, contextLabel);
    const inputIds = this._state.tokenizer.encode(processedPrompt);
    this._assertTokenIdsInRange(inputIds, `${contextLabel}.encode`);
    return inputIds;
  }

  _resolvePromptOrInputIds(prompt, useChatTemplate, contextLabel, explicitInputIds = null) {
    if (Array.isArray(explicitInputIds)) {
      this._assertTokenIdsInRange(explicitInputIds, `${contextLabel}.inputIds`);
      return explicitInputIds;
    }
    return this._resolvePromptTokenIds(prompt, useChatTemplate, contextLabel);
  }

  _sampleNextTokenFromLogits(logits, generatedIds, opts) {
    const sampledLogits = Float32Array.from(logits);
    applyRepetitionPenalty(sampledLogits, generatedIds, opts.repetitionPenalty);
    // Optional pre-sample logit mask. Callers pass `opts.logitMaskFn` to
    // implement grammar/schema-constrained decoding. The hook receives the
    // mutable logit buffer (after repetition penalty) plus the running token
    // sequence so it can track parse state across decode steps.
    if (typeof opts?.logitMaskFn === "function") {
      try {
        opts.logitMaskFn(sampledLogits, {
          generatedIds,
          tokenizer: this._state.tokenizer ?? null,
          vocabSize: this._state.modelConfig?.vocabSize ?? sampledLogits.length,
        });
      } catch (maskError) {
        log.warn("Pipeline", `logitMaskFn threw; continuing without mask: ${maskError}`);
      }
    }
    const padTokenId = this._state.tokenizer?.getSpecialTokens?.()?.pad;
    const tokenId = sample(sampledLogits, {
      temperature: opts.temperature,
      topP: opts.topP,
      topK: opts.topK,
      padTokenId,
      seed: opts.seed,
      suppressTokenIds: opts.suppressTokenIds,
    });
    if (typeof opts.onLogits === 'function') {
      opts.onLogits(sampledLogits, {
        tokenId,
        inputTokenCount: Array.isArray(generatedIds) ? generatedIds.length : null,
      });
    }
    return tokenId;
  }

  _resolvePrefillTokenChunkSize(inputIds) {
    const chunkSize = resolveEffectivePrefillTokenChunkSize(this._state);
    if (chunkSize === undefined) {
      throw new Error('inference.session.prefillTokenChunkSize is required; use null to disable token-chunked prefill.');
    }
    if (chunkSize === null) {
      return null;
    }
    if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
      throw new Error('inference.session.prefillTokenChunkSize must be null or a positive integer.');
    }
    return chunkSize < inputIds.length ? chunkSize : null;
  }

  async _commitPrefillHiddenChunk(prefillResult) {
    return _commitPrefillHiddenChunkImpl.apply(this, arguments);
  }

  async _prefillInputIdsToLogits(inputIds, opts) {
    return _prefillInputIdsToLogitsImpl.apply(this, arguments);
  }

  async _prefillPromptToLogits(prompt, opts, contextLabel) {
    return _prefillPromptToLogitsImpl.apply(this, arguments);
  }

  async _decodeStepToLogits(currentIds, opts) {
    if (usesReplayPrefillDecode(this._state)) {
      return this._replayPrefillDecodeLogits(currentIds, opts);
    }
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim)
      : undefined;
    return decodeStepLogits(this._state, currentIds, opts, this._getDecodeHelpers(debugCheckBuffer));
  }

  async _decodeNextTokenViaLogits(currentIds, opts) {
    const stepResult = await this._decodeStepToLogits(currentIds, opts);
    return this._sampleNextTokenFromLogits(stepResult.logits, currentIds, opts);
  }

  _matchesStopSequence(generatedIds, stopSequenceStart, stopSequences) {
    if (!Array.isArray(stopSequences) || stopSequences.length === 0) {
      return false;
    }
    const fullText = this._state.tokenizer.decode(generatedIds.slice(stopSequenceStart), false);
    return stopSequences.some((sequence) => fullText.endsWith(sequence));
  }

  _shouldStopAfterAppendedToken(generatedIds, tokenId, opts, runtime) {
    if (isStopToken(tokenId, runtime.stopTokenIds, runtime.eosToken)) {
      return true;
    }
    return this._matchesStopSequence(generatedIds, runtime.stopSequenceStart, opts.stopSequences);
  }

  _recordStopReason(reason, tokenId = null) {
    this._state.stats.stopReason = reason;
    this._state.stats.stopTokenId = Number.isInteger(tokenId) ? tokenId : null;
  }

  async *_generateTokensInternal(prompt, options = {}, mode = 'text') {
    return yield* _generateTokensInternalImpl.apply(this, arguments);
  }

  _beginFinitenessFallback(opts, reasonLabel, rollbackSeqLen = undefined) {
    return _beginFinitenessFallbackImpl.apply(this, arguments);
  }

  _endFinitenessFallback(opts, original) {
    return _endFinitenessFallbackImpl.apply(this, arguments);
  }

  async _retryWithFinitenessFallback(opts, reasonLabel, retryFn, rollbackSeqLen = undefined) {
    return _retryWithFinitenessFallbackImpl.apply(this, arguments);
  }

  async _retryWithPersistentFinitenessFallback(opts, reasonLabel, tokenBudget, retryFn, rollbackSeqLen = undefined) {
    return _retryWithPersistentFinitenessFallbackImpl.apply(this, arguments);
  }

  async _retryDecodeStepWithFinitenessWindow(generatedIds, opts, reasonLabel) {
    return _retryDecodeStepWithFinitenessWindowImpl.apply(this, arguments);
  }

  // ==========================================================================
  // Generation Public API
  // ==========================================================================

  /*
   * Truncate the KV cache back to `seqLen` tokens and set `currentSeqLen` to
   * match. Intended for "prefix-reuse" workflows where a caller wants to run
   * several decodes that share a common prompt prefix: prefill once with the
   * shared prefix, decode the first tail, then `resetToSeqLen(prefixLen)` to
   * drop the tail's KV entries and reuse the prefix KV for the next tail.
   *
   * Only valid when no decode is in progress.
   */
  resetToSeqLen(seqLen) {
    if (this._state.isGenerating) {
      throw new Error('InferencePipeline.resetToSeqLen: cannot reset while generation is in progress');
    }
    const target = Math.max(0, Math.floor(Number(seqLen) || 0));
    if (!Number.isFinite(target)) {
      throw new Error('InferencePipeline.resetToSeqLen: seqLen must be a finite non-negative integer');
    }
    if (target > this._state.currentSeqLen) {
      throw new Error(
        `InferencePipeline.resetToSeqLen: target ${target} exceeds currentSeqLen ${this._state.currentSeqLen}`
      );
    }
    this._state.kvCache?.truncate?.(target);
    this._state.currentSeqLen = target;
  }

  async _createDiffusionGemmaSelfConditioningEmbeddings(canvasIds, selfConditioningLogits, opts) {
    return _createDiffusionGemmaSelfConditioningEmbeddingsImpl.apply(this, arguments);
  }

  async _createDiffusionGemmaSelfConditioningSoftEmbeddingState(logitsState, canvasLength, hiddenSize, vocabSize) {
    return _createDiffusionGemmaSelfConditioningSoftEmbeddingStateImpl.apply(this, arguments);
  }

  async computeDiffusionGemmaCanvasLogits(args, options = {}) {
    return computeDiffusionGemmaCanvasLogitsImpl.apply(this, arguments);
  }

  async computeDiffusionGemmaCanvasStep(args, options = {}) {
    return computeDiffusionGemmaCanvasStepImpl.apply(this, arguments);
  }

  async *generate(prompt, options = {}) {
    yield* this._generateTokensInternal(prompt, options, 'text');
  }

  async *generateTokens(prompt, options = {}) {
    yield* this._generateTokensInternal(prompt, options, 'token');
  }

  async generateTokenIds(prompt, options = {}) {
    return generateTokenIdsImpl.apply(this, arguments);
  }

  async prefillKVOnly(prompt, options = {}) {
    return prefillKVOnlyImpl.apply(this, arguments);
  }

  async prefillForLoRATraining(inputIds, options = {}) {
    return prefillForLoRATrainingImpl.apply(this, arguments);
  }

  async prefillWithEmbedding(prompt, options = {}) {
    return prefillWithEmbeddingImpl.apply(this, arguments);
  }

  async prefillWithLogits(prompt, options = {}) {
    return prefillWithLogitsImpl.apply(this, arguments);
  }

  async prefillWithTokenLogits(prompt, tokenIds, options = {}) {
    return prefillWithTokenLogitsImpl.apply(this, arguments);
  }

  async prefillWithTokenLogitsFromKV(prefix, prompt, tokenIds, options = {}) {
    return prefillWithTokenLogitsFromKVImpl.apply(this, arguments);
  }

  async *generateWithPrefixKV(prefix, prompt, options = {}) {
    return yield* generateWithPrefixKVImpl.apply(this, arguments);
  }

  // ==========================================================================
  // Internal Methods (Prefill, Decode, Helpers)
  // ==========================================================================

  async *_runDecodeLoop(generatedIds, opts, options, runtime) {
    return yield* _runDecodeLoopImpl.apply(this, arguments);
  }

  async _prefillToHidden(inputIds, opts) {
    return _prefillToHiddenImpl.apply(this, arguments);
  }

  async _prefill(inputIds, opts) {
    return _prefillImpl.apply(this, arguments);
  }

  async _decodeStep(currentIds, opts) {
    return _decodeStepImpl.apply(this, arguments);
  }

  async decodeStepLogits(currentIds, options = {}) {
    return decodeStepLogitsImpl.apply(this, arguments);
  }

  async advanceWithToken(tokenId, options = {}) {
    return advanceWithTokenImpl.apply(this, arguments);
  }

  async advanceWithTokenAndEmbedding(tokenId, options = {}) {
    return advanceWithTokenAndEmbeddingImpl.apply(this, arguments);
  }

  async _generateNTokensGPU(startToken, N, currentIds, opts) {
    return _generateNTokensGPUImpl.apply(this, arguments);
  }
}
