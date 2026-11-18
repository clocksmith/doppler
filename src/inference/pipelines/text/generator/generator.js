import { CAPTURE_LEVELS, createDefaultCaptureConfig, validateCaptureConfig } from "../../../../debug/index.js";
import { OperatorEventEmitter } from "../observation/operator-events.js";
import { createTsirFixtureState } from "../observation/tsir-fixture-writer.js";
import { resolveSuppressedSamplingTokenIds } from "./session/text.js";
import { withHyperConnectionTransaction, assertClassicSequenceState, assertHyperConnectionOperation } from "../hyper-connection/sequence-transaction.js";
import "../../../../gpu/device.js"; // Preserve declared device initialization side effects.
import { beginDiagnosticSequence, endDiagnosticSequence } from "./session/diagnostic-sequence.js";
import { prefillWithToken, decodeStepWithToken } from "./adapters/capsule-generation.js";
import { releaseBuffer } from "../../../../memory/buffer-pool.js";

// Pipeline sub-modules
import { sample, applyRepetitionPenalty, applyPresencePenalty, applyLogitMask } from "./sampling/sampling.js";
import { isStopToken } from "../init.js";
import { decodeStepLogits } from './generator-steps.js';
import { projectHiddenStates as projectHiddenStatesImpl } from "../logits/hidden-projection.js";
import { buildLayerContext, debugCheckBuffer as debugCheckBufferHelper } from "./session/session-context.js";
import { getLogitsConfig, getLogitsWeights } from "../logits/logits-config.js";
import { releaseSharedAttentionState } from "./session/attention-lifecycle.js";
import { resetSequenceState } from "./session/sequence-state.js";
import {
  settleGenerationStats,
  assertTokenIdsInRange,
  assertTokenIdInRange,
  resolveStepOptions,
  getFinalNormWeights,
  extractEmbeddingFromHidden,
  extractTokenEmbeddingsFromHidden,
} from "./session/generator-runtime.js";

import { samplingSeedForToken } from "../../../token-sampling.js";
import { createGenerationOutputDecoder, decodeGenerationStream } from "../../../generation-output.js";
import { matchesStopSequence } from "./sampling/stopping.js";
import { resolveDeferredRoundingWindowTokens } from "../policy/finiteness-policy.js";
import { resetActiveExecutionPlan, resolveActiveExecutionPlan } from "../execution/execution-plan.js";
import { resetLinearAttentionSequence } from "../linear-attention/linear-attention.js";

// Extracted standalone helpers
import { resolvePromptInput } from "./request/prompt-input.js";
import { resolvePrefillTokenChunkSize } from "./prefill/prefill-policy.js";
import { usesReplayPrefillDecode, summarizeExecutionPlan } from "../policy/generator-decode-policy.js";
import {
  _createDiffusionGemmaSelfConditioningEmbeddings as _createDiffusionGemmaSelfConditioningEmbeddingsImpl,
  _createDiffusionGemmaSelfConditioningSoftEmbeddingState as _createDiffusionGemmaSelfConditioningSoftEmbeddingStateImpl,
  computeDiffusionGemmaCanvasLogits as computeDiffusionGemmaCanvasLogitsImpl,
  computeDiffusionGemmaCanvasStep as computeDiffusionGemmaCanvasStepImpl,
} from "./adapters/diffusion.js";
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
} from "./session/recovery.js";
import {
  prefillKVOnly as prefillKVOnlyImpl,
  prefillForLoRATraining as prefillForLoRATrainingImpl,
  prefillWithEmbedding as prefillWithEmbeddingImpl,
  prefillWithLogits as prefillWithLogitsImpl,
  prefillWithTokenLogits as prefillWithTokenLogitsImpl,
  prefillWithTokenLogitsFromKV as prefillWithTokenLogitsFromKVImpl,
  generateWithPrefixKV as generateWithPrefixKVImpl,
} from "./adapters/training.js";
import {
  _generateTokensInternal as _generateTokensInternalImpl,
  generateTokenIds as generateTokenIdsImpl,
  _runDecodeLoop as _runDecodeLoopImpl,
  _decodeStep as _decodeStepImpl,
  decodeStepLogits as decodeStepLogitsImpl,
  advanceWithToken as advanceWithTokenImpl,
  advanceWithTokenAndEmbedding as advanceWithTokenAndEmbeddingImpl,
  _generateNTokensGPU as _generateNTokensGPUImpl,
} from "./decode/decode-runtime.js";
import {
  _commitPrefillHiddenChunk as _commitPrefillHiddenChunkImpl,
  _prefillInputIdsToLogits as _prefillInputIdsToLogitsImpl,
  _prefillPromptToLogits as _prefillPromptToLogitsImpl,
  _prefillToHidden as _prefillToHiddenImpl,
  _prefill as _prefillImpl,
} from "./prefill/prefill-runtime.js";
export { resolvePrefillChunkSubmitMode } from "./session/text.js";

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

  beginGeneration(opts, options) {
    assertHyperConnectionOperation(this._state, opts);
    this._resetReplayPrefillRuntimeState();
    this._state.isGenerating = true;
    this._resetDecodeRuntimeState();
    delete this._state.stats.outputUsage;
    this._state.stats.speculationAttempts = 0;
    this._state.stats.speculationAccepted = 0;
    this._state.stats.gpuTimePrefillMs = undefined;
    this._state.stats.gpuTimeDecodeMs = undefined;
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
    this._state.stats.singleTokenSubmitWaitMs = 0;
    this._state.stats.singleTokenReadbackWaitMs = 0;
    this._state.stats.singleTokenReadbackMapWaitMs = 0;
    this._state.stats.singleTokenReadbackCleanupMs = 0;
    this._state.stats.singleTokenReadbackCopyMs = 0;
    this._state.stats.singleTokenOrchestrationMs = 0;
    this._state.stats.pleHotVocabularyHits = 0;
    this._state.stats.pleHotVocabularyMisses = 0;
    this._state.stats.plePreparedTokenCacheHits = 0;
    this._state.stats.plePreparedTokenCacheMisses = 0;
    this._state.stats.plePreparedTokenCacheEntries = 0;
    this._state.stats.plePreparedTokenCacheBytes = 0;
    this._state.stats.decodeMode = null;
    this._state.stats.batchGuardReason = null;
    this._state.stats.stopReason = null;
    this._state.stats.stopTokenId = null;
    this._state.stats.ttftMs = 0;
    this._state.stats.prefillTimeMs = 0;
    this._state.stats.decodeTimeMs = 0;
    this._state.stats.prefillTokens = 0;
    this._state.stats.reusedPrefixTokens = 0;
    this._state.stats.prefillComputedTokens = 0;
    this._state.stats.tokensGenerated = 0;
    this._state.stats.decodeTokens = 0;
    this._state.stats.totalTimeMs = 0;
    const startTime = performance.now();
  
    opts.onLogits = typeof options.onLogits === 'function' ? options.onLogits : null;
    // Validate and normalize sampling parameters through single source of truth
    const samplingConfig = opts;
    opts.suppressTokenIds = resolveSuppressedSamplingTokenIds(this._state, samplingConfig);
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
    if (opts.__activationTrace?.activity) {
      this._state.operatorDiagnostics ??= { enabled: false };
      this._state.operatorDiagnostics.activity = opts.__activationTrace.activity;
    }
    const activePlan = resolveActiveExecutionPlan(this._state);
    this._state.stats.executionPlan = {
      primary: summarizeExecutionPlan(this._state.executionPlanState?.primaryPlan ?? null),
      fallback: summarizeExecutionPlan(this._state.executionPlanState?.fallbackPlan ?? null),
      activePlanIdAtStart: activePlan?.id ?? null,
      finalActivePlanId: activePlan?.id ?? null,
      transitions: [],
    };
    this._state.stats.kernelPathId = activePlan?.kernelPathId ?? this._state.resolvedKernelPath?.id ?? null;
    this._state.stats.kernelPathSource = activePlan?.kernelPathSource ?? this._state.kernelPathSource ?? 'none';
    return startTime;
  }

  finishGeneration(options, startTime, generatedIds, promptTokenCount) {
    settleGenerationStats(this._state.stats, startTime,
      Math.max(0, generatedIds.length - promptTokenCount), options.signal?.aborted === true);
    this._state.stats.batching = { ...this._state.batchingStats };
    if (this._state.stats.executionPlan) {
      this._state.stats.executionPlan.finalActivePlanId = this._state.executionPlanState?.activePlanId ?? null;
    }
    resetActiveExecutionPlan(this._state);
    this._state.stats.operatorDiagnostics = this._state.operatorDiagnostics?.emitter
      ? {
        enabled: true,
        timeline: this._state.operatorDiagnostics.emitter.getTimeline(),
        recordCount: this._state.operatorDiagnostics.emitter.length,
      }
      : null;
    this._state.operatorDiagnostics = null;
    this._state.isGenerating = false;
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
    this._state.hyperConnectionSession?.reset();
    this._state.kvCache?.clear?.();
    this._state.linearAttentionRuntime = resetLinearAttentionSequence(this._state.linearAttentionRuntime);
    this._state.currentSeqLen = 0;
  }

  resetGenerationState() {
    if (this._state.isGenerating) {
      throw new Error('InferencePipeline.resetGenerationState: cannot reset while generation is in progress');
    }
    this._state.channelGainRequest?.close();
    this._state.channelGainRequest = null;
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
    delete this._state.stats.wholeTokenProgram;
    this._state.stats.wholeTokenProgramWallMs = 0;
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
      getLogitsConfig: () => getLogitsConfig(this._state, 'decode'),
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
    applyRepetitionPenalty(sampledLogits, generatedIds, opts.repetitionPenalty, opts.repetitionPenaltyWindow);
    if (opts.presencePenalty) {
      applyPresencePenalty(sampledLogits, generatedIds, opts.presencePenalty, opts.repetitionPenaltyWindow);
    }
    // Optional pre-sample logit mask for grammar/schema-constrained decoding; shared with the decode step.
    applyLogitMask(sampledLogits, generatedIds, opts, this._state);
    const padTokenId = this._state.tokenizer?.getSpecialTokens?.()?.pad;
    // Sampling masks pad/suppressed entries in place. Observers retain the
    // post-penalty/user-mask vector, matching the decode observation boundary.
    const observedLogits = typeof opts.onLogits === 'function' ? sampledLogits.slice() : null;
    const tokenId = sample(sampledLogits, {
      temperature: opts.temperature,
      topP: opts.topP,
      topK: opts.topK,
      padTokenId,
      seed: opts.temperature === 0 ? opts.seed
        : samplingSeedForToken(opts.seed, generatedIds.length - opts.promptTokenCount),
      suppressTokenIds: opts.suppressTokenIds,
    });
    if (typeof opts.onLogits === 'function') {
      opts.onLogits(observedLogits, {
        tokenId,
        inputTokenCount: Array.isArray(generatedIds) ? generatedIds.length : null,
      });
    }
    return tokenId;
  }

  _resolvePrefillTokenChunkSize(inputIds) {
    return resolvePrefillTokenChunkSize(this._state, inputIds.length);
  }

  async _commitPrefillHiddenChunk(prefillResult) {
    return _commitPrefillHiddenChunkImpl.apply(this, arguments);
  }

  async _prefillInputIdsToLogits(inputIds, opts) {
    return withHyperConnectionTransaction(this._state, opts, scoped => _prefillInputIdsToLogitsImpl.call(this, inputIds, scoped));
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
    return withHyperConnectionTransaction(this._state, opts, scoped => decodeStepLogits(this._state, currentIds, scoped, this._getDecodeHelpers(debugCheckBuffer)));
  }

  async _decodeNextTokenViaLogits(currentIds, opts) {
    const stepResult = await this._decodeStepToLogits(currentIds, opts);
    try {
      return this._sampleNextTokenFromLogits(stepResult.logits, currentIds, opts);
    } finally {
      if (stepResult.logitsBuffer) releaseBuffer(stepResult.logitsBuffer);
    }
  }

  _matchesStopSequence(generatedIds, stopSequenceStart, stopSequences) {
    if (!Array.isArray(stopSequences) || stopSequences.length === 0) {
      return false;
    }
    return matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, stopSequences,
      this._state.modelConfig?.reasoningOutput);
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
    assertClassicSequenceState(this._state, 'resetToSeqLen; use resetGenerationState for a full reset');
    resetSequenceState(this._state, seqLen);
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

  async projectHiddenStates(hiddenStates, rowCount) {
    return projectHiddenStatesImpl(this._state, hiddenStates, rowCount);
  }

  async computeDiffusionGemmaCanvasStep(args, options = {}) {
    return computeDiffusionGemmaCanvasStepImpl.apply(this, arguments);
  }

  async *generate(prompt, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');
    const decoder = createGenerationOutputDecoder(this._state.tokenizer,
      this._state.modelConfig?.reasoningOutput,
      [...this._state.modelConfig.stopTokenIds, this._state.tokenizer.getSpecialTokens?.()?.eos]
        .filter(Number.isSafeInteger));
    try {
      yield* decodeGenerationStream(this._generateTokensInternal(prompt, { ...options, onToken: undefined }, 'token'),
        decoder, { ...options, getStopReason: () => this._state.stats.stopReason });
    } finally {
      const usage = decoder.getUsage?.();
      if (usage?.rawTokens > 0) this._state.stats.outputUsage = usage;
    }
  }

  async *generateTokens(prompt, options = {}) {
    yield* this._generateTokensInternal(prompt, options, 'token');
  }

  async generateTokenIds(prompt, options = {}) {
    return generateTokenIdsImpl.apply(this, arguments);
  }

  async prefillKVOnly(prompt, options = {}) {
    assertClassicSequenceState(this._state, 'prefillKVOnly');
    return prefillKVOnlyImpl.apply(this, arguments);
  }

  async prefillForLoRATraining(inputIds, options = {}) {
    assertClassicSequenceState(this._state, 'prefillForLoRATraining');
    return prefillForLoRATrainingImpl.apply(this, arguments);
  }

  async prefillWithEmbedding(prompt, options = {}) {
    assertClassicSequenceState(this._state, 'prefillWithEmbedding');
    return prefillWithEmbeddingImpl.apply(this, arguments);
  }

  async prefillWithLogits(prompt, options = {}) {
    assertClassicSequenceState(this._state, 'prefillWithLogits');
    return prefillWithLogitsImpl.apply(this, arguments);
  }

  prefillWithToken(prompt, options, tokenContract) {
    return prefillWithToken.call(this, prompt, options, tokenContract);
  }

  decodeStepWithToken(currentIds, options, tokenContract) {
    return decodeStepWithToken.call(this, currentIds, options, tokenContract);
  }

  async prefillWithTokenLogits(prompt, tokenIds, options = {}) {
    assertClassicSequenceState(this._state, 'prefillWithTokenLogits');
    return prefillWithTokenLogitsImpl.apply(this, arguments);
  }

  async prefillWithTokenLogitsFromKV(prefix, prompt, tokenIds, options = {}) {
    assertClassicSequenceState(this._state, 'prefillWithTokenLogitsFromKV');
    return prefillWithTokenLogitsFromKVImpl.apply(this, arguments);
  }

  async *generateWithPrefixKV(prefix, prompt, options = {}) {
    assertClassicSequenceState(this._state, 'generateWithPrefixKV');
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');
    const decoder = createGenerationOutputDecoder(this._state.tokenizer,
      this._state.modelConfig?.reasoningOutput,
      [...this._state.modelConfig.stopTokenIds, this._state.tokenizer.getSpecialTokens?.()?.eos]
        .filter(Number.isSafeInteger));
    try {
      yield* decodeGenerationStream(generateWithPrefixKVImpl.call(this, prefix, prompt,
        { ...options, onToken: undefined }, 'token'), decoder,
        { ...options, getStopReason: () => this._state.stats.stopReason });
    } finally {
      const usage = decoder.getUsage?.();
      if (usage?.rawTokens > 0) this._state.stats.outputUsage = usage;
    }
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
    return withHyperConnectionTransaction(this._state, opts, scoped => _prefillImpl.call(this, inputIds, scoped));
  }

  async _decodeStep(currentIds, opts) {
    return withHyperConnectionTransaction(this._state, opts, scoped => _decodeStepImpl.call(this, currentIds, scoped));
  }

  async decodeStepLogits(currentIds, options = {}) {
    return decodeStepLogitsImpl.apply(this, arguments);
  }

  beginDiagnosticSequence(options = {}) {
    return beginDiagnosticSequence(this._state, options);
  }

  endDiagnosticSequence() {
    return endDiagnosticSequence(this._state);
  }

  async advanceWithToken(tokenId, options = {}) {
    return withHyperConnectionTransaction(this._state, options, scoped => advanceWithTokenImpl.call(this, tokenId, scoped));
  }

  async advanceWithTokenAndEmbedding(tokenId, options = {}) {
    assertClassicSequenceState(this._state, 'advanceWithTokenAndEmbedding');
    return advanceWithTokenAndEmbeddingImpl.apply(this, arguments);
  }

  async _generateNTokensGPU(startToken, N, currentIds, opts) {
    return _generateNTokensGPUImpl.apply(this, arguments);
  }
}
