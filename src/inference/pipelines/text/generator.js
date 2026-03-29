

import { getDevice, setTrackSubmits } from '../../../gpu/device.js';
import { releaseBuffer, readBuffer, readBufferSlice, uploadData } from '../../../memory/buffer-pool.js';
import { isGPUSamplingAvailable } from '../../../gpu/kernels/sample.js';
import { markWarmed as markKernelCacheWarmed } from '../../../gpu/kernel-selection-cache.js';
import { resetSubmitStats, logSubmitStats } from '../../../gpu/submit-tracker.js';
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from '../../../gpu/command-recorder.js';
import { allowReadback } from '../../../gpu/perf-guards.js';
import { log, trace } from '../../../debug/index.js';
import {
  CAPTURE_LEVELS,
  createDefaultCaptureConfig,
  validateCaptureConfig,
} from '../../../debug/index.js';
import { validateCallTimeOptions } from '../../../config/param-validator.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

// Pipeline sub-modules
import { sample, applyRepetitionPenalty, logitsSanity, getTopK } from './sampling.js';
import { enforceLogitDrift } from '../../../hotswap/intent-bundle.js';
import { applyChatTemplate, isStopToken } from './init.js';
import { formatChatMessages } from './chat-format.js';
import { embed } from './embed.js';
import { processLayer } from './layer.js';
import { computeLogits, recordLogitsGPU, extractLastPositionLogits, applySoftcapping } from './logits/index.js';
import { OperatorEventEmitter } from './operator-events.js';
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, getWeightDtype } from '../../../gpu/weight-buffer.js';
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
import { buildLayerContext, debugCheckBuffer as debugCheckBufferHelper, getLogitsConfig, getLogitsWeights } from './generator-helpers.js';
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
} from './generator-runtime.js';

import { resolveSamplingConfig } from './sampling-config.js';
import { decodeReadback, getLogitsHealth } from './debug-utils/index.js';
import { parseFinitenessStatusWords } from './finiteness-guard-status.js';
import { resolveDeferredRoundingWindowTokens } from './finiteness-policy.js';
import {
  activateFallbackExecutionPlan,
  hasFallbackExecutionPlan,
  rebaseExecutionSessionPlan,
  resetActiveExecutionPlan,
  resolveActiveExecutionPlan,
  setActiveExecutionPlan,
} from './execution-plan.js';
import {
  cloneLinearAttentionRuntime,
  hasLinearAttentionLayers,
  resetLinearAttentionRuntime,
  restoreLinearAttentionRuntime,
} from './linear-attention.js';

function isStructuredChatRequest(prompt) {
  return prompt != null
    && typeof prompt === 'object'
    && !Array.isArray(prompt)
    && Array.isArray(prompt.messages);
}

function resolvePromptInput(state, prompt, useChatTemplate, contextLabel) {
  if (typeof prompt === 'string') {
    if (useChatTemplate && state.modelConfig.chatTemplateType) {
      if (state.modelConfig.chatTemplateType === 'translategemma') {
        throw new Error(
          `[Pipeline] ${contextLabel}: translategemma chat template requires structured messages. ` +
          'Pass { messages: [...] } instead of a plain string prompt.'
        );
      }
      return applyChatTemplate(prompt, state.modelConfig.chatTemplateType);
    }
    return prompt;
  }

  if (prompt != null && typeof prompt === 'object' && !Array.isArray(prompt) && 'messages' in prompt && !Array.isArray(prompt.messages)) {
    throw new Error(
      `[Pipeline] ${contextLabel}: prompt.messages must be an array of chat messages, got ${typeof prompt.messages}. ` +
      'Pass { messages: [{ role: "user", content: "..." }, ...] }.'
    );
  }
  const messages = isStructuredChatRequest(prompt)
    ? prompt.messages
    : (Array.isArray(prompt) ? prompt : null);
  if (!messages) {
    throw new Error(
      `[Pipeline] ${contextLabel}: prompt must be a string, chat message array, or { messages: [...] }.`
    );
  }
  const templateType = useChatTemplate ? state.modelConfig.chatTemplateType : null;
  return formatChatMessages(messages, templateType);
}

/**
 * Resolve display text for a token ID sequence.
 *
 * Preference order:
 *   1. renderTokenText (custom primary renderer) — if provided and returns non-empty string.
 *   2. tokenizer.decode (default primary) — if renderTokenText is not provided.
 *   3. renderFallbackTokenText (custom fallback) — if provided and returns non-empty string,
 *      unless primary returned empty string and fallback looks like a special token (<...>),
 *      in which case empty string is preserved to keep skip-special behavior deterministic.
 *   4. tokenizer.decode with skipSpecialTokens=false (default fallback) — same special-token guard.
 *   5. fallbackText — static fallback string (default '?').
 */
function resolveTokenText(tokenizer, tokenIds, fallbackText = '?', renderTokenText, renderFallbackTokenText) {
  const renderPrimary = typeof renderTokenText === 'function'
    ? renderTokenText
    : (ids) => tokenizer?.decode?.(ids);
  const renderFallback = typeof renderFallbackTokenText === 'function'
    ? renderFallbackTokenText
    : (ids) => tokenizer?.decode?.(ids, false);

  const primaryText = renderPrimary(tokenIds);
  if (typeof primaryText === 'string' && primaryText.length > 0) {
    return primaryText;
  }

  const fallback = renderFallback(tokenIds);
  if (typeof fallback === 'string' && fallback.length > 0) {
    // Keep skip-special behavior deterministic: if primary decoding filtered this
    // token to empty, do not reintroduce obvious special-token text via fallback.
    if (
      primaryText === ''
      && /^<[^>\n]{1,80}>$/.test(fallback.trim())
    ) {
      return '';
    }
    return fallback;
  }

  return fallbackText;
}

function summarizeExecutionPlan(plan) {
  if (!plan) {
    return null;
  }
  if (typeof plan !== 'object') {
    log.warn('Pipeline', `summarizeExecutionPlan: expected object, got ${typeof plan}`);
    return null;
  }
  if (typeof plan.id !== 'string') {
    log.warn('Pipeline', 'summarizeExecutionPlan: plan is missing required string property "id"');
  }
  if (typeof plan.activationDtype !== 'string') {
    log.warn('Pipeline', 'summarizeExecutionPlan: plan is missing required string property "activationDtype"');
  }
  return {
    id: plan.id,
    kernelPathId: plan.kernelPathId ?? null,
    kernelPathSource: plan.kernelPathSource ?? 'none',
    activationDtype: plan.activationDtype,
    readbackInterval: plan.readbackInterval ?? null,
    batchSize: plan.defaultBatchSize,
    stopCheckMode: plan.defaultStopCheckMode,
    ringTokens: plan.ringTokens ?? null,
    ringStop: plan.ringStop ?? null,
    ringStaging: plan.ringStaging ?? null,
  };
}

export function shouldRetryWithFinitenessFallback(error) {
  if (error?.name === 'FinitenessError') {
    return true;
  }
  const message = typeof error?.message === 'string'
    ? error.message
    : (typeof error === 'string' ? error : '');
  if (!message.startsWith('[Sampling]')) {
    return false;
  }
  return message.includes('no finite candidate logits after masking the pad token')
    || message.includes('Softmax produced no finite candidate probabilities');
}

function createUnhandledFinitenessPolicyError(state, contextLabel, error) {
  const activePlan = resolveActiveExecutionPlan(state);
  const wrapped = new Error(
    `[Pipeline] ${contextLabel}: finiteness guard triggered for kernelPath ` +
    `"${activePlan.kernelPathId ?? 'none'}" under fail-fast policy. ` +
    'Resolve the unstable path with an explicit capability-aware execution override, ' +
    'or opt into alternate-plan recovery with ' +
    'runtime.inference.compute.rangeAwareSelectiveWidening.onTrigger="fallback-plan".',
    error instanceof Error ? { cause: error } : undefined
  );
  wrapped.name = error?.name === 'FinitenessError' ? error.name : 'FinitenessError';
  return wrapped;
}

export class PipelineGenerator {

  #state;
  #finitenessFallbackWindow;

  _assertTokenIdsInRange(tokenIds, context = 'encode') {
    assertTokenIdsInRange(this.#state, tokenIds, context);
  }

  _assertTokenIdInRange(tokenId, context = 'token') {
    assertTokenIdInRange(this.#state, tokenId, context);
  }


  constructor(state) {
    this.#state = state;
    this.#finitenessFallbackWindow = null;
  }

  _resolveDeferredRoundingWindowTokens() {
    const activePlan = resolveActiveExecutionPlan(this.#state);
    return activePlan?.deferredRoundingWindowTokens
      ?? resolveDeferredRoundingWindowTokens(this.#state.runtimeConfig?.inference?.compute);
  }

  _getEffectiveActivationDtype() {
    return resolveActiveExecutionPlan(this.#state).activationDtype;
  }

  _hasFinitenessFallbackWindow() {
    return this.#finitenessFallbackWindow !== null;
  }

  _shouldUseFinitenessFallback(error, contextLabel) {
    if (!shouldRetryWithFinitenessFallback(error)) {
      return false;
    }
    if (!hasFallbackExecutionPlan(this.#state)) {
      throw createUnhandledFinitenessPolicyError(this.#state, contextLabel, error);
    }
    return true;
  }

  _openFinitenessFallbackWindow(opts, reasonLabel, tokenCount) {
    const normalizedCount = Number.isFinite(tokenCount)
      ? Math.max(1, Math.floor(tokenCount))
      : 1;
    if (this.#finitenessFallbackWindow) {
      this.#finitenessFallbackWindow.remainingTokens = Math.max(
        this.#finitenessFallbackWindow.remainingTokens,
        normalizedCount
      );
      return;
    }
    const original = this._beginFinitenessFallback(opts, reasonLabel);
    this.#finitenessFallbackWindow = {
      original,
      remainingTokens: normalizedCount,
    };
  }

  _closeFinitenessFallbackWindow(opts) {
    if (!this.#finitenessFallbackWindow) {
      return;
    }
    const original = this.#finitenessFallbackWindow.original;
    this.#finitenessFallbackWindow = null;
    this._endFinitenessFallback(opts, original);
  }

  _consumeFinitenessFallbackToken(opts) {
    if (!this.#finitenessFallbackWindow) {
      return;
    }
    this.#finitenessFallbackWindow.remainingTokens -= 1;
    if (this.#finitenessFallbackWindow.remainingTokens <= 0) {
      this._closeFinitenessFallbackWindow(opts);
    }
  }

  _resolveStepOptions(options = {}) {
    return resolveStepOptions(this.#state, options);
  }

  _resetDecodeRuntimeState() {
    this.#state.stats.decodeMode = null;
    this.#state.stats.batchGuardReason = null;
    this.#state.stats.decodeProfileSteps = [];
    this.#state.stats.ttftMs = 0;
    this.#state.stats.decodeTimeMs = 0;
    this.#state.stats.decodeRecordMs = 0;
    this.#state.stats.decodeSubmitWaitMs = 0;
    this.#state.stats.decodeReadbackWaitMs = 0;
    this.#state.decodeStepCount = 0;
    this.#state.disableRecordedLogits = false;
    this.#state.disableFusedDecode = false;
    this.#state.batchingStats = {
      batchedForwardCalls: 0,
      unbatchedForwardCalls: 0,
      totalBatchedTimeMs: 0,
      totalUnbatchedTimeMs: 0,
      gpuSubmissions: 0,
    };
    resetActiveExecutionPlan(this.#state);
    this.#state.decodeRing?.reset();
  }

  _getDecodeHelpers(debugCheckBuffer) {
    return {
      buildLayerContext: (recorder, isDecodeMode, debugLayers, executionPlan) =>
        buildLayerContext(this.#state, recorder, isDecodeMode, debugLayers, debugCheckBuffer, executionPlan),
      getLogitsWeights: () => getLogitsWeights(this.#state),
      getLogitsConfig: () => getLogitsConfig(this.#state),
      debugCheckBuffer,
    };
  }

  async _getFinalNormWeights() {
    return getFinalNormWeights(this.#state);
  }

  _extractEmbeddingFromHidden(hiddenStates, numTokens, hiddenSize, embeddingMode, finalNormWeights, config) {
    return extractEmbeddingFromHidden(
      hiddenStates,
      numTokens,
      hiddenSize,
      embeddingMode,
      finalNormWeights,
      config,
      this.#state.embeddingPostprocessor
    );
  }

  _resolvePromptTokenIds(prompt, useChatTemplate, contextLabel) {
    const processedPrompt = resolvePromptInput(this.#state, prompt, useChatTemplate, contextLabel);
    const inputIds = this.#state.tokenizer.encode(processedPrompt);
    this._assertTokenIdsInRange(inputIds, `${contextLabel}.encode`);
    return inputIds;
  }

  _sampleNextTokenFromLogits(logits, generatedIds, opts) {
    const sampledLogits = Float32Array.from(logits);
    applyRepetitionPenalty(sampledLogits, generatedIds, opts.repetitionPenalty);
    const padTokenId = this.#state.tokenizer?.getSpecialTokens?.()?.pad;
    return sample(sampledLogits, {
      temperature: opts.temperature,
      topP: opts.topP,
      topK: opts.topK,
      padTokenId,
      seed: opts.seed,
    });
  }

  async _prefillPromptToLogits(prompt, opts, contextLabel) {
    const inputIds = this._resolvePromptTokenIds(prompt, opts.useChatTemplate, contextLabel);
    if (opts.debug) {
      log.debug('Pipeline', `${contextLabel}: ${inputIds.length} tokens`);
    }

    let logits;
    try {
      logits = await this._prefill(inputIds, opts);
    } catch (error) {
      if (!this._shouldUseFinitenessFallback(error, contextLabel)) {
        throw error;
      }
      log.warn('Pipeline', `FinitenessGuard caught NaN/Inf during ${contextLabel}. Retrying with F32 precision.`);
      logits = await this._retryWithFinitenessFallback(
        opts,
        contextLabel,
        () => this._prefill(inputIds, opts)
      );
    }

    return { inputIds, logits };
  }

  async _decodeStepToLogits(currentIds, opts) {
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;
    return decodeStepLogits(this.#state, currentIds, opts, this._getDecodeHelpers(debugCheckBuffer));
  }

  async _decodeNextTokenViaLogits(currentIds, opts) {
    const stepResult = await this._decodeStepToLogits(currentIds, opts);
    return this._sampleNextTokenFromLogits(stepResult.logits, currentIds, opts);
  }

  async *_generateTokensInternal(prompt, options = {}, mode = 'text') {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);

    this.#state.isGenerating = true;
    this._resetDecodeRuntimeState();
    this.#state.stats.gpuTimePrefillMs = undefined;
    this.#state.stats.gpuTimeDecodeMs = undefined;
    this.#state.stats.decodeRecordMs = 0;
    this.#state.stats.decodeSubmitWaitMs = 0;
    this.#state.stats.decodeReadbackWaitMs = 0;
    this.#state.stats.singleTokenSubmitWaitMs = 0;
    this.#state.stats.singleTokenReadbackWaitMs = 0;
    this.#state.stats.singleTokenOrchestrationMs = 0;
    this.#state.stats.decodeMode = null;
    this.#state.stats.batchGuardReason = null;
    this.#state.stats.ttftMs = 0;
    const startTime = performance.now();

    const opts = resolveGenerateOptions(this.#state, options);
    // Validate and normalize sampling parameters through single source of truth
    const samplingConfig = resolveSamplingConfig(options, this.#state.runtimeConfig);
    opts.temperature = samplingConfig.temperature;
    opts.topP = samplingConfig.topP;
    opts.topK = samplingConfig.topK;
    opts.repetitionPenalty = samplingConfig.repetitionPenalty;
    const diagnosticsEnabled = options?.diagnostics?.enabled === true
      || this.#state.runtimeConfig?.shared?.harness?.mode === 'diagnose';
    if (diagnosticsEnabled) {
      const captureConfig = {
        ...createDefaultCaptureConfig(),
        enabled: true,
        defaultLevel: CAPTURE_LEVELS.SLICE,
        ...(options?.diagnostics?.captureConfig ?? {}),
      };
      validateCaptureConfig(captureConfig);
      this.#state.operatorDiagnostics = {
        enabled: true,
        captureConfig,
        emitter: new OperatorEventEmitter({
          modelHash: this.#state.manifest?.modelId ?? null,
          runtimeConfigHash: this.#state.resolvedKernelPath?.id ?? null,
          executionPlanHash: opts.executionPlan?.id ?? null,
        }),
      };
    }
    const activePlan = opts.executionPlan ?? resolveActiveExecutionPlan(this.#state);
    this.#state.stats.executionPlan = {
      primary: summarizeExecutionPlan(this.#state.executionPlanState?.primaryPlan ?? null),
      fallback: summarizeExecutionPlan(this.#state.executionPlanState?.fallbackPlan ?? null),
      activePlanIdAtStart: activePlan?.id ?? null,
      finalActivePlanId: activePlan?.id ?? null,
      transitions: [],
    };
    this.#state.stats.kernelPathId = activePlan?.kernelPathId ?? this.#state.resolvedKernelPath?.id ?? null;
    this.#state.stats.kernelPathSource = activePlan?.kernelPathSource ?? this.#state.kernelPathSource ?? 'none';

    if (opts.debug) {
      log.debug('Pipeline', `ChatTemplate: options=${options.useChatTemplate}, final=${opts.useChatTemplate}`);
    }

    const emitToken = async function* (generator, tokenId, textDecoder) {
      if (mode === 'token') {
        yield tokenId;
        if (options.onToken) options.onToken(tokenId, '');
        return;
      }
      const tokenText = textDecoder(tokenId);
      yield tokenText;
      if (options.onToken) options.onToken(tokenId, tokenText);
    };

    try {
      const prefillStart = performance.now();
      const { inputIds, logits: initialPrefillLogits } = await this._prefillPromptToLogits(prompt, opts, 'generate');
      let prefillLogits = initialPrefillLogits;
      this.#state.stats.prefillTimeMs = performance.now() - prefillStart;
      this._assertTokenIdsInRange(inputIds, 'generate.prefillTokens');
      const generatedIds = [...inputIds];
      this.#state.stats.prefillTokens = inputIds.length;

      if (opts.debug) {
        log.debug('Pipeline', `Input: ${inputIds.length} tokens`);
      }

      const intentBundleConfig = this.#state.runtimeConfig.shared.intentBundle;
      const intentBundle = intentBundleConfig?.bundle;
      const expectedTopK = intentBundle?.payload?.expectedTopK
        ?? intentBundle?.payload?.expected_top_k;
      const maxDriftThreshold = intentBundle?.constraints?.maxDriftThreshold
        ?? intentBundle?.constraints?.max_drift_threshold;

      if (intentBundleConfig?.enabled && Array.isArray(expectedTopK) && expectedTopK.length > 0) {
        const actualTopK = getTopK(
          prefillLogits,
          expectedTopK.length,
          (tokens) => resolveTokenText(this.#state.tokenizer, tokens),
        ).map((token) => token.token);
        const driftResult = enforceLogitDrift(expectedTopK, actualTopK, maxDriftThreshold);
        if (!driftResult.ok) {
          throw new Error(`Intent bundle drift check failed: ${driftResult.reason}`);
        }
      }

      if (opts.debug) {
        const topAfterPenalty = getTopK(
          Float32Array.from(prefillLogits),
          5,
          (tokens) => resolveTokenText(this.#state.tokenizer, tokens)
        );
        log.debug('Pipeline', `After rep penalty top-5: ${topAfterPenalty.map(t => `"${t.text}"(${(t.prob * 100).toFixed(1)}%)`).join(', ')}`);
      }

      let firstToken;
      try {
        firstToken = this._sampleNextTokenFromLogits(prefillLogits, generatedIds, opts);
      } catch (error) {
        if (!this._shouldUseFinitenessFallback(error, 'prefill-sample')) {
          throw error;
        }
        log.warn('Pipeline', 'FinitenessGuard caught non-finite prefill logits at sampling. Retrying with F32 precision.');
        prefillLogits = await this._retryWithFinitenessFallback(
          opts,
          'prefill-sample',
          () => this._prefill(inputIds, opts)
        );
        firstToken = this._sampleNextTokenFromLogits(prefillLogits, generatedIds, opts);
      }

      if (opts.debug) {
        const firstTokenText = resolveTokenText(this.#state.tokenizer, [firstToken], `[${firstToken}]`, (tokens) => this.#state.tokenizer?.decode?.(tokens, true, false));
        log.debug('Pipeline', `First token sampled: id=${firstToken} text="${firstTokenText}"`);
      }

      generatedIds.push(firstToken);
      this.#state.stats.ttftMs = performance.now() - startTime;

      const decodeToken = (tokenId) => resolveTokenText(
        this.#state.tokenizer,
        [tokenId],
        `[${tokenId}]`,
        (tokens) => this.#state.tokenizer?.decode?.(tokens, true, false),
        (tokens) => this.#state.tokenizer?.decode?.(tokens, false, false)
      );

      yield* emitToken(this, firstToken, decodeToken);

      yield* this._runDecodeLoop(generatedIds, opts, options, {
        stopTokenIds: this.#state.modelConfig.stopTokenIds,
        eosToken: this.#state.tokenizer.getSpecialTokens?.()?.eos,
        stopSequenceStart: inputIds.length,
        decodeToken,
        logBatchPath: opts.debug,
        emitMode: mode,
      });
      const tokensGenerated = this.#state.stats.decodeTokens;
      this.#state.stats.totalTimeMs = performance.now() - startTime;

      if (opts.debug) {
        log.debug('Pipeline', `Generated ${tokensGenerated} tokens in ${this.#state.stats.totalTimeMs.toFixed(0)}ms`);
      }

      const ttft = this.#state.stats.ttftMs ?? this.#state.stats.prefillTimeMs;
      const decodeTokens = Math.max(0, tokensGenerated - 1);
      const decodeSpeed = decodeTokens > 0 ? (decodeTokens / this.#state.stats.decodeTimeMs * 1000) : 0;
      const loadMs = this.#state.stats.modelLoadMs;
      const loadLabel = Number.isFinite(loadMs) ? `Load: ${loadMs.toFixed(0)}ms | ` : '';
      if (opts.benchmark) {
        log.info('Benchmark', `${loadLabel}TTFT: ${ttft.toFixed(0)}ms | Prefill: ${this.#state.stats.prefillTimeMs.toFixed(0)}ms | Decode: ${this.#state.stats.decodeTimeMs.toFixed(0)}ms (${decodeTokens} tokens @ ${decodeSpeed.toFixed(1)} tok/s)`);
      } else {
        log.info('Perf', `${loadLabel}TTFT: ${ttft.toFixed(0)}ms | Prefill: ${this.#state.stats.prefillTimeMs.toFixed(0)}ms | Decode: ${this.#state.stats.decodeTimeMs.toFixed(0)}ms (${decodeTokens} tokens @ ${decodeSpeed.toFixed(1)} tok/s)`);
      }
      trace.perf('Decode summary', {
        ttftMs: ttft,
        prefillMs: this.#state.stats.prefillTimeMs,
        decodeMs: this.#state.stats.decodeTimeMs,
        decodeTokens,
        decodeSpeed,
        totalMs: this.#state.stats.totalTimeMs,
      });
    } finally {
      this._closeFinitenessFallbackWindow(opts);
      resetActiveExecutionPlan(this.#state);
      this.#state.stats.operatorDiagnostics = this.#state.operatorDiagnostics?.emitter
        ? {
          enabled: true,
          timeline: this.#state.operatorDiagnostics.emitter.getTimeline(),
          recordCount: this.#state.operatorDiagnostics.emitter.length,
        }
        : null;
      this.#state.operatorDiagnostics = null;
      this.#state.isGenerating = false;
    }
  }

  _beginFinitenessFallback(opts, reasonLabel) {
    const originalPlan = resolveActiveExecutionPlan(this.#state);
    const original = {
      activePlanId: this.#state.executionPlanState?.activePlanId ?? 'primary',
      seed: opts.seed,
    };

    const fallbackPlan = activateFallbackExecutionPlan(this.#state);
    if (!fallbackPlan) {
      throw new Error(
        '[Pipeline] Explicit alternate-plan finiteness recovery is unavailable for this model/runtime configuration.'
      );
    }
    log.warn(
      'Pipeline',
      `FinitenessGuard fallback (${reasonLabel}): ` +
      `${originalPlan.kernelPathId ?? 'none'} -> ${fallbackPlan.kernelPathId ?? 'none'}`
    );

    this.#state.decodeBuffers?.ensureBuffers({
      hiddenSize: this.#state.modelConfig.hiddenSize,
      intermediateSize: this.#state.modelConfig.intermediateSize,
      activationDtype: fallbackPlan.activationDtype,
      enablePingPong: true,
    });

    if (opts.seed == null) {
      const fallbackSeedBase = (this.#state.decodeStepCount + this.#state.currentSeqLen + 1) >>> 0;
      opts.seed = (fallbackSeedBase * 2654435761) >>> 0;
    }
    opts.executionPlan = rebaseExecutionSessionPlan(this.#state, opts.executionPlan);
    if (this.#state.stats.executionPlan) {
      this.#state.stats.executionPlan.finalActivePlanId = fallbackPlan.id;
      this.#state.stats.executionPlan.transitions.push({
        kind: 'activate-finiteness-fallback',
        reason: reasonLabel ?? null,
        decodeStep: this.#state.decodeStepCount,
        seqLen: this.#state.currentSeqLen,
        fromPlanId: originalPlan.id,
        toPlanId: fallbackPlan.id,
        fromKernelPathId: originalPlan.kernelPathId ?? null,
        toKernelPathId: fallbackPlan.kernelPathId ?? null,
      });
    }
    this.#state.stats.kernelPathId = fallbackPlan.kernelPathId ?? null;
    this.#state.stats.kernelPathSource = fallbackPlan.kernelPathSource ?? 'none';

    return original;
  }

  _endFinitenessFallback(opts, original) {
    opts.seed = original.seed;
    setActiveExecutionPlan(this.#state, original.activePlanId);
    opts.executionPlan = rebaseExecutionSessionPlan(this.#state, opts.executionPlan);
    const restoredPlan = resolveActiveExecutionPlan(this.#state);
    if (this.#state.stats.executionPlan) {
      this.#state.stats.executionPlan.finalActivePlanId = restoredPlan.id;
      this.#state.stats.executionPlan.transitions.push({
        kind: 'restore-primary-plan',
        reason: null,
        decodeStep: this.#state.decodeStepCount,
        seqLen: this.#state.currentSeqLen,
        fromPlanId: this.#state.executionPlanState?.fallbackPlan?.id ?? null,
        toPlanId: restoredPlan.id,
        fromKernelPathId: this.#state.executionPlanState?.fallbackPlan?.kernelPathId ?? null,
        toKernelPathId: restoredPlan.kernelPathId ?? null,
      });
    }
    this.#state.stats.kernelPathId = restoredPlan.kernelPathId ?? this.#state.resolvedKernelPath?.id ?? null;
    this.#state.stats.kernelPathSource = restoredPlan.kernelPathSource ?? this.#state.kernelPathSource ?? 'none';
    const nextActivationDtype = this._getEffectiveActivationDtype();
    this.#state.decodeBuffers?.ensureBuffers({
      hiddenSize: this.#state.modelConfig.hiddenSize,
      intermediateSize: this.#state.modelConfig.intermediateSize,
      activationDtype: nextActivationDtype,
      enablePingPong: true,
    });
  }

  async _retryWithFinitenessFallback(opts, reasonLabel, retryFn) {
    if (this._hasFinitenessFallbackWindow()) {
      return retryFn();
    }
    this.#state.kvCache?.truncate(this.#state.currentSeqLen);
    const original = this._beginFinitenessFallback(opts, reasonLabel);
    try {
      return await retryFn();
    } finally {
      this._endFinitenessFallback(opts, original);
    }
  }

  async _retryDecodeStepWithFinitenessWindow(generatedIds, opts, reasonLabel) {
    const windowTokens = this._resolveDeferredRoundingWindowTokens();
    if (windowTokens <= 1) {
      return this._retryWithFinitenessFallback(
        opts,
        reasonLabel,
        () => this._decodeStep(generatedIds, opts)
      );
    }

    this.#state.kvCache?.truncate(this.#state.currentSeqLen);
    this._openFinitenessFallbackWindow(opts, reasonLabel, windowTokens);
    try {
      return await this._decodeStep(generatedIds, opts);
    } catch (error) {
      this._closeFinitenessFallbackWindow(opts);
      throw error;
    }
  }

  // ==========================================================================
  // Generation Public API
  // ==========================================================================


  async *generate(prompt, options = {}) {
    yield* this._generateTokensInternal(prompt, options, 'text');
  }

  async *generateTokens(prompt, options = {}) {
    yield* this._generateTokensInternal(prompt, options, 'token');
  }

  async generateTokenIds(prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);

    this.#state.isGenerating = true;
    this._resetDecodeRuntimeState();
    this.#state.stats.gpuTimePrefillMs = undefined;
    this.#state.stats.gpuTimeDecodeMs = undefined;
    this.#state.stats.decodeRecordMs = 0;
    this.#state.stats.decodeSubmitWaitMs = 0;
    this.#state.stats.decodeReadbackWaitMs = 0;
    this.#state.stats.singleTokenSubmitWaitMs = 0;
    this.#state.stats.singleTokenReadbackWaitMs = 0;
    this.#state.stats.singleTokenOrchestrationMs = 0;
    this.#state.stats.decodeMode = null;
    this.#state.stats.batchGuardReason = null;
    this.#state.stats.ttftMs = 0;
    const startTime = performance.now();
    const opts = resolveGenerateOptions(this.#state, options);
    // Validate and normalize sampling parameters through single source of truth
    const samplingConfig = resolveSamplingConfig(options, this.#state.runtimeConfig);
    opts.temperature = samplingConfig.temperature;
    opts.topP = samplingConfig.topP;
    opts.topK = samplingConfig.topK;
    opts.repetitionPenalty = samplingConfig.repetitionPenalty;

    try {
      const prefillStart = performance.now();
      const { inputIds, logits: initialPrefillLogits } = await this._prefillPromptToLogits(prompt, opts, 'generateTokenIds');
      let prefillLogits = initialPrefillLogits;
      this.#state.stats.prefillTimeMs = performance.now() - prefillStart;
      this._assertTokenIdsInRange(inputIds, 'generateTokenIds.prefillTokens');
      const generatedIds = [...inputIds];
      this.#state.stats.prefillTokens = inputIds.length;

      let firstToken;
      try {
        firstToken = this._sampleNextTokenFromLogits(prefillLogits, generatedIds, opts);
      } catch (error) {
        if (!this._shouldUseFinitenessFallback(error, 'prefill-sample')) {
          throw error;
        }
        prefillLogits = await this._retryWithFinitenessFallback(
          opts,
          'prefill-sample',
          () => this._prefill(inputIds, opts)
        );
        firstToken = this._sampleNextTokenFromLogits(prefillLogits, generatedIds, opts);
      }

      generatedIds.push(firstToken);
      const tokenIds = [firstToken];
      this.#state.stats.ttftMs = performance.now() - startTime;

      const stopTokenIds = this.#state.modelConfig.stopTokenIds;
      const eosToken = this.#state.tokenizer.getSpecialTokens?.()?.eos;
      const stopSequenceStart = inputIds.length;
      markKernelCacheWarmed();
      const decodeStart = performance.now();

      while (tokenIds.length < opts.maxTokens) {
        if (options.signal?.aborted) break;
        let nextToken;
        try {
          nextToken = await this._decodeNextTokenViaLogits(generatedIds, opts);
        } catch (error) {
          if (this._shouldUseFinitenessFallback(error, `decode-step-${tokenIds.length}`)) {
            nextToken = await this._retryDecodeStepWithFinitenessWindow(
              generatedIds,
              opts,
              `decode-step-${tokenIds.length}`
            );
          } else {
            throw error;
          }
        }
        generatedIds.push(nextToken);
        tokenIds.push(nextToken);
        this._consumeFinitenessFallbackToken(opts);
        if (isStopToken(nextToken, stopTokenIds, eosToken)) {
          break;
        }
        if (opts.stopSequences.length > 0) {
          const fullText = this.#state.tokenizer.decode(generatedIds.slice(stopSequenceStart), false);
          if (opts.stopSequences.some((seq) => fullText.endsWith(seq))) break;
        }
      }

      this.#state.stats.decodeTimeMs = performance.now() - decodeStart;
      this.#state.stats.tokensGenerated = tokenIds.length;
      this.#state.stats.decodeTokens = tokenIds.length;
      this.#state.stats.totalTimeMs = performance.now() - startTime;

      return {
        tokenIds,
        stats: this.#state.stats,
      };
    } finally {
      this._closeFinitenessFallbackWindow(opts);
      if (this.#state.stats.executionPlan) {
        this.#state.stats.executionPlan.finalActivePlanId = this.#state.executionPlanState?.activePlanId ?? null;
      }
      resetActiveExecutionPlan(this.#state);
      this.#state.isGenerating = false;
    }
  }


  async prefillKVOnly(prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    this._resetDecodeRuntimeState();
    this.#state.stats.gpuTimePrefillMs = undefined;
    const opts = resolvePrefillOptions(this.#state, options);
    const inputIds = this._resolvePromptTokenIds(prompt, opts.useChatTemplate, 'prefillKVOnly');
    if (opts.debug) {
      log.debug('Pipeline', `PrefillKVOnly: ${inputIds.length} tokens`);
    }

    let prefillResult;
    try {
      prefillResult = await this._prefillToHidden(inputIds, opts);
    } catch (error) {
      if (this._shouldUseFinitenessFallback(error, 'prefillKVOnly')) {
        log.warn('Pipeline', `FinitenessGuard caught NaN/Inf during prefillKVOnly. Retrying with F32 precision.`);
        prefillResult = await this._retryWithFinitenessFallback(
          opts,
          'prefillKVOnly',
          () => this._prefillToHidden(inputIds, opts)
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
      await currentRecorder.submitAndWait();
      await recordProfile(currentRecorder);
    } else {
      const device = getDevice();
      if (device) {
        await device.queue.onSubmittedWorkDone();
      }
    }

    this.#state.currentSeqLen = startPos + numTokens;
    releaseBuffer(currentHiddenBuffer);

    const snapshot = this.#state.kvCache?.clone();
    if (!snapshot) {
      throw new Error('KV cache unavailable after prefill');
    }

    return {
      cache: snapshot,
      seqLen: this.#state.currentSeqLen,
      tokens: inputIds,
      linearAttention: await cloneLinearAttentionRuntime(this.#state.linearAttentionRuntime),
    };
  }

  async prefillWithEmbedding(prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    this._resetDecodeRuntimeState();
    this.#state.stats.gpuTimePrefillMs = undefined;
    const opts = resolvePrefillEmbeddingOptions(this.#state, options);
    const inputIds = this._resolvePromptTokenIds(prompt, opts.useChatTemplate, 'prefillWithEmbedding');
    if (opts.debug) {
      log.debug('Pipeline', `PrefillWithEmbedding: ${inputIds.length} tokens (mode=${opts.embeddingMode})`);
    }

    let prefillResult;
    try {
      prefillResult = await this._prefillToHidden(inputIds, opts);
    } catch (error) {
      if (shouldRetryWithFinitenessFallback(error)) {
        log.warn('Pipeline', `FinitenessGuard caught NaN/Inf during prefillWithEmbedding. Retrying with F32 precision.`);
        prefillResult = await this._retryWithFinitenessFallback(
          opts,
          'prefillWithEmbedding',
          () => this._prefillToHidden(inputIds, opts)
        );
      } else {
        throw error;
      }
    }

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
    if (currentRecorder) {
      await currentRecorder.submitAndWait();
      await recordProfile(currentRecorder);
    } else {
      const device = getDevice();
      if (device) {
        await device.queue.onSubmittedWorkDone();
      }
    }

    if (!allowReadback('pipeline.prefill.embedding')) {
      throw new Error('GPU readback disabled; cannot return embedding');
    }

    let embedding;
    try {
      const hiddenSize = config.hiddenSize;
      const hiddenBytes = numTokens * hiddenSize * activationBytes;
      const hiddenData = await readBuffer(currentHiddenBuffer, hiddenBytes);
      if (hiddenData.byteLength === 0) {
        throw new Error('GPU readback disabled; cannot return embedding');
      }
      const hiddenStates = decodeReadback(hiddenData, activationDtype);
      const finalNormWeights = await this._getFinalNormWeights();
      embedding = this._extractEmbeddingFromHidden(
        hiddenStates,
        numTokens,
        hiddenSize,
        opts.embeddingMode,
        finalNormWeights,
        config
      );
    } finally {
      releaseBuffer(currentHiddenBuffer);
    }

    this.#state.currentSeqLen = startPos + numTokens;

    const snapshot = this.#state.kvCache?.clone();
    if (!snapshot) {
      throw new Error('KV cache unavailable after prefill');
    }

    return {
      cache: snapshot,
      seqLen: this.#state.currentSeqLen,
      tokens: inputIds,
      embedding,
      embeddingMode: opts.embeddingMode,
      linearAttention: await cloneLinearAttentionRuntime(this.#state.linearAttentionRuntime),
    };
  }

  async prefillWithLogits(prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    this._resetDecodeRuntimeState();
    this.#state.stats.gpuTimePrefillMs = undefined;
    const opts = resolvePrefillOptions(this.#state, options);
    const { inputIds, logits } = await this._prefillPromptToLogits(prompt, opts, 'prefillWithLogits');

    const snapshot = this.#state.kvCache?.clone();
    if (!snapshot) {
      throw new Error('KV cache unavailable after prefill');
    }

    return {
      cache: snapshot,
      seqLen: this.#state.currentSeqLen,
      tokens: inputIds,
      logits,
      linearAttention: await cloneLinearAttentionRuntime(this.#state.linearAttentionRuntime),
    };
  }


  async *generateWithPrefixKV(prefix, prompt, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);

    // Apply snapshot
    this.#state.kvCache = prefix.cache.clone();
    if (this.#state.useGPU && this.#state.kvCache) {
      const device = getDevice();
      if (device) {
        this.#state.kvCache.setGPUContext({ device });
      }
    }
    if (
      hasLinearAttentionLayers(this.#state.modelConfig.layerTypes)
      && prefix.linearAttention == null
    ) {
      throw new Error(
        'Prefix snapshot is missing linear_attention recurrent state. ' +
        'Regenerate the prefix snapshot using the current runtime.'
      );
    }
    this.#state.linearAttentionRuntime = restoreLinearAttentionRuntime(
      this.#state.linearAttentionRuntime,
      prefix.linearAttention ?? null
    );
    this.#state.currentSeqLen = prefix.seqLen;

    this.#state.isGenerating = true;
    this.#state.decodeStepCount = 0;
    resetActiveExecutionPlan(this.#state);
    this.#state.stats.gpuTimePrefillMs = undefined;
    this.#state.stats.gpuTimeDecodeMs = undefined;
    this.#state.decodeRing?.reset();
    this.#state.stats.decodeRecordMs = 0;
    this.#state.stats.decodeSubmitWaitMs = 0;
    this.#state.stats.decodeReadbackWaitMs = 0;
    this.#state.stats.ttftMs = 0;
    const startTime = performance.now();

    const opts = resolveGenerateOptions(this.#state, options);

    try {
      const processedPrompt = resolvePromptInput(this.#state, prompt, opts.useChatTemplate, 'generateWithPrefixKV');

      const inputIds = this.#state.tokenizer.encode(processedPrompt);
      this._assertTokenIdsInRange(inputIds, 'generateWithPrefixKV.encode');
      const generatedIds = [...prefix.tokens, ...inputIds];
      const promptTokenCount = generatedIds.length;
      this.#state.stats.prefillTokens = inputIds.length;

      const prefillStart = performance.now();
      const prefillLogits = await this._prefill(inputIds, opts);
      this.#state.stats.prefillTimeMs = performance.now() - prefillStart;

      applyRepetitionPenalty(prefillLogits, generatedIds, opts.repetitionPenalty);
      const padTokenId = this.#state.tokenizer?.getSpecialTokens?.()?.pad;
      const firstToken = sample(prefillLogits, {
        temperature: opts.temperature,
        topP: opts.topP,
        topK: opts.topK,
        padTokenId,
        seed: opts.seed,
      });

      generatedIds.push(firstToken);
      this.#state.stats.ttftMs = performance.now() - startTime;

      const firstText = resolveTokenText(
        this.#state.tokenizer,
        [firstToken],
        `[${firstToken}]`,
        (tokens) => this.#state.tokenizer?.decode?.(tokens, true, false),
        (tokens) => this.#state.tokenizer?.decode?.(tokens, false, false)
      );
      yield firstText;
      if (options.onToken) options.onToken(firstToken, firstText);

      yield* this._runDecodeLoop(generatedIds, opts, options, {
        stopTokenIds: this.#state.modelConfig.stopTokenIds,
        eosToken: this.#state.tokenizer.getSpecialTokens?.()?.eos,
        stopSequenceStart: promptTokenCount,
        decodeToken: (tokenId) => this.#state.tokenizer.decode([tokenId], true, false),
        logBatchPath: false,
      });
      this.#state.stats.totalTimeMs = performance.now() - startTime;
    } finally {
      this._closeFinitenessFallbackWindow(opts);
      resetActiveExecutionPlan(this.#state);
      this.#state.isGenerating = false;
    }
  }

  // ==========================================================================
  // Internal Methods (Prefill, Decode, Helpers)
  // ==========================================================================

  async *_runDecodeLoop(generatedIds, opts, options, runtime) {
    const {
      stopTokenIds,
      eosToken,
      stopSequenceStart,
      decodeToken,
      logBatchPath = false,
      emitMode = 'text',
    } = runtime;

    let tokensGenerated = 1;
    markKernelCacheWarmed();

    const decodeStart = performance.now();
    const lmHead = this.#state.weights.get('lm_head');
    const embedBuffer = this.#state.weights.get('embed');
    const hasCpuWeights = isCpuWeightBuffer(lmHead)
      || isCpuWeightBuffer(embedBuffer)
      || lmHead instanceof Float32Array
      || embedBuffer instanceof Float32Array;
    const hasLinearLayers = hasLinearAttentionLayers(this.#state.modelConfig.layerTypes);
    const gpuSamplingAvailable = isGPUSamplingAvailable() && !hasCpuWeights;
    const executionPlan = opts.executionPlan;
    let useBatchPath = shouldUseBatchDecode({
      batchSize: executionPlan.batchSize,
      useGPU: this.#state.useGPU,
      gpuSamplingAvailable,
      disableMultiTokenDecode: executionPlan.disableMultiTokenDecode,
      disableCommandBatching: executionPlan.disableCommandBatching,
      isBdpaPagedLayout: this.#state.kvCache?.layout === 'bdpa_paged',
      finitenessFallbackWindowOpen: this._hasFinitenessFallbackWindow(),
    });
    if (!useBatchPath) {
      let reason = null;
      if (hasCpuWeights) reason = 'cpu_weights';
      else if (!this.#state.useGPU) reason = 'no_gpu';
      else if (!gpuSamplingAvailable) reason = 'no_gpu_sampling';
      else if (executionPlan.disableCommandBatching) reason = 'command_batching_disabled';
      else if (executionPlan.disableMultiTokenDecode) reason = 'multi_token_decode_disabled';
      else if (executionPlan.batchSize <= 1) reason = 'batch_size_1';
      else if (this.#state.kvCache?.layout === 'bdpa_paged') reason = 'bdpa_paged_layout';
      else if (this._hasFinitenessFallbackWindow()) reason = 'finiteness_fallback_window';
      this.#state.stats.decodeMode = opts.speculation?.mode === 'self' ? 'self_speculation' : 'single_token';
      this.#state.stats.batchGuardReason = reason;
    } else {
      this.#state.stats.decodeMode = 'batched_gpu';
      this.#state.stats.batchGuardReason = null;
    }

    const readbackInterval = executionPlan.readbackInterval;
    const intervalBatches = readbackInterval == null ? 1 : readbackInterval;
    const padTokenId = this.#state.tokenizer?.getSpecialTokens?.()?.pad;

    const decodeSingleTokenViaLogits = async () => this._decodeNextTokenViaLogits(generatedIds, opts);

    if (logBatchPath && useBatchPath) {
      log.debug(
        'Pipeline',
        `Using batch decode path with batchSize=${executionPlan.batchSize}, stopCheckMode=${executionPlan.stopCheckMode}, readbackInterval=${readbackInterval}`
      );
    }

    while (tokensGenerated < opts.maxTokens) {
      if (options.signal?.aborted) break;
      if (this._hasFinitenessFallbackWindow() && useBatchPath) {
        useBatchPath = false;
      }

      if (useBatchPath) {
        const remaining = opts.maxTokens - tokensGenerated;
        const thisBatchSize = Math.min(executionPlan.batchSize * intervalBatches, remaining);
        const lastToken = generatedIds[generatedIds.length - 1];

        try {
          const batchResult = await this._generateNTokensGPU(lastToken, thisBatchSize, generatedIds, opts);
          let batchTokens = [];
          let hitStop = false;
          for (const tokenId of batchResult.tokens) {
            if (isStopToken(tokenId, stopTokenIds, eosToken)) {
              hitStop = true;
              break;
            }
            generatedIds.push(tokenId);
            tokensGenerated++;
            if (emitMode === 'token') {
              yield tokenId;
              if (options.onToken) options.onToken(tokenId, '');
              batchTokens.push({ id: tokenId, text: '' });
            } else {
              const tokenText = decodeToken(tokenId);
              yield tokenText;
              if (options.onToken) options.onToken(tokenId, tokenText);
              batchTokens.push({ id: tokenId, text: tokenText });
            }
            if (batchTokens.length === executionPlan.batchSize) {
              if (options.onBatch) options.onBatch(batchTokens);
              batchTokens = [];
            }
          }
          if (batchTokens.length > 0 && options.onBatch) options.onBatch(batchTokens);
          if (hitStop || batchResult.actualCount < thisBatchSize) break;
          if (opts.stopSequences.length > 0) {
            const fullText = this.#state.tokenizer.decode(generatedIds.slice(stopSequenceStart), false);
            if (opts.stopSequences.some((seq) => fullText.endsWith(seq))) break;
          }
        } catch (error) {
          log.warn('Pipeline', `Batch decode failed, falling back to single-token: ${error}`);
          useBatchPath = false;
          let nextToken;
          try {
            nextToken = await decodeSingleTokenViaLogits();
          } catch (singleTokenError) {
            if (this._shouldUseFinitenessFallback(singleTokenError, `decode-batch-step-${tokensGenerated}`)) {
              log.warn('Pipeline', `FinitenessGuard caught NaN/Inf at batch step ${tokensGenerated}. Truncating KV cache and retrying token with F32 precision.`);
              nextToken = await this._retryDecodeStepWithFinitenessWindow(
                generatedIds,
                opts,
                `decode-batch-step-${tokensGenerated}`
              );
            } else {
              throw singleTokenError;
            }
          }
          generatedIds.push(nextToken);
          tokensGenerated++;
          if (emitMode === 'token') {
            yield nextToken;
            if (options.onToken) options.onToken(nextToken, '');
          } else {
            const tokenText = decodeToken(nextToken);
            yield tokenText;
            if (options.onToken) options.onToken(nextToken, tokenText);
          }
          this._consumeFinitenessFallbackToken(opts);
          if (isStopToken(nextToken, stopTokenIds, eosToken)) break;
        }
      } else if (opts.speculation?.mode === 'self') {
        // Self-speculation: decode 2 tokens per iteration (base + speculative).
        // Same-model speculation always accepts under greedy because the model
        // is deterministic — both base and speculative use the same weights and
        // state. The benefit is amortizing per-iteration overhead for models
        // where batch decode is disabled (e.g., linear attention).
        const doSpecDecode = hasLinearLayers
          ? () => this._decodeStep(generatedIds, opts)
          : decodeSingleTokenViaLogits;

        // Base decode
        let baseToken;
        try {
          baseToken = await doSpecDecode();
        } catch (error) {
          if (this._shouldUseFinitenessFallback(error, `spec-base-${tokensGenerated}`)) {
            log.warn('Pipeline', `FinitenessGuard caught NaN/Inf at step ${tokensGenerated} (speculation:base). Retrying.`);
            baseToken = await this._retryDecodeStepWithFinitenessWindow(generatedIds, opts, `spec-base-${tokensGenerated}`);
          } else {
            throw error;
          }
        }
        generatedIds.push(baseToken);
        tokensGenerated++;
        if (emitMode === 'token') {
          yield baseToken;
          if (options.onToken) options.onToken(baseToken, '');
        } else {
          const text = decodeToken(baseToken);
          yield text;
          if (options.onToken) options.onToken(baseToken, text);
        }
        this._consumeFinitenessFallbackToken(opts);

        if (isStopToken(baseToken, stopTokenIds, eosToken)) break;
        if (tokensGenerated >= opts.maxTokens) break;
        if (opts.stopSequences.length > 0) {
          const fullText = this.#state.tokenizer.decode(generatedIds.slice(stopSequenceStart), false);
          if (opts.stopSequences.some((seq) => fullText.endsWith(seq))) break;
        }

        // Speculative decode (tokens=1)
        let specToken;
        try {
          specToken = await doSpecDecode();
        } catch (error) {
          if (this._shouldUseFinitenessFallback(error, `spec-extra-${tokensGenerated}`)) {
            log.warn('Pipeline', `FinitenessGuard caught NaN/Inf at step ${tokensGenerated} (speculation:spec). Retrying.`);
            specToken = await this._retryDecodeStepWithFinitenessWindow(generatedIds, opts, `spec-extra-${tokensGenerated}`);
          } else {
            throw error;
          }
        }
        generatedIds.push(specToken);
        tokensGenerated++;
        this.#state.stats.speculationAttempts = (this.#state.stats.speculationAttempts ?? 0) + 1;
        this.#state.stats.speculationAccepted = (this.#state.stats.speculationAccepted ?? 0) + 1;
        if (emitMode === 'token') {
          yield specToken;
          if (options.onToken) options.onToken(specToken, '');
        } else {
          const text = decodeToken(specToken);
          yield text;
          if (options.onToken) options.onToken(specToken, text);
        }
        this._consumeFinitenessFallbackToken(opts);

        if (opts.debug || opts.benchmark) {
          const elapsedMs = performance.now() - decodeStart;
          const tokPerSec = (tokensGenerated / elapsedMs) * 1000;
          log.debug('Decode', `#${tokensGenerated} speculation:self (${tokPerSec.toFixed(2)} tok/s avg)`);
        }

        if (isStopToken(specToken, stopTokenIds, eosToken)) break;
        if (opts.stopSequences.length > 0) {
          const fullText = this.#state.tokenizer.decode(generatedIds.slice(stopSequenceStart), false);
          if (opts.stopSequences.some((seq) => fullText.endsWith(seq))) break;
        }
      } else {
        const tokenStart = performance.now();
        let nextToken;
        try {
          nextToken = hasLinearLayers
            ? await this._decodeStep(generatedIds, opts)
            : await decodeSingleTokenViaLogits();
        } catch (error) {
          if (this._shouldUseFinitenessFallback(error, `decode-step-${tokensGenerated}`)) {
            log.warn('Pipeline', `FinitenessGuard caught NaN/Inf at step ${tokensGenerated}. Truncating KV cache and retrying token with F32 precision.`);
            nextToken = await this._retryDecodeStepWithFinitenessWindow(
              generatedIds,
              opts,
              `decode-step-${tokensGenerated}`
            );
          } else {
            throw error;
          }
        }
        const tokenTime = performance.now() - tokenStart;
        generatedIds.push(nextToken);
        tokensGenerated++;
        const tokenText = emitMode === 'token' ? '' : decodeToken(nextToken);
        if (emitMode === 'token') {
          yield nextToken;
          if (options.onToken) options.onToken(nextToken, '');
        } else {
          yield tokenText;
          if (options.onToken) options.onToken(nextToken, tokenText);
        }
        this._consumeFinitenessFallbackToken(opts);

        if (opts.debug || opts.benchmark) {
          const elapsedMs = performance.now() - decodeStart;
          const tokPerSec = (tokensGenerated / elapsedMs) * 1000;
          log.debug('Decode', `#${tokensGenerated} "${tokenText}" ${tokenTime.toFixed(0)}ms (${tokPerSec.toFixed(2)} tok/s avg)`);
        }

        if (isStopToken(nextToken, stopTokenIds, eosToken)) break;
        if (opts.stopSequences.length > 0) {
          const fullText = this.#state.tokenizer.decode(generatedIds.slice(stopSequenceStart), false);
          if (opts.stopSequences.some((seq) => fullText.endsWith(seq))) break;
        }
      }
    }

    this.#state.stats.decodeTimeMs = performance.now() - decodeStart;
    this.#state.stats.tokensGenerated = tokensGenerated;
    this.#state.stats.decodeTokens = tokensGenerated;
  }

  async _prefillToHidden(inputIds, opts) {
    // Internal-only: reuse the main prefill implementation but stop before logits.
    return this._prefill(inputIds, { ...opts, _returnHidden: true });
  }


  async _prefill(inputIds, opts) {
    const numTokens = inputIds.length;
    const config = this.#state.modelConfig;
    const startPos = this.#state.currentSeqLen;
    const returnHidden = opts?._returnHidden === true;
    this.#state.stats.gpuTimePrefillMs = undefined;

    if (startPos === 0 && hasLinearAttentionLayers(config.layerTypes)) {
      this.#state.linearAttentionRuntime = resetLinearAttentionRuntime(this.#state.linearAttentionRuntime);
    }
    if (startPos === 0) {
      for (const [, convState] of this.#state.convLayerStates) {
        if (convState.convStateGPU && convState.hiddenSize && convState.kernelSize) {
          uploadData(convState.convStateGPU, new Float32Array(convState.hiddenSize * (convState.kernelSize - 1)));
        }
      }
    }

    const embedBufferRaw = this.#state.weights.get('embed');
    if (!isGpuBufferInstance(embedBufferRaw) && !isWeightBuffer(embedBufferRaw) && !isCpuWeightBuffer(embedBufferRaw) && !(embedBufferRaw instanceof Float32Array)) {
      throw new Error('Embed buffer not found or not a supported buffer type');
    }
    const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
    const embedDtype = isCpuWeightBuffer(embedBufferRaw)
      ? embedBufferRaw.dtype
      : getWeightDtype(embedBufferRaw);
    if (opts.debug) {
      const embedSize = isGpuBufferInstance(embedBuffer) ? embedBuffer.size : 'N/A';
      log.debug('Pipeline', `Embed buffer: type=${embedBuffer?.constructor?.name}, size=${embedSize}, dtype=${embedDtype}`);
    }

    const device = getDevice();
    const useCheckpoints = opts.debugLayers && opts.debugLayers.length > 0;
    const disableCommandBatching = opts.disableCommandBatching === true
      || opts.debug === true
      || this.#state.kvCache?.layout === 'bdpa_paged';
    const createRecorder = (label) => {
      if (!device || disableCommandBatching) return undefined;
      return opts.profile ? createProfilingRecorder(label) : createCommandRecorder(label);
    };
    const recorder = createRecorder('prefill');
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;
    const context = buildLayerContext(
      this.#state,
      recorder,
      false,
      opts.debugLayers,
      debugCheckBuffer,
      opts.executionPlan
    );
    context.currentTokenIds = inputIds;
    let gpuTimePrefillMs = 0;
    let hasGpuTimePrefill = false;
    const recordProfile = async (rec) => {
      if (!opts.profile || !rec?.isProfilingEnabled()) return;
      const timings = await rec.resolveProfileTimings();
      const total = sumProfileTimings(timings);
      if (total !== null) {
        gpuTimePrefillMs += total;
        hasGpuTimePrefill = true;
      }
      if (timings) {
        log.warn('Profile', `Prefill (${rec.label}):`);
        log.warn('Profile', CommandRecorder.formatProfileReport(timings));
      }
    };

    const benchmarkSubmits = opts.debug;
    if (benchmarkSubmits) {
      setTrackSubmits(true);
      resetSubmitStats();
    }

    const activationDtype = opts.executionPlan?.activationDtype ?? this._getEffectiveActivationDtype();
    const activationBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
    let hiddenStates = await embed(inputIds, embedBuffer, {
      hiddenSize: config.hiddenSize,
      vocabSize: config.vocabSize,
      scaleEmbeddings: config.scaleEmbeddings,
      debug: opts.debug,
      recorder,
      transpose: this.#state.embeddingTranspose,
      debugProbes: this.#state.runtimeConfig.shared.debug.probes,
      operatorDiagnostics: this.#state.operatorDiagnostics,
      activationDtype,
      embeddingDtype: selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', { dtype: embedDtype }),
    });

    if (opts.debug && isGpuBufferInstance(hiddenStates)) {
      if (recorder) {
        await recorder.submitAndWait();
        await recordProfile(recorder);
      }
      const debugReadbackSize = this.#state.runtimeConfig.shared.debug.pipeline.readbackSampleSize;
      const sample = await readBuffer(hiddenStates, Math.min(debugReadbackSize, hiddenStates.size));
      const f32 = decodeReadback(sample, activationDtype);
      const nanCount = f32.filter(x => !Number.isFinite(x)).length;
      let maxAbs = 0;
      for (let i = 0; i < f32.length; i++) {
        const abs = Math.abs(f32[i]);
        if (abs > maxAbs) maxAbs = abs;
      }
      const first8 = Array.from(f32).slice(0, 8).map(x => x.toFixed(4)).join(', ');
      log.debug('Pipeline', `After embed: buffer.label=${hiddenStates.label}, buffer.size=${hiddenStates.size}, maxAbs=${maxAbs.toFixed(4)}`);
      log.debug('Pipeline', `After embed first8=[${first8}], nan=${nanCount}/${f32.length}`);
    }

    if (opts.debug) {
      log.debug('Pipeline', `LAYER_LOOP_START: numLayers=${config.numLayers}, useGPU=${context.useGPU}`);
    }

    if (this.#state.finitenessBuffer) {
      const device = getDevice();
      if (device) {
        device.queue.writeBuffer(this.#state.finitenessBuffer, 0, new Uint32Array([0, 0, 0, 0]));
      }
    }

    let currentRecorder = recorder;

    let currentHiddenBuffer = hiddenStates.buffer;
    for (let l = 0; l < config.numLayers; l++) {
      context.recorder = currentRecorder;

      const prevBuffer = currentHiddenBuffer;
      const layerOutput = await processLayer(l, currentHiddenBuffer, numTokens, true, context);
      if (!isGpuBufferInstance(layerOutput)) throw new Error('Expected GPUBuffer from processLayer');
      currentHiddenBuffer = layerOutput;

      const isCheckpoint = useCheckpoints && opts.debugLayers?.includes(l);

      if (isCheckpoint && currentRecorder) {
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
        currentRecorder = undefined;
      }

      const shouldDebug = opts.debug && currentHiddenBuffer && (!recorder || isCheckpoint);
      if (shouldDebug && !currentRecorder) {
        const device = getDevice();
        if (device) {
          if (allowReadback(`pipeline.prefill.layer-${l}`)) {
            try {
              const sampleSize = config.hiddenSize * activationBytes;
              const lastTokenOffset = (numTokens - 1) * config.hiddenSize * activationBytes;
              const readback = await readBufferSlice(currentHiddenBuffer, lastTokenOffset, sampleSize);
              const data = decodeReadback(readback, activationDtype);
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
              log.debug('Pipeline', `LAYER_${l}_LAST[pos=${numTokens - 1}]: min=${min.toFixed(3)}, max=${max.toFixed(3)}, maxAbs=${maxAbs.toFixed(2)}, sample=[${sample}]`);
            } catch (e) {
              log.debug('Pipeline', `LAYER_${l}_LAST: error reading buffer: ${e}`);
            }
          }
        }
      }

      if (isCheckpoint && useCheckpoints && l < config.numLayers - 1) {
        currentRecorder = createRecorder('prefill-cont');
      }

      if (prevBuffer !== currentHiddenBuffer) {
        if (currentRecorder) {
          currentRecorder.trackTemporaryBuffer(prevBuffer);
        } else {
          releaseBuffer(prevBuffer);
        }
      }
    }

    if (this.#state.finitenessBuffer) {
      if (currentRecorder) {
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
        currentRecorder = undefined;
      }
      const isInfiniteData = await readBuffer(this.#state.finitenessBuffer, 16);
      const u32 = new Uint32Array(isInfiniteData.buffer, isInfiniteData.byteOffset, 4);
      const finitenessStatus = parseFinitenessStatusWords(u32, 0);
      if (finitenessStatus.triggered) {
        if (isGpuBufferInstance(currentHiddenBuffer)) {
          releaseBuffer(currentHiddenBuffer);
        }
        throw new FinitenessError(`F16 bounds exceeded during prefill${finitenessStatus.metadata}`);
      }
    }

    if (benchmarkSubmits) {
      logSubmitStats(`Prefill (${numTokens} tokens, ${config.numLayers} layers)`);
      setTrackSubmits(false);
    }

    if (opts.debug) {
      log.debug('Pipeline', `LAYER_LOOP_DONE, currentHiddenBuffer type=${currentHiddenBuffer?.constructor?.name}`);
      if (currentHiddenBuffer && allowReadback('pipeline.prefill.final-hidden')) {
        const lastTokenOffset = (numTokens - 1) * config.hiddenSize * activationBytes;
        const sampleSize = config.hiddenSize * activationBytes;
        const data = decodeReadback(
          await readBufferSlice(currentHiddenBuffer, lastTokenOffset, sampleSize),
          activationDtype
        );
        const nanCount = Array.from(data).filter(x => !Number.isFinite(x)).length;
        const nonZero = Array.from(data).filter(x => Number.isFinite(x) && x !== 0).slice(0, 5);
        log.debug('Pipeline', `FINAL_HIDDEN[pos=${numTokens - 1}]: nan=${nanCount}/${data.length}, sample=[${nonZero.map(x => x.toFixed(4)).join(', ')}]`);
      }
    }

    if (hasGpuTimePrefill) {
      this.#state.stats.gpuTimePrefillMs = gpuTimePrefillMs;
    }

    if (returnHidden) {
      return {
        numTokens,
        config,
        startPos,
        activationDtype,
        activationBytes,
        currentRecorder,
        recordProfile,
        debugCheckBuffer,
        currentHiddenBuffer,
      };
    }


    let lastLogits;
    let logitsVocabSize = config.vocabSize;
    let usedRecordedLogits = false;
    const lmHead = this.#state.weights.get('lm_head');
    const canRecordLogits = !!currentRecorder
      && !!lmHead
      && !isCpuWeightBuffer(lmHead)
      && !this.#state.disableRecordedLogits
      && numTokens === 1;
    if (currentRecorder && canRecordLogits) {
      const recorded = await recordLogitsGPU(
        currentRecorder,
        currentHiddenBuffer,
        numTokens,
        getLogitsWeights(this.#state),
        getLogitsConfig(this.#state),
        this.#state.operatorDiagnostics
      );
      logitsVocabSize = recorded.vocabSize;
      usedRecordedLogits = true;

      await currentRecorder.submitAndWait();
      await recordProfile(currentRecorder);

      const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: recorded.logitsDtype });
      const lastLogitsSize = logitsVocabSize * logitsBytes;
      const lastLogitsOffset = (numTokens - 1) * lastLogitsSize;
      const logitsData = await readBufferSlice(recorded.logitsBuffer, lastLogitsOffset, lastLogitsSize);
      releaseBuffer(recorded.logitsBuffer);
      lastLogits = decodeReadback(logitsData, recorded.logitsDtype);

      const health = getLogitsHealth(lastLogits);
      if (health.nanCount > 0 || health.infCount > 0 || health.nonZeroCount === 0) {
        log.warn(
          'Logits',
          `Recorded logits invalid (nan=${health.nanCount} inf=${health.infCount} nonZero=${health.nonZeroCount}, maxAbs=${health.maxAbs.toFixed(3)}); recomputing without recorder.`
        );
        this.#state.disableRecordedLogits = true;
        this.#state.disableFusedDecode = true;
        const fallbackLogits = await computeLogits(
          currentHiddenBuffer,
          numTokens,
          getLogitsWeights(this.#state),
          getLogitsConfig(this.#state),
          this.#state.useGPU,
          this.#state.debugFlags,
          undefined,
          debugCheckBuffer,
          this.#state.runtimeConfig.shared.debug.probes,
          { lastPositionOnly: true },
          this.#state.operatorDiagnostics
        );
        const fallbackHealth = getLogitsHealth(fallbackLogits);
        if (fallbackHealth.nanCount > 0 || fallbackHealth.infCount > 0 || fallbackHealth.nonZeroCount === 0) {
          throw new Error(
            `[Logits] Fallback logits invalid (nan=${fallbackHealth.nanCount} inf=${fallbackHealth.infCount} nonZero=${fallbackHealth.nonZeroCount}, maxAbs=${fallbackHealth.maxAbs.toFixed(3)}). ` +
            'This indicates upstream kernel output is NaN/Inf (often prefill attention/matmul).'
          );
        }
        logitsVocabSize = config.vocabSize;
        usedRecordedLogits = false;
        lastLogits = fallbackLogits.length === logitsVocabSize
          ? fallbackLogits
          : extractLastPositionLogits(fallbackLogits, numTokens, logitsVocabSize);
      }

      releaseBuffer(currentHiddenBuffer);
    } else {
      if (currentRecorder) {
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
      }
      const logits = await computeLogits(
        currentHiddenBuffer,
        numTokens,
        getLogitsWeights(this.#state),
        getLogitsConfig(this.#state),
        this.#state.useGPU,
        this.#state.debugFlags,
        undefined,
        debugCheckBuffer,
        this.#state.runtimeConfig.shared.debug.probes,
        { lastPositionOnly: true },
        this.#state.operatorDiagnostics
      );

      lastLogits = logits.length === logitsVocabSize
        ? logits
        : extractLastPositionLogits(logits, numTokens, logitsVocabSize);
      releaseBuffer(currentHiddenBuffer);
    }

    this.#state.currentSeqLen = startPos + numTokens;

    if (usedRecordedLogits) {
      if (logitsVocabSize < config.vocabSize) {
        const padded = new Float32Array(config.vocabSize);
        padded.set(lastLogits);
        padded.fill(-Infinity, logitsVocabSize);
        lastLogits = padded;
      }
      if (config.finalLogitSoftcapping != null) {
        applySoftcapping(lastLogits, config.finalLogitSoftcapping);
      }
    }

    if (opts.debug) {
      logitsSanity(lastLogits, 'Prefill', (tokens) => resolveTokenText(this.#state.tokenizer, tokens));
    }

    if (opts.debug) {
      if (this.#state.kvCache?.hasGPUCache?.()) {
        log.debug('Pipeline', `KV cache active after prefill: seqLen=${this.#state.kvCache.getKeyCache(0)?.constructor.name ?? '?'}`);
      } else {
        log.warn('Pipeline', `KV cache NOT active after prefill! hasGPUCache=${this.#state.kvCache?.hasGPUCache?.()}`);
      }
    }

    return lastLogits;
  }


  async _decodeStep(currentIds, opts) {
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;
    return decodeStep(this.#state, currentIds, opts, this._getDecodeHelpers(debugCheckBuffer));
  }

  async decodeStepLogits(currentIds, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    resetActiveExecutionPlan(this.#state);

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    return this._decodeStepToLogits(currentIds, opts);
  }

  async advanceWithToken(tokenId, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating) throw new Error('Generation already in progress');
    resetActiveExecutionPlan(this.#state);

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;

    this._assertTokenIdInRange(tokenId, 'advanceWithToken');
    await advanceWithToken(this.#state, tokenId, opts, this._getDecodeHelpers(debugCheckBuffer));
  }

  async advanceWithTokenAndEmbedding(tokenId, options = {}) {
    if (!this.#state.isLoaded) throw new Error('Model not loaded');
    if (this.#state.isGenerating) throw new Error('Generation already in progress');
    resetActiveExecutionPlan(this.#state);

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    const embeddingMode = resolveAdvanceEmbeddingMode(this.#state, options);
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;

    this._assertTokenIdInRange(tokenId, 'advanceWithTokenAndEmbedding');
    return runAdvanceWithTokenAndEmbedding(
      this.#state,
      tokenId,
      opts,
      this._getDecodeHelpers(debugCheckBuffer),
      embeddingMode
    );
  }

  async _generateNTokensGPU(startToken, N, currentIds, opts) {
    const debugCheckBuffer = this.#state.debug
      ? (buffer, label, numTokens, expectedDim) =>
        debugCheckBufferHelper(this.#state, buffer, label, numTokens, expectedDim)
      : undefined;
    return generateNTokensGPU(this.#state, startToken, N, currentIds, opts, this._getDecodeHelpers(debugCheckBuffer));
  }
}
