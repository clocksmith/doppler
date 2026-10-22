import { lookupDraft, verifySpeculativeTokens } from './speculative-runtime.js';
import { isGPUSamplingAvailable } from '../../../../gpu/kernels/sample.js';
import { samplingSeedForToken } from '../../../token-sampling.js';
import { matchesStopSequence } from '../stopping.js';
import { markWarmed as markKernelCacheWarmed } from '../../../../gpu/kernel-selection-cache.js';
import { log, trace } from '../../../../debug/index.js';
import {
  CAPTURE_LEVELS,
  createDefaultCaptureConfig,
  validateCaptureConfig,
} from '../../../../debug/index.js';
import { validateCallTimeOptions } from '../../../../config/param-validator.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { getTopK } from '../sampling.js';
import { isStopToken } from '../init.js';
import { OperatorEventEmitter } from '../operator-events.js';
import { isCpuWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { decodeStep, advanceWithToken as runAdvanceWithToken, generateNTokensGPU, shouldUseBatchDecode, advanceWithTokenAndEmbedding as runAdvanceWithTokenAndEmbedding } from '../generator-steps.js';
import { debugCheckBuffer as debugCheckBufferHelper, resolvePerLayerInputsSession } from './session-context.js';
import { resolveGenerateOptions, resolveAdvanceEmbeddingMode, settleGenerationStats } from '../generator-runtime.js';
import { createTsirFixtureState } from '../tsir-fixture-writer.js';
import { resetActiveExecutionPlan, resolveMaxBatchDecodeTokens, resolveActiveExecutionPlan } from '../execution-plan.js';
import { hasLinearAttentionLayers } from '../linear-attention.js';
import { createPleBufferCache, prefetchPerLayerRow, getPleHotVocabularyRuntime, hasGpuSplitPerLayerInputEmbeddings, hasRangeBackedPerLayerInputEmbeddings } from '../per-layer-inputs.js';
import { shouldDisableBatchDecodeAfterShortBatch, resolveHotVocabularyBatchDecodeAvailability, primePleDecodeRuntimeCache, resolveTokenText, usesReplayPrefillDecode, assertIncrementalDecodeSupport, summarizeExecutionPlan } from '../generator-decode-policy.js';
import { getExperimentalIntentBundleModule, resolveSuppressedSamplingTokenIds } from './text.js';
import { assertBatchRetryInPlaceAllowed } from './batch-retry-policy.js';

function prepareGenerationRuntime(generator, opts, options) {
  generator._resetReplayPrefillRuntimeState();
  generator._state.isGenerating = true;
  generator._resetDecodeRuntimeState();
  delete generator._state.stats.outputUsage;
  generator._state.stats.speculationAttempts = 0;
  generator._state.stats.speculationAccepted = 0;
  generator._state.stats.gpuTimePrefillMs = undefined;
  generator._state.stats.gpuTimeDecodeMs = undefined;
  generator._state.stats.decodeRecordMs = 0;
  generator._state.stats.decodeRecordOps = 0;
  generator._state.stats.decodeRecordPasses = 0;
  generator._state.stats.decodeRecordOpLabels = {};
  generator._state.stats.decodeSubmitWaitMs = 0;
  generator._state.stats.decodeReadbackWaitMs = 0;
  generator._state.stats.decodeReadbackMapWaitMs = 0;
  generator._state.stats.decodeReadbackCleanupMs = 0;
  generator._state.stats.decodeReadbackCopyMs = 0;
  generator._state.stats.prefillRecordMs = 0;
  generator._state.stats.prefillRecordOps = 0;
  generator._state.stats.prefillRecordPasses = 0;
  generator._state.stats.prefillRecordOpLabels = {};
  generator._state.stats.prefillSubmitWaitMs = 0;
  generator._state.stats.singleTokenSubmitWaitMs = 0;
  generator._state.stats.singleTokenReadbackWaitMs = 0;
  generator._state.stats.singleTokenReadbackMapWaitMs = 0;
  generator._state.stats.singleTokenReadbackCleanupMs = 0;
  generator._state.stats.singleTokenReadbackCopyMs = 0;
  generator._state.stats.singleTokenOrchestrationMs = 0;
  generator._state.stats.pleHotVocabularyHits = 0;
  generator._state.stats.pleHotVocabularyMisses = 0;
  generator._state.stats.plePreparedTokenCacheHits = 0;
  generator._state.stats.plePreparedTokenCacheMisses = 0;
  generator._state.stats.plePreparedTokenCacheEntries = 0;
  generator._state.stats.plePreparedTokenCacheBytes = 0;
  generator._state.stats.decodeMode = null;
  generator._state.stats.batchGuardReason = null;
  generator._state.stats.stopReason = null;
  generator._state.stats.stopTokenId = null;
  generator._state.stats.ttftMs = 0;
  generator._state.stats.prefillTimeMs = 0;
  generator._state.stats.decodeTimeMs = 0;
  generator._state.stats.prefillTokens = 0;
  generator._state.stats.tokensGenerated = 0;
  generator._state.stats.decodeTokens = 0;
  generator._state.stats.totalTimeMs = 0;
  const startTime = performance.now();

  opts.onLogits = typeof options.onLogits === 'function' ? options.onLogits : null;
  // Validate and normalize sampling parameters through single source of truth
  const samplingConfig = opts;
  opts.suppressTokenIds = resolveSuppressedSamplingTokenIds(generator._state, samplingConfig);
  const diagnosticsEnabled = options?.diagnostics?.enabled === true
    || generator._state.runtimeConfig?.shared?.harness?.mode === 'diagnose';
  const tsirFixtureCfg = generator._state.runtimeConfig?.shared?.harness?.tsirFixture ?? null;
  if (diagnosticsEnabled || tsirFixtureCfg) {
    const captureConfig = {
      ...createDefaultCaptureConfig(),
      enabled: true,
      defaultLevel: CAPTURE_LEVELS.SLICE,
      ...(options?.diagnostics?.captureConfig ?? {}),
    };
    validateCaptureConfig(captureConfig);
    generator._state.operatorDiagnostics = {
      enabled: diagnosticsEnabled === true,
      captureConfig,
      emitter: diagnosticsEnabled ? new OperatorEventEmitter({
        modelHash: generator._state.manifest?.modelId ?? null,
        runtimeConfigHash: generator._state.resolvedKernelPath?.id ?? null,
        executionPlanHash: opts.executionPlan?.id ?? null,
      }) : null,
      tsirFixture: createTsirFixtureState(tsirFixtureCfg),
    };
  }
  return startTime;
}

async function samplePrefillToken(generator, prefillLogits, generatedIds, inputIds, opts, prefillStartSeqLen, warn) {
  try {
    return generator._sampleNextTokenFromLogits(prefillLogits, generatedIds, opts);
  } catch (error) {
    if (!generator._shouldUseFinitenessFallback(error, 'prefill-sample')) {
      throw error;
    }
    if (warn) log.warn('Pipeline', 'FinitenessGuard caught non-finite prefill logits at sampling. Retrying with F32 precision.');
    prefillLogits = await generator._retryWithPersistentFinitenessFallback(
      opts,
      'prefill-sample',
      opts.maxTokens,
      () => generator._prefill(inputIds, opts),
      prefillStartSeqLen
    );
    return generator._sampleNextTokenFromLogits(prefillLogits, generatedIds, opts);
  }
}

function finishGenerationRuntime(generator, options, startTime, generatedIds, promptTokenCount) {
  settleGenerationStats(generator._state.stats, startTime,
    Math.max(0, generatedIds.length - promptTokenCount), options.signal?.aborted === true);
  generator._state.stats.batching = { ...generator._state.batchingStats };
  resetActiveExecutionPlan(generator._state);
  generator._state.stats.operatorDiagnostics = generator._state.operatorDiagnostics?.emitter
    ? {
      enabled: true,
      timeline: generator._state.operatorDiagnostics.emitter.getTimeline(),
      recordCount: generator._state.operatorDiagnostics.emitter.length,
    }
    : null;
  generator._state.operatorDiagnostics = null;
  generator._state.isGenerating = false;
}

export async function* _generateTokensInternal(prompt, options = {}, mode = 'text') {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);
    const opts = resolveGenerateOptions(this._state, options);
    const startTime = prepareGenerationRuntime(this, opts, options);
    let generatedIds = [];
    let promptTokenCount = 0;
    const activePlan = opts.executionPlan ?? resolveActiveExecutionPlan(this._state);
    this._state.stats.executionPlan = {
      primary: summarizeExecutionPlan(this._state.executionPlanState?.primaryPlan ?? null),
      fallback: summarizeExecutionPlan(this._state.executionPlanState?.fallbackPlan ?? null),
      activePlanIdAtStart: activePlan?.id ?? null,
      finalActivePlanId: activePlan?.id ?? null,
      transitions: [],
    };
    this._state.stats.kernelPathId = activePlan?.kernelPathId ?? this._state.resolvedKernelPath?.id ?? null;
    this._state.stats.kernelPathSource = activePlan?.kernelPathSource ?? this._state.kernelPathSource ?? 'none';

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
      const prefillStartSeqLen = this._state.currentSeqLen;
      const prefillStart = performance.now();
      const { inputIds, logits: initialPrefillLogits } = await this._prefillPromptToLogits(prompt, opts, 'generate');
      const prefillLogits = initialPrefillLogits;
      this._state.stats.prefillTimeMs = performance.now() - prefillStart;
      this._assertTokenIdsInRange(inputIds, 'generate.prefillTokens');
      generatedIds = [...inputIds];
      promptTokenCount = inputIds.length;
      this._state.stats.prefillTokens = opts.promptTokenCount = inputIds.length;

      if (opts.debug) {
        log.debug('Pipeline', `Input: ${inputIds.length} tokens`);
      }

      const intentBundleConfig = this._state.runtimeConfig.shared.intentBundle;
      const intentBundle = intentBundleConfig?.bundle;
      const expectedTopK = intentBundle?.payload?.expectedTopK
        ?? intentBundle?.payload?.expected_top_k;
      const maxDriftThreshold = intentBundle?.constraints?.maxDriftThreshold
        ?? intentBundle?.constraints?.max_drift_threshold;

      if (intentBundleConfig?.enabled && Array.isArray(expectedTopK) && expectedTopK.length > 0) {
        const { enforceLogitDrift } = await getExperimentalIntentBundleModule();
        const actualTopK = getTopK(
          prefillLogits,
          expectedTopK.length,
          (tokens) => resolveTokenText(this._state.tokenizer, tokens),
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
          (tokens) => resolveTokenText(this._state.tokenizer, tokens)
        );
        log.debug('Pipeline', `After rep penalty top-5: ${topAfterPenalty.map(t => `"${t.text}"(${(t.prob * 100).toFixed(1)}%)`).join(', ')}`);
      }

      const firstToken = await samplePrefillToken(this, prefillLogits, generatedIds, inputIds, opts, prefillStartSeqLen, true);

      if (opts.debug) {
        const firstTokenText = resolveTokenText(this._state.tokenizer, [firstToken], `[${firstToken}]`, (tokens) => this._state.tokenizer?.decode?.(tokens, true, false));
        log.debug('Pipeline', `First token sampled: id=${firstToken} text="${firstTokenText}"`);
      }

      const stopTokenIds = this._state.modelConfig.stopTokenIds;
      const eosToken = this._state.tokenizer.getSpecialTokens?.()?.eos;
      const stopSequenceStart = inputIds.length;
      generatedIds.push(firstToken);
      this._state.stats.ttftMs = performance.now() - startTime;

      const decodeToken = (tokenId) => resolveTokenText(
        this._state.tokenizer,
        [tokenId],
        `[${tokenId}]`,
        (tokens) => this._state.tokenizer?.decode?.(tokens, true, false),
        (tokens) => this._state.tokenizer?.decode?.(tokens, false, false)
      );
      const decodeRuntime = {
        stopTokenIds,
        eosToken,
        stopSequenceStart,
        decodeToken,
        logBatchPath: opts.debug,
        emitMode: mode,
      };

      yield* emitToken(this, firstToken, decodeToken);

      if (options.signal?.aborted || this._shouldStopAfterAppendedToken(generatedIds, firstToken, opts, decodeRuntime)) {
        this._recordStopReason(options.signal?.aborted ? 'aborted'
          : isStopToken(firstToken, decodeRuntime.stopTokenIds, decodeRuntime.eosToken) ? 'stop-token' : 'stop-sequence',
          options.signal?.aborted ? null : firstToken);
        this._state.stats.decodeTimeMs = 0;
        this._state.stats.tokensGenerated = 1;
        this._state.stats.decodeTokens = 0;
      } else {
        yield* this._runDecodeLoop(generatedIds, opts, options, decodeRuntime);
      }
      const tokensGenerated = this._state.stats.tokensGenerated ?? 1;
      this._state.stats.totalTimeMs = performance.now() - startTime;

      if (opts.debug) {
        log.debug('Pipeline', `Generated ${tokensGenerated} tokens in ${this._state.stats.totalTimeMs.toFixed(0)}ms`);
      }

      const ttft = this._state.stats.ttftMs ?? this._state.stats.prefillTimeMs;
      const decodeTokens = Math.max(0, tokensGenerated - 1);
      const decodeSpeed = decodeTokens > 0 ? (decodeTokens / this._state.stats.decodeTimeMs * 1000) : 0;
      const loadMs = this._state.stats.modelLoadMs;
      const loadLabel = Number.isFinite(loadMs) ? `Load: ${loadMs.toFixed(0)}ms | ` : '';
      if (opts.benchmark) {
        log.info('Benchmark', `${loadLabel}TTFT: ${ttft.toFixed(0)}ms | Prefill: ${this._state.stats.prefillTimeMs.toFixed(0)}ms | Decode: ${this._state.stats.decodeTimeMs.toFixed(0)}ms (${decodeTokens} tokens @ ${decodeSpeed.toFixed(1)} tok/s)`);
      } else {
        log.info('Perf', `${loadLabel}TTFT: ${ttft.toFixed(0)}ms | Prefill: ${this._state.stats.prefillTimeMs.toFixed(0)}ms | Decode: ${this._state.stats.decodeTimeMs.toFixed(0)}ms (${decodeTokens} tokens @ ${decodeSpeed.toFixed(1)} tok/s)`);
      }
      trace.perf('Decode summary', {
        ttftMs: ttft,
        prefillMs: this._state.stats.prefillTimeMs,
        decodeMs: this._state.stats.decodeTimeMs,
        decodeTokens,
        decodeSpeed,
        totalMs: this._state.stats.totalTimeMs,
      });
    } finally {
      this._closeFinitenessFallbackWindow(opts);
      finishGenerationRuntime(this, options, startTime, generatedIds, promptTokenCount);
    }
  }

export async function generateTokenIds(prompt, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');

    validateCallTimeOptions(options);
    const opts = resolveGenerateOptions(this._state, options);
    const startTime = prepareGenerationRuntime(this, opts, options);
    let generatedIds = [];
    let promptTokenCount = 0;

    try {
      const prefillStartSeqLen = this._state.currentSeqLen;
      const prefillStart = performance.now();
      const { inputIds, logits: initialPrefillLogits } = await this._prefillPromptToLogits(prompt, opts, 'generateTokenIds');
      const prefillLogits = initialPrefillLogits;
      this._state.stats.prefillTimeMs = performance.now() - prefillStart;
      this._assertTokenIdsInRange(inputIds, 'generateTokenIds.prefillTokens');
      generatedIds = [...inputIds];
      promptTokenCount = inputIds.length;
      this._state.stats.prefillTokens = opts.promptTokenCount = inputIds.length;

      const firstToken = await samplePrefillToken(this, prefillLogits, generatedIds, inputIds, opts, prefillStartSeqLen, false);

      const stopTokenIds = this._state.modelConfig.stopTokenIds;
      const eosToken = this._state.tokenizer.getSpecialTokens?.()?.eos;
      const stopSequenceStart = inputIds.length;
      generatedIds.push(firstToken);
      const tokenIds = [firstToken];
      this._state.stats.ttftMs = performance.now() - startTime;
      markKernelCacheWarmed();
      if (options.onToken) options.onToken(firstToken, '');

      const decodeRuntime = {
        stopTokenIds,
        eosToken,
        stopSequenceStart,
        decodeToken: () => '',
        emitMode: 'token',
      };

      if (!options.signal?.aborted && !this._shouldStopAfterAppendedToken(generatedIds, firstToken, opts, decodeRuntime)) {
        for await (const tokenId of this._runDecodeLoop(generatedIds, opts, options, decodeRuntime)) {
          tokenIds.push(tokenId);
        }
      } else {
        this._recordStopReason(options.signal?.aborted ? 'aborted'
          : isStopToken(firstToken, decodeRuntime.stopTokenIds, decodeRuntime.eosToken) ? 'stop-token' : 'stop-sequence',
          options.signal?.aborted ? null : firstToken);
        this._state.stats.decodeTimeMs = 0;
        this._state.stats.tokensGenerated = 1;
        this._state.stats.decodeTokens = 0;
      }

      this._state.stats.totalTimeMs = performance.now() - startTime;

      return {
        tokenIds,
        stats: this._state.stats,
      };
    } finally {
      this._closeFinitenessFallbackWindow(opts);
      if (this._state.stats.executionPlan) {
        this._state.stats.executionPlan.finalActivePlanId = this._state.executionPlanState?.activePlanId ?? null;
      }
      finishGenerationRuntime(this, options, startTime, generatedIds, promptTokenCount);
    }
  }

export async function* _runDecodeLoop(generatedIds, opts, options, runtime) {
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

    // Step 4: Lazily initialise PLE buffer cache for decode-path slice reuse.
    const pleHiddenSize = Number(this._state.modelConfig.hiddenSizePerLayerInput ?? 0);
    if (pleHiddenSize > 0 && !this._state.pleCache) {
      const activationDtype = resolveActiveExecutionPlan(this._state).activationDtype;
      const bpe = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });
      this._state.pleCache = createPleBufferCache(
        this._state.modelConfig.numLayers,
        pleHiddenSize * bpe,
      );
    }
    if (pleHiddenSize > 0) {
      await primePleDecodeRuntimeCache(this._state, generatedIds);
    }

    const decodeStart = performance.now();
    const resolvedPerLayerInputsSession = resolvePerLayerInputsSession(
      this._state.modelConfig.perLayerInputsSession ?? null,
      this._state.runtimeConfig?.inference?.session?.perLayerInputs ?? null
    );
    const lmHead = this._state.weights.get('lm_head');
    const embedBuffer = this._state.weights.get('embed');
    const hasCpuWeights = isCpuWeightBuffer(lmHead)
      || isCpuWeightBuffer(embedBuffer)
      || lmHead instanceof Float32Array
      || embedBuffer instanceof Float32Array;
    const hasLinearLayers = hasLinearAttentionLayers(this._state.modelConfig.layerTypes);
    const replayPrefillDecode = usesReplayPrefillDecode(this._state);
    const gpuSamplingAvailable = isGPUSamplingAvailable() && !hasCpuWeights;
    const hasSuppressedSamplingTokens = Array.isArray(opts.suppressTokenIds) && opts.suppressTokenIds.length > 0;
    const hasRangeBackedPerLayerInputs = hasRangeBackedPerLayerInputEmbeddings({
      config: this._state.modelConfig,
      weights: this._state.weights,
    });
    const hasGpuSplitPerLayerInputs = hasGpuSplitPerLayerInputEmbeddings({
      config: this._state.modelConfig,
      weights: this._state.weights,
      perLayerInputsSession: resolvedPerLayerInputsSession,
    });
    const pleHotVocabularyRuntime = getPleHotVocabularyRuntime({ weights: this._state.weights });
    const resolveCurrentHotVocabularyBatchDecodeAvailable = () => resolveHotVocabularyBatchDecodeAvailability({
      hasRangeBackedPerLayerInputs,
      pleHotVocabularyRuntime,
      tokenId: generatedIds[generatedIds.length - 1] ?? null,
    });
    const initialHotVocabularyBatchDecodeAvailable = resolveCurrentHotVocabularyBatchDecodeAvailable();
    const executionPlan = opts.executionPlan;
    let useBatchPath = replayPrefillDecode
      ? false
        : shouldUseBatchDecode({
          batchSize: executionPlan.batchSize,
          useGPU: this._state.useGPU,
          gpuSamplingAvailable,
          disableMultiTokenDecode: executionPlan.disableMultiTokenDecode || hasSuppressedSamplingTokens || typeof opts.logitMaskFn === 'function',
          disableCommandBatching: executionPlan.disableCommandBatching,
          isBdpaPagedLayout: this._state.kvCache?.layout === 'bdpa_paged',
          finitenessFallbackWindowOpen: this._hasFinitenessFallbackWindow(),
          hasLinearAttentionLayers: hasLinearLayers,
          selfSpeculationEnabled: opts.speculation?.mode === 'self' && !initialHotVocabularyBatchDecodeAvailable,
          hasRangeBackedPerLayerInputs,
        });
    const ordinaryBatchPath = useBatchPath;
    if (opts.speculation?.mode === 'lookup') useBatchPath = false;
    if (!useBatchPath) {
      let reason = null;
      if (replayPrefillDecode) reason = 'replay_prefill_decode';
      else if (hasCpuWeights) reason = 'cpu_weights';
      else if (!this._state.useGPU) reason = 'no_gpu';
      else if (!gpuSamplingAvailable) reason = 'no_gpu_sampling';
      else if (executionPlan.disableCommandBatching) reason = 'command_batching_disabled';
      else if (hasSuppressedSamplingTokens) reason = 'sampling_suppression_requires_cpu_logits';
      else if (typeof opts.logitMaskFn === 'function') reason = 'token_constraint_requires_per_token_logits';
      else if (executionPlan.disableMultiTokenDecode) reason = 'multi_token_decode_disabled';
      else if (executionPlan.batchSize <= 1) reason = 'batch_size_1';
      else if (this._state.kvCache?.layout === 'bdpa_paged') reason = 'bdpa_paged_layout';
      else if (this._hasFinitenessFallbackWindow()) reason = 'finiteness_fallback_window';
      else if (hasLinearLayers) reason = 'linear_attention_layers';
      this._state.stats.decodeMode = replayPrefillDecode
        ? 'replay_prefill'
        : (opts.speculation?.mode === 'self' ? 'self_speculation' : 'single_token');
      this._state.stats.batchGuardReason = reason;
    } else {
      this._state.stats.decodeMode = hasRangeBackedPerLayerInputs
        ? 'batched_gpu_stepwise_ple'
        : 'batched_gpu';
      this._state.stats.batchGuardReason = null;
    }

    const readbackInterval = executionPlan.readbackInterval;
    const intervalBatches = readbackInterval == null ? 1 : readbackInterval;

    const decodeSingleTokenViaLogits = async () => this._decodeNextTokenViaLogits(generatedIds, opts);
    const stopIfAborted = () => {
      if (!options.signal?.aborted) return false;
      this._recordStopReason('aborted');
      return true;
    };

    if (logBatchPath && useBatchPath) {
      log.debug(
        'Pipeline',
        `Using batch decode path with batchSize=${executionPlan.batchSize}, stopCheckMode=${executionPlan.stopCheckMode}, readbackInterval=${readbackInterval}`
      );
    }

    let rejectedDraftBlocks = 0;
    while (tokensGenerated < opts.maxTokens) {
      if (stopIfAborted()) break;
      if (this._hasFinitenessFallbackWindow() && useBatchPath) {
        useBatchPath = false;
      }
      const hotVocabularyBatchDecodeAvailable = resolveCurrentHotVocabularyBatchDecodeAvailable();

      if (opts.speculation?.mode === 'lookup' && rejectedDraftBlocks < opts.speculation.maxRejectedBlocks) {
        const remaining = opts.maxTokens - tokensGenerated;
        const proposal = remaining > 1 ? lookupDraft(generatedIds, Math.min(opts.speculation.tokens, remaining - 1), opts.speculation.maxMatchTokens, opts.speculation.minMatchTokens) : [];
        if (proposal.length) {
          const stop = (context, token) => isStopToken(token, stopTokenIds, eosToken)
            || matchesStopSequence(this._state.tokenizer, context, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput);
          const result = await verifySpeculativeTokens(this, generatedIds, proposal, opts, options.signal, stop);
          if (result.accepted < proposal.length) rejectedDraftBlocks++;
          if (rejectedDraftBlocks >= opts.speculation.maxRejectedBlocks) useBatchPath = ordinaryBatchPath;
          this._state.stats.decodeMode = 'target_verified_lookup';
          this._state.stats.speculationAttempts += proposal.length;
          this._state.stats.speculationAccepted += result.accepted;
          for (const token of result.tokens) {
            if (stopIfAborted()) break;
            generatedIds.push(token); tokensGenerated++;
            const text = emitMode === 'token' ? '' : decodeToken(token);
            yield emitMode === 'token' ? token : text;
            options.onToken?.(token, text);
          }
          const last = generatedIds[generatedIds.length - 1];
          if (isStopToken(last, stopTokenIds, eosToken)) { this._recordStopReason('stop-token', last); break; }
          if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
            this._recordStopReason('stop-sequence', last); break;
          }
          continue;
        }
      }

      if (useBatchPath) {
        const remaining = opts.maxTokens - tokensGenerated;
        const maxBatchDecodeTokens = resolveMaxBatchDecodeTokens({
          hasHotVocabularyBatchDecode: hotVocabularyBatchDecodeAvailable,
          hasGpuSplitPerLayerInputs,
          hasLinearAttentionLayers: hasLinearLayers,
          modelId: this._state.modelConfig.modelId ?? executionPlan.kernelPathId,
          activationDtype: executionPlan.activationDtype,
          currentSeqLen: this._state.currentSeqLen,
          maxDecodeTokens: opts.maxTokens,
          numLayers: this._state.modelConfig.numLayers,
          hiddenSize: this._state.modelConfig.hiddenSize,
          configuredMaxBatchDecodeTokens: executionPlan.maxBatchDecodeTokens,
        });
        const requestedBatchTokens = executionPlan.batchSize * intervalBatches;
        const boundedBatchTokens = maxBatchDecodeTokens == null
          ? requestedBatchTokens
          : Math.min(requestedBatchTokens, maxBatchDecodeTokens);
        const thisBatchSize = Math.min(boundedBatchTokens, remaining);
        this._state.batchingStats.requestedBatchTokens = Math.max(
          this._state.batchingStats.requestedBatchTokens ?? 0,
          requestedBatchTokens
        );
        this._state.batchingStats.effectiveBatchTokens = Math.max(
          this._state.batchingStats.effectiveBatchTokens ?? 0,
          thisBatchSize
        );
        if (maxBatchDecodeTokens != null) {
          this._state.batchingStats.maxBatchTokenCap = Math.max(
            this._state.batchingStats.maxBatchTokenCap ?? 0,
            maxBatchDecodeTokens
          );
        }
        if (thisBatchSize < requestedBatchTokens) {
          this._state.batchingStats.batchClampCount = (this._state.batchingStats.batchClampCount ?? 0) + 1;
        }
        const lastToken = generatedIds[generatedIds.length - 1];
        const boundedExecutionPlan = thisBatchSize < requestedBatchTokens
          ? {
            ...executionPlan,
            batchSize: Math.min(executionPlan.batchSize, thisBatchSize),
            readbackInterval: readbackInterval == null
              ? null
              : Math.min(intervalBatches, thisBatchSize),
          }
          : executionPlan;
        const batchOpts = boundedExecutionPlan === executionPlan
          ? opts
          : { ...opts, executionPlan: boundedExecutionPlan };

        let batchResult;
        try {
          batchResult = await this._generateNTokensGPU(lastToken, thisBatchSize, generatedIds, batchOpts);
        } catch (error) {
          if (stopIfAborted()) break;
          assertBatchRetryInPlaceAllowed(this._state, error);
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
          if (stopIfAborted()) break;
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
          if (isStopToken(nextToken, stopTokenIds, eosToken)) {
            this._recordStopReason('stop-token', nextToken);
            break;
          }
          continue;
        }
        let batchTokens = [];
        let hitStop = false;
        let stopTokenId = null;
        for (const tokenId of batchResult.tokens) {
          if (options.signal?.aborted) break;
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
          if (isStopToken(tokenId, stopTokenIds, eosToken)) {
            hitStop = true;
            stopTokenId = tokenId;
            break;
          }
        }
        if (batchTokens.length > 0 && options.onBatch) options.onBatch(batchTokens);
        if (stopIfAborted()) break;
        if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
          this._recordStopReason('stop-sequence', generatedIds[generatedIds.length - 1] ?? null);
          break;
        }
        if (hitStop) {
          this._recordStopReason('stop-token', stopTokenId);
          break;
        }
        if (shouldDisableBatchDecodeAfterShortBatch({
          hitStop,
          actualCount: batchResult.actualCount,
          requestedCount: thisBatchSize,
        })) {
          useBatchPath = false;
          continue;
        }
      } else if (opts.speculation?.mode === 'self') {
        // Self-speculation: decode one base token plus a configurable burst of
        // speculative tokens per iteration. Same-model speculation always
        // accepts under greedy because the model is deterministic — both base
        // and speculative use the same weights and state. The benefit is
        // amortizing per-iteration overhead for models where batch decode is
        // disabled (e.g., linear attention).
        const speculativeBurstTokens = opts.speculation.tokens;
        if (!Number.isInteger(speculativeBurstTokens) || speculativeBurstTokens < 1) {
          throw new Error('[Pipeline] resolved self-speculation tokens must be a positive integer.');
        }
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
        if (stopIfAborted()) break;
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

        if (isStopToken(baseToken, stopTokenIds, eosToken)) {
          this._recordStopReason('stop-token', baseToken);
          break;
        }
        if (tokensGenerated >= opts.maxTokens) break;
        if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
          this._recordStopReason('stop-sequence', baseToken);
          break;
        }

        for (let specIndex = 0; specIndex < speculativeBurstTokens; specIndex += 1) {
          if (stopIfAborted()) break;
          if (tokensGenerated >= opts.maxTokens) {
            break;
          }
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
          if (stopIfAborted()) break;
          generatedIds.push(specToken);
          tokensGenerated++;
          this._state.stats.speculationAttempts = (this._state.stats.speculationAttempts ?? 0) + 1;
          this._state.stats.speculationAccepted = (this._state.stats.speculationAccepted ?? 0) + 1;
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
            const tokPerSec = (Math.max(0, tokensGenerated - 1) / elapsedMs) * 1000;
            log.debug('Decode', `#${tokensGenerated} speculation:self (${tokPerSec.toFixed(2)} tok/s avg)`);
          }

          if (isStopToken(specToken, stopTokenIds, eosToken)) {
            this._recordStopReason('stop-token', specToken);
            break;
          }
          if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
            this._recordStopReason('stop-sequence', specToken);
            break;
          }
        }
        if (isStopToken(generatedIds[generatedIds.length - 1], stopTokenIds, eosToken)) {
          this._recordStopReason('stop-token', generatedIds[generatedIds.length - 1]);
          break;
        }
        if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
          this._recordStopReason('stop-sequence', generatedIds[generatedIds.length - 1] ?? null);
          break;
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
        if (stopIfAborted()) break;
        generatedIds.push(nextToken);
        tokensGenerated++;

        // Step 5: Fire-and-forget prefetch of next token's PLE row.
        if (pleHiddenSize > 0) {
          const pleWeights = this._state.weights.get('per_layer_inputs');
          if (pleWeights?.embedTokensPerLayer) {
            this._state.plePrefetchPending = prefetchPerLayerRow(
              nextToken,
              pleWeights.embedTokensPerLayer,
              this._state.modelConfig.numLayers * pleHiddenSize,
              resolvedPerLayerInputsSession,
            );
          }
        }

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
          const tokPerSec = (Math.max(0, tokensGenerated - 1) / elapsedMs) * 1000;
          log.debug('Decode', `#${tokensGenerated} "${tokenText}" ${tokenTime.toFixed(0)}ms (${tokPerSec.toFixed(2)} tok/s avg)`);
        }

        if (isStopToken(nextToken, stopTokenIds, eosToken)) {
          this._recordStopReason('stop-token', nextToken);
          break;
        }
        if (matchesStopSequence(this._state.tokenizer, generatedIds, stopSequenceStart, opts.stopSequences, this._state.modelConfig?.reasoningOutput)) {
          this._recordStopReason('stop-sequence', nextToken);
          break;
        }
      }
    }

    if (!this._state.stats.stopReason) {
      this._recordStopReason(tokensGenerated >= opts.maxTokens ? 'max-tokens' : 'completed');
    }
    this._state.stats.decodeTimeMs = performance.now() - decodeStart;
    this._state.stats.tokensGenerated = tokensGenerated;
    // Prefill produces the first token; decode timing covers only later tokens.
    this._state.stats.decodeTokens = Math.max(0, tokensGenerated - 1);
    this._state.stats.batching = { ...this._state.batchingStats };
  }

export async function _decodeStep(currentIds, opts) {
    if (usesReplayPrefillDecode(this._state)) {
      return this._decodeNextTokenViaLogits(currentIds, opts);
    }
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim, representation) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim, representation)
      : undefined;
    const sampling = opts.temperature === 0 ? opts : {
      ...opts, seed: samplingSeedForToken(opts.seed, currentIds.length - opts.promptTokenCount),
    };
    return decodeStep(this._state, currentIds, sampling, this._getDecodeHelpers(debugCheckBuffer));
  }

export async function decodeStepLogits(currentIds, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    resetActiveExecutionPlan(this._state);

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    return this._decodeStepToLogits(currentIds, opts);
  }

export async function advanceWithToken(tokenId, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');
    resetActiveExecutionPlan(this._state);
    assertIncrementalDecodeSupport(this._state, 'advanceWithToken');

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim, representation) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim, representation)
      : undefined;

    this._assertTokenIdInRange(tokenId, 'advanceWithToken');
    await runAdvanceWithToken(this._state, tokenId, opts, this._getDecodeHelpers(debugCheckBuffer));
  }

export async function advanceWithTokenAndEmbedding(tokenId, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating) throw new Error('Generation already in progress');
    resetActiveExecutionPlan(this._state);
    assertIncrementalDecodeSupport(this._state, 'advanceWithTokenAndEmbedding');

    validateCallTimeOptions(options);

    const opts = this._resolveStepOptions(options);
    const embeddingMode = resolveAdvanceEmbeddingMode(this._state, options);
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim, representation) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim, representation)
      : undefined;

    this._assertTokenIdInRange(tokenId, 'advanceWithTokenAndEmbedding');
    return runAdvanceWithTokenAndEmbedding(
      this._state,
      tokenId,
      opts,
      this._getDecodeHelpers(debugCheckBuffer),
      embeddingMode
    );
  }

export async function _generateNTokensGPU(startToken, N, currentIds, opts) {
    const debugCheckBuffer = this._state.debug
      ? (buffer, label, numTokens, expectedDim, representation) =>
        debugCheckBufferHelper(this._state, buffer, label, numTokens, expectedDim, representation)
      : undefined;
    return generateNTokensGPU(this._state, startToken, N, currentIds, opts, this._getDecodeHelpers(debugCheckBuffer));
  }
