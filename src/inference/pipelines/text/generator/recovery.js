import { log, trace, isTraceEnabled } from '../../../../debug/index.js';
import { createKVCache, isStopToken } from '../init.js';
import {
  decodeStep,
  decodeStepLogits,
  advanceWithToken,
  generateNTokensGPU,
  shouldUseBatchDecode,
  sumProfileTimings,
  FinitenessError,
  advanceWithTokenAndEmbedding as runAdvanceWithTokenAndEmbedding,
} from '../generator-steps.js';
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

export async function _replayPrefillDecodeLogits(currentIds, opts) {
    // Guard: cap replay-prefill sequence length to the config-owned maxSeqLen.
    // Without KV cache creation, this bound is not enforced elsewhere.
    const kvConfig = this._state.runtimeConfig?.inference?.session?.kvcache;
    const replayMaxSeqLen = kvConfig?.maxSeqLen;
    if (Number.isFinite(replayMaxSeqLen) && replayMaxSeqLen > 0 && currentIds.length > replayMaxSeqLen) {
      throw new Error(
        `[Pipeline] Replay-prefill sequence length ${currentIds.length} exceeds ` +
        `runtime.inference.session.kvcache.maxSeqLen (${replayMaxSeqLen}). ` +
        'Increase maxSeqLen in a tier profile or runtime config to allow longer sequences.'
      );
    }
    advanceDecodeStepCount(this._state);
    this._resetReplayPrefillRuntimeState();
    const logits = await this._prefill(currentIds, opts);
    return {
      logits,
      logitsBuffer: null,
      logitsDtype: null,
      rawVocabSize: this._state.modelConfig.vocabSize,
      vocabSize: this._state.modelConfig.vocabSize,
    };
  }

export function _shouldUseFinitenessFallback(error, contextLabel) {
    if (!shouldRetryWithFinitenessFallback(error)) {
      return false;
    }
    if (!hasFallbackExecutionPlan(this._state)) {
      throw createUnhandledFinitenessPolicyError(this._state, contextLabel, error);
    }
    return true;
  }

export function _recreateKVCacheForExecutionPlan(plan, reasonLabel) {
    const kvDtype = resolveTargetPlanKVDtype(plan, `${reasonLabel}: target plan`);
    const runtimeInference = cloneRuntimeInferenceWithKVDtype(this._state, kvDtype);
    this._state.kvCache?.destroy?.();
    this._state.kvCache = createKVCache(
      this._state.modelConfig,
      this._state.useGPU,
      this._state.debug,
      runtimeInference
    );
    this._state.linearAttentionRuntime = resetLinearAttentionRuntime(this._state.linearAttentionRuntime);
    this._state.currentSeqLen = 0;
    return kvDtype;
  }

export function _openFinitenessFallbackWindow(opts, reasonLabel, tokenCount, rollbackSeqLen = undefined) {
    const normalizedCount = Number.isFinite(tokenCount)
      ? Math.max(1, Math.floor(tokenCount))
      : 1;
    if (this._finitenessFallbackWindow) {
      this._finitenessFallbackWindow.remainingTokens = Math.max(
        this._finitenessFallbackWindow.remainingTokens,
        normalizedCount
      );
      return;
    }
    const original = this._beginFinitenessFallback(opts, reasonLabel, rollbackSeqLen);
    this._finitenessFallbackWindow = {
      original,
      remainingTokens: normalizedCount,
    };
  }

export function _closeFinitenessFallbackWindow(opts) {
    if (!this._finitenessFallbackWindow) {
      return;
    }
    const original = this._finitenessFallbackWindow.original;
    this._finitenessFallbackWindow = null;
    this._endFinitenessFallback(opts, original);
  }

export function _consumeFinitenessFallbackToken(opts) {
    if (!this._finitenessFallbackWindow) {
      return;
    }
    this._finitenessFallbackWindow.remainingTokens -= 1;
    if (this._finitenessFallbackWindow.remainingTokens <= 0) {
      this._closeFinitenessFallbackWindow(opts);
    }
  }

export function _beginFinitenessFallback(opts, reasonLabel, rollbackSeqLen = undefined) {
    const originalPlan = resolveActiveExecutionPlan(this._state);
    const currentKvDtype = resolveCurrentKVCacheDtype(
      this._state,
      originalPlan,
      `${reasonLabel}: current plan`
    );
    const original = {
      activePlanId: this._state.executionPlanState?.activePlanId ?? 'primary',
      seed: opts.seed,
      restoreKVCachePlan: null,
    };

    const fallbackPlan = activateFallbackExecutionPlan(this._state);
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
    const fallbackKvDtype = resolveTargetPlanKVDtype(
      fallbackPlan,
      `${reasonLabel}: fallback plan`
    );

    if (Number.isInteger(rollbackSeqLen) && rollbackSeqLen < 0) {
      setActiveExecutionPlan(this._state, original.activePlanId);
      throw new Error(
        `[Pipeline] ${reasonLabel}: rollbackSeqLen must be a non-negative integer when provided.`
      );
    }

    if (fallbackKvDtype !== currentKvDtype) {
      if (rollbackSeqLen !== 0) {
        setActiveExecutionPlan(this._state, original.activePlanId);
        throw new Error(
          `[Pipeline] ${reasonLabel}: finiteness fallback requires rebuilding the KV cache ` +
          `${currentKvDtype} -> ${fallbackKvDtype}, which is only supported from a fresh prefill (rollbackSeqLen=0).`
        );
      }
      try {
        this._recreateKVCacheForExecutionPlan(fallbackPlan, reasonLabel);
        original.restoreKVCachePlan = originalPlan;
      } catch (error) {
        setActiveExecutionPlan(this._state, original.activePlanId);
        try {
          this._recreateKVCacheForExecutionPlan(originalPlan, `${reasonLabel}: restore primary`);
        } catch (restoreError) {
          log.warn(
            'Pipeline',
            `Failed to restore primary KV cache after fallback activation error: ${restoreError}`
          );
        }
        throw error;
      }
    } else if (Number.isInteger(rollbackSeqLen)) {
      this._state.kvCache?.truncate(rollbackSeqLen);
      this._state.currentSeqLen = rollbackSeqLen;
      if (rollbackSeqLen === 0) {
        this._state.linearAttentionRuntime = resetLinearAttentionRuntime(this._state.linearAttentionRuntime);
      }
    } else {
      this._state.kvCache?.truncate(this._state.currentSeqLen);
    }

    this._state.decodeBuffers?.ensureBuffers({
      hiddenSize: this._state.modelConfig.hiddenSize,
      intermediateSize: this._state.modelConfig.maxIntermediateSize,
      activationDtype: fallbackPlan.activationDtype,
      enablePingPong: true,
    });

    if (opts.seed == null) {
      const fallbackSeedBase = (this._state.decodeStepCount + this._state.currentSeqLen + 1) >>> 0;
      opts.seed = (fallbackSeedBase * 2654435761) >>> 0;
    }
    opts.executionPlan = rebaseExecutionSessionPlan(this._state, opts.executionPlan);
    if (this._state.stats.executionPlan) {
      this._state.stats.executionPlan.finalActivePlanId = fallbackPlan.id;
      this._state.stats.executionPlan.transitions.push({
        kind: 'activate-finiteness-fallback',
        reason: reasonLabel ?? null,
        decodeStep: this._state.decodeStepCount,
        seqLen: this._state.currentSeqLen,
        fromPlanId: originalPlan.id,
        toPlanId: fallbackPlan.id,
        fromKernelPathId: originalPlan.kernelPathId ?? null,
        toKernelPathId: fallbackPlan.kernelPathId ?? null,
      });
    }
    this._state.stats.kernelPathId = fallbackPlan.kernelPathId ?? null;
    this._state.stats.kernelPathSource = fallbackPlan.kernelPathSource ?? 'none';

    return original;
  }

export function _endFinitenessFallback(opts, original) {
    opts.seed = original.seed;
    setActiveExecutionPlan(this._state, original.activePlanId);
    opts.executionPlan = rebaseExecutionSessionPlan(this._state, opts.executionPlan);
    const restoredPlan = resolveActiveExecutionPlan(this._state);
    if (original.restoreKVCachePlan) {
      this._recreateKVCacheForExecutionPlan(restoredPlan, 'restore-primary-plan');
    }
    if (this._state.stats.executionPlan) {
      this._state.stats.executionPlan.finalActivePlanId = restoredPlan.id;
      this._state.stats.executionPlan.transitions.push({
        kind: 'restore-primary-plan',
        reason: null,
        decodeStep: this._state.decodeStepCount,
        seqLen: this._state.currentSeqLen,
        fromPlanId: this._state.executionPlanState?.fallbackPlan?.id ?? null,
        toPlanId: restoredPlan.id,
        fromKernelPathId: this._state.executionPlanState?.fallbackPlan?.kernelPathId ?? null,
        toKernelPathId: restoredPlan.kernelPathId ?? null,
      });
    }
    this._state.stats.kernelPathId = restoredPlan.kernelPathId ?? this._state.resolvedKernelPath?.id ?? null;
    this._state.stats.kernelPathSource = restoredPlan.kernelPathSource ?? this._state.kernelPathSource ?? 'none';
    const nextActivationDtype = this._getEffectiveActivationDtype();
    this._state.decodeBuffers?.ensureBuffers({
      hiddenSize: this._state.modelConfig.hiddenSize,
      intermediateSize: this._state.modelConfig.maxIntermediateSize,
      activationDtype: nextActivationDtype,
      enablePingPong: true,
    });
  }

export async function _retryWithFinitenessFallback(opts, reasonLabel, retryFn, rollbackSeqLen = undefined) {
    if (this._hasFinitenessFallbackWindow()) {
      return retryFn();
    }
    const original = this._beginFinitenessFallback(opts, reasonLabel, rollbackSeqLen);
    try {
      return await retryFn();
    } finally {
      this._endFinitenessFallback(opts, original);
    }
  }

export async function _retryWithPersistentFinitenessFallback(opts, reasonLabel, tokenBudget, retryFn, rollbackSeqLen = undefined) {
    if (this._hasFinitenessFallbackWindow()) {
      return retryFn();
    }
    this._openFinitenessFallbackWindow(opts, reasonLabel, tokenBudget, rollbackSeqLen);
    try {
      return await retryFn();
    } catch (error) {
      this._closeFinitenessFallbackWindow(opts);
      throw error;
    }
  }

export async function _retryDecodeStepWithFinitenessWindow(generatedIds, opts, reasonLabel) {
    const windowTokens = this._resolveDeferredRoundingWindowTokens();
    if (windowTokens <= 1) {
      return this._retryWithFinitenessFallback(
        opts,
        reasonLabel,
        () => this._decodeStep(generatedIds, opts)
      );
    }

    this._openFinitenessFallbackWindow(opts, reasonLabel, windowTokens);
    try {
      return await this._decodeStep(generatedIds, opts);
    } catch (error) {
      this._closeFinitenessFallbackWindow(opts);
      throw error;
    }
  }
