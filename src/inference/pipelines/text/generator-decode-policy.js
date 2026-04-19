
import { log } from '../../../debug/index.js';
import { mergeRuntimeValues } from '../../../config/runtime-merge.js';
import { resolveActiveExecutionPlan } from './execution-plan.js';
import { resolvePerLayerInputsSession } from './generator-helpers.js';
import {
  ensurePleGpuHotVocabularyRuntime,
  ensurePleGpuSplitTablesRuntime,
  ensurePleScaledProjectionNormWeight,
} from './per-layer-inputs.js';

export function shouldDisableBatchDecodeAfterShortBatch({ hitStop, actualCount, requestedCount }) {
  return hitStop !== true
    && Number.isInteger(actualCount)
    && Number.isInteger(requestedCount)
    && actualCount > 0
    && actualCount < requestedCount;
}

export function resolveHotVocabularyBatchDecodeAvailability({
  hasRangeBackedPerLayerInputs,
  pleHotVocabularyRuntime,
  tokenId,
}) {
  if (hasRangeBackedPerLayerInputs !== true) {
    return false;
  }
  if (!pleHotVocabularyRuntime || typeof pleHotVocabularyRuntime !== 'object') {
    return false;
  }
  const sentinelIndex = pleHotVocabularyRuntime.sentinelIndex;
  if (!Number.isInteger(sentinelIndex)) {
    return false;
  }
  if (!Number.isInteger(tokenId)) {
    return false;
  }
  const hotTokenIndex = pleHotVocabularyRuntime.hotTokenIndexMap?.[tokenId] ?? sentinelIndex;
  return hotTokenIndex !== sentinelIndex;
}

export async function primePleDecodeRuntimeCache(state, seedTokenIds = null) {
  const perLayerInputsSession = resolvePerLayerInputsSession(
    state.modelConfig?.perLayerInputsSession ?? null,
    state.runtimeConfig?.inference?.session?.perLayerInputs ?? null
  );
  const materialization = perLayerInputsSession?.materialization;
  if (state.debug) {
    log.debug(
      'Pipeline',
      `PLE materialization session=${materialization ?? 'null'} ` +
      `modelId=${state.modelConfig?.modelId ?? 'unknown'} ` +
      `runtimePerLayerInputs=${Boolean(state.runtimeConfig.inference.session?.perLayerInputs)} ` +
      `manifestPerLayerInputs=${Boolean(state.modelConfig?.perLayerInputsSession)}`
    );
  }
  await ensurePleGpuHotVocabularyRuntime({
    config: state.modelConfig,
    weights: state.weights,
    perLayerInputsSession,
    debugFlags: state.debugFlags,
    tokenizer: state.tokenizer ?? null,
    seedTokenIds: Array.isArray(seedTokenIds) ? seedTokenIds : null,
  });
  await ensurePleGpuSplitTablesRuntime({
    config: state.modelConfig,
    weights: state.weights,
    perLayerInputsSession,
    debugFlags: state.debugFlags,
  });
  await ensurePleScaledProjectionNormWeight({
    config: state.modelConfig,
    weights: state.weights,
    weightConfig: {
      rmsNormWeightOffset: state.modelConfig?.rmsNormWeightOffset ?? false,
    },
    debugFlags: state.debugFlags,
  });
}

export function recordPrefillProfileStep(state, entry) {
  if (!entry?.timings || Object.keys(entry.timings).length === 0) return;
  if (!state.stats.prefillProfileSteps) {
    state.stats.prefillProfileSteps = [];
  }
  state.stats.prefillProfileSteps.push(entry);
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
export function resolveTokenText(tokenizer, tokenIds, fallbackText = '?', renderTokenText, renderFallbackTokenText) {
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

export function usesReplayPrefillDecode(state) {
  return state?.modelConfig?.decodeStrategy === 'replay_prefill';
}

export function assertIncrementalDecodeSupport(state, operation) {
  if (!usesReplayPrefillDecode(state)) {
    return;
  }
  throw new Error(
    `[Pipeline] ${operation} is not supported for models that require replay-prefill decode. ` +
    'Incremental KV-cache decode and prefix snapshots stay disabled when the model config does not resolve ' +
    'explicit layerTypes for mixed-geometry/shared-KV decode.'
  );
}

export function summarizeExecutionPlan(plan) {
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

export function createUnhandledFinitenessPolicyError(state, contextLabel, error) {
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

function assertResolvedKVDtype(kvDtype, contextLabel) {
  if (kvDtype === 'f16' || kvDtype === 'f32') {
    return kvDtype;
  }
  throw new Error(
    `[Pipeline] ${contextLabel}: expected execution-plan kvDtype to resolve to "f16" or "f32", ` +
    `got ${JSON.stringify(kvDtype)}.`
  );
}

export function resolveTargetPlanKVDtype(plan, contextLabel) {
  return assertResolvedKVDtype(
    plan?.kernelPath?.kvDtype ?? plan?.activationDtype ?? null,
    contextLabel
  );
}

export function resolveCurrentKVCacheDtype(state, plan, contextLabel) {
  return assertResolvedKVDtype(
    state?.kvCache?.kvDtype
      ?? plan?.kernelPath?.kvDtype
      ?? state?.runtimeConfig?.inference?.session?.kvcache?.kvDtype
      ?? plan?.activationDtype
      ?? null,
    contextLabel
  );
}

export function cloneRuntimeInferenceWithKVDtype(state, kvDtype) {
  const runtimeInference = state?.runtimeConfig?.inference;
  if (!runtimeInference?.session?.kvcache) {
    throw new Error(
      '[Pipeline] runtime.inference.session.kvcache is required for finiteness fallback KV-cache recovery.'
    );
  }
  return mergeRuntimeValues(runtimeInference, {
    session: {
      kvcache: {
        kvDtype,
      },
    },
  });
}
