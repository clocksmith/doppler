const DEFAULT_FINITENESS_ENABLED = true;
const DEFAULT_COMPUTED_FINITENESS_ABS_THRESHOLD = 65500;
const DEFAULT_INCLUDE_NON_FINITE = true;
const DEFAULT_FINITENESS_ON_TRIGGER = 'error';
const DEFAULT_COMPUTED_DEFERRED_ROUNDING_WINDOW_TOKENS = 1;

export const DEFAULT_FINITENESS_ABS_THRESHOLD = DEFAULT_COMPUTED_FINITENESS_ABS_THRESHOLD;
export const DEFAULT_DEFERRED_ROUNDING_WINDOW_TOKENS = DEFAULT_COMPUTED_DEFERRED_ROUNDING_WINDOW_TOKENS;
export const DEFAULT_FINITENESS_ON_TRIGGER_MODE = DEFAULT_FINITENESS_ON_TRIGGER;

export function resolveRangeAwareSelectiveWideningConfig(computeConfig) {
  const policy = computeConfig?.rangeAwareSelectiveWidening;
  const enabled = policy?.enabled ?? DEFAULT_FINITENESS_ENABLED;
  const includeNonFinite = policy?.includeNonFinite ?? DEFAULT_INCLUDE_NON_FINITE;
  const onTrigger = policy?.onTrigger ?? DEFAULT_FINITENESS_ON_TRIGGER;
  const absThreshold = Number.isFinite(policy?.absThreshold) && policy.absThreshold > 0
    ? policy.absThreshold
    : DEFAULT_FINITENESS_ABS_THRESHOLD;

  if (onTrigger !== 'error' && onTrigger !== 'fallback-plan') {
    throw new Error(
      `[FinitenessPolicy] rangeAwareSelectiveWidening.onTrigger must be "error" or "fallback-plan"; got ${JSON.stringify(onTrigger)}.`
    );
  }

  return {
    enabled,
    includeNonFinite,
    onTrigger,
    absThreshold,
  };
}

export function resolveDeferredRoundingWindowTokens(computeConfig) {
  const raw = computeConfig?.deferredRoundingWindowTokens;
  if (!Number.isFinite(raw)) {
    return DEFAULT_DEFERRED_ROUNDING_WINDOW_TOKENS;
  }
  const normalized = Math.floor(raw);
  return normalized >= 1 ? normalized : DEFAULT_DEFERRED_ROUNDING_WINDOW_TOKENS;
}

export function shouldRunFinitenessGuard(activationDtype, computeConfig) {
  if (activationDtype !== 'f16') {
    return false;
  }
  return resolveRangeAwareSelectiveWideningConfig(computeConfig).enabled;
}
