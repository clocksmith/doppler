export const DEFAULT_FINITENESS_ABS_THRESHOLD = 65500;
export const DEFAULT_DEFERRED_ROUNDING_WINDOW_TOKENS = 1;

export function resolveRangeAwareSelectiveWideningConfig(computeConfig) {
  const policy = computeConfig?.rangeAwareSelectiveWidening;
  const enabled = policy?.enabled !== false;
  const includeNonFinite = policy?.includeNonFinite !== false;
  const absThreshold = Number.isFinite(policy?.absThreshold) && policy.absThreshold > 0
    ? policy.absThreshold
    : DEFAULT_FINITENESS_ABS_THRESHOLD;

  return {
    enabled,
    includeNonFinite,
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
