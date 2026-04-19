import { getRuntimeConfig } from '../../../config/runtime.js';

/**
 * Resolve a session-level flag with runtime-over-manifest precedence:
 *
 *   getRuntimeConfig().inference.session[field]
 *     (the merged manifest base + runtime override session)
 *   -> manifest sessionSettings[field] only for legacy callers whose runtime
 *      config has not been patched yet
 *
 * New call sites should generally read getRuntimeConfig() directly. This helper
 * is retained for older paths that still accept a modelConfig fallback.
 */
export function resolveSessionFlag(modelConfig, field) {
  const runtimeValue = getRuntimeConfig().inference?.session?.[field];
  if (runtimeValue !== null && runtimeValue !== undefined) {
    return runtimeValue;
  }
  const manifestValue = modelConfig?.sessionSettings?.[field];
  if (manifestValue !== null && manifestValue !== undefined) {
    return manifestValue;
  }
  return runtimeValue;
}

export function resolveLargeWeightOverrides(modelConfig) {
  const manifestOverrides = modelConfig?.largeWeightsConfig?.gpuResidentOverrides;
  if (manifestOverrides !== null && manifestOverrides !== undefined) {
    return manifestOverrides;
  }
  return getRuntimeConfig().inference?.largeWeights?.gpuResidentOverrides ?? null;
}
