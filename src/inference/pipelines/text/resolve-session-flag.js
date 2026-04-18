import { getRuntimeConfig } from '../../../config/runtime.js';

/**
 * Resolve a session-level flag with manifest-first precedence:
 *
 *   manifest.inference.session[field]  (if set, i.e. not null/undefined)
 *   -> runtime.inference.session[field] (explicit runtime profile)
 *   -> schema default                   (baked into runtime config)
 *
 * The merge layer in src/config/merge.js already computes manifest × runtime
 * for session fields when a modelConfig is parsed via parseModelConfigFromManifest.
 * This helper exists for pipeline call sites that don't have the merged
 * modelConfig in hand — typically early init or deep kernel-selection paths —
 * so they can consult the manifest's sessionSettings first.
 */
export function resolveSessionFlag(modelConfig, field) {
  const manifestValue = modelConfig?.sessionSettings?.[field];
  if (manifestValue !== null && manifestValue !== undefined) {
    return manifestValue;
  }
  return getRuntimeConfig().inference?.session?.[field];
}

export function resolveLargeWeightOverrides(modelConfig) {
  const manifestOverrides = modelConfig?.largeWeightsConfig?.gpuResidentOverrides;
  if (manifestOverrides !== null && manifestOverrides !== undefined) {
    return manifestOverrides;
  }
  return getRuntimeConfig().inference?.largeWeights?.gpuResidentOverrides ?? null;
}
