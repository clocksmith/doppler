import { GENERATION_CONTRACT, validateGenerationField } from '../../../config/generation-contract.js';
/** @typedef {import('./sampling-config.js').ResolvedSamplingConfig} ResolvedSamplingConfig */

/** @type {import('./sampling-config.js').resolveSamplingConfig} */
export function resolveSamplingConfig(opts, runtimeConfig) {
  const defaults = runtimeConfig?.inference?.sampling;
  if (!defaults || typeof defaults !== 'object' || Array.isArray(defaults)) {
    throw new Error('[Sampling] runtimeConfig.inference.sampling is required.');
  }
  const resolved = /** @type {ResolvedSamplingConfig} */ ({});
  for (const [rawName, rule] of Object.entries(GENERATION_CONTRACT.options)) {
    if (!rule.runtimeSampling) continue;
    const name = /** @type {keyof import('../../../config/generation-contract.js').RuntimeSamplingOptions} */ (rawName);
    if (defaults[name] === undefined) throw new Error(`[Sampling] runtimeConfig.inference.sampling.${name} is required.`);
    const value = opts?.[name] === undefined ? defaults[name] : opts[name];
    validateGenerationField(name, value, rule);
    Object.assign(resolved, { [name]: Array.isArray(value) ? [...value] : value });
  }
  const greedyThreshold = defaults.greedyThreshold;
  if (!Number.isFinite(greedyThreshold) || greedyThreshold < 0) {
    throw new Error('[Sampling] greedyThreshold must be a non-negative finite number.');
  }
  resolved.greedyThreshold = greedyThreshold;
  for (const name of /** @type {const} */ (['suppressSpecialTokens', 'suppressSpecialLikeTokens'])) {
    const value = opts?.[name] === undefined ? defaults[name] : opts[name];
    if (typeof value !== 'boolean') throw new Error(`[Sampling] ${name} must be a boolean and cannot be null.`);
    resolved[name] = value;
  }
  return resolved;
}
