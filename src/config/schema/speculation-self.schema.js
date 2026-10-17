// =============================================================================
// Self-speculation config
//
// Same-model speculative decoding: run extra forward passes to produce
// multiple tokens per decode loop iteration. Distinct from draft-model
// speculation (speculative.schema.js) which requires a separate model.
// =============================================================================

export const SPECULATION_MODES = Object.freeze(['none', 'self', 'lookup', 'draft', 'medusa']);
export const SPECULATION_VERIFY_MODES = Object.freeze(['greedy']);

export const DEFAULT_SELF_SPECULATION_CONFIG = Object.freeze({
  mode: 'none',
  tokens: 1,
  verify: 'greedy',
  threshold: null,
  rollbackOnReject: true,
});

/** @type {import('./speculation-self.schema.js').validateSelfSpeculationConfig} */
export function validateSelfSpeculationConfig(config) {
  if (!config || typeof config !== 'object') {
    throw new Error('[Speculation] config must be a non-null object.');
  }
  if (!SPECULATION_MODES.includes(config.mode)) {
    throw new Error(
      `[Speculation] mode="${config.mode}" is not supported. Expected one of: ${SPECULATION_MODES.join(', ')}.`
    );
  }
  if (config.mode === 'draft') {
    throw new Error(
      '[Speculation] mode="draft" requires a separate draft model and is not yet supported. Use mode="self" for same-model speculation.'
    );
  }
  if (config.mode === 'medusa') {
    throw new Error(
      '[Speculation] mode="medusa" (multi-head prediction) is not yet supported. Use mode="self" for same-model speculation.'
    );
  }
  if (config.mode === 'lookup') {
    if (typeof config.maxMatchTokens !== 'number' || typeof config.minMatchTokens !== 'number'
      || typeof config.maxRejectedBlocks !== 'number' || config.rollbackOnReject !== true || config.threshold !== null
      || !Number.isSafeInteger(config.maxMatchTokens) || config.maxMatchTokens < 1
      || !Number.isSafeInteger(config.minMatchTokens) || config.minMatchTokens < 1 || config.minMatchTokens > config.maxMatchTokens
      || !Number.isSafeInteger(config.maxRejectedBlocks) || config.maxRejectedBlocks < 1) {
      throw new Error('[Speculation] lookup requires rollback, threshold=null, positive match/rejection bounds, and minMatchTokens <= maxMatchTokens.');
    }
  }
  if (config.mode === 'self' || config.mode === 'lookup') {
    if (!Number.isInteger(config.tokens) || config.tokens < 1) {
      throw new Error('[Speculation] tokens must be a positive integer.');
    }
    if (!SPECULATION_VERIFY_MODES.includes(config.verify)) {
      throw new Error(
        `[Speculation] verify="${config.verify}" is not supported. Expected one of: ${SPECULATION_VERIFY_MODES.join(', ')}.`
      );
    }
  }
}

// Resolve once before request identity is frozen. Null explicitly disables.
/** @type {import('./speculation-self.schema.js').resolveSelfSpeculationConfig} */
export function resolveSelfSpeculationConfig(session, override) {
  if (override === null) return null;
  if (session === undefined) throw new Error('[Pipeline] runtime.inference.session.speculation is required.');
  if (override !== undefined && (!override || typeof override !== 'object')) {
    throw new Error('[Pipeline] options.speculation must be an object or null.');
  }
  if (override !== undefined && !session) {
    throw new Error('[Pipeline] options.speculation requires runtime.inference.session.speculation defaults.');
  }
  if (session === null) return null;
  const resolved = { ...session, ...override };
  validateSelfSpeculationConfig(resolved);
  return resolved.mode === 'none' ? null : Object.freeze(resolved);
}
