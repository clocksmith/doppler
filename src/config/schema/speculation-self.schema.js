// =============================================================================
// Self-speculation config
//
// Same-model speculative decoding: run extra forward passes to produce
// multiple tokens per decode loop iteration. Distinct from draft-model
// speculation (speculative.schema.js) which requires a separate model.
// =============================================================================

export const SPECULATION_MODES = Object.freeze(['none', 'self', 'draft', 'medusa']);
export const SPECULATION_VERIFY_MODES = Object.freeze(['greedy']);

export const DEFAULT_SELF_SPECULATION_CONFIG = Object.freeze({
  mode: 'none',
  tokens: 1,
  verify: 'greedy',
  threshold: null,
  rollbackOnReject: true,
});

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
  if (config.mode === 'self') {
    if (!Number.isInteger(config.tokens) || config.tokens < 1) {
      throw new Error('[Speculation] tokens must be a positive integer.');
    }
    if (config.tokens !== 1) {
      throw new Error(
        `[Speculation] Only tokens=1 (batch-2) is currently supported for self-speculation. Got tokens=${config.tokens}.`
      );
    }
    if (!SPECULATION_VERIFY_MODES.includes(config.verify)) {
      throw new Error(
        `[Speculation] verify="${config.verify}" is not supported. Expected one of: ${SPECULATION_VERIFY_MODES.join(', ')}.`
      );
    }
  }
}
