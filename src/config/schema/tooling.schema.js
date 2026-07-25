// =============================================================================
// Tooling Observation Config
// =============================================================================

export const TOOLING_INTENTS = ['verify', 'investigate', 'calibrate'];
export const TOOLING_DIAGNOSTICS = ['off', 'on_failure', 'always'];
export const REFACTOR_RECEIPT_POLICIES = ['off', 'on_failure', 'always'];

export const DEFAULT_TOOLING_CONFIG = {
  diagnostics: 'on_failure',
  refactorReceipts: 'off',
  converter: null,
};
