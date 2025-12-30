// =============================================================================
// Tooling Intent Config
// =============================================================================

export const TOOLING_INTENTS = ['verify', 'investigate', 'calibrate'];
export const TOOLING_DIAGNOSTICS = ['off', 'on_failure', 'always'];

export const DEFAULT_TOOLING_CONFIG = {
  intent: null,
  diagnostics: 'on_failure',
  converter: null,
};
