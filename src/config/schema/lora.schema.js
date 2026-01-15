import { VALID_LORA_TARGET_MODULES } from './adapter.schema.js';

// =============================================================================
// LoRA Defaults
// =============================================================================

export const DEFAULT_LORA_CONFIG = {
  rank: 16,
  alpha: 32,
  dropout: 0.0,
  targetModules: [...VALID_LORA_TARGET_MODULES],
};
