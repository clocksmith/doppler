// =============================================================================
// Target Modules
// =============================================================================

export const VALID_LORA_TARGET_MODULES = [
  'q_proj',
  'k_proj',
  'v_proj',
  'o_proj',
  'gate_proj',
  'up_proj',
  'down_proj',
  'gate_up_proj',
];

// =============================================================================
// Validation Thresholds
// =============================================================================

export const DEFAULT_ADAPTER_VALIDATION_CONFIG = {
  minRank: 1,
  maxRank: 1024,
  minAlpha: 0.1,
  maxNameLength: 256,
  maxDescriptionLength: 4096,
};

// =============================================================================
// Stacking Config
// =============================================================================

export const DEFAULT_ADAPTER_STACK_CONFIG = {
  strategy: 'sum',
  normalizeWeights: false,
  minWeight: 0.0,
  maxWeight: 2.0,
};

// =============================================================================
// Registry Config
// =============================================================================

export const DEFAULT_ADAPTER_REGISTRY_CONFIG = {
  dbName: 'doppler-adapter-registry',
  dbVersion: 1,
  storeName: 'adapters',
  defaultSortBy: 'name',
  defaultSortOrder: 'asc',
};

// =============================================================================
// Complete Adapter Config
// =============================================================================

export const DEFAULT_ADAPTER_CONFIG = {
  validation: DEFAULT_ADAPTER_VALIDATION_CONFIG,
  stacking: DEFAULT_ADAPTER_STACK_CONFIG,
  registry: DEFAULT_ADAPTER_REGISTRY_CONFIG,
  targetModules: VALID_LORA_TARGET_MODULES,
};
