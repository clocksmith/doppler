// =============================================================================
// Execution v0 Schema
// =============================================================================

export const EXECUTION_V0_SCHEMA_ID = 'doppler.execution/v0';

export const EXECUTION_V0_HASH_PATTERN = /^sha256:[0-9a-f]{64}$/;
export const EXECUTION_V0_SEMVER_PATTERN = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;

export const DEFAULT_EXECUTION_V0_COMPUTE_DEFAULTS = {
  activationDtype: 'f16',
  mathDtype: 'f16',
  accumDtype: 'f32',
  outputDtype: 'f16',
};

export const DEFAULT_EXECUTION_V0_SESSION_DEFAULTS = {
  compute: {
    defaults: { ...DEFAULT_EXECUTION_V0_COMPUTE_DEFAULTS },
    kernelProfiles: [],
  },
  kvcache: null,
  decodeLoop: null,
};

export const DEFAULT_EXECUTION_V0_POLICIES = {
  precisionPrecedence: 'step_then_kernel_profile_then_session_default',
  unsupportedPrecision: 'error',
  dtypeTransition: 'require_cast_step',
  unresolvedKernel: 'error',
};

export const DEFAULT_EXECUTION_V0_CONFIG = {
  model: null,
  sessionDefaults: { ...DEFAULT_EXECUTION_V0_SESSION_DEFAULTS },
  execution: {
    steps: [],
    policies: { ...DEFAULT_EXECUTION_V0_POLICIES },
  },
};

export const DEFAULT_EXECUTION_V0_PATCH = {
  set: [],
  remove: [],
  add: [],
};

export function isExecutionV0Digest(value) {
  return typeof value === 'string' && EXECUTION_V0_HASH_PATTERN.test(value);
}

export function isExecutionV0Semver(value) {
  return typeof value === 'string' && EXECUTION_V0_SEMVER_PATTERN.test(value);
}

