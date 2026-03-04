// =============================================================================
// Hot-Swap Config
// =============================================================================

export const DEFAULT_HOTSWAP_CONFIG = {
  enabled: false,
  localOnly: false,
  allowUnsignedLocal: false,
  policyVersion: 1,
  rollout: {
    mode: 'shadow',
    canaryPercent: 0,
    cohortSalt: 'doppler-hotswap-v1',
    optInAllowlist: [],
  },
  trustedSigners: [],
  manifestUrl: null,
};
