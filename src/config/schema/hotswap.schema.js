/**
 * Hot-Swap Config Schema
 *
 * Security policy for swapping JS/WGSL/JSON artifacts at runtime.
 *
 * @module config/schema/hotswap
 */

// =============================================================================
// Hot-Swap Config
// =============================================================================

/** Default hot-swap configuration */
export const DEFAULT_HOTSWAP_CONFIG = {
  enabled: false,
  localOnly: false,
  allowUnsignedLocal: false,
  trustedSigners: [],
  manifestUrl: null,
};
