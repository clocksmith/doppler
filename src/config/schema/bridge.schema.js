/**
 * Bridge Config Schema
 *
 * Configuration for the native messaging bridge between DOPPLER and the
 * Chrome extension. Controls security boundaries and resource limits.
 *
 * @module config/schema/bridge
 */

// =============================================================================
// Bridge Config
// =============================================================================

/** Default bridge configuration */
export const DEFAULT_BRIDGE_CONFIG = {
  maxReadSizeBytes: 100 * 1024 * 1024, // 100MB
  allowedDirectories: '/Users:/home:/tmp:/var/tmp',
};
