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

/**
 * Configuration for the native messaging bridge.
 *
 * Controls file access permissions and resource limits for the native host
 * process that provides filesystem access to the browser extension.
 */
export interface BridgeConfigSchema {
  /** Maximum bytes to read per request to prevent OOM (default: 100MB) */
  maxReadSizeBytes: number;

  /** Colon-separated list of allowed directory paths for file access */
  allowedDirectories: string;
}

/** Default bridge configuration */
export const DEFAULT_BRIDGE_CONFIG: BridgeConfigSchema = {
  maxReadSizeBytes: 100 * 1024 * 1024, // 100MB
  allowedDirectories: '/Users:/home:/tmp:/var/tmp',
};
