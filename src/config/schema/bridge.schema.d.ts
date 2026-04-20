/**
 * Bridge Config Schema
 *
 * Configuration for the native messaging bridge between DOPPLER and the
 * Chrome extension. Controls security boundaries and resource limits.
 *
 * @module config/schema/bridge
 */

/**
 * Per-operation timeouts for the native messaging bridge.
 */
export interface BridgeTimeoutConfigSchema {
  pingTimeoutMs: number;
  readTimeoutMs: number;
  listTimeoutMs: number;
  defaultTimeoutMs: number;
}

/** Default bridge timeout configuration */
export declare const DEFAULT_BRIDGE_TIMEOUT_CONFIG: BridgeTimeoutConfigSchema;

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

  /** Per-operation timeouts for bridge requests */
  timeouts: BridgeTimeoutConfigSchema;
}

/** Default bridge configuration */
export declare const DEFAULT_BRIDGE_CONFIG: BridgeConfigSchema;
