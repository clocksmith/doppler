// =============================================================================
// Timeout Config
// =============================================================================

export const DEFAULT_BRIDGE_TIMEOUT_CONFIG = {
  pingTimeoutMs: 5000,
  readTimeoutMs: 60000,
  listTimeoutMs: 30000,
  defaultTimeoutMs: 30000,
};

// =============================================================================
// Bridge Config
// =============================================================================

export const DEFAULT_BRIDGE_CONFIG = {
  maxReadSizeBytes: 100 * 1024 * 1024, // 100MB
  allowedDirectories: '/Users:/home:/tmp:/var/tmp',
  timeouts: DEFAULT_BRIDGE_TIMEOUT_CONFIG,
};
