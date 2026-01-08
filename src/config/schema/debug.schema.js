/**
 * Debug Config Schema
 *
 * Configuration for the DOPPLER debug module, including log history limits,
 * default log levels, trace categories, and decode step limits.
 *
 * @module config/schema/debug
 */

// =============================================================================
// Log Output Config
// =============================================================================

/** Default log output configuration */
export const DEFAULT_LOG_OUTPUT_CONFIG = {
  stdout: true,
  file: null,
  append: true,
};

// =============================================================================
// Log History Config
// =============================================================================

/** Default log history configuration */
export const DEFAULT_LOG_HISTORY_CONFIG = {
  maxLogHistoryEntries: 1000,
};

// =============================================================================
// Log Level Config
// =============================================================================

/** Valid log levels */
export const LOG_LEVELS = ['debug', 'verbose', 'info', 'warn', 'error', 'silent'];

/** Default log level configuration */
export const DEFAULT_LOG_LEVEL_CONFIG = {
  defaultLogLevel: 'info',
};

// =============================================================================
// Trace Config
// =============================================================================

/** Default trace configuration */
export const DEFAULT_TRACE_CONFIG = {
  enabled: false,
  categories: ['all'],
  layers: null,
  maxDecodeSteps: 0,
  file: null,
};

// =============================================================================
// Pipeline Debug Config (debug-utils)
// =============================================================================

/** Default pipeline debug configuration */
export const DEFAULT_PIPELINE_DEBUG_CONFIG = {
  enabled: false,
  categories: [],
  layers: null,
  maxDecodeSteps: 0,
  maxAbsThreshold: 10000,
  bufferStats: false,
  readbackSampleSize: 512,
};

// =============================================================================
// Complete Debug Config
// =============================================================================

/** Default debug configuration */
export const DEFAULT_DEBUG_CONFIG = {
  logOutput: DEFAULT_LOG_OUTPUT_CONFIG,
  logHistory: DEFAULT_LOG_HISTORY_CONFIG,
  logLevel: DEFAULT_LOG_LEVEL_CONFIG,
  trace: DEFAULT_TRACE_CONFIG,
  pipeline: DEFAULT_PIPELINE_DEBUG_CONFIG,
  probes: [],
};
