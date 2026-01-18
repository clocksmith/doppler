// =============================================================================
// Log Output Config
// =============================================================================

export const DEFAULT_LOG_OUTPUT_CONFIG = {
  stdout: true,
  file: null,
  append: true,
};

// =============================================================================
// Log History Config
// =============================================================================

export const DEFAULT_LOG_HISTORY_CONFIG = {
  maxLogHistoryEntries: 1000,
};

// =============================================================================
// Log Level Config
// =============================================================================

export const LOG_LEVELS = ['debug', 'verbose', 'info', 'warn', 'error', 'silent'];

export const DEFAULT_LOG_LEVEL_CONFIG = {
  defaultLogLevel: 'info',
};

// =============================================================================
// Trace Config
// =============================================================================

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
// Profiler Config
// =============================================================================

export const DEFAULT_PROFILER_CONFIG = {
  enabled: false,
  queryCapacity: 256,
  maxQueries: 16384,
  defaultQueryLimit: 4096,
  maxSamples: 100,
  maxDurationMs: 60000,
};

// =============================================================================
// Complete Debug Config
// =============================================================================

export const DEFAULT_DEBUG_CONFIG = {
  logOutput: DEFAULT_LOG_OUTPUT_CONFIG,
  logHistory: DEFAULT_LOG_HISTORY_CONFIG,
  logLevel: DEFAULT_LOG_LEVEL_CONFIG,
  trace: DEFAULT_TRACE_CONFIG,
  pipeline: DEFAULT_PIPELINE_DEBUG_CONFIG,
  probes: [],
  profiler: DEFAULT_PROFILER_CONFIG,
};
