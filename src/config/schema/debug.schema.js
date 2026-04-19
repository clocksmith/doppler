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
// Loader Debug Config
// =============================================================================

export const DEFAULT_LOADER_DEBUG_CONFIG = {
  enabled: false,
  forceGpuDequant: false,
  preferCpuDequant: false,
  failOnCpuDequantPath: false,
  runQ4KDequantParity: false,
  q4kDequantParitySamples: 256,
};

// =============================================================================
// Kernel Debug Config
// =============================================================================

export const DEFAULT_MATMUL_DEBUG_CONFIG = {
  enabled: false,
  forceSplitQKV: false,
  validateAttentionWeightBuffer: false,
  failOnSmallAttentionWeightBuffer: false,
  logAttentionWeightBuffer: false,
  logProjectionValues: false,
};

// =============================================================================
// Kernel Trace Config (kernel-trace.js anomaly detection)
// =============================================================================

export const DEFAULT_KERNEL_TRACE_CONFIG = {
  layers: [],
  breakOnAnomaly: false,
  explosionThreshold: 10,
  collapseThreshold: 1e-6,
  maxSteps: 5000,
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
  logEveryDecodeSteps: 64,
};

// =============================================================================
// Perf Guard Config
// =============================================================================

export const DEFAULT_PERF_GUARDS_CONFIG = {
  allowGPUReadback: true,
  trackSubmitCount: false,
  trackAllocations: false,
  logExpensiveOps: false,
  strictMode: false,
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
  loader: DEFAULT_LOADER_DEBUG_CONFIG,
  matmul: DEFAULT_MATMUL_DEBUG_CONFIG,
  kernelTrace: DEFAULT_KERNEL_TRACE_CONFIG,
  probes: [],
  profiler: DEFAULT_PROFILER_CONFIG,
  perfGuards: DEFAULT_PERF_GUARDS_CONFIG,
};
