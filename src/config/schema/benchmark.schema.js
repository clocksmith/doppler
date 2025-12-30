// =============================================================================
// Benchmark Output Schema
// =============================================================================

export const DEFAULT_BENCHMARK_OUTPUT_CONFIG = {
  schemaVersion: 1,
};

// =============================================================================
// Benchmark Run Config
// =============================================================================

export const DEFAULT_BENCHMARK_RUN_CONFIG = {
  promptName: 'medium',
  customPrompt: null,
  maxNewTokens: 128,
  runType: 'warm',
  warmupRuns: 2,
  timedRuns: 3,
  sampling: {
    temperature: 0,
    topK: 1,
    topP: 1,
  },
  debug: false,
  profile: false,
  useChatTemplate: undefined,
  // Memory time series capture (VRAM over time)
  captureMemoryTimeSeries: false,
  memoryTimeSeriesIntervalMs: 100,
};

// =============================================================================
// Benchmark Stats Config
// =============================================================================

export const DEFAULT_BENCHMARK_STATS_CONFIG = {
  outlierIqrMultiplier: 1.5,
  warmupStabilityPercent: 10,
  thermalSlowdownPercent: 10,
  minSamplesForComparison: 3,
};

// =============================================================================
// Benchmark Comparison Config
// =============================================================================

export const DEFAULT_BENCHMARK_COMPARISON_CONFIG = {
  regressionThresholdPercent: 10,
  failOnRegression: true,
};

// =============================================================================
// Benchmark Baseline Registry Config
// =============================================================================

export const DEFAULT_BENCHMARK_BASELINE_CONFIG = {
  enabled: true,
  file: 'tests/baselines.json',
  failOnOutOfRange: true,
  requireQualityOk: true,
};

// =============================================================================
// Combined Benchmark Config
// =============================================================================

export const DEFAULT_BENCHMARK_CONFIG = {
  output: DEFAULT_BENCHMARK_OUTPUT_CONFIG,
  run: DEFAULT_BENCHMARK_RUN_CONFIG,
  stats: DEFAULT_BENCHMARK_STATS_CONFIG,
  comparison: DEFAULT_BENCHMARK_COMPARISON_CONFIG,
  baselines: DEFAULT_BENCHMARK_BASELINE_CONFIG,
};
