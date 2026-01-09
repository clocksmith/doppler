/**
 * Benchmark Types
 *
 * JavaScript module for the DOPPLER benchmark harness.
 * Follows the JSON schema defined in docs/spec/BENCHMARK_HARNESS.md
 *
 * @module tests/benchmark/types
 */

/**
 * Default benchmark configuration
 */
export const DEFAULT_BENCHMARK_CONFIG = {
  promptName: 'medium',
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
  useChatTemplate: undefined,  // Auto-detect based on model name
};
