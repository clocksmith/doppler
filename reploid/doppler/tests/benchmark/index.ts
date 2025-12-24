/**
 * DOPPLER Benchmark Harness
 *
 * Public API for benchmarking.
 *
 * @module tests/benchmark
 */

// Types
export * from './types.js';

// Standard prompts
export * from './prompts.js';

// Pipeline benchmarks
export {
  PipelineBenchmark,
  runQuickBenchmark,
  runFullBenchmark,
  formatBenchmarkSummary,
} from './pipeline-benchmark.js';

// Debug utilities (bundled by build:benchmark so CLI --trace works without extra HTTP modules)
export { setDebugCategories, DEBUG_PRESETS } from '../../inference/pipeline/debug-utils.js';

// Benchmark mode (silences console.log for clean timing)
export { setBenchmarkMode, isBenchmarkMode } from '../../debug/index.js';

// System benchmarks
export {
  SystemBenchmark,
  runSystemBenchmark,
  formatSystemSummary,
  type SystemBenchmarkConfig,
  type SystemBenchmarkResult,
  type StorageMetrics,
  type DownloadMetrics,
  type OPFSMetrics,
} from './system-benchmark.js';

// Results storage
export {
  // File naming
  generateResultFilename,
  generateSessionFilename,
  // IndexedDB storage
  saveResult,
  loadAllResults,
  loadResultsBySuite,
  loadResultsByModel,
  clearAllResults,
  // JSON export/import
  exportToJSON,
  exportResultToJSON,
  importFromJSON,
  downloadAsJSON,
  // Comparison
  comparePipelineResults,
  formatComparison,
  type ComparisonDelta,
  // Sessions
  createSession,
  addResultToSession,
  computeSessionSummary,
} from './results-storage.js';
