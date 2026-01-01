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

// Debug utilities (unified in debug/index.ts)
export { setTrace, setLogLevel, setBenchmarkMode, isBenchmarkMode, TRACE_CATEGORIES } from '../../src/debug/index.js';

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
