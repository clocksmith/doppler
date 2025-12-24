/**
 * Test Harness Exports
 */

export {
  compareArrays,
  compareIntArrays,
  generateTestData,
  verifySumTo,
  verifyRange,
  KERNEL_TOLERANCES,
  type ToleranceConfig,
  type TopKToleranceConfig,
  type KernelTolerances,
  type Mismatch,
  type ComparisonResult,
  type IntMismatch,
  type IntComparisonResult,
  type DataType,
  type GenerateOptions,
  type SumVerificationResult,
  type RangeVerificationResult,
} from './tolerance.js';

export {
  createBuffer,
  createEmptyBuffer,
  readGPUBuffer,
  readAsFloat32,
  readAsUint32,
  uploadToBuffer,
  clearBuffer,
  type BufferUsage,
} from './buffer-utils.js';

export {
  KernelBenchmark,
  computeMetrics,
  formatBenchmarkResult,
  BENCHMARK_CONFIGS,
  type BenchmarkOptions,
  type BenchmarkStats,
  type BenchmarkStatsWithMetrics,
  type OperationType,
  type MatmulWorkload,
  type AttentionWorkload,
  type SoftmaxWorkload,
  type GenericWorkload,
  type Workload,
  type MatmulConfig,
  type AttentionConfig,
  type MoeConfig,
  type SoftmaxConfig,
} from './benchmark.js';
