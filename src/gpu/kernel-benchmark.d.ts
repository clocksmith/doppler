/**
 * Kernel Benchmark Harness (Tier 2 P0)
 *
 * Infrastructure to measure kernel performance with comprehensive metrics:
 * - Throughput (GB/s)
 * - Latency (ms)
 * - FLOPS (computational efficiency)
 * - Before/after comparisons
 *
 * Outputs JSON results for tracking performance over time.
 */

/** Benchmark result for a single kernel */
export interface KernelBenchmarkResult {
  kernel: string;
  variant: string;
  config: Record<string, number>;
  latency: {
    median_ms: number;
    min_ms: number;
    max_ms: number;
    p95_ms: number;
    p99_ms: number;
    stddev_ms: number;
  };
  throughput: {
    gb_per_sec: number;
    elements_per_sec: number;
  };
  flops: {
    gflops: number;
    theoretical_gflops: number;
    efficiency_pct: number;
  };
  memory: {
    read_bytes: number;
    write_bytes: number;
    total_bytes: number;
  };
  iterations: number;
  warmup_iterations: number;
  timestamp: string;
}

/** Comparison result between two benchmark runs */
export interface BenchmarkComparison {
  baseline: KernelBenchmarkResult;
  optimized: KernelBenchmarkResult;
  speedup: number;
  latency_reduction_pct: number;
  throughput_increase_pct: number;
}

/** Full benchmark report */
export interface BenchmarkReport {
  device_info: {
    vendor: string;
    architecture: string;
    max_workgroup_size: number;
    max_shared_memory: number;
    has_f16: boolean;
    has_subgroups: boolean;
  };
  model_config: {
    name: string;
    hidden_size: number;
    intermediate_size: number;
    num_heads: number;
    num_kv_heads: number;
    head_dim: number;
    num_layers: number;
    vocab_size: number;
  };
  results: KernelBenchmarkResult[];
  comparisons: BenchmarkComparison[];
  summary: {
    total_decode_latency_ms: number;
    estimated_tok_per_sec: number;
    bottleneck_kernel: string;
    bottleneck_percentage: number;
  };
  generated_at: string;
}

/** Benchmark configuration */
export interface BenchmarkConfig {
  warmupIterations?: number;
  timedIterations?: number;
  modelConfig?: {
    hiddenSize?: number;
    intermediateSize?: number;
    numHeads?: number;
    numKVHeads?: number;
    headDim?: number;
    vocabSize?: number;
    numLayers?: number;
  };
}

/**
 * Benchmark matmul kernel
 */
export function benchmarkMatmul(
  M: number,
  N: number,
  K: number,
  options?: BenchmarkConfig
): Promise<KernelBenchmarkResult>;

/**
 * Benchmark attention decode kernel
 */
export function benchmarkAttentionDecode(
  numHeads: number,
  headDim: number,
  kvLen: number,
  options?: BenchmarkConfig
): Promise<KernelBenchmarkResult>;

/**
 * Benchmark RMSNorm kernel
 */
export function benchmarkRMSNorm(
  batchSize: number,
  hiddenSize: number,
  options?: BenchmarkConfig
): Promise<KernelBenchmarkResult>;

/**
 * Benchmark SiLU kernel
 */
export function benchmarkSiLU(
  size: number,
  options?: BenchmarkConfig
): Promise<KernelBenchmarkResult>;

/**
 * Benchmark fused Matmul+RMSNorm kernel
 *
 * Compares separate matmul + rmsnorm vs fused kernel for decode (M=1)
 */
export function benchmarkMatmulRMSNormFused(
  N: number,
  K: number,
  options?: BenchmarkConfig
): Promise<{
  separate: KernelBenchmarkResult;
  fused: KernelBenchmarkResult;
  comparison: BenchmarkComparison;
}>;

/**
 * Run comprehensive decode benchmark (one token generation)
 */
export function benchmarkDecodePass(
  options?: BenchmarkConfig
): Promise<BenchmarkReport>;

/**
 * Compare two benchmark results
 */
export function compareBenchmarks(
  baseline: KernelBenchmarkResult,
  optimized: KernelBenchmarkResult
): BenchmarkComparison;

/**
 * Export benchmark report as JSON
 */
export function exportBenchmarkJSON(report: BenchmarkReport): string;

/**
 * Print benchmark report to console
 */
export function printBenchmarkReport(report: BenchmarkReport): void;
