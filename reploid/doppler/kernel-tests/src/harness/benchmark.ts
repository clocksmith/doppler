/**
 * GPU Benchmark Harness
 */

export interface BenchmarkOptions {
  warmupRuns?: number;
  timedRuns?: number;
  label?: string;
}

export interface BenchmarkStats {
  label: string;
  medianMs: number;
  meanMs: number;
  minMs: number;
  maxMs: number;
  p95Ms: number;
  p99Ms: number;
  stdDevMs: number;
  ci95Ms: number; // 95% confidence interval half-width
  samples: number;
  samplesAfterOutlierRemoval: number;
  outliersRemoved: number;
  rawTimes: number[];
  warnings: string[];
}

export interface BenchmarkStatsWithMetrics extends BenchmarkStats {
  flops?: number;
  gflops?: number;
  bytesTransferred?: number;
  throughputGbps?: number;
}

// ============================================================================
// Statistical Helper Functions
// ============================================================================

/**
 * Compute percentile using linear interpolation
 * @param sorted Sorted array of values
 * @param p Percentile (0-100)
 */
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/**
 * Compute median (average of middle two for even-length arrays)
 * @param sorted Sorted array of values
 */
function median(sorted: number[]): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const mid = Math.floor(n / 2);
  return n % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

/**
 * Remove outliers using IQR method
 * @param times Array of timing samples
 * @returns Object with filtered array and count of removed outliers
 */
function removeOutliers(times: number[]): { filtered: number[]; removed: number } {
  if (times.length < 4) return { filtered: times, removed: 0 };

  const sorted = [...times].sort((a, b) => a - b);
  const q1 = percentile(sorted, 25);
  const q3 = percentile(sorted, 75);
  const iqr = q3 - q1;
  const lower = q1 - 1.5 * iqr;
  const upper = q3 + 1.5 * iqr;

  const filtered = times.filter((t) => t >= lower && t <= upper);
  return { filtered, removed: times.length - filtered.length };
}

/**
 * Compute sample standard deviation (n-1 denominator)
 */
function sampleStdDev(values: number[], mean: number): number {
  const n = values.length;
  if (n < 2) return 0;
  const variance = values.reduce((sum, t) => sum + (t - mean) ** 2, 0) / (n - 1);
  return Math.sqrt(variance);
}

/**
 * Compute 95% confidence interval half-width
 */
function confidenceInterval95(stdDev: number, n: number): number {
  if (n < 2) return 0;
  // For n >= 30, use 1.96 (z-score). For smaller n, use t-distribution approximation.
  const tValue = n >= 30 ? 1.96 : 2.0 + 3.0 / n; // Rough approximation for t-distribution
  return tValue * (stdDev / Math.sqrt(n));
}

/**
 * Kernel benchmark runner with proper GPU synchronization
 */
export class KernelBenchmark {
  device: GPUDevice;
  hasTimestamp: boolean;

  constructor(device: GPUDevice) {
    this.device = device;
    this.hasTimestamp = device.features?.has('timestamp-query') || false;
  }

  /**
   * Run benchmark with warmup and multiple iterations
   * @param kernelFn Async function that runs the kernel
   * @param options Benchmark options
   * @returns Promise resolving to benchmark statistics
   */
  async runBenchmark(kernelFn: () => Promise<void>, options: BenchmarkOptions = {}): Promise<BenchmarkStats> {
    const { warmupRuns = 5, timedRuns = 20, label = 'kernel' } = options;
    const warnings: string[] = [];

    // Warmup (compile shaders, warm caches)
    const warmupTimes: number[] = [];
    for (let i = 0; i < warmupRuns; i++) {
      const start = performance.now();
      await kernelFn();
      await this.device.queue.onSubmittedWorkDone();
      warmupTimes.push(performance.now() - start);
    }

    // Verify warmup stabilized (last two runs within 10% of each other)
    if (warmupTimes.length >= 2) {
      const last = warmupTimes[warmupTimes.length - 1];
      const prev = warmupTimes[warmupTimes.length - 2];
      if (Math.abs(last - prev) / Math.max(prev, 0.001) > 0.1) {
        warnings.push('Warmup may not have stabilized (>10% variance in last 2 runs)');
      }
    }

    // Timed runs
    const times: number[] = [];
    for (let i = 0; i < timedRuns; i++) {
      const start = performance.now();
      await kernelFn();
      await this.device.queue.onSubmittedWorkDone();
      const end = performance.now();
      times.push(end - start);
    }

    // Detect thermal throttling (last 3 runs >10% slower than first 3)
    if (times.length >= 6) {
      const firstThree = times.slice(0, 3);
      const lastThree = times.slice(-3);
      const firstAvg = firstThree.reduce((a, b) => a + b, 0) / 3;
      const lastAvg = lastThree.reduce((a, b) => a + b, 0) / 3;
      if (lastAvg > firstAvg * 1.1) {
        warnings.push(`Possible thermal throttling detected (last runs ${((lastAvg / firstAvg - 1) * 100).toFixed(1)}% slower)`);
      }
    }

    const stats = this.computeStats(times, label);
    stats.warnings.push(...warnings);
    return stats;
  }

  /**
   * Compute statistics from timing samples
   * @param times Array of timing samples in ms
   * @param label Benchmark label
   * @returns Benchmark statistics
   */
  computeStats(times: number[], label: string): BenchmarkStats {
    const warnings: string[] = [];

    // Remove outliers using IQR method
    const { filtered, removed } = removeOutliers(times);
    if (removed > 0) {
      warnings.push(`Removed ${removed} outlier(s)`);
    }

    const sorted = [...filtered].sort((a, b) => a - b);
    const n = sorted.length;

    if (n === 0) {
      return {
        label,
        medianMs: 0,
        meanMs: 0,
        minMs: 0,
        maxMs: 0,
        p95Ms: 0,
        p99Ms: 0,
        stdDevMs: 0,
        ci95Ms: 0,
        samples: times.length,
        samplesAfterOutlierRemoval: 0,
        outliersRemoved: removed,
        rawTimes: times,
        warnings: ['No valid samples after outlier removal'],
      };
    }

    const medianVal = median(sorted);
    const meanVal = filtered.reduce((a, b) => a + b, 0) / n;
    const minVal = sorted[0];
    const maxVal = sorted[n - 1];
    const p95Val = percentile(sorted, 95);
    const p99Val = percentile(sorted, 99);

    // Sample standard deviation (n-1 denominator for unbiased estimate)
    const stdDev = sampleStdDev(filtered, meanVal);

    // 95% confidence interval
    const ci95 = confidenceInterval95(stdDev, n);

    return {
      label,
      medianMs: medianVal,
      meanMs: meanVal,
      minMs: minVal,
      maxMs: maxVal,
      p95Ms: p95Val,
      p99Ms: p99Val,
      stdDevMs: stdDev,
      ci95Ms: ci95,
      samples: times.length,
      samplesAfterOutlierRemoval: n,
      outliersRemoved: removed,
      rawTimes: times,
      warnings,
    };
  }
}

export type OperationType = 'matmul' | 'attention' | 'softmax' | string;

export interface MatmulWorkload {
  M: number;
  N: number;
  K: number;
  elementSize?: number;
  operation: 'matmul';
}

export interface AttentionWorkload {
  seqLen: number;
  numHeads: number;
  headDim: number;
  kvLen?: number;
  elementSize?: number;
  operation: 'attention';
}

export interface SoftmaxWorkload {
  innerSize: number;
  outerSize: number;
  elementSize?: number;
  operation: 'softmax';
}

export interface GenericWorkload {
  M: number;
  N: number;
  elementSize?: number;
  operation?: string;
}

export type Workload = MatmulWorkload | AttentionWorkload | SoftmaxWorkload | GenericWorkload;

/**
 * Compute theoretical performance metrics
 * @param stats Benchmark statistics
 * @param workload Workload parameters
 * @returns Stats with added metrics
 */
export function computeMetrics(stats: BenchmarkStats, workload: Workload): BenchmarkStatsWithMetrics {
  const { elementSize = 4, operation = 'matmul' } = workload;

  let flops = 0;
  let bytesTransferred = 0;

  switch (operation) {
    case 'matmul': {
      const { M, N, K } = workload as MatmulWorkload;
      // FLOPs for matmul: 2*M*N*K (multiply-add for each output element)
      flops = 2 * M * N * K;
      // Memory: read A + B, write C
      bytesTransferred = (M * K + K * N + M * N) * elementSize;
      break;
    }

    case 'attention': {
      const { seqLen, numHeads, headDim, kvLen = seqLen } = workload as AttentionWorkload;
      // Q @ K^T: 2*seqLen*kvLen*headDim per head
      // Softmax: ~5*seqLen*kvLen per head
      // Scores @ V: 2*seqLen*headDim*kvLen per head
      flops = numHeads * (2 * seqLen * kvLen * headDim + 5 * seqLen * kvLen + 2 * seqLen * headDim * kvLen);
      bytesTransferred = numHeads * (seqLen + kvLen * 2 + seqLen) * headDim * elementSize;
      break;
    }

    case 'softmax': {
      const { innerSize, outerSize } = workload as SoftmaxWorkload;
      // ~5 ops per element (exp, sum, div, max finding)
      flops = 5 * innerSize * outerSize;
      bytesTransferred = 2 * innerSize * outerSize * elementSize;
      break;
    }

    default: {
      const { M, N } = workload as GenericWorkload;
      // Generic: assume 1 flop per element
      flops = M * N;
      bytesTransferred = 2 * M * N * elementSize;
    }
  }

  const seconds = stats.medianMs / 1000;
  const gflops = (flops / 1e9) / seconds;
  const throughputGbps = (bytesTransferred / 1e9) / seconds;

  return {
    ...stats,
    flops,
    gflops,
    bytesTransferred,
    throughputGbps,
  };
}

/**
 * Format benchmark results for display
 * @param stats Benchmark statistics
 * @returns Formatted string
 */
export function formatBenchmarkResult(stats: BenchmarkStatsWithMetrics): string {
  const lines = [
    `${stats.label}:`,
    `  Median: ${stats.medianMs.toFixed(3)} ms (+/- ${stats.ci95Ms.toFixed(3)} ms, 95% CI)`,
    `  Mean: ${stats.meanMs.toFixed(3)} ms`,
    `  Min/Max: ${stats.minMs.toFixed(3)} / ${stats.maxMs.toFixed(3)} ms`,
    `  P95/P99: ${stats.p95Ms.toFixed(3)} / ${stats.p99Ms.toFixed(3)} ms`,
    `  StdDev: ${stats.stdDevMs.toFixed(3)} ms`,
    `  Samples: ${stats.samplesAfterOutlierRemoval}/${stats.samples} (${stats.outliersRemoved} outliers removed)`,
  ];

  if (stats.gflops) {
    lines.push(`  GFLOPS: ${stats.gflops.toFixed(1)}`);
  }
  if (stats.throughputGbps) {
    lines.push(`  Throughput: ${stats.throughputGbps.toFixed(1)} GB/s`);
  }

  if (stats.warnings && stats.warnings.length > 0) {
    lines.push(`  Warnings: ${stats.warnings.join('; ')}`);
  }

  return lines.join('\n');
}

export interface MatmulConfig {
  M: number;
  N: number;
  K: number;
  label: string;
}

export interface AttentionConfig {
  seqLen: number;
  numHeads: number;
  headDim: number;
  kvLen?: number;
  label: string;
}

export interface MoeConfig {
  numTokens: number;
  hiddenSize: number;
  numExperts: number;
  topK: number;
  label: string;
}

export interface SoftmaxConfig {
  innerSize: number;
  outerSize: number;
  label: string;
}

/**
 * Standard benchmark workload configurations
 */
export const BENCHMARK_CONFIGS = {
  matmul: [
    // Decode (single token)
    { M: 1, N: 4096, K: 4096, label: 'decode-qkv' },
    { M: 1, N: 14336, K: 4096, label: 'decode-ffn-up' },
    { M: 1, N: 4096, K: 14336, label: 'decode-ffn-down' },
    // Prefill
    { M: 128, N: 4096, K: 4096, label: 'prefill-128' },
    { M: 512, N: 4096, K: 4096, label: 'prefill-512' },
    { M: 2048, N: 4096, K: 4096, label: 'prefill-2k' },
  ] as MatmulConfig[],

  attention: [
    { seqLen: 128, numHeads: 32, headDim: 128, label: 'prefill-128' },
    { seqLen: 512, numHeads: 32, headDim: 128, label: 'prefill-512' },
    { seqLen: 1, kvLen: 2048, numHeads: 32, headDim: 128, label: 'decode-2k' },
  ] as AttentionConfig[],

  moe: [
    { numTokens: 1, hiddenSize: 4096, numExperts: 8, topK: 2, label: 'decode-8e' },
    { numTokens: 128, hiddenSize: 4096, numExperts: 8, topK: 2, label: 'prefill-128-8e' },
  ] as MoeConfig[],

  softmax: [
    { innerSize: 128, outerSize: 32, label: 'attention-scores' },
    { innerSize: 32000, outerSize: 1, label: 'lm-head-vocab' },
    { innerSize: 128000, outerSize: 1, label: 'lm-head-large' },
  ] as SoftmaxConfig[],
};
