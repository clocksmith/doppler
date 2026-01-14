/**
 * GPU Benchmark Harness
 */

import { computeSampleStats } from '../../../src/debug/stats.js';
import { DEFAULT_BENCHMARK_STATS_CONFIG } from '../../../src/config/schema/benchmark.schema.js';

/**
 * Kernel benchmark runner with proper GPU synchronization
 */
export class KernelBenchmark {
  constructor(device) {
    this.device = device;
    this.hasTimestamp = device.features?.has('timestamp-query') || false;
  }

  /**
   * Run benchmark with warmup and multiple iterations
   * @param kernelFn Async function that runs the kernel
   * @param options Benchmark options
   * @returns Promise resolving to benchmark statistics
   */
  async runBenchmark(kernelFn, options = {}) {
    const { warmupRuns = 5, timedRuns = 20, label = 'kernel' } = options;
    const warnings = [];

    // Warmup (compile shaders, warm caches)
    const warmupTimes = [];
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
      const thresholdPercent = DEFAULT_BENCHMARK_STATS_CONFIG.warmupStabilityPercent;
      const threshold = thresholdPercent / 100;
      if (Math.abs(last - prev) / Math.max(prev, 0.001) > threshold) {
        warnings.push(`Warmup may not have stabilized (>${thresholdPercent}% variance in last 2 runs)`);
      }
    }

    // Timed runs
    const times = [];
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
      const threshold = DEFAULT_BENCHMARK_STATS_CONFIG.thermalSlowdownPercent / 100;
      if (lastAvg > firstAvg * (1 + threshold)) {
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
  computeStats(times, label) {
    const warnings = [];

    const stats = computeSampleStats(times, {
      outlierIqrMultiplier: DEFAULT_BENCHMARK_STATS_CONFIG.outlierIqrMultiplier,
    });
    if (stats.outliersRemoved > 0) {
      warnings.push(`Removed ${stats.outliersRemoved} outlier(s)`);
    }

    if (stats.samplesAfterOutlierRemoval === 0) {
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
        outliersRemoved: stats.outliersRemoved,
        rawTimes: times,
        warnings: ['No valid samples after outlier removal'],
      };
    }

    return {
      label,
      medianMs: stats.median,
      meanMs: stats.mean,
      minMs: stats.min,
      maxMs: stats.max,
      p95Ms: stats.p95,
      p99Ms: stats.p99,
      stdDevMs: stats.stdDev,
      ci95Ms: stats.ci95,
      samples: stats.samples,
      samplesAfterOutlierRemoval: stats.samplesAfterOutlierRemoval,
      outliersRemoved: stats.outliersRemoved,
      rawTimes: times,
      warnings,
    };
  }
}

/**
 * Compute theoretical performance metrics
 * @param stats Benchmark statistics
 * @param workload Workload parameters
 * @returns Stats with added metrics
 */
export function computeMetrics(stats, workload) {
  const { elementSize = 4, operation = 'matmul' } = workload;

  let flops = 0;
  let bytesTransferred = 0;

  switch (operation) {
    case 'matmul': {
      const { M, N, K } = workload;
      // FLOPs for matmul: 2*M*N*K (multiply-add for each output element)
      flops = 2 * M * N * K;
      // Memory: read A + B, write C
      bytesTransferred = (M * K + K * N + M * N) * elementSize;
      break;
    }

    case 'attention': {
      const { seqLen, numHeads, headDim, kvLen = seqLen } = workload;
      // Q @ K^T: 2*seqLen*kvLen*headDim per head
      // Softmax: ~5*seqLen*kvLen per head
      // Scores @ V: 2*seqLen*headDim*kvLen per head
      flops = numHeads * (2 * seqLen * kvLen * headDim + 5 * seqLen * kvLen + 2 * seqLen * headDim * kvLen);
      bytesTransferred = numHeads * (seqLen + kvLen * 2 + seqLen) * headDim * elementSize;
      break;
    }

    case 'softmax': {
      const { innerSize, outerSize } = workload;
      // ~5 ops per element (exp, sum, div, max finding)
      flops = 5 * innerSize * outerSize;
      bytesTransferred = 2 * innerSize * outerSize * elementSize;
      break;
    }

    default: {
      const { M, N } = workload;
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
export function formatBenchmarkResult(stats) {
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
  ],

  attention: [
    { seqLen: 128, numHeads: 32, headDim: 128, label: 'prefill-128' },
    { seqLen: 512, numHeads: 32, headDim: 128, label: 'prefill-512' },
    { seqLen: 1, kvLen: 2048, numHeads: 32, headDim: 128, label: 'decode-2k' },
  ],

  moe: [
    { numTokens: 1, hiddenSize: 4096, numExperts: 8, topK: 2, label: 'decode-8e' },
    { numTokens: 128, hiddenSize: 4096, numExperts: 8, topK: 2, label: 'prefill-128-8e' },
  ],

  softmax: [
    { innerSize: 128, outerSize: 32, label: 'attention-scores' },
    { innerSize: 32000, outerSize: 1, label: 'lm-head-vocab' },
    { innerSize: 128000, outerSize: 1, label: 'lm-head-large' },
  ],
};
