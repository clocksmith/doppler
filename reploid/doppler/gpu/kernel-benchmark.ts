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

import { getDevice, getDeviceLimits, getKernelCapabilities } from './device.js';
import { acquireBuffer, releaseBuffer } from './buffer-pool.js';
import { runMatmul } from './kernels/matmul.js';
import { runRMSNorm } from './kernels/rmsnorm.js';
import { runAttention } from './kernels/attention.js';
import { runSiLU } from './kernels/silu.js';
import { PERFORMANCE } from './kernels/constants.js';

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

const DEFAULT_CONFIG: Required<BenchmarkConfig> = {
  warmupIterations: PERFORMANCE.WARMUP_RUNS,
  timedIterations: PERFORMANCE.TIMED_RUNS,
  modelConfig: {
    hiddenSize: 1152,       // Gemma 3 1B
    intermediateSize: 6912, // Gemma 3 1B
    numHeads: 4,            // Gemma 3 1B
    numKVHeads: 1,          // Gemma 3 1B (GQA)
    headDim: 256,           // Gemma 3 1B
    vocabSize: 262144,      // Gemma 3 1B
    numLayers: 26,          // Gemma 3 1B
  },
};

/**
 * Calculate statistics from timing array
 */
function calculateStats(times: number[]): {
  median: number;
  min: number;
  max: number;
  p95: number;
  p99: number;
  stddev: number;
  mean: number;
} {
  const sorted = [...times].sort((a, b) => a - b);
  const n = sorted.length;

  const mean = times.reduce((a, b) => a + b, 0) / n;
  const variance = times.reduce((sum, t) => sum + (t - mean) ** 2, 0) / n;
  const stddev = Math.sqrt(variance);

  return {
    median: sorted[Math.floor(n / 2)],
    min: sorted[0],
    max: sorted[n - 1],
    p95: sorted[Math.floor(n * 0.95)],
    p99: sorted[Math.floor(n * 0.99)],
    stddev,
    mean,
  };
}

/**
 * Estimate FLOPS for different kernel types
 */
function estimateFLOPS(
  kernel: string,
  config: Record<string, number>,
  latencyMs: number
): { gflops: number; theoretical: number } {
  let flops = 0;

  switch (kernel) {
    case 'matmul':
      // GEMM: 2 * M * N * K FLOPs
      flops = 2 * (config.M || 1) * (config.N || 1) * (config.K || 1);
      break;
    case 'attention':
      // QK^T: 2 * seqLen * kvLen * headDim * numHeads
      // softmax: ~5 * seqLen * kvLen * numHeads
      // V: 2 * seqLen * kvLen * headDim * numHeads
      const qk = 2 * (config.seqLen || 1) * (config.kvLen || 1) * (config.headDim || 128) * (config.numHeads || 1);
      const sm = 5 * (config.seqLen || 1) * (config.kvLen || 1) * (config.numHeads || 1);
      const v = 2 * (config.seqLen || 1) * (config.kvLen || 1) * (config.headDim || 128) * (config.numHeads || 1);
      flops = qk + sm + v;
      break;
    case 'rmsnorm':
      // 2 * size (for variance) + 2 * size (for normalize)
      flops = 4 * (config.size || 1);
      break;
    case 'silu':
      // 2 ops per element (sigmoid, multiply)
      flops = 2 * (config.size || 1);
      break;
    default:
      flops = config.size || config.elements || 1;
  }

  const gflops = flops / (latencyMs * 1e6);
  // Theoretical peak: ~10 TFLOPS for M1 Pro, ~20 TFLOPS for M1 Max
  const theoretical = 10000; // Conservative estimate

  return { gflops, theoretical };
}

/**
 * Create random test buffer
 */
function createTestBuffer(size: number, label: string): GPUBuffer {
  const buffer = acquireBuffer(size, undefined, label);
  const device = getDevice();

  // Initialize with random data
  const data = new Float32Array(size / 4);
  for (let i = 0; i < data.length; i++) {
    data[i] = (Math.random() - 0.5) * 2;
  }
  device.queue.writeBuffer(buffer, 0, data);

  return buffer;
}

/**
 * Benchmark a single kernel execution
 */
async function benchmarkKernel(
  name: string,
  variant: string,
  config: Record<string, number>,
  runFn: () => Promise<void>,
  warmupIterations: number,
  timedIterations: number
): Promise<KernelBenchmarkResult> {
  const device = getDevice();

  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    await runFn();
    await device.queue.onSubmittedWorkDone();
  }

  // Timed runs
  const times: number[] = [];
  for (let i = 0; i < timedIterations; i++) {
    const start = performance.now();
    await runFn();
    await device.queue.onSubmittedWorkDone();
    times.push(performance.now() - start);
  }

  const stats = calculateStats(times);
  const flopsInfo = estimateFLOPS(name, config, stats.median);

  // Calculate memory bandwidth
  const readBytes = config.readBytes || config.size * 4 || 0;
  const writeBytes = config.writeBytes || config.size * 4 || 0;
  const totalBytes = readBytes + writeBytes;
  const gbPerSec = totalBytes / (stats.median * 1e6);

  return {
    kernel: name,
    variant,
    config,
    latency: {
      median_ms: stats.median,
      min_ms: stats.min,
      max_ms: stats.max,
      p95_ms: stats.p95,
      p99_ms: stats.p99,
      stddev_ms: stats.stddev,
    },
    throughput: {
      gb_per_sec: gbPerSec,
      elements_per_sec: (config.size || config.elements || 1) / (stats.median / 1000),
    },
    flops: {
      gflops: flopsInfo.gflops,
      theoretical_gflops: flopsInfo.theoretical,
      efficiency_pct: (flopsInfo.gflops / flopsInfo.theoretical) * 100,
    },
    memory: {
      read_bytes: readBytes,
      write_bytes: writeBytes,
      total_bytes: totalBytes,
    },
    iterations: timedIterations,
    warmup_iterations: warmupIterations,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Benchmark matmul kernel
 */
export async function benchmarkMatmul(
  M: number,
  N: number,
  K: number,
  options: BenchmarkConfig = {}
): Promise<KernelBenchmarkResult> {
  const config = { ...DEFAULT_CONFIG, ...options };
  const device = getDevice();

  const A = createTestBuffer(M * K * 4, 'bench_A');
  const B = createTestBuffer(K * N * 4, 'bench_B');

  const result = await benchmarkKernel(
    'matmul',
    'f32',
    {
      M, N, K,
      readBytes: (M * K + K * N) * 4,
      writeBytes: M * N * 4,
    },
    async () => {
      const C = await runMatmul(A, B, M, N, K);
      releaseBuffer(C);
    },
    config.warmupIterations,
    config.timedIterations
  );

  releaseBuffer(A);
  releaseBuffer(B);

  return result;
}

/**
 * Benchmark attention decode kernel
 */
export async function benchmarkAttentionDecode(
  numHeads: number,
  headDim: number,
  kvLen: number,
  options: BenchmarkConfig = {}
): Promise<KernelBenchmarkResult> {
  const config = { ...DEFAULT_CONFIG, ...options };

  const Q = createTestBuffer(numHeads * headDim * 4, 'bench_Q');
  const K = createTestBuffer(kvLen * numHeads * headDim * 4, 'bench_K');
  const V = createTestBuffer(kvLen * numHeads * headDim * 4, 'bench_V');

  const result = await benchmarkKernel(
    'attention',
    'decode',
    {
      seqLen: 1,
      kvLen,
      numHeads,
      headDim,
      readBytes: (numHeads * headDim + 2 * kvLen * numHeads * headDim) * 4,
      writeBytes: numHeads * headDim * 4,
    },
    async () => {
      const out = await runAttention(Q, K, V, null, numHeads, headDim, {
        seqLen: 1,
        kvLen,
        numKVHeads: numHeads,
      });
      releaseBuffer(out);
    },
    config.warmupIterations,
    config.timedIterations
  );

  releaseBuffer(Q);
  releaseBuffer(K);
  releaseBuffer(V);

  return result;
}

/**
 * Benchmark RMSNorm kernel
 */
export async function benchmarkRMSNorm(
  batchSize: number,
  hiddenSize: number,
  options: BenchmarkConfig = {}
): Promise<KernelBenchmarkResult> {
  const config = { ...DEFAULT_CONFIG, ...options };

  const size = batchSize * hiddenSize;
  const input = createTestBuffer(size * 4, 'bench_input');
  const weight = createTestBuffer(hiddenSize * 4, 'bench_weight');

  const result = await benchmarkKernel(
    'rmsnorm',
    'default',
    {
      batchSize,
      hiddenSize,
      size,
      readBytes: (size + hiddenSize) * 4,
      writeBytes: size * 4,
    },
    async () => {
      const out = await runRMSNorm(input, weight, 1e-6, { batchSize, hiddenSize });
      releaseBuffer(out);
    },
    config.warmupIterations,
    config.timedIterations
  );

  releaseBuffer(input);
  releaseBuffer(weight);

  return result;
}

/**
 * Benchmark SiLU kernel
 */
export async function benchmarkSiLU(
  size: number,
  options: BenchmarkConfig = {}
): Promise<KernelBenchmarkResult> {
  const config = { ...DEFAULT_CONFIG, ...options };

  const input = createTestBuffer(size * 4, 'bench_input');

  const result = await benchmarkKernel(
    'silu',
    'default',
    {
      size,
      readBytes: size * 4,
      writeBytes: size * 4,
    },
    async () => {
      const out = await runSiLU(input, { size });
      releaseBuffer(out);
    },
    config.warmupIterations,
    config.timedIterations
  );

  releaseBuffer(input);

  return result;
}

/**
 * Benchmark fused Matmul+RMSNorm kernel
 *
 * Compares separate matmul + rmsnorm vs fused kernel for decode (M=1)
 */
export async function benchmarkMatmulRMSNormFused(
  N: number,  // hiddenSize (output)
  K: number,  // intermediateSize (input)
  options: BenchmarkConfig = {}
): Promise<{
  separate: KernelBenchmarkResult;
  fused: KernelBenchmarkResult;
  comparison: BenchmarkComparison;
}> {
  const config = { ...DEFAULT_CONFIG, ...options };

  // Import fused kernel
  const { runMatmulRMSNormFused, shouldUseFusedMatmulRMSNorm } = await import('./kernels/fused_matmul_rmsnorm.js');

  if (!shouldUseFusedMatmulRMSNorm(1, N)) {
    throw new Error(`Fused kernel not supported for N=${N} (max 4096)`);
  }

  const input = createTestBuffer(K * 4, 'bench_input');
  const weight = createTestBuffer(K * N * 4, 'bench_weight');
  const normWeight = createTestBuffer(N * 4, 'bench_norm_weight');
  const residual = createTestBuffer(N * 4, 'bench_residual');

  // Benchmark separate: matmul + rmsnorm
  const separateResult = await benchmarkKernel(
    'matmul+rmsnorm',
    'separate',
    {
      M: 1, N, K,
      readBytes: (K + K * N + N + N) * 4,  // input + weight + norm_weight + residual
      writeBytes: N * 4,
    },
    async () => {
      const matmulOut = await runMatmul(input, weight, 1, N, K);
      const normOut = await runRMSNorm(matmulOut, normWeight, 1e-6, {
        batchSize: 1,
        hiddenSize: N,
        residual,
      });
      releaseBuffer(matmulOut);
      releaseBuffer(normOut);
    },
    config.warmupIterations,
    config.timedIterations
  );

  // Benchmark fused: matmul+rmsnorm in one kernel
  const fusedResult = await benchmarkKernel(
    'matmul+rmsnorm',
    'fused',
    {
      M: 1, N, K,
      readBytes: (K + K * N + N + N) * 4,
      writeBytes: N * 4,
    },
    async () => {
      const out = await runMatmulRMSNormFused(input, weight, normWeight, {
        N, K,
        eps: 1e-6,
        residual,
      });
      releaseBuffer(out);
    },
    config.warmupIterations,
    config.timedIterations
  );

  releaseBuffer(input);
  releaseBuffer(weight);
  releaseBuffer(normWeight);
  releaseBuffer(residual);

  // Calculate comparison
  const speedup = separateResult.latency.median_ms / fusedResult.latency.median_ms;
  const comparison: BenchmarkComparison = {
    baseline: separateResult,
    optimized: fusedResult,
    speedup,
    latency_reduction_pct: (1 - fusedResult.latency.median_ms / separateResult.latency.median_ms) * 100,
    throughput_increase_pct: (speedup - 1) * 100,
  };

  console.log(`[Benchmark] Matmul+RMSNorm (N=${N}, K=${K}):`);
  console.log(`  Separate: ${separateResult.latency.median_ms.toFixed(3)}ms`);
  console.log(`  Fused:    ${fusedResult.latency.median_ms.toFixed(3)}ms`);
  console.log(`  Speedup:  ${speedup.toFixed(2)}x`);

  return { separate: separateResult, fused: fusedResult, comparison };
}

/**
 * Run comprehensive decode benchmark (one token generation)
 */
export async function benchmarkDecodePass(
  options: BenchmarkConfig = {}
): Promise<BenchmarkReport> {
  const config = { ...DEFAULT_CONFIG.modelConfig, ...options.modelConfig };
  const device = getDevice();
  const limits = getDeviceLimits();
  const caps = getKernelCapabilities();

  const results: KernelBenchmarkResult[] = [];

  console.log('[Benchmark] Starting decode pass benchmark...');
  console.log(`[Benchmark] Model config: hidden=${config.hiddenSize}, intermediate=${config.intermediateSize}, heads=${config.numHeads}`);

  // 1. RMSNorm (input normalization)
  console.log('[Benchmark] Running RMSNorm...');
  results.push(await benchmarkRMSNorm(1, config.hiddenSize, options));

  // 2. QKV projection matmul
  console.log('[Benchmark] Running QKV projection...');
  const qkvDim = (config.numHeads + 2 * config.numKVHeads) * config.headDim;
  results.push(await benchmarkMatmul(1, qkvDim, config.hiddenSize, options));

  // 3. Attention (decode)
  console.log('[Benchmark] Running Attention decode...');
  const kvLen = 512; // Simulate 512 token context
  results.push(await benchmarkAttentionDecode(config.numHeads, config.headDim, kvLen, options));

  // 4. Output projection matmul
  console.log('[Benchmark] Running output projection...');
  results.push(await benchmarkMatmul(1, config.hiddenSize, config.numHeads * config.headDim, options));

  // 5. FFN gate+up projection
  console.log('[Benchmark] Running FFN gate+up...');
  results.push(await benchmarkMatmul(1, config.intermediateSize * 2, config.hiddenSize, options));

  // 6. SiLU activation
  console.log('[Benchmark] Running SiLU...');
  results.push(await benchmarkSiLU(config.intermediateSize, options));

  // 7. FFN down projection
  console.log('[Benchmark] Running FFN down...');
  results.push(await benchmarkMatmul(1, config.hiddenSize, config.intermediateSize, options));

  // 8. Final RMSNorm
  console.log('[Benchmark] Running final RMSNorm...');
  results.push(await benchmarkRMSNorm(1, config.hiddenSize, options));

  // 9. LM head projection
  console.log('[Benchmark] Running LM head...');
  results.push(await benchmarkMatmul(1, config.vocabSize, config.hiddenSize, options));

  // Calculate summary
  const perLayerLatency = results.slice(0, 8).reduce((sum, r) => sum + r.latency.median_ms, 0);
  const lmHeadLatency = results[8].latency.median_ms;
  const totalDecodeLatency = perLayerLatency * config.numLayers + lmHeadLatency;
  const tokPerSec = 1000 / totalDecodeLatency;

  // Find bottleneck
  const sortedByLatency = [...results].sort((a, b) => b.latency.median_ms - a.latency.median_ms);
  const bottleneck = sortedByLatency[0];
  const bottleneckPct = (bottleneck.latency.median_ms / totalDecodeLatency) * 100;

  const report: BenchmarkReport = {
    device_info: {
      vendor: 'WebGPU',
      architecture: 'Unknown',
      max_workgroup_size: limits?.maxComputeInvocationsPerWorkgroup || 256,
      max_shared_memory: limits?.maxComputeWorkgroupStorageSize || 16384,
      has_f16: caps.hasF16,
      has_subgroups: caps.hasSubgroups,
    },
    model_config: {
      name: 'Gemma 3 1B',
      hidden_size: config.hiddenSize,
      intermediate_size: config.intermediateSize,
      num_heads: config.numHeads,
      num_kv_heads: config.numKVHeads,
      head_dim: config.headDim,
      num_layers: config.numLayers,
      vocab_size: config.vocabSize,
    },
    results,
    comparisons: [],
    summary: {
      total_decode_latency_ms: totalDecodeLatency,
      estimated_tok_per_sec: tokPerSec,
      bottleneck_kernel: `${bottleneck.kernel}/${bottleneck.variant}`,
      bottleneck_percentage: bottleneckPct,
    },
    generated_at: new Date().toISOString(),
  };

  console.log('\n=== Benchmark Summary ===');
  console.log(`Total decode latency: ${totalDecodeLatency.toFixed(2)}ms`);
  console.log(`Estimated tokens/sec: ${tokPerSec.toFixed(1)}`);
  console.log(`Bottleneck: ${bottleneck.kernel} (${bottleneckPct.toFixed(1)}%)`);

  return report;
}

/**
 * Compare two benchmark results
 */
export function compareBenchmarks(
  baseline: KernelBenchmarkResult,
  optimized: KernelBenchmarkResult
): BenchmarkComparison {
  const speedup = baseline.latency.median_ms / optimized.latency.median_ms;
  const latencyReduction = ((baseline.latency.median_ms - optimized.latency.median_ms) / baseline.latency.median_ms) * 100;
  const throughputIncrease = ((optimized.throughput.gb_per_sec - baseline.throughput.gb_per_sec) / baseline.throughput.gb_per_sec) * 100;

  return {
    baseline,
    optimized,
    speedup,
    latency_reduction_pct: latencyReduction,
    throughput_increase_pct: throughputIncrease,
  };
}

/**
 * Export benchmark report as JSON
 */
export function exportBenchmarkJSON(report: BenchmarkReport): string {
  return JSON.stringify(report, null, 2);
}

/**
 * Print benchmark report to console
 */
export function printBenchmarkReport(report: BenchmarkReport): void {
  console.log('\n' + '='.repeat(60));
  console.log('KERNEL BENCHMARK REPORT');
  console.log('='.repeat(60));

  console.log('\nDevice Info:');
  console.log(`  Max Workgroup Size: ${report.device_info.max_workgroup_size}`);
  console.log(`  Max Shared Memory: ${(report.device_info.max_shared_memory / 1024).toFixed(1)}KB`);
  console.log(`  F16 Support: ${report.device_info.has_f16}`);
  console.log(`  Subgroup Support: ${report.device_info.has_subgroups}`);

  console.log('\nModel Config:');
  console.log(`  Name: ${report.model_config.name}`);
  console.log(`  Hidden Size: ${report.model_config.hidden_size}`);
  console.log(`  Intermediate Size: ${report.model_config.intermediate_size}`);
  console.log(`  Heads: ${report.model_config.num_heads} (KV: ${report.model_config.num_kv_heads})`);

  console.log('\nKernel Results:');
  console.log('-'.repeat(60));
  console.log('Kernel           | Latency (ms) | GB/s    | GFLOPS');
  console.log('-'.repeat(60));

  for (const r of report.results) {
    console.log(
      `${(r.kernel + '/' + r.variant).padEnd(16)} | ` +
      `${r.latency.median_ms.toFixed(3).padStart(12)} | ` +
      `${r.throughput.gb_per_sec.toFixed(2).padStart(7)} | ` +
      `${r.flops.gflops.toFixed(1).padStart(7)}`
    );
  }

  console.log('-'.repeat(60));
  console.log('\nSummary:');
  console.log(`  Total Decode Latency: ${report.summary.total_decode_latency_ms.toFixed(2)}ms`);
  console.log(`  Estimated Tokens/sec: ${report.summary.estimated_tok_per_sec.toFixed(1)}`);
  console.log(`  Bottleneck: ${report.summary.bottleneck_kernel} (${report.summary.bottleneck_percentage.toFixed(1)}%)`);

  if (report.comparisons.length > 0) {
    console.log('\nComparisons:');
    for (const c of report.comparisons) {
      console.log(`  ${c.baseline.kernel}: ${c.speedup.toFixed(2)}x speedup`);
    }
  }

  console.log('\n' + '='.repeat(60));
}
