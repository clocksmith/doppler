/**
 * Matrix Multiplication Benchmarks
 */

import { test, expect } from '@playwright/test';
import {
  MODEL_CONFIGS,
  BENCHMARK_SETTINGS,
  getWorkloadConfig,
  BENCHMARK_MATRIX,
} from './config.js';
import type { WorkloadConfig, BenchmarkSettings } from './config.js';

interface BenchmarkResult {
  M: number;
  K: number;
  N: number;
  median: string;
  mean: string;
  min: string;
  max: string;
  gflops: string;
  samples: number;
}

interface SizeBenchResult {
  M: number;
  K: number;
  N: number;
  median: string;
  gflops: string;
}

test.describe('Matmul Benchmarks', () => {
  // Skip if not running benchmarks explicitly
  test.beforeEach(({}, testInfo) => {
    test.skip(!testInfo.project.name.includes('bench'), 'Only run in bench project');
  });

  for (const { model, seq, batch } of BENCHMARK_MATRIX.matmul) {
    const config = getWorkloadConfig('matmul', model, seq, batch || 'single');

    test(`${config.name}`, async ({ page }) => {
      await page.goto('/');

      const result = await page.evaluate(async (cfg: WorkloadConfig & { settings: BenchmarkSettings }): Promise<BenchmarkResult> => {
        const { M, K, N } = cfg;
        const { warmupIterations, timedIterations } = cfg.settings;

        // Create test matrices
        const A = new Float32Array(M! * K!);
        const B = new Float32Array(K! * N!);

        for (let i = 0; i < A.length; i++) A[i] = Math.random() * 2 - 1;
        for (let i = 0; i < B.length; i++) B[i] = Math.random() * 2 - 1;

        const gpu = await window.testHarness.getGPU();

        // Warmup
        for (let i = 0; i < warmupIterations; i++) {
          await window.testHarness.runMatmul(gpu.device, A, B, M!, N!, K!);
        }

        // Timed runs
        const times = [];
        for (let i = 0; i < timedIterations; i++) {
          const start = performance.now();
          await window.testHarness.runMatmul(gpu.device, A, B, M!, N!, K!);
          await gpu.device.queue.onSubmittedWorkDone();
          const end = performance.now();
          times.push(end - start);
        }

        // Compute stats
        times.sort((a, b) => a - b);
        const median = times[Math.floor(times.length / 2)];
        const mean = times.reduce((a, b) => a + b, 0) / times.length;
        const min = times[0];
        const max = times[times.length - 1];

        // Compute throughput
        const flops = 2 * M! * N! * K!; // matmul FLOP count
        const gflops = (flops / (median * 1e-3)) / 1e9;

        return {
          M: M!,
          K: K!,
          N: N!,
          median: median.toFixed(3),
          mean: mean.toFixed(3),
          min: min.toFixed(3),
          max: max.toFixed(3),
          gflops: gflops.toFixed(2),
          samples: times.length,
        };
      }, { ...config, settings: BENCHMARK_SETTINGS });

      console.log(`
Matmul Benchmark: ${config.name}
  Dimensions: M=${result.M}, K=${result.K}, N=${result.N}
  Time (ms): median=${result.median}, mean=${result.mean}, min=${result.min}, max=${result.max}
  Throughput: ${result.gflops} GFLOP/s
  Samples: ${result.samples}
`);

      // Basic sanity check - should complete in reasonable time
      expect(parseFloat(result.median)).toBeLessThan(10000);
    });
  }

  test.describe('Size sweep', () => {
    const sizes = [
      { M: 1, K: 4096, N: 4096, name: 'matvec' },
      { M: 32, K: 4096, N: 4096, name: '32 tokens' },
      { M: 128, K: 4096, N: 4096, name: '128 tokens' },
      { M: 512, K: 4096, N: 4096, name: '512 tokens' },
      { M: 4096, K: 4096, N: 4096, name: 'square 4K' },
    ];

    for (const size of sizes) {
      test(`Size: ${size.name}`, async ({ page }) => {
        await page.goto('/');

        const result = await page.evaluate(async (cfg: { M: number; K: number; N: number }): Promise<SizeBenchResult> => {
          const { M, K, N } = cfg;

          const A = new Float32Array(M * K);
          const B = new Float32Array(K * N);

          for (let i = 0; i < A.length; i++) A[i] = Math.random() * 2 - 1;
          for (let i = 0; i < B.length; i++) B[i] = Math.random() * 2 - 1;

          const gpu = await window.testHarness.getGPU();

          // Warmup
          for (let i = 0; i < 3; i++) {
            await window.testHarness.runMatmul(gpu.device, A, B, M, N, K);
          }

          // Timed
          const times = [];
          for (let i = 0; i < 10; i++) {
            const start = performance.now();
            await window.testHarness.runMatmul(gpu.device, A, B, M, N, K);
            await gpu.device.queue.onSubmittedWorkDone();
            times.push(performance.now() - start);
          }

          const median = times.sort((a, b) => a - b)[5];
          const flops = 2 * M * N * K;
          const gflops = (flops / (median * 1e-3)) / 1e9;

          return { M, K, N, median: median.toFixed(3), gflops: gflops.toFixed(2) };
        }, size);

        console.log(`${size.name}: ${result.median}ms, ${result.gflops} GFLOP/s`);
      });
    }
  });
});
