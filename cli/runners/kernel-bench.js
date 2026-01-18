

/**
 * Kernel microbenchmarks.
 */

import { setHarnessConfig, appendRuntimeConfigParams } from '../args/index.js';
import { KERNEL_BENCHMARKS } from '../suites.js';

/**
 * Run kernel benchmarks.
 * @param {import('playwright').Page} page
 * @param {import('../args/index.js').CLIOptions} opts
 * @returns {Promise<import('../output.js').SuiteResult>}
 */
export async function runKernelBenchmarks(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('KERNEL BENCHMARKS');
  console.log('='.repeat(60));

  const runtimeConfig = opts.runtimeConfig;
  if (!runtimeConfig) {
    throw new Error('Runtime config is required for benchmarks.');
  }
  const benchmarkRun = runtimeConfig.shared.benchmark.run;

  await page.addInitScript(() => {
    /** @type {any} */
    (window).__name = (target) => target;
  });

  setHarnessConfig(opts, {
    mode: 'kernels',
    autorun: false,
    skipLoad: false,
    modelId: null,
  });
  const harnessParams = new URLSearchParams();
  appendRuntimeConfigParams(harnessParams, opts);
  await page.goto(`${opts.baseUrl}/doppler/tests/harness.html?${harnessParams.toString()}`, {
    timeout: opts.timeout,
  });

  await page.waitForFunction(
    () => /** @type {any} */ (window).testHarness && /** @type {any} */ (window).testHarness.references,
    { timeout: 10000 }
  );

  /** @type {Array<{name: string, passed: boolean, duration: number, error?: string}>} */
  const results = [];
  const startTime = Date.now();

  const benchmarks = opts.filter
    ? KERNEL_BENCHMARKS.filter((b) => b.includes(opts.filter))
    : KERNEL_BENCHMARKS;

  for (const benchName of benchmarks) {
    console.log(`\n  Benchmarking: ${benchName}...`);

    try {
      const result = await page.evaluate(
        async (config) => {
          const __name = (target) => target;
          const { name, warmup, runs } = config;
          const harness = /** @type {any} */ (window).testHarness;
          const gpu = await harness.getGPU();

          /** @type {Record<string, () => Promise<void>>} */
          const benchmarks = {
            matmul: async () => {
              const M = 1, N = 4096, K = 4096;
              const A = new Float32Array(M * K).fill(1);
              const B = new Float32Array(K * N).fill(1);
              await harness.runMatmul(gpu.device, A, B, M, N, K);
            },
            rmsnorm: async () => {
              const batchSize = 1, hiddenSize = 4096;
              const input = new Float32Array(batchSize * hiddenSize).fill(1);
              const weight = new Float32Array(hiddenSize).fill(1);
              await harness.runRMSNorm(gpu.device, input, weight, batchSize, hiddenSize);
            },
            softmax: async () => {
              const innerSize = 32000, outerSize = 1;
              const input = new Float32Array(innerSize * outerSize).fill(1);
              await harness.runSoftmax(gpu.device, input, innerSize, outerSize);
            },
            silu: async () => {
              const size = 14336;
              const input = new Float32Array(size).fill(1);
              await harness.runSiLU(gpu.device, input);
            },
            rope: async () => {
              const seqLen = 1, numHeads = 32, headDim = 128;
              const input = new Float32Array(seqLen * numHeads * headDim).fill(1);
              await harness.runRoPE(gpu.device, input, seqLen, numHeads, headDim);
            },
            attention: async () => {
              const seqLen = 1, kvLen = 128, numHeads = 32, headDim = 128;
              const Q = new Float32Array(seqLen * numHeads * headDim).fill(0.1);
              const K = new Float32Array(kvLen * numHeads * headDim).fill(0.1);
              const V = new Float32Array(kvLen * numHeads * headDim).fill(0.1);
              await harness.runAttention(gpu.device, Q, K, V, seqLen, kvLen, numHeads, numHeads, headDim);
            },
            moe: async () => {
              const numTokens = 1, numExperts = 8, topK = 2;
              const logits = new Float32Array(numTokens * numExperts).fill(1);
              await harness.runSoftmaxTopK(gpu.device, logits, numTokens, numExperts, topK);
            },
          };

          const fn = benchmarks[name];
          if (!fn) return { error: `Unknown benchmark: ${name}` };

          for (let i = 0; i < warmup; i++) {
            await fn();
            await gpu.device.queue.onSubmittedWorkDone();
          }

          /** @type {number[]} */
          const times = [];
          for (let i = 0; i < runs; i++) {
            const start = performance.now();
            await fn();
            await gpu.device.queue.onSubmittedWorkDone();
            times.push(performance.now() - start);
          }

          const sorted = [...times].sort((a, b) => a - b);
          const median = sorted[Math.floor(sorted.length / 2)];
          const mean = times.reduce((a, b) => a + b, 0) / times.length;
          const min = sorted[0];
          const max = sorted[sorted.length - 1];

          return { median, mean, min, max, samples: times.length };
        },
        { name: benchName, warmup: benchmarkRun.warmupRuns, runs: benchmarkRun.timedRuns }
      );

      if ('error' in result) {
        results.push({
          name: benchName,
          passed: false,
          duration: 0,
          error: result.error,
        });
        console.log(`  \x1b[31mFAIL\x1b[0m ${benchName}: ${result.error}`);
      } else {
        results.push({
          name: benchName,
          passed: true,
          duration: result.median,
        });
        console.log(
          `  ${benchName}: median=${result.median.toFixed(3)}ms ` +
          `mean=${result.mean.toFixed(3)}ms ` +
          `min=${result.min.toFixed(3)}ms max=${result.max.toFixed(3)}ms`
        );
      }
    } catch (err) {
      results.push({
        name: benchName,
        passed: false,
        duration: 0,
        error: /** @type {Error} */ (err).message,
      });
      console.log(`  \x1b[31mFAIL\x1b[0m ${benchName}: ${/** @type {Error} */ (err).message}`);
    }
  }

  const totalDuration = Date.now() - startTime;

  return {
    suite: 'bench:kernels',
    passed: results.filter((r) => r.passed).length,
    failed: results.filter((r) => !r.passed).length,
    skipped: 0,
    duration: totalDuration,
    results,
  };
}
