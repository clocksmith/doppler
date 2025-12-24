/**
 * All Kernels Summary Benchmark
 *
 * Quick benchmark of all kernels for comparison
 */

import { test, expect } from '@playwright/test';
import { MODEL_CONFIGS, SEQ_CONFIGS } from './config.js';

interface BenchmarkResult {
  name: string;
  median: number;
  min: number;
  max: number;
}

interface Config {
  seqLen: number;
  batchSize: number;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  headDim: number;
  numExperts: number;
  topK: number;
}

test.describe('All Kernels Summary', () => {
  test.beforeEach(({}, testInfo) => {
    test.skip(!testInfo.project.name.includes('bench'), 'Only run in bench project');
  });

  test('7B model - decode (1 token)', async ({ page }) => {
    await page.goto('/');

    const result = await page.evaluate(async (): Promise<BenchmarkResult[]> => {
      const gpu = await window.testHarness.getGPU();

      const config: Config = {
        seqLen: 1,
        batchSize: 1,
        hiddenSize: 4096,
        intermediateSize: 11008,
        numHeads: 32,
        headDim: 128,
        numExperts: 8,
        topK: 2,
      };

      const numTokens = config.seqLen * config.batchSize;

      // Benchmark helper
      async function benchmark(name: string, fn: () => Promise<void>, iterations: number = 20): Promise<BenchmarkResult> {
        // Warmup
        for (let i = 0; i < 3; i++) await fn();

        // Timed
        const times = [];
        for (let i = 0; i < iterations; i++) {
          const start = performance.now();
          await fn();
          await gpu.device.queue.onSubmittedWorkDone();
          times.push(performance.now() - start);
        }

        const sorted = times.sort((a, b) => a - b);
        return {
          name,
          median: sorted[Math.floor(sorted.length / 2)],
          min: sorted[0],
          max: sorted[sorted.length - 1],
        };
      }

      const results: BenchmarkResult[] = [];

      // 1. RMSNorm
      {
        const input = new Float32Array(numTokens * config.hiddenSize);
        const weight = new Float32Array(config.hiddenSize);
        for (let i = 0; i < input.length; i++) input[i] = Math.random();
        for (let i = 0; i < weight.length; i++) weight[i] = Math.random();

        results.push(await benchmark('rmsnorm', async () => {
          await window.testHarness.runRMSNorm?.(
            gpu.device, input, weight, numTokens, config.hiddenSize
          );
        }));
      }

      // 2. Matmul (QKV projection)
      {
        const input = new Float32Array(numTokens * config.hiddenSize);
        const weight = new Float32Array(config.hiddenSize * 3 * config.hiddenSize);
        for (let i = 0; i < input.length; i++) input[i] = Math.random();
        for (let i = 0; i < weight.length; i++) weight[i] = Math.random() * 0.1;

        results.push(await benchmark('matmul (QKV)', async () => {
          await window.testHarness.runMatmul(
            gpu.device, input, weight,
            numTokens, 3 * config.hiddenSize, config.hiddenSize
          );
        }));
      }

      // 3. RoPE
      {
        const qk = new Float32Array(numTokens * config.numHeads * config.headDim);
        for (let i = 0; i < qk.length; i++) qk[i] = Math.random();

        results.push(await benchmark('rope', async () => {
          await window.testHarness.runRoPE?.(
            gpu.device, qk, config.seqLen, config.numHeads, config.headDim
          );
        }));
      }

      // 4. Attention (softmax component)
      {
        const scores = new Float32Array(numTokens * config.numHeads * config.seqLen);
        for (let i = 0; i < scores.length; i++) scores[i] = Math.random();

        results.push(await benchmark('softmax (attn)', async () => {
          await window.testHarness.runSoftmax?.(
            gpu.device, scores, config.seqLen, numTokens * config.numHeads
          );
        }));
      }

      // 5. SiLU (FFN activation)
      {
        const x = new Float32Array(numTokens * config.intermediateSize);
        for (let i = 0; i < x.length; i++) x[i] = Math.random();

        results.push(await benchmark('silu', async () => {
          await window.testHarness.runSiLU?.(gpu.device, x);
        }));
      }

      // 6. Matmul (FFN up)
      {
        const input = new Float32Array(numTokens * config.hiddenSize);
        const weight = new Float32Array(config.hiddenSize * config.intermediateSize);
        for (let i = 0; i < input.length; i++) input[i] = Math.random();
        for (let i = 0; i < weight.length; i++) weight[i] = Math.random() * 0.1;

        results.push(await benchmark('matmul (FFN up)', async () => {
          await window.testHarness.runMatmul(
            gpu.device, input, weight,
            numTokens, config.intermediateSize, config.hiddenSize
          );
        }));
      }

      // 7. Top-K (MoE routing)
      {
        const logits = new Float32Array(numTokens * config.numExperts);
        for (let i = 0; i < logits.length; i++) logits[i] = Math.random() * 4 - 2;

        results.push(await benchmark('topk', async () => {
          await window.testHarness.runSoftmaxTopK(
            gpu.device, logits, numTokens, config.numExperts, config.topK
          );
        }));
      }

      // 8. Scatter-Add (MoE combine)
      {
        const expertOutputs = new Float32Array(
          config.numExperts * numTokens * config.hiddenSize
        );
        const indices = new Uint32Array(numTokens * config.topK);
        const weights = new Float32Array(numTokens * config.topK);

        for (let i = 0; i < expertOutputs.length; i++) expertOutputs[i] = Math.random();
        for (let t = 0; t < numTokens; t++) {
          for (let k = 0; k < config.topK; k++) {
            indices[t * config.topK + k] = Math.floor(Math.random() * config.numExperts);
            weights[t * config.topK + k] = 1.0 / config.topK;
          }
        }

        results.push(await benchmark('scatter_add', async () => {
          await window.testHarness.runScatterAdd(
            gpu.device, expertOutputs, indices, weights,
            numTokens, config.hiddenSize, config.numExperts, config.topK
          );
        }));
      }

      // 9. Residual add
      {
        const x = new Float32Array(numTokens * config.hiddenSize);
        const residual = new Float32Array(numTokens * config.hiddenSize);
        for (let i = 0; i < x.length; i++) x[i] = Math.random();
        for (let i = 0; i < residual.length; i++) residual[i] = Math.random();

        results.push(await benchmark('residual', async () => {
          await window.testHarness.runResidual?.(gpu.device, x, residual);
        }));
      }

      return results.filter(r => r.median !== undefined);
    });

    console.log('\n=== Kernel Benchmark Summary (7B decode, 1 token) ===\n');
    console.log('Kernel           | Median (ms) | Min    | Max');
    console.log('-----------------+-------------+--------+--------');

    let totalTime = 0;
    for (const r of result) {
      if (r.median) {
        totalTime += r.median;
        console.log(
          `${r.name.padEnd(16)} | ${r.median.toFixed(3).padStart(11)} | ` +
          `${r.min.toFixed(3).padStart(6)} | ${r.max.toFixed(3).padStart(6)}`
        );
      }
    }

    console.log('-----------------+-------------+--------+--------');
    console.log(`${'TOTAL'.padEnd(16)} | ${totalTime.toFixed(3).padStart(11)} |`);
    console.log(`\nEstimated tokens/sec: ${(1000 / totalTime).toFixed(1)}`);
  });

  test('Mixtral - prefill (128 tokens)', async ({ page }) => {
    await page.goto('/');

    const result = await page.evaluate(async (): Promise<BenchmarkResult[]> => {
      const gpu = await window.testHarness.getGPU();

      const config: Config = {
        seqLen: 128,
        batchSize: 1,
        hiddenSize: 4096,
        intermediateSize: 14336,
        numHeads: 32,
        headDim: 128,
        numExperts: 8,
        topK: 2,
      };

      async function benchmark(name: string, fn: () => Promise<void>, iterations: number = 10): Promise<BenchmarkResult> {
        for (let i = 0; i < 2; i++) await fn();

        const times = [];
        for (let i = 0; i < iterations; i++) {
          const start = performance.now();
          await fn();
          await gpu.device.queue.onSubmittedWorkDone();
          times.push(performance.now() - start);
        }

        const sorted = times.sort((a, b) => a - b);
        return {
          name,
          median: sorted[Math.floor(sorted.length / 2)],
          min: 0,
          max: 0,
        };
      }

      const results: BenchmarkResult[] = [];

      // Key operations for prefill
      // 1. Matmul (attention QKV)
      {
        const input = new Float32Array(config.seqLen * config.hiddenSize);
        const weight = new Float32Array(config.hiddenSize * 3 * config.hiddenSize);
        for (let i = 0; i < input.length; i++) input[i] = Math.random();
        for (let i = 0; i < weight.length; i++) weight[i] = Math.random() * 0.1;

        results.push(await benchmark('matmul (QKV)', async () => {
          await window.testHarness.runMatmul(
            gpu.device, input, weight,
            config.seqLen, 3 * config.hiddenSize, config.hiddenSize
          );
        }));
      }

      // 2. MoE routing (topk)
      {
        const logits = new Float32Array(config.seqLen * config.numExperts);
        for (let i = 0; i < logits.length; i++) logits[i] = Math.random() * 4 - 2;

        results.push(await benchmark('topk (MoE)', async () => {
          await window.testHarness.runSoftmaxTopK(
            gpu.device, logits, config.seqLen, config.numExperts, config.topK
          );
        }));
      }

      // 3. MoE FFN matmul (expert up projection)
      {
        const input = new Float32Array(config.seqLen * config.hiddenSize);
        const weight = new Float32Array(config.hiddenSize * config.intermediateSize);
        for (let i = 0; i < input.length; i++) input[i] = Math.random();
        for (let i = 0; i < weight.length; i++) weight[i] = Math.random() * 0.1;

        results.push(await benchmark('matmul (expert)', async () => {
          await window.testHarness.runMatmul(
            gpu.device, input, weight,
            config.seqLen, config.intermediateSize, config.hiddenSize
          );
        }));
      }

      // 4. Scatter-add
      {
        const expertOutputs = new Float32Array(
          config.numExperts * config.seqLen * config.hiddenSize
        );
        const indices = new Uint32Array(config.seqLen * config.topK);
        const weights = new Float32Array(config.seqLen * config.topK);

        for (let i = 0; i < expertOutputs.length; i++) expertOutputs[i] = Math.random();
        for (let t = 0; t < config.seqLen; t++) {
          for (let k = 0; k < config.topK; k++) {
            indices[t * config.topK + k] = Math.floor(Math.random() * config.numExperts);
            weights[t * config.topK + k] = 0.5;
          }
        }

        results.push(await benchmark('scatter_add', async () => {
          await window.testHarness.runScatterAdd(
            gpu.device, expertOutputs, indices, weights,
            config.seqLen, config.hiddenSize, config.numExperts, config.topK
          );
        }));
      }

      return results;
    });

    console.log('\n=== Mixtral Prefill (128 tokens) ===\n');
    let total = 0;
    for (const r of result) {
      console.log(`${r.name}: ${r.median.toFixed(3)}ms`);
      total += r.median;
    }
    console.log(`\nTotal: ${total.toFixed(3)}ms`);
    console.log(`Throughput: ${(128 / total * 1000).toFixed(1)} tokens/sec`);
  });
});
