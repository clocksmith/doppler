/**
 * MoE Pipeline Benchmarks
 *
 * End-to-end benchmark of the MoE routing pipeline:
 * router logits -> softmax+topk -> expert FFN -> scatter-add
 */

import { test, expect } from '@playwright/test';
import {
  MODEL_CONFIGS,
  SEQ_CONFIGS,
  BENCHMARK_SETTINGS,
} from './config.js';
import type { ModelConfig, SeqConfig } from './config.js';

interface PipelineConfig {
  numTokens: number;
  hiddenSize: number;
  intermediateSize: number;
  numExperts: number;
  topK: number;
}

interface PipelineResult {
  numTokens: number;
  hiddenSize: number;
  numExperts: number;
  topK: number;
  stages: {
    router: string;
    topk: string;
    scatterAdd: string;
    total: string;
  };
}

interface ScalingResult {
  numExperts?: number;
  numTokens?: number;
  median: string;
  tokensPerMs?: string;
  throughput?: string;
}

test.describe('MoE Pipeline Benchmarks', () => {
  test.beforeEach(({}, testInfo) => {
    test.skip(!testInfo.project.name.includes('bench'), 'Only run in bench project');
  });

  const configs: Array<{ model: string; seq: string; name: string }> = [
    { model: 'mixtral', seq: 'single', name: 'Decode (1 token)' },
    { model: 'mixtral', seq: 'short', name: 'Short prompt (32)' },
    { model: 'mixtral', seq: 'medium', name: 'Medium prompt (128)' },
    { model: 'mixtral', seq: 'long', name: 'Long prompt (512)' },
  ];

  for (const { model, seq, name } of configs) {
    const m = MODEL_CONFIGS[model];
    const s = SEQ_CONFIGS[seq];

    test(`Full MoE layer: ${name}`, async ({ page }) => {
      await page.goto('/');

      const result = await page.evaluate(async (cfg: PipelineConfig): Promise<PipelineResult> => {
        const {
          numTokens,
          hiddenSize,
          intermediateSize,
          numExperts,
          topK,
        } = cfg;

        const gpu = await window.testHarness.getGPU();

        // Setup: create input tensors
        const input = new Float32Array(numTokens * hiddenSize);
        for (let i = 0; i < input.length; i++) {
          input[i] = Math.random() * 2 - 1;
        }

        // Router weights [hiddenSize, numExperts]
        const routerWeights = new Float32Array(hiddenSize * numExperts);
        for (let i = 0; i < routerWeights.length; i++) {
          routerWeights[i] = Math.random() * 0.1;
        }

        // Expert weights (simplified - just use random)
        const expertGate = new Float32Array(numExperts * hiddenSize * intermediateSize);
        const expertUp = new Float32Array(numExperts * hiddenSize * intermediateSize);
        const expertDown = new Float32Array(numExperts * intermediateSize * hiddenSize);

        for (let i = 0; i < expertGate.length; i++) expertGate[i] = Math.random() * 0.1;
        for (let i = 0; i < expertUp.length; i++) expertUp[i] = Math.random() * 0.1;
        for (let i = 0; i < expertDown.length; i++) expertDown[i] = Math.random() * 0.1;

        // Benchmark functions
        const benchmarks = {
          router: async () => {
            // Compute router logits: input @ router_weights
            return await window.testHarness.runMatmul(
              gpu.device, input, routerWeights,
              numTokens, numExperts, hiddenSize
            );
          },

          topk: async (logits: Float32Array) => {
            return await window.testHarness.runSoftmaxTopK(
              gpu.device, logits, numTokens, numExperts, topK
            );
          },

          // Full expert forward would be complex - benchmark individual ops
          scatterAdd: async (expertOutputs: Float32Array, indices: Uint32Array, weights: Float32Array) => {
            return await window.testHarness.runScatterAdd(
              gpu.device, expertOutputs, indices, weights,
              numTokens, hiddenSize, numExperts, topK
            );
          },
        };

        // Warmup
        const warmupLogits = await benchmarks.router();
        const warmupTopk = await benchmarks.topk(warmupLogits);

        // Measure each stage
        const stages: Record<string, number> = {};

        // Router benchmark
        const routerTimes = [];
        for (let i = 0; i < 10; i++) {
          const start = performance.now();
          await benchmarks.router();
          await gpu.device.queue.onSubmittedWorkDone();
          routerTimes.push(performance.now() - start);
        }
        stages.router = routerTimes.sort((a, b) => a - b)[5];

        // TopK benchmark
        const topkTimes = [];
        for (let i = 0; i < 10; i++) {
          const start = performance.now();
          await benchmarks.topk(warmupLogits);
          await gpu.device.queue.onSubmittedWorkDone();
          topkTimes.push(performance.now() - start);
        }
        stages.topk = topkTimes.sort((a, b) => a - b)[5];

        // Scatter-add benchmark (using dummy expert outputs)
        const dummyExpertOutputs = new Float32Array(numExperts * numTokens * hiddenSize);
        for (let i = 0; i < dummyExpertOutputs.length; i++) {
          dummyExpertOutputs[i] = Math.random();
        }

        const scatterTimes = [];
        for (let i = 0; i < 10; i++) {
          const start = performance.now();
          await benchmarks.scatterAdd(
            dummyExpertOutputs,
            warmupTopk.indices,
            warmupTopk.weights
          );
          await gpu.device.queue.onSubmittedWorkDone();
          scatterTimes.push(performance.now() - start);
        }
        stages.scatterAdd = scatterTimes.sort((a, b) => a - b)[5];

        // Full pipeline timing (simulated)
        stages.total = stages.router + stages.topk + stages.scatterAdd;

        return {
          numTokens,
          hiddenSize,
          numExperts,
          topK,
          stages: {
            router: stages.router.toFixed(3),
            topk: stages.topk.toFixed(3),
            scatterAdd: stages.scatterAdd.toFixed(3),
            total: stages.total.toFixed(3),
          },
        };
      }, {
        numTokens: s.seqLen,
        hiddenSize: m.hiddenSize,
        intermediateSize: m.intermediateSize,
        numExperts: m.numExperts,
        topK: m.topK,
      });

      console.log(`
MoE Pipeline: ${name}
  Config: ${result.numTokens} tokens, ${result.numExperts} experts, top-${result.topK}
  Router:     ${result.stages.router}ms
  TopK:       ${result.stages.topk}ms
  ScatterAdd: ${result.stages.scatterAdd}ms
  -----------
  Total:      ${result.stages.total}ms
`);
    });
  }

  test.describe('Expert count scaling', () => {
    const expertCounts = [4, 8, 16, 32];

    for (const numExperts of expertCounts) {
      test(`${numExperts} experts`, async ({ page }) => {
        await page.goto('/');

        const result = await page.evaluate(async (cfg: { numTokens: number; numExperts: number; hiddenSize: number; topK: number }): Promise<ScalingResult> => {
          const { numTokens, numExperts, hiddenSize, topK } = cfg;

          const gpu = await window.testHarness.getGPU();

          // Router logits
          const logits = new Float32Array(numTokens * numExperts);
          for (let i = 0; i < logits.length; i++) {
            logits[i] = Math.random() * 4 - 2;
          }

          // Warmup
          for (let i = 0; i < 3; i++) {
            await window.testHarness.runSoftmaxTopK(
              gpu.device, logits, numTokens, numExperts, topK
            );
          }

          // Benchmark
          const times = [];
          for (let i = 0; i < 20; i++) {
            const start = performance.now();
            await window.testHarness.runSoftmaxTopK(
              gpu.device, logits, numTokens, numExperts, topK
            );
            await gpu.device.queue.onSubmittedWorkDone();
            times.push(performance.now() - start);
          }

          const median = times.sort((a, b) => a - b)[10];

          return {
            numExperts,
            numTokens,
            median: median.toFixed(3),
            tokensPerMs: (numTokens / median).toFixed(1),
          };
        }, {
          numTokens: 128,
          numExperts,
          hiddenSize: 4096,
          topK: 2,
        });

        console.log(
          `${numExperts} experts: ${result.median}ms, ` +
          `${result.tokensPerMs} tokens/ms`
        );
      });
    }
  });

  test.describe('Token count scaling', () => {
    const tokenCounts = [1, 8, 32, 128, 512];

    for (const numTokens of tokenCounts) {
      test(`${numTokens} tokens`, async ({ page }) => {
        await page.goto('/');

        const result = await page.evaluate(async (cfg: { numTokens: number; numExperts: number; hiddenSize: number; topK: number }): Promise<ScalingResult> => {
          const { numTokens, numExperts, hiddenSize, topK } = cfg;

          const gpu = await window.testHarness.getGPU();

          const logits = new Float32Array(numTokens * numExperts);
          for (let i = 0; i < logits.length; i++) {
            logits[i] = Math.random() * 4 - 2;
          }

          // Expert outputs for scatter-add
          const expertOutputs = new Float32Array(numExperts * numTokens * hiddenSize);
          for (let i = 0; i < expertOutputs.length; i++) {
            expertOutputs[i] = Math.random();
          }

          // Warmup
          const { indices, weights } = await window.testHarness.runSoftmaxTopK(
            gpu.device, logits, numTokens, numExperts, topK
          );

          // Benchmark full routing
          const times = [];
          for (let i = 0; i < 10; i++) {
            const start = performance.now();

            // TopK
            const topkResult = await window.testHarness.runSoftmaxTopK(
              gpu.device, logits, numTokens, numExperts, topK
            );

            // Scatter-add
            await window.testHarness.runScatterAdd(
              gpu.device, expertOutputs, topkResult.indices, topkResult.weights,
              numTokens, hiddenSize, numExperts, topK
            );

            await gpu.device.queue.onSubmittedWorkDone();
            times.push(performance.now() - start);
          }

          const median = times.sort((a, b) => a - b)[5];

          return {
            numTokens,
            median: median.toFixed(3),
            throughput: (numTokens / median * 1000).toFixed(0),
          };
        }, {
          numTokens,
          numExperts: 8,
          hiddenSize: 4096,
          topK: 2,
        });

        console.log(
          `${numTokens} tokens: ${result.median}ms, ` +
          `${result.throughput} tokens/sec`
        );
      });
    }
  });
});
