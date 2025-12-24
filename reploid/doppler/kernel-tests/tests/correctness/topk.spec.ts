/**
 * Top-K Kernel Correctness Tests
 *
 * Validates the topk.wgsl GPU kernel against reference JS implementation.
 * Priority: HIGH - newly implemented MoE routing kernel
 */

import { test, expect, TEST_SIZES } from './setup.js';
import type { TestSize } from './setup.js';

interface TopKResult {
  gpuIndices?: number[];
  gpuWeights?: number[];
  refIndices?: number[];
  refWeights?: number[];
  indices?: number[];
  expected?: number[];
  weights?: number[];
  expectedWeight?: number;
  indicesMatch?: boolean;
  numTokens?: number;
  numExperts?: number;
  topK?: number;
  hasNaN?: boolean;
  hasInf?: boolean;
  weightSums?: number[];
  maxError?: number;
}

test.describe('Top-K Selection Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should select correct top-2 from 8 experts', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<TopKResult> => {
        const { topkRef, softmaxRef } = window.testHarness.references;

        const numTokens = 4;
        const numExperts = 8;
        const topK = 2;

        // Create logits with known pattern
        const logits = new Float32Array(numTokens * numExperts);
        for (let t = 0; t < numTokens; t++) {
          for (let e = 0; e < numExperts; e++) {
            // Token 0: experts 7,6 should be top
            // Token 1: experts 0,1 should be top
            // etc.
            logits[t * numExperts + e] = (t === 0) ? e : (numExperts - 1 - e);
          }
        }

        // Run GPU kernel
        const gpu = await window.testHarness.getGPU();
        const gpuResult = await window.testHarness.runSoftmaxTopK(
          gpu.device, logits, numTokens, numExperts, topK
        );

        // Run reference
        const refResult = topkRef(
          softmaxRef(logits, numExperts, numTokens),
          numTokens, numExperts, topK, true
        );

        return {
          gpuIndices: Array.from(gpuResult.indices),
          gpuWeights: Array.from(gpuResult.weights),
          refIndices: Array.from(refResult.indices),
          refWeights: Array.from(refResult.weights),
        };
      });

      // Indices should match exactly
      expect(result.gpuIndices).toEqual(result.refIndices);

      // Weights should be close (renormalized probabilities)
      for (let i = 0; i < result.gpuWeights!.length; i++) {
        expect(result.gpuWeights![i]).toBeCloseTo(result.refWeights![i], 4);
      }
    });

    test('should handle uniform distribution', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<TopKResult> => {
        const { softmaxTopkRef } = window.testHarness.references;

        const numTokens = 8;
        const numExperts = 4;
        const topK = 2;

        // Uniform logits - all experts equally likely
        const logits = new Float32Array(numTokens * numExperts).fill(0);

        const gpu = await window.testHarness.getGPU();
        const gpuResult = await window.testHarness.runSoftmaxTopK(
          gpu.device, logits, numTokens, numExperts, topK
        );

        const refResult = softmaxTopkRef(logits, numTokens, numExperts, topK, true);

        // With uniform distribution, weights should be equal (0.5 each for top-2)
        const expectedWeight = 0.5;

        return {
          gpuWeights: Array.from(gpuResult.weights),
          refWeights: Array.from(refResult.weights),
          expectedWeight,
        };
      });

      // Each selected expert should have weight ~0.5 after renormalization
      for (const w of result.gpuWeights!) {
        expect(w).toBeCloseTo(result.expectedWeight!, 3);
      }
    });
  });

  test.describe('Edge cases', () => {
    test('should handle topK=1 (argmax)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<TopKResult> => {
        const numTokens = 16;
        const numExperts = 8;
        const topK = 1;

        // Each token has a clear winner
        const logits = new Float32Array(numTokens * numExperts);
        const expectedIndices = [];

        for (let t = 0; t < numTokens; t++) {
          const winner = t % numExperts;
          expectedIndices.push(winner);
          for (let e = 0; e < numExperts; e++) {
            logits[t * numExperts + e] = e === winner ? 10 : 0;
          }
        }

        const gpu = await window.testHarness.getGPU();
        const gpuResult = await window.testHarness.runSoftmaxTopK(
          gpu.device, logits, numTokens, numExperts, topK
        );

        return {
          indices: Array.from(gpuResult.indices),
          expected: expectedIndices,
          weights: Array.from(gpuResult.weights),
        };
      });

      expect(result.indices).toEqual(result.expected);

      // With topK=1, weight should be 1.0 after normalization
      for (const w of result.weights!) {
        expect(w).toBeCloseTo(1.0, 4);
      }
    });

    test('should handle 64 experts', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<TopKResult> => {
        const { softmaxTopkRef } = window.testHarness.references;

        const numTokens = 4;
        const numExperts = 64;
        const topK = 4;

        // Random logits
        const logits = new Float32Array(numTokens * numExperts);
        for (let i = 0; i < logits.length; i++) {
          logits[i] = Math.random() * 10 - 5;
        }

        const gpu = await window.testHarness.getGPU();
        const gpuResult = await window.testHarness.runSoftmaxTopK(
          gpu.device, logits, numTokens, numExperts, topK
        );

        const refResult = softmaxTopkRef(logits, numTokens, numExperts, topK, true);

        // Check that indices match (may be in different order if probabilities are close)
        let indicesMatch = true;
        for (let t = 0; t < numTokens; t++) {
          const gpuSet = new Set<number>();
          const refSet = new Set<number>();
          for (let k = 0; k < topK; k++) {
            gpuSet.add(gpuResult.indices[t * topK + k]);
            refSet.add(refResult.indices[t * topK + k]);
          }
          // Check if same experts were selected
          for (const idx of gpuSet) {
            if (!refSet.has(idx)) indicesMatch = false;
          }
        }

        return { indicesMatch, numTokens, numExperts, topK };
      });

      expect(result.indicesMatch).toBe(true);
    });

    test('should handle 128 experts (Snowflake Arctic)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<TopKResult> => {
        const { softmaxTopkRef } = window.testHarness.references;

        const numTokens = 2;
        const numExperts = 128;
        const topK = 2;

        const logits = new Float32Array(numTokens * numExperts);
        for (let i = 0; i < logits.length; i++) {
          logits[i] = Math.random() * 10 - 5;
        }

        const gpu = await window.testHarness.getGPU();
        const gpuResult = await window.testHarness.runSoftmaxTopK(
          gpu.device, logits, numTokens, numExperts, topK
        );

        const refResult = softmaxTopkRef(logits, numTokens, numExperts, topK, true);

        let indicesMatch = true;
        for (let t = 0; t < numTokens; t++) {
          const gpuSet = new Set<number>();
          const refSet = new Set<number>();
          for (let k = 0; k < topK; k++) {
            gpuSet.add(gpuResult.indices[t * topK + k]);
            refSet.add(refResult.indices[t * topK + k]);
          }
          for (const idx of gpuSet) {
            if (!refSet.has(idx)) indicesMatch = false;
          }
        }

        return { indicesMatch };
      });

      expect(result.indicesMatch).toBe(true);
    });

    test('should handle 160 experts (DeepSeek-V2)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<TopKResult> => {
        const { softmaxTopkRef } = window.testHarness.references;

        const numTokens = 2;
        const numExperts = 160;
        const topK = 6;

        const logits = new Float32Array(numTokens * numExperts);
        for (let i = 0; i < logits.length; i++) {
          logits[i] = Math.random() * 10 - 5;
        }

        const gpu = await window.testHarness.getGPU();
        const gpuResult = await window.testHarness.runSoftmaxTopK(
          gpu.device, logits, numTokens, numExperts, topK
        );

        const refResult = softmaxTopkRef(logits, numTokens, numExperts, topK, true);

        let indicesMatch = true;
        for (let t = 0; t < numTokens; t++) {
          const gpuSet = new Set<number>();
          const refSet = new Set<number>();
          for (let k = 0; k < topK; k++) {
            gpuSet.add(gpuResult.indices[t * topK + k]);
            refSet.add(refResult.indices[t * topK + k]);
          }
          for (const idx of gpuSet) {
            if (!refSet.has(idx)) indicesMatch = false;
          }
        }

        return { indicesMatch };
      });

      expect(result.indicesMatch).toBe(true);
    });
  });

  test.describe('Numerical stability', () => {
    test('should handle very large logits', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<TopKResult> => {
        const numTokens = 4;
        const numExperts = 8;
        const topK = 2;

        // Very large logits that could cause overflow without numerical stability
        const logits = new Float32Array(numTokens * numExperts);
        for (let t = 0; t < numTokens; t++) {
          for (let e = 0; e < numExperts; e++) {
            logits[t * numExperts + e] = 100 + e * 10; // Large values
          }
        }

        const gpu = await window.testHarness.getGPU();
        const gpuResult = await window.testHarness.runSoftmaxTopK(
          gpu.device, logits, numTokens, numExperts, topK
        );

        // Check for NaN/Inf
        let hasNaN = false;
        let hasInf = false;
        for (const w of gpuResult.weights) {
          if (isNaN(w)) hasNaN = true;
          if (!isFinite(w)) hasInf = true;
        }

        // Weights should sum to 1 per token
        const weightSums = [];
        for (let t = 0; t < numTokens; t++) {
          let sum = 0;
          for (let k = 0; k < topK; k++) {
            sum += gpuResult.weights[t * topK + k];
          }
          weightSums.push(sum);
        }

        return { hasNaN, hasInf, weightSums };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.hasInf).toBe(false);

      for (const sum of result.weightSums!) {
        expect(sum).toBeCloseTo(1.0, 4);
      }
    });

    test('should handle very negative logits', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<TopKResult> => {
        const numTokens = 4;
        const numExperts = 8;
        const topK = 2;

        // Very negative logits
        const logits = new Float32Array(numTokens * numExperts);
        for (let i = 0; i < logits.length; i++) {
          logits[i] = -100 - Math.random() * 100;
        }

        const gpu = await window.testHarness.getGPU();
        const gpuResult = await window.testHarness.runSoftmaxTopK(
          gpu.device, logits, numTokens, numExperts, topK
        );

        let hasNaN = false;
        for (const w of gpuResult.weights) {
          if (isNaN(w)) hasNaN = true;
        }

        return { hasNaN, weights: Array.from(gpuResult.weights) };
      });

      expect(result.hasNaN).toBe(false);
    });
  });

  test.describe('Performance sizes', () => {
    for (const [name, size] of Object.entries(TEST_SIZES)) {
      test(`should handle ${name} workload`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (params: TestSize): Promise<TopKResult> => {
          const { softmaxTopkRef } = window.testHarness.references;

          const { tokens: numTokens, experts: numExperts } = params;
          const topK = 2;

          const logits = new Float32Array(numTokens * numExperts);
          for (let i = 0; i < logits.length; i++) {
            logits[i] = Math.random() * 4 - 2;
          }

          const gpu = await window.testHarness.getGPU();
          const gpuResult = await window.testHarness.runSoftmaxTopK(
            gpu.device, logits, numTokens, numExperts, topK
          );

          const refResult = softmaxTopkRef(logits, numTokens, numExperts, topK, true);

          // Compare weights
          let maxError = 0;
          for (let i = 0; i < gpuResult.weights.length; i++) {
            const error = Math.abs(gpuResult.weights[i] - refResult.weights[i]);
            maxError = Math.max(maxError, error);
          }

          return { maxError, numTokens, numExperts };
        }, size);

        // Tolerance for FP32
        expect(result.maxError).toBeLessThan(1e-4);
      });
    }
  });
});
