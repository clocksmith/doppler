/**
 * Scatter-Add Kernel Correctness Tests
 *
 * Validates the scatter_add.wgsl GPU kernel against reference JS implementation.
 * Priority: HIGH - newly implemented MoE output combination kernel
 */

import { test, expect, TEST_SIZES } from './setup.js';
import type { TestSize } from './setup.js';

interface ScatterAddResult {
  maxError: number;
  gpuOutput?: number[];
  refOutput?: number[];
  avgError?: number;
  size?: { numTokens: number; numExperts: number; hiddenSize: number };
  expected?: number;
  gpuSample?: number;
  refSample?: number;
}

test.describe('Scatter-Add Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should combine expert outputs with weights', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<ScatterAddResult> => {
        const { scatterAddRef } = window.testHarness.references;

        const numTokens = 4;
        const hiddenSize = 8;
        const numExperts = 4;
        const topK = 2;

        // Create expert outputs (all experts process all tokens for simplicity)
        const expertOutputs = new Float32Array(numExperts * numTokens * hiddenSize);
        for (let e = 0; e < numExperts; e++) {
          for (let t = 0; t < numTokens; t++) {
            for (let d = 0; d < hiddenSize; d++) {
              // Each expert has unique scaling
              expertOutputs[e * numTokens * hiddenSize + t * hiddenSize + d] = (e + 1) * (d + 1);
            }
          }
        }

        // Token routing: each token uses experts (0,1), (1,2), (2,3), (3,0)
        const indices = new Uint32Array(numTokens * topK);
        const weights = new Float32Array(numTokens * topK);

        for (let t = 0; t < numTokens; t++) {
          indices[t * topK + 0] = t % numExperts;
          indices[t * topK + 1] = (t + 1) % numExperts;
          weights[t * topK + 0] = 0.6;
          weights[t * topK + 1] = 0.4;
        }

        // Run reference
        const refOutput = scatterAddRef(
          expertOutputs, indices, weights,
          numTokens, hiddenSize, numExperts, topK
        );

        // Run GPU kernel
        const gpu = await window.testHarness.getGPU();
        const gpuOutput = await window.testHarness.runScatterAdd(
          gpu.device, expertOutputs, indices, weights,
          numTokens, hiddenSize, numExperts, topK
        );

        // Compare
        let maxError = 0;
        for (let i = 0; i < refOutput.length; i++) {
          const error = Math.abs(gpuOutput[i] - refOutput[i]);
          maxError = Math.max(maxError, error);
        }

        return {
          maxError,
          gpuOutput: Array.from(gpuOutput.slice(0, 16)),
          refOutput: Array.from(refOutput.slice(0, 16)),
        };
      });

      expect(result.maxError).toBeLessThan(1e-5);
    });

    test('should handle single expert (topK=1)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<ScatterAddResult> => {
        const { scatterAddRef } = window.testHarness.references;

        const numTokens = 8;
        const hiddenSize = 16;
        const numExperts = 4;
        const topK = 1;

        // Random expert outputs
        const expertOutputs = new Float32Array(numExperts * numTokens * hiddenSize);
        for (let i = 0; i < expertOutputs.length; i++) {
          expertOutputs[i] = Math.random() * 2 - 1;
        }

        // Each token routed to one expert with weight 1.0
        const indices = new Uint32Array(numTokens);
        const weights = new Float32Array(numTokens);

        for (let t = 0; t < numTokens; t++) {
          indices[t] = t % numExperts;
          weights[t] = 1.0;
        }

        const refOutput = scatterAddRef(
          expertOutputs, indices, weights,
          numTokens, hiddenSize, numExperts, topK
        );

        const gpu = await window.testHarness.getGPU();
        const gpuOutput = await window.testHarness.runScatterAdd(
          gpu.device, expertOutputs, indices, weights,
          numTokens, hiddenSize, numExperts, topK
        );

        let maxError = 0;
        for (let i = 0; i < refOutput.length; i++) {
          maxError = Math.max(maxError, Math.abs(gpuOutput[i] - refOutput[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-5);
    });
  });

  test.describe('Weight handling', () => {
    test('should correctly apply non-uniform weights', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<ScatterAddResult> => {
        const { scatterAddRef } = window.testHarness.references;

        const numTokens = 4;
        const hiddenSize = 4;
        const numExperts = 2;
        const topK = 2;

        // Simple expert outputs: expert 0 = all 1s, expert 1 = all 2s
        const expertOutputs = new Float32Array(numExperts * numTokens * hiddenSize);
        for (let e = 0; e < numExperts; e++) {
          for (let t = 0; t < numTokens; t++) {
            for (let d = 0; d < hiddenSize; d++) {
              expertOutputs[e * numTokens * hiddenSize + t * hiddenSize + d] = e + 1;
            }
          }
        }

        // All tokens use both experts with weights (0.7, 0.3)
        const indices = new Uint32Array(numTokens * topK);
        const weights = new Float32Array(numTokens * topK);

        for (let t = 0; t < numTokens; t++) {
          indices[t * topK + 0] = 0;
          indices[t * topK + 1] = 1;
          weights[t * topK + 0] = 0.7;
          weights[t * topK + 1] = 0.3;
        }

        // Expected: 0.7 * 1 + 0.3 * 2 = 1.3
        const expected = 0.7 * 1 + 0.3 * 2;

        const refOutput = scatterAddRef(
          expertOutputs, indices, weights,
          numTokens, hiddenSize, numExperts, topK
        );

        const gpu = await window.testHarness.getGPU();
        const gpuOutput = await window.testHarness.runScatterAdd(
          gpu.device, expertOutputs, indices, weights,
          numTokens, hiddenSize, numExperts, topK
        );

        return {
          expected,
          gpuSample: gpuOutput[0],
          refSample: refOutput[0],
          maxError: 0,
        };
      });

      expect(result.gpuSample).toBeCloseTo(result.expected!, 5);
      expect(result.refSample).toBeCloseTo(result.expected!, 5);
    });

    test('should handle zero weights correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<ScatterAddResult> => {
        const numTokens = 4;
        const hiddenSize = 8;
        const numExperts = 4;
        const topK = 2;

        const expertOutputs = new Float32Array(numExperts * numTokens * hiddenSize);
        for (let i = 0; i < expertOutputs.length; i++) {
          expertOutputs[i] = Math.random() * 100; // Large values
        }

        // First expert has weight 1.0, second has weight 0.0
        const indices = new Uint32Array(numTokens * topK);
        const weights = new Float32Array(numTokens * topK);

        for (let t = 0; t < numTokens; t++) {
          indices[t * topK + 0] = 0;
          indices[t * topK + 1] = 1;
          weights[t * topK + 0] = 1.0;
          weights[t * topK + 1] = 0.0; // Should not contribute
        }

        const gpu = await window.testHarness.getGPU();
        const gpuOutput = await window.testHarness.runScatterAdd(
          gpu.device, expertOutputs, indices, weights,
          numTokens, hiddenSize, numExperts, topK
        );

        // Output should equal expert 0's output exactly
        let maxError = 0;
        for (let t = 0; t < numTokens; t++) {
          for (let d = 0; d < hiddenSize; d++) {
            const expected = expertOutputs[0 * numTokens * hiddenSize + t * hiddenSize + d];
            const actual = gpuOutput[t * hiddenSize + d];
            maxError = Math.max(maxError, Math.abs(actual - expected));
          }
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-5);
    });
  });

  test.describe('Performance sizes', () => {
    for (const [name, size] of Object.entries(TEST_SIZES)) {
      test(`should handle ${name} workload`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (params: TestSize): Promise<ScatterAddResult> => {
          const { scatterAddRef } = window.testHarness.references;

          const { tokens: numTokens, experts: numExperts, hidden: hiddenSize } = params;
          const topK = 2;

          // Random data
          const expertOutputs = new Float32Array(numExperts * numTokens * hiddenSize);
          for (let i = 0; i < expertOutputs.length; i++) {
            expertOutputs[i] = Math.random() * 2 - 1;
          }

          const indices = new Uint32Array(numTokens * topK);
          const weights = new Float32Array(numTokens * topK);

          for (let t = 0; t < numTokens; t++) {
            indices[t * topK + 0] = Math.floor(Math.random() * numExperts);
            indices[t * topK + 1] = Math.floor(Math.random() * numExperts);
            weights[t * topK + 0] = 0.5 + Math.random() * 0.3;
            weights[t * topK + 1] = 1.0 - weights[t * topK + 0];
          }

          const refOutput = scatterAddRef(
            expertOutputs, indices, weights,
            numTokens, hiddenSize, numExperts, topK
          );

          const gpu = await window.testHarness.getGPU();
          const gpuOutput = await window.testHarness.runScatterAdd(
            gpu.device, expertOutputs, indices, weights,
            numTokens, hiddenSize, numExperts, topK
          );

          let maxError = 0;
          let sumError = 0;
          for (let i = 0; i < refOutput.length; i++) {
            const error = Math.abs(gpuOutput[i] - refOutput[i]);
            maxError = Math.max(maxError, error);
            sumError += error;
          }

          return {
            maxError,
            avgError: sumError / refOutput.length,
            size: { numTokens, numExperts, hiddenSize },
          };
        }, size);

        expect(result.maxError).toBeLessThan(1e-4);
        expect(result.avgError).toBeLessThan(1e-5);
      });
    }
  });

  test.describe('Accumulation variant', () => {
    test('should accumulate to existing buffer', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<ScatterAddResult> => {
        const { scatterAddAccumulateRef } = window.testHarness.references;

        const numTokens = 4;
        const hiddenSize = 8;
        const numExperts = 2;
        const topK = 2;

        // Initial values
        const existing = new Float32Array(numTokens * hiddenSize);
        for (let i = 0; i < existing.length; i++) {
          existing[i] = 1.0; // Start with all 1s
        }

        // Expert outputs
        const expertOutputs = new Float32Array(numExperts * numTokens * hiddenSize);
        for (let i = 0; i < expertOutputs.length; i++) {
          expertOutputs[i] = 0.5;
        }

        // Equal weights
        const indices = new Uint32Array(numTokens * topK);
        const weights = new Float32Array(numTokens * topK);

        for (let t = 0; t < numTokens; t++) {
          indices[t * topK + 0] = 0;
          indices[t * topK + 1] = 1;
          weights[t * topK + 0] = 0.5;
          weights[t * topK + 1] = 0.5;
        }

        // Reference: existing + weighted sum = 1.0 + (0.5 * 0.5 + 0.5 * 0.5) = 1.5
        const refOutput = scatterAddAccumulateRef(
          expertOutputs, indices, weights,
          numTokens, hiddenSize, numExperts, topK, existing
        );

        return {
          expected: 1.5,
          refSample: refOutput[0],
          maxError: 0,
        };
      });

      expect(result.refSample).toBeCloseTo(result.expected!, 5);
    });
  });
});
