/**
 * RMSNorm Kernel Correctness Tests
 */

import { test, expect } from './setup.js';

interface RMSNormResult {
  maxError: number;
  avgError?: number;
  hasNaN?: boolean;
  hasInf?: boolean;
}

interface RMSNormParams {
  batch: number;
  hidden: number;
}

test.describe('RMSNorm Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should compute RMSNorm correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<RMSNormResult> => {
        const { rmsNormRef } = window.testHarness.references;

        const batchSize = 4;
        const hiddenSize = 64;

        const input = new Float32Array(batchSize * hiddenSize);
        const weight = new Float32Array(hiddenSize);

        for (let i = 0; i < input.length; i++) {
          input[i] = Math.random() * 2 - 1;
        }
        for (let i = 0; i < weight.length; i++) {
          weight[i] = Math.random() * 0.5 + 0.75; // Around 1.0
        }

        const expected = rmsNormRef(input, weight, batchSize, hiddenSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runRMSNorm(
          gpu.device, input, weight, batchSize, hiddenSize
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-4);
    });

    test('should handle unit weights', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<RMSNormResult> => {
        const { rmsNormRef } = window.testHarness.references;

        const batchSize = 2;
        const hiddenSize = 32;

        const input = new Float32Array(batchSize * hiddenSize);
        const weight = new Float32Array(hiddenSize).fill(1.0);

        for (let i = 0; i < input.length; i++) {
          input[i] = Math.random() * 2 - 1;
        }

        const expected = rmsNormRef(input, weight, batchSize, hiddenSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runRMSNorm(
          gpu.device, input, weight, batchSize, hiddenSize
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-4);
    });
  });

  test.describe('Size variations', () => {
    const sizes = [
      { batch: 1, hidden: 64 },
      { batch: 1, hidden: 512 },
      { batch: 1, hidden: 4096 },
      { batch: 32, hidden: 256 },
      { batch: 128, hidden: 512 },
    ];

    for (const { batch, hidden } of sizes) {
      test(`should handle ${batch}x${hidden}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (params: RMSNormParams): Promise<RMSNormResult> => {
          const { rmsNormRef } = window.testHarness.references;

          const { batch, hidden } = params;

          const input = new Float32Array(batch * hidden);
          const weight = new Float32Array(hidden);

          for (let i = 0; i < input.length; i++) {
            input[i] = Math.random() * 2 - 1;
          }
          for (let i = 0; i < weight.length; i++) {
            weight[i] = Math.random() + 0.5;
          }

          const expected = rmsNormRef(input, weight, batch, hidden);

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runRMSNorm(
            gpu.device, input, weight, batch, hidden
          );

          let maxError = 0;
          let sumError = 0;
          for (let i = 0; i < expected.length; i++) {
            const err = Math.abs(actual[i] - expected[i]);
            maxError = Math.max(maxError, err);
            sumError += err;
          }

          return { maxError, avgError: sumError / expected.length };
        }, { batch, hidden });

        expect(result.maxError).toBeLessThan(1e-3);
      });
    }
  });

  test.describe('Numerical stability', () => {
    test('should handle small input values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<RMSNormResult> => {
        const { rmsNormRef } = window.testHarness.references;

        const batchSize = 4;
        const hiddenSize = 64;

        // Very small values
        const input = new Float32Array(batchSize * hiddenSize);
        const weight = new Float32Array(hiddenSize);

        for (let i = 0; i < input.length; i++) {
          input[i] = (Math.random() - 0.5) * 1e-5;
        }
        for (let i = 0; i < weight.length; i++) {
          weight[i] = 1.0;
        }

        const expected = rmsNormRef(input, weight, batchSize, hiddenSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runRMSNorm(
          gpu.device, input, weight, batchSize, hiddenSize
        );

        let hasNaN = false;
        let hasInf = false;
        for (const v of actual) {
          if (isNaN(v)) hasNaN = true;
          if (!isFinite(v)) hasInf = true;
        }

        return { hasNaN, hasInf, maxError: 0 };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.hasInf).toBe(false);
    });

    test('should handle large input values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<RMSNormResult> => {
        const batchSize = 2;
        const hiddenSize = 64;

        const input = new Float32Array(batchSize * hiddenSize);
        const weight = new Float32Array(hiddenSize);

        for (let i = 0; i < input.length; i++) {
          input[i] = (Math.random() - 0.5) * 1000;
        }
        for (let i = 0; i < weight.length; i++) {
          weight[i] = 1.0;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runRMSNorm(
          gpu.device, input, weight, batchSize, hiddenSize
        );

        let hasNaN = false;
        let hasInf = false;
        for (const v of actual) {
          if (isNaN(v)) hasNaN = true;
          if (!isFinite(v)) hasInf = true;
        }

        return { hasNaN, hasInf, maxError: 0 };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.hasInf).toBe(false);
    });
  });
});
