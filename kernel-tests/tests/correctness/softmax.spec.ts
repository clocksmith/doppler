/**
 * Softmax Kernel Correctness Tests
 */

import { test, expect } from './setup.js';

interface SoftmaxResult {
  maxError: number;
  rowSums?: number[];
  hasNaN?: boolean;
  hasInf?: boolean;
}

interface SoftmaxParams {
  inner: number;
  outer: number;
}

test.describe('Softmax Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should compute softmax correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<SoftmaxResult> => {
        const { softmaxRef } = window.testHarness.references;

        const innerSize = 8;
        const outerSize = 4;

        const input = new Float32Array(outerSize * innerSize);
        for (let i = 0; i < input.length; i++) {
          input[i] = Math.random() * 4 - 2;
        }

        const expected = softmaxRef(input, innerSize, outerSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSoftmax(
          gpu.device, input, innerSize, outerSize
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        // Check rows sum to 1
        const rowSums = [];
        for (let r = 0; r < outerSize; r++) {
          let sum = 0;
          for (let c = 0; c < innerSize; c++) {
            sum += actual[r * innerSize + c];
          }
          rowSums.push(sum);
        }

        return { maxError, rowSums };
      });

      expect(result.maxError).toBeLessThan(1e-5);
      for (const sum of result.rowSums!) {
        expect(sum).toBeCloseTo(1.0, 4);
      }
    });

    test('should handle temperature scaling', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<SoftmaxResult> => {
        const { softmaxRef } = window.testHarness.references;

        const innerSize = 16;
        const outerSize = 2;
        const temperature = 2.0;

        const input = new Float32Array(outerSize * innerSize);
        for (let i = 0; i < input.length; i++) {
          input[i] = Math.random() * 10 - 5;
        }

        const expected = softmaxRef(input, innerSize, outerSize, temperature);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSoftmax(
          gpu.device, input, innerSize, outerSize, temperature
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-5);
    });
  });

  test.describe('Numerical stability', () => {
    test('should handle large values without overflow', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<SoftmaxResult> => {
        const innerSize = 32;
        const outerSize = 4;

        // Large values that would overflow without max subtraction
        const input = new Float32Array(outerSize * innerSize);
        for (let i = 0; i < input.length; i++) {
          input[i] = 80 + Math.random() * 10;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSoftmax(
          gpu.device, input, innerSize, outerSize
        );

        let hasNaN = false;
        let hasInf = false;
        for (const v of actual) {
          if (isNaN(v)) hasNaN = true;
          if (!isFinite(v)) hasInf = true;
        }

        // Check sums
        const rowSums = [];
        for (let r = 0; r < outerSize; r++) {
          let sum = 0;
          for (let c = 0; c < innerSize; c++) {
            sum += actual[r * innerSize + c];
          }
          rowSums.push(sum);
        }

        return { hasNaN, hasInf, rowSums, maxError: 0 };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.hasInf).toBe(false);
      for (const sum of result.rowSums!) {
        expect(sum).toBeCloseTo(1.0, 4);
      }
    });

    test('should handle very negative values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<SoftmaxResult> => {
        const innerSize = 16;
        const outerSize = 2;

        const input = new Float32Array(outerSize * innerSize);
        for (let i = 0; i < input.length; i++) {
          input[i] = -80 - Math.random() * 10;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSoftmax(
          gpu.device, input, innerSize, outerSize
        );

        let hasNaN = false;
        for (const v of actual) {
          if (isNaN(v)) hasNaN = true;
        }

        return { hasNaN, maxError: 0 };
      });

      expect(result.hasNaN).toBe(false);
    });
  });

  test.describe('Size variations', () => {
    const sizes = [
      { inner: 2, outer: 1 },
      { inner: 8, outer: 1 },
      { inner: 64, outer: 1 },
      { inner: 256, outer: 1 },
      { inner: 32, outer: 16 },
      { inner: 128, outer: 64 },
    ];

    for (const { inner, outer } of sizes) {
      test(`should handle size ${inner}x${outer}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (params: SoftmaxParams): Promise<SoftmaxResult> => {
          const { softmaxRef } = window.testHarness.references;

          const { inner, outer } = params;
          const input = new Float32Array(outer * inner);
          for (let i = 0; i < input.length; i++) {
            input[i] = Math.random() * 4 - 2;
          }

          const expected = softmaxRef(input, inner, outer);

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runSoftmax(
            gpu.device, input, inner, outer
          );

          let maxError = 0;
          for (let i = 0; i < expected.length; i++) {
            maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
          }

          return { maxError };
        }, { inner, outer });

        expect(result.maxError).toBeLessThan(1e-4);
      });
    }
  });
});
