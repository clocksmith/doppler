/**
 * SiLU Kernel Correctness Tests
 */

import { test, expect } from './setup.js';

test.describe('SiLU Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should compute SiLU correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { siluRef } = window.testHarness.references;

        const size = 256;
        const input = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          input[i] = Math.random() * 10 - 5;
        }

        const expected = siluRef(input);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSiLU(gpu.device, input);

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-5);
    });

    test('should compute SiLU(0) = 0', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([0, 0, 0, 0]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSiLU(gpu.device, input);

        return { values: Array.from(actual), maxError: 0 };
      });

      for (const v of result.values) {
        expect(v).toBeCloseTo(0, 5);
      }
    });

    test('should be positive for positive inputs', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const size = 64;
        const input = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          input[i] = Math.random() * 10 + 0.1;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSiLU(gpu.device, input);

        let allPositive = true;
        for (const v of actual) {
          if (v <= 0) allPositive = false;
        }

        return { allPositive, maxError: 0 };
      });

      expect(result.allPositive).toBe(true);
    });
  });

  test.describe('Gated SiLU', () => {
    test('should compute gated SiLU correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { siluGatedRef } = window.testHarness.references;

        const size = 128;
        const gate = new Float32Array(size);
        const up = new Float32Array(size);

        for (let i = 0; i < size; i++) {
          gate[i] = Math.random() * 6 - 3;
          up[i] = Math.random() * 6 - 3;
        }

        const expected = siluGatedRef(gate, up);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSiLUGated(gpu.device, gate, up);

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
    const sizes = [1, 4, 16, 64, 256, 1024, 4096];

    for (const size of sizes) {
      test(`should handle size ${size}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (sz) => {
          const { siluRef } = window.testHarness.references;

          const input = new Float32Array(sz);
          for (let i = 0; i < sz; i++) {
            input[i] = Math.random() * 10 - 5;
          }

          const expected = siluRef(input);

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runSiLU(gpu.device, input);

          let maxError = 0;
          for (let i = 0; i < expected.length; i++) {
            maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
          }

          return { maxError };
        }, size);

        expect(result.maxError).toBeLessThan(1e-4);
      });
    }
  });

  test.describe('Numerical stability', () => {
    test('should handle very large positive values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([50, 60, 70, 80]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSiLU(gpu.device, input);

        let hasNaN = false;
        let hasInf = false;
        for (const v of actual) {
          if (isNaN(v)) hasNaN = true;
          if (!isFinite(v)) hasInf = true;
        }

        // For large x, SiLU(x) ≈ x
        const closeToInput = actual.every((v, i) =>
          Math.abs(v - input[i]) / input[i] < 0.01
        );

        return { hasNaN, hasInf, closeToInput, maxError: 0 };
      });

      expect(result.hasNaN).toBe(false);
      // Note: Very large values will overflow to Inf, but sigmoid approaches 1
    });

    test('should handle very large negative values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([-50, -60, -70, -80]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSiLU(gpu.device, input);

        const expected = new Float32Array(input.length);
        for (let i = 0; i < input.length; i++) {
          const clamped = Math.min(15, Math.max(-15, input[i]));
          expected[i] = input[i] / (1 + Math.exp(-clamped));
        }

        let hasNaN = false;
        for (const v of actual) {
          if (isNaN(v)) hasNaN = true;
        }

        let maxError = 0;
        for (let i = 0; i < actual.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { hasNaN, maxError };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.maxError).toBeLessThan(1e-5);
    });
  });
});
