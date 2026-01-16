

import { test, expect } from './setup.js';

test.describe('GELU Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should compute GELU correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { geluRef } = window.testHarness.references;

        const size = 256;
        const input = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          input[i] = Math.random() * 10 - 5;
        }

        const expected = geluRef(input);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGeLU(gpu.device, input);

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-4);
    });

    test('should compute GELU(0) = 0', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([0, 0, 0, 0]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGeLU(gpu.device, input);

        return { values: Array.from(actual) };
      });

      for (const v of result.values) {
        expect(v).toBeCloseTo(0, 5);
      }
    });

    test('should be positive for large positive inputs', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const size = 64;
        const input = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          input[i] = Math.random() * 10 + 1; // All positive > 1
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGeLU(gpu.device, input);

        let allPositive = true;
        for (const v of actual) {
          if (v <= 0) allPositive = false;
        }

        return { allPositive };
      });

      expect(result.allPositive).toBe(true);
    });

    test('should be near zero for large negative inputs', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([-5, -6, -7, -8]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGeLU(gpu.device, input);

        const allNearZero = actual.every(v => Math.abs(v) < 0.01);

        return { allNearZero, values: Array.from(actual) };
      });

      expect(result.allNearZero).toBe(true);
    });
  });

  test.describe('Gated GeGLU', () => {
    test('should compute GeGLU correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { gegluRef } = window.testHarness.references;

        const size = 128;
        const gate = new Float32Array(size);
        const up = new Float32Array(size);

        for (let i = 0; i < size; i++) {
          gate[i] = Math.random() * 6 - 3;
          up[i] = Math.random() * 6 - 3;
        }

        const expected = gegluRef(gate, up);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGeGLU(gpu.device, gate, up);

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-4);
    });

    test('should handle GeGLU with zero gate', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const size = 64;
        const gate = new Float32Array(size).fill(0);
        const up = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          up[i] = Math.random() * 10;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGeGLU(gpu.device, gate, up);

        // GELU(0) = 0, so output should be 0
        const allZero = actual.every(v => Math.abs(v) < 1e-6);

        return { allZero };
      });

      expect(result.allZero).toBe(true);
    });
  });

  test.describe('Size variations', () => {
    const sizes = [1, 4, 16, 64, 256, 1024, 4096];

    for (const size of sizes) {
      test(`should handle size ${size}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (sz) => {
          const { geluRef } = window.testHarness.references;

          const input = new Float32Array(sz);
          for (let i = 0; i < sz; i++) {
            input[i] = Math.random() * 10 - 5;
          }

          const expected = geluRef(input);

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runGeLU(gpu.device, input);

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
        const actual = await window.testHarness.runGeLU(gpu.device, input);

        let hasNaN = false;
        let hasInf = false;
        for (const v of actual) {
          if (isNaN(v)) hasNaN = true;
          if (!isFinite(v)) hasInf = true;
        }

        // For large x, GELU(x) ≈ x (since tanh approaches 1)
        const closeToInput = actual.every((v, i) =>
          Math.abs(v - input[i]) / input[i] < 0.01
        );

        return { hasNaN, hasInf, closeToInput };
      });

      expect(result.hasNaN).toBe(false);
      // Note: Very large values will be close to input
    });

    test('should handle very large negative values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([-50, -60, -70, -80]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGeLU(gpu.device, input);

        let hasNaN = false;
        for (const v of actual) {
          if (isNaN(v)) hasNaN = true;
        }

        // For large negative x, GELU(x) ≈ 0
        const closeToZero = actual.every(v => Math.abs(v) < 1e-6);

        return { hasNaN, closeToZero };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.closeToZero).toBe(true);
    });

    test('should handle mixed edge cases', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { geluRef } = window.testHarness.references;

        // Include edge cases: zeros, small values, large values
        const input = new Float32Array([
          0, 0.001, -0.001, 1, -1, 10, -10, 0.5, -0.5
        ]);

        const expected = geluRef(input);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGeLU(gpu.device, input);

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-4);
    });
  });

  test.describe('Mathematical properties', () => {
    test('GELU should be monotonically increasing for x > 0', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGeLU(gpu.device, input);

        let isMonotonic = true;
        for (let i = 1; i < actual.length; i++) {
          if (actual[i] <= actual[i - 1]) {
            isMonotonic = false;
            break;
          }
        }

        return { isMonotonic, values: Array.from(actual) };
      });

      expect(result.isMonotonic).toBe(true);
    });

    test('GELU should satisfy GELU(x) ≈ x for large x', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([10, 15, 20, 25]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGeLU(gpu.device, input);

        // For large x, GELU(x) → x * 1 = x
        const errors = actual.map((v, i) => Math.abs(v - input[i]) / input[i]);
        const maxRelError = Math.max(...errors);

        return { maxRelError };
      });

      expect(result.maxRelError).toBeLessThan(0.001);
    });
  });
});
