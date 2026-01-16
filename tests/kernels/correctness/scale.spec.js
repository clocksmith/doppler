

import { test, expect } from './setup.js';

test.describe('Scale Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should scale by 1.0 (identity)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const size = 256;
        const input = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          input[i] = Math.random() * 10 - 5;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runScale(gpu.device, input, 1.0);

        let maxError = 0;
        for (let i = 0; i < input.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - input[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-6);
    });

    test('should scale by 2.0 (double)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([1, 2, 3, 4, 5]);
        const scale = 2.0;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runScale(gpu.device, input, scale);

        const expected = input.map(v => v * scale);

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError, actual: Array.from(actual) };
      });

      expect(result.maxError).toBeLessThan(1e-6);
      expect(result.actual).toEqual([2, 4, 6, 8, 10]);
    });

    test('should scale by 0.0 (zero output)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([1, 2, 3, 4, 5]);
        const scale = 0.0;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runScale(gpu.device, input, scale);

        const allZero = actual.every(v => v === 0);

        return { allZero };
      });

      expect(result.allZero).toBe(true);
    });

    test('should scale by -1.0 (negate)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([1, -2, 3, -4, 5]);
        const scale = -1.0;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runScale(gpu.device, input, scale);

        const expected = input.map(v => v * scale);

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError, actual: Array.from(actual) };
      });

      expect(result.maxError).toBeLessThan(1e-6);
      expect(result.actual).toEqual([-1, 2, -3, 4, -5]);
    });

    test('should scale by fractional value (0.5)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([2, 4, 6, 8, 10]);
        const scale = 0.5;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runScale(gpu.device, input, scale);

        const expected = input.map(v => v * scale);

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-6);
    });
  });

  test.describe('Random data', () => {
    const scales = [0.001, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0];

    for (const scale of scales) {
      test(`should correctly scale by ${scale}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (s) => {
          const size = 1024;
          const input = new Float32Array(size);
          for (let i = 0; i < size; i++) {
            input[i] = Math.random() * 20 - 10;
          }

          const expected = new Float32Array(size);
          for (let i = 0; i < size; i++) {
            expected[i] = input[i] * s;
          }

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runScale(gpu.device, input, s);

          let maxError = 0;
          for (let i = 0; i < expected.length; i++) {
            maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
          }

          return { maxError };
        }, scale);

        expect(result.maxError).toBeLessThan(1e-5);
      });
    }
  });

  test.describe('Size variations', () => {
    const sizes = [1, 4, 16, 64, 256, 1024, 4096, 16384];

    for (const size of sizes) {
      test(`should handle size ${size}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (sz) => {
          const input = new Float32Array(sz);
          for (let i = 0; i < sz; i++) {
            input[i] = Math.random() * 10 - 5;
          }
          const scale = 0.7;

          const expected = new Float32Array(sz);
          for (let i = 0; i < sz; i++) {
            expected[i] = input[i] * scale;
          }

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runScale(gpu.device, input, scale);

          let maxError = 0;
          for (let i = 0; i < expected.length; i++) {
            maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
          }

          return { maxError, actualLength: actual.length };
        }, size);

        expect(result.actualLength).toBe(size);
        expect(result.maxError).toBeLessThan(1e-5);
      });
    }
  });

  test.describe('Edge cases', () => {
    test('should handle zero input', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([0, 0, 0, 0]);
        const scale = 5.0;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runScale(gpu.device, input, scale);

        const allZero = actual.every(v => v === 0);

        return { allZero };
      });

      expect(result.allZero).toBe(true);
    });

    test('should handle very small scale', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([1, 2, 3, 4]);
        const scale = 1e-7;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runScale(gpu.device, input, scale);

        const expected = input.map(v => v * scale);

        let maxRelError = 0;
        for (let i = 0; i < expected.length; i++) {
          if (expected[i] !== 0) {
            maxRelError = Math.max(maxRelError, Math.abs(actual[i] - expected[i]) / Math.abs(expected[i]));
          }
        }

        return { maxRelError };
      });

      expect(result.maxRelError).toBeLessThan(1e-5);
    });

    test('should handle very large scale', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const input = new Float32Array([0.001, 0.01, 0.1, 1.0]);
        const scale = 1e6;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runScale(gpu.device, input, scale);

        const expected = input.map(v => v * scale);

        let maxRelError = 0;
        for (let i = 0; i < expected.length; i++) {
          if (expected[i] !== 0) {
            maxRelError = Math.max(maxRelError, Math.abs(actual[i] - expected[i]) / Math.abs(expected[i]));
          }
        }

        return { maxRelError, hasNaN: actual.some(v => isNaN(v)) };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.maxRelError).toBeLessThan(1e-5);
    });

    test('should not produce NaN or Inf for normal inputs', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const size = 1000;
        const input = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          input[i] = Math.random() * 1000 - 500;
        }
        const scale = 2.5;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runScale(gpu.device, input, scale);

        let hasNaN = false;
        let hasInf = false;
        for (const v of actual) {
          if (isNaN(v)) hasNaN = true;
          if (!isFinite(v)) hasInf = true;
        }

        return { hasNaN, hasInf };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.hasInf).toBe(false);
    });
  });

  test.describe('Attention scaling use case', () => {
    test('should correctly apply 1/sqrt(head_dim) scaling', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        // Simulate attention scaling: scale = 1/sqrt(64) = 0.125
        const headDim = 64;
        const scale = 1 / Math.sqrt(headDim);

        const size = 256;
        const input = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          input[i] = Math.random() * 10 - 5;
        }

        const expected = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          expected[i] = input[i] * scale;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runScale(gpu.device, input, scale);

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError, scale };
      });

      expect(result.maxError).toBeLessThan(1e-5);
    });
  });
});
