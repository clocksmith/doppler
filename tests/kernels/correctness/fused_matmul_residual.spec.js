

import { test, expect } from './setup.js';

test.describe('Fused Matmul + Residual Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should compute matmul + residual correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 64;
        const K = 32;

        // Create test data
        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const residual = new Float32Array(N);

        for (let i = 0; i < K; i++) {
          input[i] = Math.random() * 2 - 1;
        }
        for (let i = 0; i < N * K; i++) {
          weight[i] = Math.random() * 0.1 - 0.05;
        }
        for (let i = 0; i < N; i++) {
          residual[i] = Math.random() * 2 - 1;
        }

        const expected = window.testHarness.fusedMatmulResidualRef(input, weight, residual, N, K);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulResidual(gpu.device, input, weight, residual, N, K);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError, actualLength: actual.length };
      });

      expect(result.maxError).toBeLessThan(1e-4);
      expect(result.actualLength).toBe(64);
    });

    test('should handle alpha scaling', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 32;
        const K = 16;
        const alpha = 0.5;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const residual = new Float32Array(N);

        for (let i = 0; i < K; i++) input[i] = 1.0;
        for (let i = 0; i < N * K; i++) weight[i] = 0.1;
        for (let i = 0; i < N; i++) residual[i] = 1.0;

        const expected = window.testHarness.fusedMatmulResidualRef(input, weight, residual, N, K, alpha);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulResidual(gpu.device, input, weight, residual, N, K, alpha);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        // Expected: 0.1 * 16 * 0.5 + 1.0 = 0.8 + 1.0 = 1.8
        return { maxError, sampleValue: actual[0], expectedValue: expected[0] };
      });

      expect(result.maxError).toBeLessThan(1e-4);
      expect(result.sampleValue).toBeCloseTo(result.expectedValue, 4);
    });

    test('should handle zero residual', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 32;
        const K = 16;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const residual = new Float32Array(N).fill(0);

        for (let i = 0; i < K; i++) input[i] = Math.random() * 2 - 1;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.1 - 0.05;

        const expected = window.testHarness.fusedMatmulResidualRef(input, weight, residual, N, K);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulResidual(gpu.device, input, weight, residual, N, K);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-4);
    });

    test('should handle zero input', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 32;
        const K = 16;

        const input = new Float32Array(K).fill(0);
        const weight = new Float32Array(N * K);
        const residual = new Float32Array(N);

        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.1;
        for (let i = 0; i < N; i++) residual[i] = Math.random() * 2 - 1;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulResidual(gpu.device, input, weight, residual, N, K);

        // With zero input, output should equal residual
        let maxError = 0;
        for (let i = 0; i < N; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - residual[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-6);
    });
  });

  test.describe('Size variations', () => {
    const configs = [
      { N: 64, K: 64 },
      { N: 128, K: 64 },
      { N: 256, K: 128 },
      { N: 64, K: 256 },
    ];

    for (const { N, K } of configs) {
      test(`should handle N=${N}, K=${K}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (cfg) => {
          const { N, K } = cfg;

          const input = new Float32Array(K);
          const weight = new Float32Array(N * K);
          const residual = new Float32Array(N);

          for (let i = 0; i < K; i++) input[i] = Math.random() * 2 - 1;
          for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.1 - 0.05;
          for (let i = 0; i < N; i++) residual[i] = Math.random() * 2 - 1;

          const expected = window.testHarness.fusedMatmulResidualRef(input, weight, residual, N, K);

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runFusedMatmulResidual(gpu.device, input, weight, residual, N, K);

          let maxError = 0;
          for (let i = 0; i < N; i++) {
            maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
          }

          return { maxError, outputLength: actual.length };
        }, { N, K });

        expect(result.maxError).toBeLessThan(1e-3);
        expect(result.outputLength).toBe(N);
      });
    }
  });

  test.describe('Numerical stability', () => {
    test('should handle small values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 64;
        const K = 32;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const residual = new Float32Array(N);

        for (let i = 0; i < K; i++) input[i] = Math.random() * 0.001;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.001;
        for (let i = 0; i < N; i++) residual[i] = Math.random() * 0.001;

        const expected = window.testHarness.fusedMatmulResidualRef(input, weight, residual, N, K);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulResidual(gpu.device, input, weight, residual, N, K);

        let maxError = 0;
        let hasNaN = false;
        for (let i = 0; i < N; i++) {
          if (isNaN(actual[i])) hasNaN = true;
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError, hasNaN };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.maxError).toBeLessThan(1e-6);
    });

    test('should not produce NaN or Inf', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 128;
        const K = 64;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const residual = new Float32Array(N);

        for (let i = 0; i < K; i++) input[i] = Math.random() * 10 - 5;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 2 - 1;
        for (let i = 0; i < N; i++) residual[i] = Math.random() * 10 - 5;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulResidual(gpu.device, input, weight, residual, N, K);

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

  test.describe('Typical model configurations', () => {
    test('should handle Gemma-2B hidden size (2048)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        // Gemma-2B: hidden_size=2048
        // Note: This kernel requires N <= colsPerWg (typically 256)
        // For larger N, use regular matmul + residual_add
        const N = 256; // Within single workgroup limit
        const K = 256;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const residual = new Float32Array(N);

        for (let i = 0; i < K; i++) input[i] = Math.random() * 0.5 - 0.25;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.02 - 0.01;
        for (let i = 0; i < N; i++) residual[i] = Math.random() * 0.5 - 0.25;

        const expected = window.testHarness.fusedMatmulResidualRef(input, weight, residual, N, K);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulResidual(gpu.device, input, weight, residual, N, K);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-3);
    });
  });
});
