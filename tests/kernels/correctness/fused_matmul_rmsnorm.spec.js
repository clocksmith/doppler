

import { test, expect } from './setup.js';

test.describe('Fused Matmul + RMSNorm Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should compute matmul + rmsnorm correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 64;
        const K = 32;

        // Create test data
        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const normWeight = new Float32Array(N);

        for (let i = 0; i < K; i++) {
          input[i] = Math.random() * 2 - 1;
        }
        for (let i = 0; i < N * K; i++) {
          weight[i] = Math.random() * 0.1 - 0.05;
        }
        for (let i = 0; i < N; i++) {
          normWeight[i] = Math.random() * 0.5 + 0.5; // Positive weights
        }

        const expected = window.testHarness.fusedMatmulRMSNormRef(input, weight, normWeight, N, K);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulRMSNorm(gpu.device, input, weight, normWeight, N, K);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError, actualLength: actual.length };
      });

      expect(result.maxError).toBeLessThan(1e-3);
      expect(result.actualLength).toBe(64);
    });

    test('should handle unit norm weights', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 64;
        const K = 32;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const normWeight = new Float32Array(N).fill(1.0); // Unit weights

        for (let i = 0; i < K; i++) input[i] = Math.random() * 2 - 1;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.1 - 0.05;

        const expected = window.testHarness.fusedMatmulRMSNormRef(input, weight, normWeight, N, K);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulRMSNorm(gpu.device, input, weight, normWeight, N, K);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-3);
    });

    test('should handle different epsilon values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 64;
        const K = 32;
        const eps = 1e-6;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const normWeight = new Float32Array(N);

        for (let i = 0; i < K; i++) input[i] = Math.random() * 2 - 1;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.1 - 0.05;
        for (let i = 0; i < N; i++) normWeight[i] = Math.random() * 0.5 + 0.5;

        const expected = window.testHarness.fusedMatmulRMSNormRef(input, weight, normWeight, N, K, eps);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulRMSNorm(gpu.device, input, weight, normWeight, N, K, eps);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-3);
    });
  });

  test.describe('With residual', () => {
    test('should add residual after rmsnorm', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 64;
        const K = 32;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const normWeight = new Float32Array(N);
        const residual = new Float32Array(N);

        for (let i = 0; i < K; i++) input[i] = Math.random() * 2 - 1;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.1 - 0.05;
        for (let i = 0; i < N; i++) normWeight[i] = Math.random() * 0.5 + 0.5;
        for (let i = 0; i < N; i++) residual[i] = Math.random() * 2 - 1;

        const expected = window.testHarness.fusedMatmulRMSNormRef(input, weight, normWeight, N, K, 1e-5, residual);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulRMSNorm(gpu.device, input, weight, normWeight, N, K, 1e-5, residual);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-3);
    });

    test('should match non-residual when residual is zero', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 64;
        const K = 32;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const normWeight = new Float32Array(N);
        const zeroResidual = new Float32Array(N).fill(0);

        for (let i = 0; i < K; i++) input[i] = Math.random() * 2 - 1;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.1 - 0.05;
        for (let i = 0; i < N; i++) normWeight[i] = Math.random() * 0.5 + 0.5;

        const gpu = await window.testHarness.getGPU();

        const withoutResidual = await window.testHarness.runFusedMatmulRMSNorm(gpu.device, input, weight, normWeight, N, K);
        const withZeroResidual = await window.testHarness.runFusedMatmulRMSNorm(gpu.device, input, weight, normWeight, N, K, 1e-5, zeroResidual);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          maxError = Math.max(maxError, Math.abs(withoutResidual[i] - withZeroResidual[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-5);
    });
  });

  test.describe('Size variations', () => {
    const configs = [
      { N: 32, K: 32 },
      { N: 64, K: 64 },
      { N: 128, K: 64 },
      { N: 256, K: 128 },
    ];

    for (const { N, K } of configs) {
      test(`should handle N=${N}, K=${K}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (cfg) => {
          const { N, K } = cfg;

          const input = new Float32Array(K);
          const weight = new Float32Array(N * K);
          const normWeight = new Float32Array(N);

          for (let i = 0; i < K; i++) input[i] = Math.random() * 2 - 1;
          for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.1 - 0.05;
          for (let i = 0; i < N; i++) normWeight[i] = Math.random() * 0.5 + 0.5;

          const expected = window.testHarness.fusedMatmulRMSNormRef(input, weight, normWeight, N, K);

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runFusedMatmulRMSNorm(gpu.device, input, weight, normWeight, N, K);

          let maxError = 0;
          for (let i = 0; i < N; i++) {
            maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
          }

          return { maxError, outputLength: actual.length };
        }, { N, K });

        expect(result.maxError).toBeLessThan(1e-2);
        expect(result.outputLength).toBe(N);
      });
    }
  });

  test.describe('RMSNorm properties', () => {
    test('should produce normalized output (approximate unit RMS)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 64;
        const K = 32;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const normWeight = new Float32Array(N).fill(1.0); // Unit norm weights

        for (let i = 0; i < K; i++) input[i] = Math.random() * 2 - 1;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.1 - 0.05;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulRMSNorm(gpu.device, input, weight, normWeight, N, K);

        // Compute RMS of output (should be ~1 with unit norm weights)
        let sumSq = 0;
        for (let i = 0; i < N; i++) {
          sumSq += actual[i] * actual[i];
        }
        const rms = Math.sqrt(sumSq / N);

        return { rms };
      });

      // RMS should be approximately 1 with unit norm weights
      expect(result.rms).toBeCloseTo(1.0, 0);
    });
  });

  test.describe('Numerical stability', () => {
    test('should handle small intermediate values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 64;
        const K = 32;

        // Small values that might cause numerical issues
        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const normWeight = new Float32Array(N);

        for (let i = 0; i < K; i++) input[i] = Math.random() * 0.001;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.001;
        for (let i = 0; i < N; i++) normWeight[i] = 1.0;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulRMSNorm(gpu.device, input, weight, normWeight, N, K);

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

    test('should not produce NaN or Inf with typical values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const N = 128;
        const K = 64;

        const input = new Float32Array(K);
        const weight = new Float32Array(N * K);
        const normWeight = new Float32Array(N);

        for (let i = 0; i < K; i++) input[i] = Math.random() * 4 - 2;
        for (let i = 0; i < N * K; i++) weight[i] = Math.random() * 0.2 - 0.1;
        for (let i = 0; i < N; i++) normWeight[i] = Math.random() * 0.5 + 0.5;

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedMatmulRMSNorm(gpu.device, input, weight, normWeight, N, K);

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
});
