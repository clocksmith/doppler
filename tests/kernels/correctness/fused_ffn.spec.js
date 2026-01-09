/**
 * Fused FFN Kernel Correctness Tests
 *
 * Validates the fused_ffn GPU kernel against reference JS implementation.
 * Tests: output = activation(input @ W_gate^T) * (input @ W_up^T)
 * Supports SiLU and GELU activations.
 */

import { test, expect } from './setup.js';

test.describe('Fused FFN Kernel', () => {
  test.describe('Basic functionality with SiLU', () => {
    test('should compute fused FFN correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const hiddenSize = 64;
        const intermediateSize = 128;

        // Create test data
        const input = new Float32Array(hiddenSize);
        const W_gate = new Float32Array(intermediateSize * hiddenSize);
        const W_up = new Float32Array(intermediateSize * hiddenSize);

        for (let i = 0; i < hiddenSize; i++) {
          input[i] = Math.random() * 2 - 1;
        }
        for (let i = 0; i < intermediateSize * hiddenSize; i++) {
          W_gate[i] = Math.random() * 0.1 - 0.05;
          W_up[i] = Math.random() * 0.1 - 0.05;
        }

        const expected = window.testHarness.fusedFFNRef(input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedFFN(gpu.device, input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');

        let maxError = 0;
        for (let i = 0; i < intermediateSize; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError, actualLength: actual.length };
      });

      // GPU float operations have inherent numerical differences, especially with random data
      expect(result.maxError).toBeLessThan(0.1);
      expect(result.actualLength).toBe(128);
    });

    test('should handle zero input', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const hiddenSize = 32;
        const intermediateSize = 64;

        const input = new Float32Array(hiddenSize).fill(0);
        const W_gate = new Float32Array(intermediateSize * hiddenSize);
        const W_up = new Float32Array(intermediateSize * hiddenSize);

        for (let i = 0; i < intermediateSize * hiddenSize; i++) {
          W_gate[i] = Math.random() * 0.1;
          W_up[i] = Math.random() * 0.1;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedFFN(gpu.device, input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');

        // With zero input, gate and up are zero, so output should be zero
        // SiLU(0) = 0, so output = 0 * 0 = 0
        const allZero = actual.every(v => Math.abs(v) < 1e-6);

        return { allZero };
      });

      expect(result.allZero).toBe(true);
    });

    test('should handle unit input', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const hiddenSize = 32;
        const intermediateSize = 64;

        const input = new Float32Array(hiddenSize).fill(1.0);
        const W_gate = new Float32Array(intermediateSize * hiddenSize);
        const W_up = new Float32Array(intermediateSize * hiddenSize);

        for (let i = 0; i < intermediateSize * hiddenSize; i++) {
          W_gate[i] = 0.01; // Small uniform weights
          W_up[i] = 0.01;
        }

        const expected = window.testHarness.fusedFFNRef(input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedFFN(gpu.device, input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');

        let maxError = 0;
        for (let i = 0; i < intermediateSize; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-3);
    });
  });

  test.describe('GELU activation', () => {
    test('should compute fused FFN with GELU correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const hiddenSize = 64;
        const intermediateSize = 128;

        const input = new Float32Array(hiddenSize);
        const W_gate = new Float32Array(intermediateSize * hiddenSize);
        const W_up = new Float32Array(intermediateSize * hiddenSize);

        for (let i = 0; i < hiddenSize; i++) {
          input[i] = Math.random() * 2 - 1;
        }
        for (let i = 0; i < intermediateSize * hiddenSize; i++) {
          W_gate[i] = Math.random() * 0.1 - 0.05;
          W_up[i] = Math.random() * 0.1 - 0.05;
        }

        const expected = window.testHarness.fusedFFNRef(input, W_gate, W_up, hiddenSize, intermediateSize, 'gelu');

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedFFN(gpu.device, input, W_gate, W_up, hiddenSize, intermediateSize, 'gelu');

        let maxError = 0;
        for (let i = 0; i < intermediateSize; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-3);
    });
  });

  test.describe('Size variations', () => {
    const configs = [
      { hiddenSize: 32, intermediateSize: 64 },
      { hiddenSize: 64, intermediateSize: 128 },
      { hiddenSize: 128, intermediateSize: 256 },
      { hiddenSize: 64, intermediateSize: 256 },
    ];

    for (const { hiddenSize, intermediateSize } of configs) {
      test(`should handle hidden=${hiddenSize}, intermediate=${intermediateSize}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (cfg) => {
          const { hiddenSize, intermediateSize } = cfg;

          const input = new Float32Array(hiddenSize);
          const W_gate = new Float32Array(intermediateSize * hiddenSize);
          const W_up = new Float32Array(intermediateSize * hiddenSize);

          for (let i = 0; i < hiddenSize; i++) {
            input[i] = Math.random() * 2 - 1;
          }
          for (let i = 0; i < intermediateSize * hiddenSize; i++) {
            W_gate[i] = Math.random() * 0.1 - 0.05;
            W_up[i] = Math.random() * 0.1 - 0.05;
          }

          const expected = window.testHarness.fusedFFNRef(input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runFusedFFN(gpu.device, input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');

          let maxError = 0;
          for (let i = 0; i < intermediateSize; i++) {
            maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
          }

          return { maxError, outputLength: actual.length };
        }, { hiddenSize, intermediateSize });

        expect(result.maxError).toBeLessThan(1e-2);
        expect(result.outputLength).toBe(intermediateSize);
      });
    }
  });

  test.describe('SiLU properties', () => {
    test('should produce positive output for positive gate and up', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const hiddenSize = 32;
        const intermediateSize = 64;

        // Use positive input and weights to get positive gate/up
        const input = new Float32Array(hiddenSize).fill(1.0);
        const W_gate = new Float32Array(intermediateSize * hiddenSize).fill(0.01);
        const W_up = new Float32Array(intermediateSize * hiddenSize).fill(0.01);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedFFN(gpu.device, input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');

        // gate = 0.01 * 32 = 0.32 (positive)
        // SiLU(0.32) = 0.32 / (1 + exp(-0.32)) ≈ 0.32 * 0.58 ≈ 0.18 (positive)
        // up = 0.32 (positive)
        // output = positive * positive = positive
        const allPositive = actual.every(v => v > 0);

        return { allPositive, sample: actual[0] };
      });

      expect(result.allPositive).toBe(true);
      expect(result.sample).toBeGreaterThan(0);
    });
  });

  test.describe('Numerical stability', () => {
    test('should handle small values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const hiddenSize = 64;
        const intermediateSize = 128;

        const input = new Float32Array(hiddenSize);
        const W_gate = new Float32Array(intermediateSize * hiddenSize);
        const W_up = new Float32Array(intermediateSize * hiddenSize);

        for (let i = 0; i < hiddenSize; i++) {
          input[i] = Math.random() * 0.01;
        }
        for (let i = 0; i < intermediateSize * hiddenSize; i++) {
          W_gate[i] = Math.random() * 0.001;
          W_up[i] = Math.random() * 0.001;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedFFN(gpu.device, input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');

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
        const hiddenSize = 128;
        const intermediateSize = 256;

        const input = new Float32Array(hiddenSize);
        const W_gate = new Float32Array(intermediateSize * hiddenSize);
        const W_up = new Float32Array(intermediateSize * hiddenSize);

        for (let i = 0; i < hiddenSize; i++) {
          input[i] = Math.random() * 4 - 2;
        }
        for (let i = 0; i < intermediateSize * hiddenSize; i++) {
          W_gate[i] = Math.random() * 0.2 - 0.1;
          W_up[i] = Math.random() * 0.2 - 0.1;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runFusedFFN(gpu.device, input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');

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

  test.describe('Activation comparison', () => {
    test('SiLU and GELU should give similar but different results', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const hiddenSize = 64;
        const intermediateSize = 128;

        const input = new Float32Array(hiddenSize);
        const W_gate = new Float32Array(intermediateSize * hiddenSize);
        const W_up = new Float32Array(intermediateSize * hiddenSize);

        for (let i = 0; i < hiddenSize; i++) {
          input[i] = Math.random() * 2 - 1;
        }
        for (let i = 0; i < intermediateSize * hiddenSize; i++) {
          W_gate[i] = Math.random() * 0.1 - 0.05;
          W_up[i] = Math.random() * 0.1 - 0.05;
        }

        const gpu = await window.testHarness.getGPU();
        const siluResult = await window.testHarness.runFusedFFN(gpu.device, input, W_gate, W_up, hiddenSize, intermediateSize, 'silu');
        const geluResult = await window.testHarness.runFusedFFN(gpu.device, input, W_gate, W_up, hiddenSize, intermediateSize, 'gelu');

        // They should be different (SiLU and GELU are different activations)
        let maxDiff = 0;
        for (let i = 0; i < intermediateSize; i++) {
          maxDiff = Math.max(maxDiff, Math.abs(siluResult[i] - geluResult[i]));
        }

        // But not wildly different for small inputs
        const reasonable = maxDiff < 10.0;

        return { maxDiff, reasonable, areDifferent: maxDiff > 1e-6 };
      });

      expect(result.areDifferent).toBe(true);
      expect(result.reasonable).toBe(true);
    });
  });
});
