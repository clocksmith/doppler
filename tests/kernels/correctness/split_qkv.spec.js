/**
 * Split QKV Kernel Correctness Tests
 *
 * Validates the split_qkv.wgsl GPU kernel against reference JS implementation.
 * Tests splitting fused QKV tensor [numTokens, qSize+kSize+vSize] into Q, K, V.
 */

import { test, expect } from './setup.js';

test.describe('Split QKV Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should split equal-sized Q, K, V', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { splitQkvRef, fuseQkvRef } = window.testHarness.references;

        const numTokens = 4;
        const qSize = 64;
        const kSize = 64;
        const vSize = 64;
        const qkvSize = qSize + kSize + vSize;

        // Create test data
        const qkv = new Float32Array(numTokens * qkvSize);
        for (let i = 0; i < qkv.length; i++) {
          qkv[i] = Math.random() * 2 - 1;
        }

        const expected = splitQkvRef(qkv, numTokens, qSize, kSize, vSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        let qMaxError = 0;
        let kMaxError = 0;
        let vMaxError = 0;

        for (let i = 0; i < expected.Q.length; i++) {
          qMaxError = Math.max(qMaxError, Math.abs(actual.Q[i] - expected.Q[i]));
        }
        for (let i = 0; i < expected.K.length; i++) {
          kMaxError = Math.max(kMaxError, Math.abs(actual.K[i] - expected.K[i]));
        }
        for (let i = 0; i < expected.V.length; i++) {
          vMaxError = Math.max(vMaxError, Math.abs(actual.V[i] - expected.V[i]));
        }

        return {
          qMaxError,
          kMaxError,
          vMaxError,
          qLength: actual.Q.length,
          kLength: actual.K.length,
          vLength: actual.V.length,
        };
      });

      expect(result.qMaxError).toBeLessThan(1e-6);
      expect(result.kMaxError).toBeLessThan(1e-6);
      expect(result.vMaxError).toBeLessThan(1e-6);
      expect(result.qLength).toBe(4 * 64);
      expect(result.kLength).toBe(4 * 64);
      expect(result.vLength).toBe(4 * 64);
    });

    test('should split with different Q, K, V sizes', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { splitQkvRef } = window.testHarness.references;

        // Typical attention config: Q larger due to more heads
        const numTokens = 2;
        const qSize = 256; // 8 heads * 32 head_dim
        const kSize = 64;  // 2 KV heads * 32 head_dim
        const vSize = 64;  // 2 KV heads * 32 head_dim
        const qkvSize = qSize + kSize + vSize;

        const qkv = new Float32Array(numTokens * qkvSize);
        for (let i = 0; i < qkv.length; i++) {
          qkv[i] = Math.random() * 2 - 1;
        }

        const expected = splitQkvRef(qkv, numTokens, qSize, kSize, vSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        let maxError = 0;
        for (let i = 0; i < expected.Q.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.Q[i] - expected.Q[i]));
        }
        for (let i = 0; i < expected.K.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.K[i] - expected.K[i]));
        }
        for (let i = 0; i < expected.V.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.V[i] - expected.V[i]));
        }

        return {
          maxError,
          qLength: actual.Q.length,
          kLength: actual.K.length,
          vLength: actual.V.length,
        };
      });

      expect(result.maxError).toBeLessThan(1e-6);
      expect(result.qLength).toBe(2 * 256);
      expect(result.kLength).toBe(2 * 64);
      expect(result.vLength).toBe(2 * 64);
    });

    test('should handle single token', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { splitQkvRef } = window.testHarness.references;

        const numTokens = 1;
        const qSize = 128;
        const kSize = 128;
        const vSize = 128;
        const qkvSize = qSize + kSize + vSize;

        const qkv = new Float32Array(numTokens * qkvSize);
        for (let i = 0; i < qkv.length; i++) {
          qkv[i] = i / qkvSize; // Sequential for easy verification
        }

        const expected = splitQkvRef(qkv, numTokens, qSize, kSize, vSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        // Verify first and last elements
        const qFirst = actual.Q[0];
        const qLast = actual.Q[qSize - 1];
        const kFirst = actual.K[0];
        const kLast = actual.K[kSize - 1];
        const vFirst = actual.V[0];
        const vLast = actual.V[vSize - 1];

        let maxError = 0;
        for (let i = 0; i < expected.Q.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.Q[i] - expected.Q[i]));
        }
        for (let i = 0; i < expected.K.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.K[i] - expected.K[i]));
        }
        for (let i = 0; i < expected.V.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.V[i] - expected.V[i]));
        }

        return {
          maxError,
          qFirst,
          qLast,
          kFirst,
          kLast,
          vFirst,
          vLast,
        };
      });

      expect(result.maxError).toBeLessThan(1e-6);
      // Verify sequential values were correctly split
      expect(result.qFirst).toBeCloseTo(0, 5);
      expect(result.kFirst).toBeCloseTo(128 / 384, 5);
      expect(result.vFirst).toBeCloseTo(256 / 384, 5);
    });
  });

  test.describe('Data preservation', () => {
    test('should preserve exact values with known data', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const numTokens = 2;
        const qSize = 3;
        const kSize = 2;
        const vSize = 2;

        // Known input: [Q0, K0, V0, Q1, K1, V1] for each token
        const qkv = new Float32Array([
          // Token 0: Q=[1,2,3], K=[4,5], V=[6,7]
          1, 2, 3, 4, 5, 6, 7,
          // Token 1: Q=[8,9,10], K=[11,12], V=[13,14]
          8, 9, 10, 11, 12, 13, 14,
        ]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        return {
          Q: Array.from(actual.Q),
          K: Array.from(actual.K),
          V: Array.from(actual.V),
        };
      });

      // Q should be [1,2,3, 8,9,10]
      expect(result.Q).toEqual([1, 2, 3, 8, 9, 10]);
      // K should be [4,5, 11,12]
      expect(result.K).toEqual([4, 5, 11, 12]);
      // V should be [6,7, 13,14]
      expect(result.V).toEqual([6, 7, 13, 14]);
    });

    test('should handle negative values', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { splitQkvRef } = window.testHarness.references;

        const numTokens = 3;
        const qSize = 32;
        const kSize = 32;
        const vSize = 32;
        const qkvSize = qSize + kSize + vSize;

        const qkv = new Float32Array(numTokens * qkvSize);
        for (let i = 0; i < qkv.length; i++) {
          qkv[i] = Math.random() * 4 - 2; // Range [-2, 2]
        }

        const expected = splitQkvRef(qkv, numTokens, qSize, kSize, vSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        let maxError = 0;
        for (let i = 0; i < expected.Q.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.Q[i] - expected.Q[i]));
        }
        for (let i = 0; i < expected.K.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.K[i] - expected.K[i]));
        }
        for (let i = 0; i < expected.V.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.V[i] - expected.V[i]));
        }

        // Check that we have some negative values
        const hasNegQ = actual.Q.some(v => v < 0);
        const hasNegK = actual.K.some(v => v < 0);
        const hasNegV = actual.V.some(v => v < 0);

        return { maxError, hasNegQ, hasNegK, hasNegV };
      });

      expect(result.maxError).toBeLessThan(1e-6);
      expect(result.hasNegQ).toBe(true);
      expect(result.hasNegK).toBe(true);
      expect(result.hasNegV).toBe(true);
    });

    test('should handle zeros', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const numTokens = 2;
        const qSize = 16;
        const kSize = 16;
        const vSize = 16;
        const qkvSize = qSize + kSize + vSize;

        const qkv = new Float32Array(numTokens * qkvSize).fill(0);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        const qAllZero = actual.Q.every(v => v === 0);
        const kAllZero = actual.K.every(v => v === 0);
        const vAllZero = actual.V.every(v => v === 0);

        return { qAllZero, kAllZero, vAllZero };
      });

      expect(result.qAllZero).toBe(true);
      expect(result.kAllZero).toBe(true);
      expect(result.vAllZero).toBe(true);
    });
  });

  test.describe('Size variations', () => {
    const configs = [
      { numTokens: 1, qSize: 64, kSize: 64, vSize: 64 },
      { numTokens: 4, qSize: 64, kSize: 64, vSize: 64 },
      { numTokens: 16, qSize: 64, kSize: 64, vSize: 64 },
      { numTokens: 1, qSize: 512, kSize: 128, vSize: 128 },
      { numTokens: 8, qSize: 256, kSize: 64, vSize: 64 },
      { numTokens: 32, qSize: 128, kSize: 32, vSize: 32 },
    ];

    for (const cfg of configs) {
      test(`should handle numTokens=${cfg.numTokens}, Q=${cfg.qSize}, K=${cfg.kSize}, V=${cfg.vSize}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (config) => {
          const { splitQkvRef } = window.testHarness.references;
          const { numTokens, qSize, kSize, vSize } = config;
          const qkvSize = qSize + kSize + vSize;

          const qkv = new Float32Array(numTokens * qkvSize);
          for (let i = 0; i < qkv.length; i++) {
            qkv[i] = Math.random() * 2 - 1;
          }

          const expected = splitQkvRef(qkv, numTokens, qSize, kSize, vSize);

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

          let maxError = 0;
          for (let i = 0; i < expected.Q.length; i++) {
            maxError = Math.max(maxError, Math.abs(actual.Q[i] - expected.Q[i]));
          }
          for (let i = 0; i < expected.K.length; i++) {
            maxError = Math.max(maxError, Math.abs(actual.K[i] - expected.K[i]));
          }
          for (let i = 0; i < expected.V.length; i++) {
            maxError = Math.max(maxError, Math.abs(actual.V[i] - expected.V[i]));
          }

          return {
            maxError,
            qCorrectLength: actual.Q.length === numTokens * qSize,
            kCorrectLength: actual.K.length === numTokens * kSize,
            vCorrectLength: actual.V.length === numTokens * vSize,
          };
        }, cfg);

        expect(result.maxError).toBeLessThan(1e-6);
        expect(result.qCorrectLength).toBe(true);
        expect(result.kCorrectLength).toBe(true);
        expect(result.vCorrectLength).toBe(true);
      });
    }
  });

  test.describe('Typical attention configurations', () => {
    test('should handle Gemma-2B config (8 heads, 256 head_dim)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { splitQkvRef } = window.testHarness.references;

        // Gemma-2B: 8 Q heads, 1 KV head, 256 head_dim
        const numTokens = 4;
        const numHeads = 8;
        const numKVHeads = 1;
        const headDim = 256;
        const qSize = numHeads * headDim;
        const kSize = numKVHeads * headDim;
        const vSize = numKVHeads * headDim;
        const qkvSize = qSize + kSize + vSize;

        const qkv = new Float32Array(numTokens * qkvSize);
        for (let i = 0; i < qkv.length; i++) {
          qkv[i] = Math.random() * 2 - 1;
        }

        const expected = splitQkvRef(qkv, numTokens, qSize, kSize, vSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        let maxError = 0;
        for (let i = 0; i < expected.Q.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.Q[i] - expected.Q[i]));
        }
        for (let i = 0; i < expected.K.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.K[i] - expected.K[i]));
        }
        for (let i = 0; i < expected.V.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.V[i] - expected.V[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-6);
    });

    test('should handle LLaMA config (32 heads, 128 head_dim)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { splitQkvRef } = window.testHarness.references;

        // LLaMA-style: 32 Q heads, 8 KV heads, 128 head_dim
        const numTokens = 2;
        const numHeads = 32;
        const numKVHeads = 8;
        const headDim = 128;
        const qSize = numHeads * headDim;
        const kSize = numKVHeads * headDim;
        const vSize = numKVHeads * headDim;
        const qkvSize = qSize + kSize + vSize;

        const qkv = new Float32Array(numTokens * qkvSize);
        for (let i = 0; i < qkv.length; i++) {
          qkv[i] = Math.random() * 2 - 1;
        }

        const expected = splitQkvRef(qkv, numTokens, qSize, kSize, vSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        let maxError = 0;
        for (let i = 0; i < expected.Q.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.Q[i] - expected.Q[i]));
        }
        for (let i = 0; i < expected.K.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.K[i] - expected.K[i]));
        }
        for (let i = 0; i < expected.V.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.V[i] - expected.V[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-6);
    });

    test('should handle MHA config (Q=K=V heads)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { splitQkvRef } = window.testHarness.references;

        // Standard MHA: same number of Q, K, V heads
        const numTokens = 8;
        const numHeads = 12;
        const headDim = 64;
        const qSize = numHeads * headDim;
        const kSize = numHeads * headDim;
        const vSize = numHeads * headDim;
        const qkvSize = qSize + kSize + vSize;

        const qkv = new Float32Array(numTokens * qkvSize);
        for (let i = 0; i < qkv.length; i++) {
          qkv[i] = Math.random() * 2 - 1;
        }

        const expected = splitQkvRef(qkv, numTokens, qSize, kSize, vSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        let maxError = 0;
        for (let i = 0; i < expected.Q.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.Q[i] - expected.Q[i]));
        }
        for (let i = 0; i < expected.K.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.K[i] - expected.K[i]));
        }
        for (let i = 0; i < expected.V.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.V[i] - expected.V[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-6);
    });
  });

  test.describe('Edge cases', () => {
    test('should handle very small sizes', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { splitQkvRef } = window.testHarness.references;

        const numTokens = 1;
        const qSize = 4;
        const kSize = 4;
        const vSize = 4;
        const qkvSize = qSize + kSize + vSize;

        const qkv = new Float32Array(numTokens * qkvSize);
        for (let i = 0; i < qkv.length; i++) {
          qkv[i] = i + 1;
        }

        const expected = splitQkvRef(qkv, numTokens, qSize, kSize, vSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        let maxError = 0;
        for (let i = 0; i < expected.Q.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.Q[i] - expected.Q[i]));
        }
        for (let i = 0; i < expected.K.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.K[i] - expected.K[i]));
        }
        for (let i = 0; i < expected.V.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.V[i] - expected.V[i]));
        }

        return {
          maxError,
          Q: Array.from(actual.Q),
          K: Array.from(actual.K),
          V: Array.from(actual.V),
        };
      });

      expect(result.maxError).toBeLessThan(1e-6);
      expect(result.Q).toEqual([1, 2, 3, 4]);
      expect(result.K).toEqual([5, 6, 7, 8]);
      expect(result.V).toEqual([9, 10, 11, 12]);
    });

    test('should handle large token count', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { splitQkvRef } = window.testHarness.references;

        const numTokens = 128;
        const qSize = 64;
        const kSize = 64;
        const vSize = 64;
        const qkvSize = qSize + kSize + vSize;

        const qkv = new Float32Array(numTokens * qkvSize);
        for (let i = 0; i < qkv.length; i++) {
          qkv[i] = Math.random() * 2 - 1;
        }

        const expected = splitQkvRef(qkv, numTokens, qSize, kSize, vSize);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        let maxError = 0;
        for (let i = 0; i < expected.Q.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.Q[i] - expected.Q[i]));
        }
        for (let i = 0; i < expected.K.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.K[i] - expected.K[i]));
        }
        for (let i = 0; i < expected.V.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual.V[i] - expected.V[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-6);
    });

    test('should not produce NaN or Inf', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const numTokens = 8;
        const qSize = 128;
        const kSize = 128;
        const vSize = 128;
        const qkvSize = qSize + kSize + vSize;

        const qkv = new Float32Array(numTokens * qkvSize);
        for (let i = 0; i < qkv.length; i++) {
          qkv[i] = Math.random() * 1000 - 500;
        }

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runSplitQKV(gpu.device, qkv, numTokens, qSize, kSize, vSize);

        let hasNaN = false;
        let hasInf = false;

        for (const v of actual.Q) {
          if (isNaN(v)) hasNaN = true;
          if (!isFinite(v)) hasInf = true;
        }
        for (const v of actual.K) {
          if (isNaN(v)) hasNaN = true;
          if (!isFinite(v)) hasInf = true;
        }
        for (const v of actual.V) {
          if (isNaN(v)) hasNaN = true;
          if (!isFinite(v)) hasInf = true;
        }

        return { hasNaN, hasInf };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.hasInf).toBe(false);
    });
  });

  test.describe('Round-trip verification', () => {
    test('should preserve data through fuse -> split cycle', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { fuseQkvRef } = window.testHarness.references;

        const numTokens = 4;
        const qSize = 64;
        const kSize = 32;
        const vSize = 32;

        // Create original Q, K, V
        const originalQ = new Float32Array(numTokens * qSize);
        const originalK = new Float32Array(numTokens * kSize);
        const originalV = new Float32Array(numTokens * vSize);

        for (let i = 0; i < originalQ.length; i++) {
          originalQ[i] = Math.random() * 2 - 1;
        }
        for (let i = 0; i < originalK.length; i++) {
          originalK[i] = Math.random() * 2 - 1;
        }
        for (let i = 0; i < originalV.length; i++) {
          originalV[i] = Math.random() * 2 - 1;
        }

        // Fuse them
        const fused = fuseQkvRef(originalQ, originalK, originalV, numTokens, qSize, kSize, vSize);

        // Split via GPU
        const gpu = await window.testHarness.getGPU();
        const split = await window.testHarness.runSplitQKV(gpu.device, fused, numTokens, qSize, kSize, vSize);

        // Compare
        let qMaxError = 0;
        let kMaxError = 0;
        let vMaxError = 0;

        for (let i = 0; i < originalQ.length; i++) {
          qMaxError = Math.max(qMaxError, Math.abs(split.Q[i] - originalQ[i]));
        }
        for (let i = 0; i < originalK.length; i++) {
          kMaxError = Math.max(kMaxError, Math.abs(split.K[i] - originalK[i]));
        }
        for (let i = 0; i < originalV.length; i++) {
          vMaxError = Math.max(vMaxError, Math.abs(split.V[i] - originalV[i]));
        }

        return { qMaxError, kMaxError, vMaxError };
      });

      expect(result.qMaxError).toBeLessThan(1e-6);
      expect(result.kMaxError).toBeLessThan(1e-6);
      expect(result.vMaxError).toBeLessThan(1e-6);
    });
  });
});
