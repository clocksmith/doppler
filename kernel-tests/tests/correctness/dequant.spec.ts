/**
 * Dequantization Kernel Correctness Tests
 */

import { test, expect } from './setup.js';

interface DequantResult {
  allMatch?: boolean;
  verifications?: Array<{ expected: number; actual: number; match: boolean }>;
  allZero?: boolean;
  correctSize?: boolean;
  hasNonZero?: boolean;
  outputLength?: number;
  values?: number[];
  expected?: number[];
  expectedLength?: number;
  hasNaN?: boolean;
}

test.describe('Dequantization Kernels', () => {
  test.describe('INT8 Dequantization', () => {
    test('should dequantize INT8 correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<DequantResult> => {
        const { dequantInt8Ref } = window.testHarness.references;

        const numChannels = 4;
        const channelSize = 64;
        const totalSize = numChannels * channelSize;

        // Create quantized data
        const quantized = new Int8Array(totalSize);
        for (let i = 0; i < totalSize; i++) {
          quantized[i] = Math.floor(Math.random() * 256) - 128;
        }

        // Create scales and zero points
        const scales = new Float32Array(numChannels);
        const zeroPoints = new Int8Array(numChannels);
        for (let c = 0; c < numChannels; c++) {
          scales[c] = Math.random() * 0.1;
          zeroPoints[c] = Math.floor(Math.random() * 10) - 5;
        }

        const expected = dequantInt8Ref(quantized, scales, zeroPoints, numChannels, channelSize);

        // Manual verification for a few values
        const verifications = [];
        for (let c = 0; c < numChannels; c++) {
          const idx = c * channelSize;
          const expectedVal = (quantized[idx] - zeroPoints[c]) * scales[c];
          verifications.push({
            expected: expectedVal,
            actual: expected[idx],
            match: Math.abs(expectedVal - expected[idx]) < 1e-6,
          });
        }

        return {
          allMatch: verifications.every(v => v.match),
          verifications,
        };
      });

      expect(result.allMatch).toBe(true);
    });

    test('should handle zero scale', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<DequantResult> => {
        const { dequantInt8Ref } = window.testHarness.references;

        const size = 64;
        const quantized = new Int8Array(size);
        for (let i = 0; i < size; i++) {
          quantized[i] = Math.floor(Math.random() * 256) - 128;
        }

        const scales = new Float32Array([0.0]); // Zero scale

        const expected = dequantInt8Ref(quantized, scales, null, 1, size);

        // All outputs should be zero with zero scale
        let allZero = true;
        for (const v of expected) {
          if (v !== 0) allZero = false;
        }

        return { allZero };
      });

      expect(result.allZero).toBe(true);
    });
  });

  test.describe('INT4 Dequantization', () => {
    test('should dequantize INT4 correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<DequantResult> => {
        const { dequantInt4Ref } = window.testHarness.references;

        const numElements = 64;
        const groupSize = 32;
        const numGroups = numElements / groupSize;

        // Create packed INT4 data (2 values per byte)
        const numBytes = numElements / 2;
        const quantized = new Uint8Array(numBytes);
        for (let i = 0; i < numBytes; i++) {
          const low = Math.floor(Math.random() * 16);
          const high = Math.floor(Math.random() * 16);
          quantized[i] = (high << 4) | low;
        }

        // Scales per group
        const scales = new Float32Array(numGroups);
        for (let g = 0; g < numGroups; g++) {
          scales[g] = Math.random() * 0.1;
        }

        const expected = dequantInt4Ref(quantized, scales, numElements, groupSize);

        // Verify output size
        const correctSize = expected.length === numElements;

        // Verify some values are non-zero (unless all inputs map to zero)
        let hasNonZero = false;
        for (const v of expected) {
          if (v !== 0) hasNonZero = true;
        }

        return { correctSize, hasNonZero, outputLength: expected.length };
      });

      expect(result.correctSize).toBe(true);
    });

    test('should unpack INT4 values correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<DequantResult> => {
        const { dequantInt4Ref } = window.testHarness.references;

        // Simple test: 4 elements (2 bytes)
        const numElements = 4;
        const groupSize = 4;

        // Pack known values: [3, -2, 7, -8] as INT4
        // 3 = 0011, -2 (14 as unsigned) = 1110
        // 7 = 0111, -8 (8 as unsigned) = 1000
        // Byte 0: high nibble = -2 (14), low nibble = 3 -> 0xE3
        // Byte 1: high nibble = -8 (8), low nibble = 7 -> 0x87
        const quantized = new Uint8Array([0xE3, 0x87]);

        const scales = new Float32Array([1.0]); // Unit scale for easy verification

        const expected = dequantInt4Ref(quantized, scales, numElements, groupSize);

        // Values are: 3, -2, 7, -8 (converted from unsigned 4-bit)
        return {
          values: Array.from(expected),
          expected: [3, -2, 7, -8],
        };
      });

      expect(result.values![0]).toBeCloseTo(3, 4);
      expect(result.values![1]).toBeCloseTo(-2, 4);
      expect(result.values![2]).toBeCloseTo(7, 4);
      expect(result.values![3]).toBeCloseTo(-8, 4);
    });
  });

  test.describe('Q4_0 Block Dequantization', () => {
    test('should dequantize Q4_0 blocks', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<DequantResult> => {
        const { dequantQ4_0Ref } = window.testHarness.references;

        // Q4_0: 32 values per block, 2 bytes scale + 16 bytes data = 18 bytes per block
        const numBlocks = 2;
        const blockSize = 32;
        const bytesPerBlock = 18;

        const quantized = new Uint8Array(numBlocks * bytesPerBlock);

        // Fill with test data
        for (let b = 0; b < numBlocks; b++) {
          const offset = b * bytesPerBlock;

          // Set scale (fp16) - just use simple value
          // For testing, set to a known fp16 pattern
          quantized[offset] = 0x00;
          quantized[offset + 1] = 0x3C; // 1.0 in fp16

          // Set data (16 bytes = 32 int4 values)
          for (let i = 0; i < 16; i++) {
            quantized[offset + 2 + i] = 0x88; // Both nibbles = 8, which is 0 after -8 offset
          }
        }

        const expected = dequantQ4_0Ref(quantized, numBlocks);

        // All values should be 0 (since 8 - 8 = 0, and 0 * scale = 0)
        let allZero = true;
        for (const v of expected) {
          if (Math.abs(v) > 1e-5) allZero = false;
        }

        return {
          outputLength: expected.length,
          expectedLength: numBlocks * blockSize,
          allZero,
        };
      });

      expect(result.outputLength).toBe(result.expectedLength);
      expect(result.allZero).toBe(true);
    });
  });

  test.describe('Q4_K (Q4_K_M) GPU Dequantization', () => {
    test('should dequantize Q4_K blocks on GPU (mixed sign)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const { quantizeQ4_KRef, dequantQ4_KRef } = window.testHarness.references;
        const numBlocks = 4;
        const blockElems = 256;

        // Deterministic mixed-sign input to avoid flukes
        const values = new Float32Array(numBlocks * blockElems);
        for (let i = 0; i < values.length; i++) {
          values[i] = Math.sin(i * 0.1) * 0.75 + Math.cos(i * 0.03) * 0.25;
        }

        const quantized = quantizeQ4_KRef(values, numBlocks);
        const expected = dequantQ4_KRef(quantized, numBlocks);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runDequantQ4K(gpu.device, quantized, numBlocks);

        let maxError = 0;
        let minVal = Infinity;
        let maxVal = -Infinity;
        for (let i = 0; i < expected.length; i++) {
          const e = expected[i];
          const a = actual[i];
          const err = Math.abs(e - a);
          if (err > maxError) maxError = err;
          if (a < minVal) minVal = a;
          if (a > maxVal) maxVal = a;
        }

        return { maxError, minVal, maxVal, length: actual.length };
      });

      expect(result.length).toBe(4 * 256);
      expect(result.maxError).toBeLessThan(1e-3);
      expect(result.minVal).toBeLessThan(0);
      expect(result.maxVal).toBeGreaterThan(0);
    });
  });

  test.describe('Size variations', () => {
    const sizes = [32, 64, 256, 1024, 4096];

    for (const size of sizes) {
      test(`INT8 dequant size ${size}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (sz: number): Promise<DequantResult> => {
          const { dequantInt8Ref } = window.testHarness.references;

          const quantized = new Int8Array(sz);
          for (let i = 0; i < sz; i++) {
            quantized[i] = Math.floor(Math.random() * 256) - 128;
          }

          const scales = new Float32Array([0.05]);

          // dequantInt8Ref(quantized, scales, zeroPoints, numChannels, channelSize)
          const expected = dequantInt8Ref(quantized, scales, null, 1, sz);

          // Verify size and no NaN
          let hasNaN = false;
          for (const v of expected) {
            if (isNaN(v)) hasNaN = true;
          }

          return {
            correctSize: expected.length === sz,
            hasNaN,
          };
        }, size);

        expect(result.correctSize).toBe(true);
        expect(result.hasNaN).toBe(false);
      });
    }
  });
});
