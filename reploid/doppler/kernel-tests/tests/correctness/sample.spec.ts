/**
 * Sample Kernel Correctness Tests
 *
 * Tests GPU-side argmax and top-k sampling against reference implementations.
 */

import { test, expect } from './setup.js';

interface SampleResult {
  gpuToken: number;
  refToken: number;
  match: boolean;
  topKMatches?: boolean;
}

test.describe('Argmax Kernel', () => {
  test('should find max index correctly', async ({ gpuPage }) => {
    const result = await gpuPage.evaluate(async (): Promise<SampleResult> => {
      const { argmaxRef } = window.testHarness.references;

      // Create logits with known max
      const vocabSize = 1000;
      const logits = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        logits[i] = Math.random() * 10 - 5;
      }
      // Set known max at specific position
      const maxPos = 42;
      logits[maxPos] = 100;

      const gpu = await window.testHarness.getGPU();
      const gpuToken = await window.testHarness.runArgmax(gpu.device, logits);
      const refToken = argmaxRef(logits);

      return {
        gpuToken,
        refToken,
        match: gpuToken === refToken,
      };
    });

    expect(result.match).toBe(true);
    expect(result.gpuToken).toBe(42);
    expect(result.refToken).toBe(42);
  });

  test('should handle max at position 0', async ({ gpuPage }) => {
    const result = await gpuPage.evaluate(async (): Promise<SampleResult> => {
      const { argmaxRef } = window.testHarness.references;

      const vocabSize = 1000;
      const logits = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        logits[i] = -i; // Descending, max at 0
      }

      const gpu = await window.testHarness.getGPU();
      const gpuToken = await window.testHarness.runArgmax(gpu.device, logits);
      const refToken = argmaxRef(logits);

      return {
        gpuToken,
        refToken,
        match: gpuToken === refToken,
      };
    });

    expect(result.match).toBe(true);
    expect(result.gpuToken).toBe(0);
  });

  test('should handle max at last position', async ({ gpuPage }) => {
    const result = await gpuPage.evaluate(async (): Promise<SampleResult> => {
      const { argmaxRef } = window.testHarness.references;

      const vocabSize = 1000;
      const logits = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        logits[i] = i; // Ascending, max at end
      }

      const gpu = await window.testHarness.getGPU();
      const gpuToken = await window.testHarness.runArgmax(gpu.device, logits);
      const refToken = argmaxRef(logits);

      return {
        gpuToken,
        refToken,
        match: gpuToken === refToken,
      };
    });

    expect(result.match).toBe(true);
    expect(result.gpuToken).toBe(999);
  });

  test('should handle large vocabulary (256K)', async ({ gpuPage }) => {
    const result = await gpuPage.evaluate(async (): Promise<SampleResult> => {
      const { argmaxRef } = window.testHarness.references;

      const vocabSize = 256000; // Typical LLM vocab size
      const logits = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        logits[i] = Math.random() * 2 - 1;
      }
      // Set known max deep in the vocabulary
      const maxPos = 128000;
      logits[maxPos] = 50;

      const gpu = await window.testHarness.getGPU();
      const gpuToken = await window.testHarness.runArgmax(gpu.device, logits);
      const refToken = argmaxRef(logits);

      return {
        gpuToken,
        refToken,
        match: gpuToken === refToken,
      };
    });

    expect(result.match).toBe(true);
    expect(result.gpuToken).toBe(128000);
  });

  test('should handle negative logits', async ({ gpuPage }) => {
    const result = await gpuPage.evaluate(async (): Promise<SampleResult> => {
      const { argmaxRef } = window.testHarness.references;

      const vocabSize = 500;
      const logits = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        logits[i] = -100 - i; // All negative
      }
      // Least negative at position 100
      logits[100] = -10;

      const gpu = await window.testHarness.getGPU();
      const gpuToken = await window.testHarness.runArgmax(gpu.device, logits);
      const refToken = argmaxRef(logits);

      return {
        gpuToken,
        refToken,
        match: gpuToken === refToken,
      };
    });

    expect(result.match).toBe(true);
    expect(result.gpuToken).toBe(100);
  });
});

test.describe('Top-K Sampling Kernel', () => {
  test('should sample from top-k with temperature', async ({ gpuPage }) => {
    const result = await gpuPage.evaluate(async (): Promise<{
      gpuToken: number;
      inTopK: boolean;
      topKIndices: number[];
    }> => {
      const { topkArgmaxRef } = window.testHarness.references;

      const vocabSize = 1000;
      const logits = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        logits[i] = Math.random() * 10 - 5;
      }
      // Set clear top-k values
      logits[10] = 100;
      logits[20] = 95;
      logits[30] = 90;
      logits[40] = 85;
      logits[50] = 80;

      const gpu = await window.testHarness.getGPU();
      const gpuToken = await window.testHarness.runSampleTopK(
        gpu.device,
        logits,
        1.0, // temperature
        5,   // topK
        0.5  // randomValue
      );

      const { indices } = topkArgmaxRef(logits, 5);

      return {
        gpuToken,
        inTopK: indices.includes(gpuToken),
        topKIndices: indices,
      };
    });

    // GPU should sample from top-k
    expect(result.inTopK).toBe(true);
    expect(result.topKIndices).toContain(result.gpuToken);
  });

  test('should use greedy at temperature ~0', async ({ gpuPage }) => {
    const result = await gpuPage.evaluate(async (): Promise<SampleResult> => {
      const { argmaxRef } = window.testHarness.references;

      const vocabSize = 1000;
      const logits = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        logits[i] = Math.random() * 10;
      }
      logits[42] = 1000; // Clear max

      const gpu = await window.testHarness.getGPU();
      // Very low temperature should act like greedy
      const gpuToken = await window.testHarness.runSampleTopK(
        gpu.device,
        logits,
        0.001, // near-zero temperature
        40,
        0.99   // high random value shouldn't matter
      );
      const refToken = argmaxRef(logits);

      return {
        gpuToken,
        refToken,
        match: gpuToken === refToken,
      };
    });

    expect(result.match).toBe(true);
    expect(result.gpuToken).toBe(42);
  });

  test('should handle high temperature (more random)', async ({ gpuPage }) => {
    const result = await gpuPage.evaluate(async (): Promise<{
      gpuToken: number;
      inTopK: boolean;
    }> => {
      const vocabSize = 100;
      const logits = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        logits[i] = Math.random() * 2; // Small differences
      }

      const gpu = await window.testHarness.getGPU();
      // High temperature spreads probability
      const gpuToken = await window.testHarness.runSampleTopK(
        gpu.device,
        logits,
        2.0,  // high temperature
        10,   // topK
        0.5
      );

      // With high temp, should still be in vocabulary range
      return {
        gpuToken,
        inTopK: gpuToken >= 0 && gpuToken < vocabSize,
      };
    });

    expect(result.gpuToken).toBeGreaterThanOrEqual(0);
    expect(result.gpuToken).toBeLessThan(100);
  });
});

test.describe('Vocabulary sizes', () => {
  const sizes = [100, 256, 1000, 4096, 32000, 128000];

  for (const vocabSize of sizes) {
    test(`argmax with vocab size ${vocabSize}`, async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (vs: number): Promise<SampleResult> => {
        const { argmaxRef } = window.testHarness.references;

        const logits = new Float32Array(vs);
        for (let i = 0; i < vs; i++) {
          logits[i] = Math.random() * 2 - 1;
        }
        // Known max position
        const maxPos = Math.floor(vs / 2);
        logits[maxPos] = 100;

        const gpu = await window.testHarness.getGPU();
        const gpuToken = await window.testHarness.runArgmax(gpu.device, logits);
        const refToken = argmaxRef(logits);

        return {
          gpuToken,
          refToken,
          match: gpuToken === refToken,
        };
      }, vocabSize);

      expect(result.match).toBe(true);
      expect(result.gpuToken).toBe(Math.floor(vocabSize / 2));
    });
  }
});
