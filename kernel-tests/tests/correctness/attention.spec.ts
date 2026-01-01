/**
 * Attention Kernel Correctness Tests
 */

import { test, expect } from './setup.js';

interface AttentionResult {
  maxError: number;
  outputSize?: number;
  hasNaN?: boolean;
  hasInf?: boolean;
}

test.describe('Attention Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should compute self-attention correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<AttentionResult> => {
        const { attentionRef } = window.testHarness.references;

        const seqLen = 8;
        const kvLen = 8;
        const numHeads = 4;
        const numKVHeads = 4;
        const headDim = 32;

        const Q = new Float32Array(seqLen * numHeads * headDim);
        const K = new Float32Array(kvLen * numKVHeads * headDim);
        const V = new Float32Array(kvLen * numKVHeads * headDim);

        for (let i = 0; i < Q.length; i++) Q[i] = Math.random() * 2 - 1;
        for (let i = 0; i < K.length; i++) K[i] = Math.random() * 2 - 1;
        for (let i = 0; i < V.length; i++) V[i] = Math.random() * 2 - 1;

        const expected = attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runAttention(
          gpu.device, Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-3);
    });

    test('should handle causal masking', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<AttentionResult> => {
        const { attentionRef, createCausalMask } = window.testHarness.references;

        const seqLen = 8;
        const numHeads = 2;
        const headDim = 16;

        const Q = new Float32Array(seqLen * numHeads * headDim);
        const K = new Float32Array(seqLen * numHeads * headDim);
        const V = new Float32Array(seqLen * numHeads * headDim);

        for (let i = 0; i < Q.length; i++) Q[i] = Math.random() * 2 - 1;
        for (let i = 0; i < K.length; i++) K[i] = Math.random() * 2 - 1;
        for (let i = 0; i < V.length; i++) V[i] = Math.random() * 2 - 1;

        const mask = createCausalMask(seqLen);
        const expected = attentionRef(Q, K, V, seqLen, seqLen, numHeads, numHeads, headDim, mask);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runAttention(
          gpu.device, Q, K, V, seqLen, seqLen, numHeads, numHeads, headDim, mask
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-3);
    });
  });

  test.describe('GQA (Grouped Query Attention)', () => {
    test('should handle GQA with 4:1 ratio', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<AttentionResult> => {
        const { attentionRef } = window.testHarness.references;

        const seqLen = 8;
        const numHeads = 8;
        const numKVHeads = 2;
        const headDim = 32;

        const Q = new Float32Array(seqLen * numHeads * headDim);
        const K = new Float32Array(seqLen * numKVHeads * headDim);
        const V = new Float32Array(seqLen * numKVHeads * headDim);

        for (let i = 0; i < Q.length; i++) Q[i] = Math.random() * 2 - 1;
        for (let i = 0; i < K.length; i++) K[i] = Math.random() * 2 - 1;
        for (let i = 0; i < V.length; i++) V[i] = Math.random() * 2 - 1;

        const expected = attentionRef(Q, K, V, seqLen, seqLen, numHeads, numKVHeads, headDim);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runAttention(
          gpu.device, Q, K, V, seqLen, seqLen, numHeads, numKVHeads, headDim
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError, outputSize: actual.length };
      });

      expect(result.maxError).toBeLessThan(1e-3);
    });

    test('should handle MQA (single KV head)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<AttentionResult> => {
        const { attentionRef } = window.testHarness.references;

        const seqLen = 8;
        const numHeads = 8;
        const numKVHeads = 1;
        const headDim = 32;

        const Q = new Float32Array(seqLen * numHeads * headDim);
        const K = new Float32Array(seqLen * numKVHeads * headDim);
        const V = new Float32Array(seqLen * numKVHeads * headDim);

        for (let i = 0; i < Q.length; i++) Q[i] = Math.random() * 2 - 1;
        for (let i = 0; i < K.length; i++) K[i] = Math.random() * 2 - 1;
        for (let i = 0; i < V.length; i++) V[i] = Math.random() * 2 - 1;

        const expected = attentionRef(Q, K, V, seqLen, seqLen, numHeads, numKVHeads, headDim);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runAttention(
          gpu.device, Q, K, V, seqLen, seqLen, numHeads, numKVHeads, headDim
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-3);
    });
  });

  test.describe('Numerical stability', () => {
    test('should handle large attention scores', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<AttentionResult> => {
        const seqLen = 16;
        const numHeads = 4;
        const headDim = 32;

        const Q = new Float32Array(seqLen * numHeads * headDim);
        const K = new Float32Array(seqLen * numHeads * headDim);
        const V = new Float32Array(seqLen * numHeads * headDim);

        for (let i = 0; i < Q.length; i++) Q[i] = Math.random() * 10;
        for (let i = 0; i < K.length; i++) K[i] = Math.random() * 10;
        for (let i = 0; i < V.length; i++) V[i] = Math.random();

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runAttention(
          gpu.device, Q, K, V, seqLen, seqLen, numHeads, numHeads, headDim
        );

        let hasNaN = false;
        let hasInf = false;
        for (const v of actual) {
          if (isNaN(v)) hasNaN = true;
          if (!isFinite(v)) hasInf = true;
        }

        return { hasNaN, hasInf, maxError: 0 };
      });

      expect(result.hasNaN).toBe(false);
      expect(result.hasInf).toBe(false);
    });
  });
});
