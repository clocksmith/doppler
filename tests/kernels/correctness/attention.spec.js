

import { test, expect } from './setup.js';

test.describe('Attention Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should compute self-attention correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
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
      const result = await gpuPage.evaluate(async () => {
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
      const result = await gpuPage.evaluate(async () => {
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
      const result = await gpuPage.evaluate(async () => {
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

  test.describe('Q/K normalization with weight offset', () => {
    test('should apply (1+w) formula when rmsNormWeightOffset=true', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const seqLen = 4;
        const numHeads = 2;
        const headDim = 8;

        // Q values that will be normalized
        const Q = new Float32Array(seqLen * numHeads * headDim);
        for (let i = 0; i < Q.length; i++) Q[i] = 1.0;

        // Q norm weight = zeros
        const qNormWeight = new Float32Array(headDim);
        for (let i = 0; i < headDim; i++) qNormWeight[i] = 0.0;

        const gpu = await window.testHarness.getGPU();

        // With offset=false: result = x/rms * 0 = 0
        const qNoOffset = await window.testHarness.runRMSNorm(
          gpu.device, Q, qNormWeight, seqLen * numHeads, headDim, 1e-6,
          { rmsNormWeightOffset: false }
        );

        // With offset=true: result = x/rms * (1+0) = x/rms
        const qWithOffset = await window.testHarness.runRMSNorm(
          gpu.device, Q, qNormWeight, seqLen * numHeads, headDim, 1e-6,
          { rmsNormWeightOffset: true }
        );

        // Without offset, output should be ~0
        const noOffsetMaxAbs = Math.max(...Array.from(qNoOffset).map(Math.abs));
        // With offset, output should be non-zero (~1.0 since input is ones)
        const withOffsetMaxAbs = Math.max(...Array.from(qWithOffset).map(Math.abs));

        return { noOffsetMaxAbs, withOffsetMaxAbs };
      });

      // No offset: weight of 0 → output ≈ 0
      expect(result.noOffsetMaxAbs).toBeLessThan(1e-5);
      // With offset: (1+0)=1 → output ≈ 1.0 (normalized ones)
      expect(result.withOffsetMaxAbs).toBeGreaterThan(0.9);
    });
  });

  test.describe('Numerical stability', () => {
    test('should handle large attention scores', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
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
