/**
 * RoPE Kernel Correctness Tests
 */

import { test, expect } from './setup.js';

interface RoPEResult {
  maxError: number;
  maxNormDiff?: number;
}

interface RoPEConfig {
  seqLen: number;
  numHeads: number;
  headDim: number;
}

test.describe('RoPE Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should apply RoPE correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<RoPEResult> => {
        const { ropeRef, computeRopeFreqs } = window.testHarness.references;

        const seqLen = 8;
        const numHeads = 4;
        const headDim = 32;

        const input = new Float32Array(seqLen * numHeads * headDim);
        for (let i = 0; i < input.length; i++) {
          input[i] = Math.random() * 2 - 1;
        }

        const { cos, sin } = computeRopeFreqs(headDim, seqLen);
        const expected = ropeRef(input, cos, sin, seqLen, numHeads, headDim, 0);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runRoPE(
          gpu.device, input, seqLen, numHeads, headDim
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-4);
    });

    test('should handle position offset', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<RoPEResult> => {
        const { ropeRef, computeRopeFreqs } = window.testHarness.references;

        const seqLen = 4;
        const numHeads = 2;
        const headDim = 16;
        const startPos = 10;

        const input = new Float32Array(seqLen * numHeads * headDim);
        for (let i = 0; i < input.length; i++) {
          input[i] = Math.random() * 2 - 1;
        }

        const { cos, sin } = computeRopeFreqs(headDim, seqLen + startPos);
        const expected = ropeRef(input, cos, sin, seqLen, numHeads, headDim, startPos);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runRoPE(
          gpu.device, input, seqLen, numHeads, headDim, startPos
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBeLessThan(1e-4);
    });
  });

  test.describe('Properties', () => {
    test('should preserve vector norms', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<RoPEResult> => {
        const seqLen = 4;
        const numHeads = 2;
        const headDim = 32;

        const input = new Float32Array(seqLen * numHeads * headDim);
        for (let i = 0; i < input.length; i++) {
          input[i] = Math.random() * 2 - 1;
        }

        const gpu = await window.testHarness.getGPU();
        const output = await window.testHarness.runRoPE(
          gpu.device, input, seqLen, numHeads, headDim
        );

        // Compute norms for each head
        const inputNorms = [];
        const outputNorms = [];

        for (let s = 0; s < seqLen; s++) {
          for (let h = 0; h < numHeads; h++) {
            let inputNorm = 0;
            let outputNorm = 0;
            const offset = s * numHeads * headDim + h * headDim;

            for (let d = 0; d < headDim; d++) {
              inputNorm += input[offset + d] ** 2;
              outputNorm += output[offset + d] ** 2;
            }

            inputNorms.push(Math.sqrt(inputNorm));
            outputNorms.push(Math.sqrt(outputNorm));
          }
        }

        let maxNormDiff = 0;
        for (let i = 0; i < inputNorms.length; i++) {
          maxNormDiff = Math.max(maxNormDiff,
            Math.abs(inputNorms[i] - outputNorms[i]));
        }

        return { maxNormDiff, maxError: 0 };
      });

      // RoPE is a rotation, should preserve norms
      expect(result.maxNormDiff).toBeLessThan(1e-4);
    });
  });

  test.describe('Size variations', () => {
    const configs = [
      { seqLen: 1, numHeads: 1, headDim: 64 },
      { seqLen: 16, numHeads: 8, headDim: 64 },
      { seqLen: 32, numHeads: 32, headDim: 128 },
      { seqLen: 128, numHeads: 8, headDim: 128 },
    ];

    for (const config of configs) {
      test(`should handle ${config.seqLen}x${config.numHeads}x${config.headDim}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (cfg: RoPEConfig): Promise<RoPEResult> => {
          const { ropeRef, computeRopeFreqs } = window.testHarness.references;

          const { seqLen, numHeads, headDim } = cfg;

          const input = new Float32Array(seqLen * numHeads * headDim);
          for (let i = 0; i < input.length; i++) {
            input[i] = Math.random() * 2 - 1;
          }

          const { cos, sin } = computeRopeFreqs(headDim, seqLen);
          const expected = ropeRef(input, cos, sin, seqLen, numHeads, headDim, 0);

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runRoPE(
            gpu.device, input, seqLen, numHeads, headDim
          );

          let maxError = 0;
          for (let i = 0; i < expected.length; i++) {
            maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
          }

          return { maxError };
        }, config);

        expect(result.maxError).toBeLessThan(1e-3);
      });
    }
  });
});
