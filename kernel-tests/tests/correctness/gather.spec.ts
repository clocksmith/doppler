/**
 * Gather (Embedding Lookup) Kernel Correctness Tests
 */

import { test, expect } from './setup.js';

interface GatherResult {
  maxError: number;
  firstMatch?: boolean;
  lastMatch?: boolean;
  match?: boolean;
  length?: number;
  expectedLength?: number;
}

interface GatherConfig {
  vocab: number;
  dim: number;
  seq: number;
}

test.describe('Gather Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should gather embeddings correctly', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<GatherResult> => {
        const { gatherRef } = window.testHarness.references;

        const vocabSize = 100;
        const embedDim = 64;
        const seqLen = 8;

        // Create embedding table
        const embeddings = new Float32Array(vocabSize * embedDim);
        for (let i = 0; i < embeddings.length; i++) {
          embeddings[i] = Math.random() * 2 - 1;
        }

        // Create indices
        const indices = new Uint32Array(seqLen);
        for (let i = 0; i < seqLen; i++) {
          indices[i] = Math.floor(Math.random() * vocabSize);
        }

        const expected = gatherRef(embeddings, indices, vocabSize, embedDim);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGather(
          gpu.device, embeddings, indices, vocabSize, embedDim
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      // Gather should be exact (no computation)
      expect(result.maxError).toBe(0);
    });

    test('should handle repeated indices', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<GatherResult> => {
        const { gatherRef } = window.testHarness.references;

        const vocabSize = 50;
        const embedDim = 32;
        const seqLen = 16;

        const embeddings = new Float32Array(vocabSize * embedDim);
        for (let i = 0; i < embeddings.length; i++) {
          embeddings[i] = Math.random();
        }

        // Use repeated indices
        const indices = new Uint32Array(seqLen);
        for (let i = 0; i < seqLen; i++) {
          indices[i] = i % 5; // Only use indices 0-4
        }

        const expected = gatherRef(embeddings, indices, vocabSize, embedDim);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGather(
          gpu.device, embeddings, indices, vocabSize, embedDim
        );

        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
        }

        return { maxError };
      });

      expect(result.maxError).toBe(0);
    });
  });

  test.describe('Size variations', () => {
    const configs = [
      { vocab: 32000, dim: 512, seq: 1 },
      { vocab: 32000, dim: 512, seq: 32 },
      { vocab: 32000, dim: 4096, seq: 1 },
      { vocab: 32000, dim: 4096, seq: 16 },
      // Skipping very large vocab - would require 2GB+ embedding table
      // { vocab: 128000, dim: 4096, seq: 8 },
    ];

    for (const { vocab, dim, seq } of configs) {
      test(`should handle vocab=${vocab}, dim=${dim}, seq=${seq}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (cfg: GatherConfig): Promise<GatherResult> => {
          const { gatherRef } = window.testHarness.references;

          const { vocab, dim, seq } = cfg;

          const embeddings = new Float32Array(vocab * dim);
          for (let i = 0; i < embeddings.length; i++) {
            embeddings[i] = Math.random() * 2 - 1;
          }

          const indices = new Uint32Array(seq);
          for (let i = 0; i < seq; i++) {
            indices[i] = Math.floor(Math.random() * vocab);
          }

          const expected = gatherRef(embeddings, indices, vocab, dim);

          const gpu = await window.testHarness.getGPU();
          const actual = await window.testHarness.runGather(
            gpu.device, embeddings, indices, vocab, dim
          );

          let maxError = 0;
          for (let i = 0; i < expected.length; i++) {
            maxError = Math.max(maxError, Math.abs(actual[i] - expected[i]));
          }

          return { maxError };
        }, { vocab, dim, seq });

        expect(result.maxError).toBe(0);
      });
    }
  });

  test.describe('Edge cases', () => {
    test('should handle first and last vocab entries', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<GatherResult> => {
        const vocabSize = 1000;
        const embedDim = 64;

        const embeddings = new Float32Array(vocabSize * embedDim);
        for (let i = 0; i < embeddings.length; i++) {
          embeddings[i] = Math.random();
        }

        // Test first and last entries
        const indices = new Uint32Array([0, vocabSize - 1, 0, vocabSize - 1]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGather(
          gpu.device, embeddings, indices, vocabSize, embedDim
        );

        // Verify first entry
        let firstMatch = true;
        for (let d = 0; d < embedDim; d++) {
          if (actual[d] !== embeddings[d]) firstMatch = false;
        }

        // Verify last entry
        let lastMatch = true;
        const lastOffset = (vocabSize - 1) * embedDim;
        for (let d = 0; d < embedDim; d++) {
          if (actual[embedDim + d] !== embeddings[lastOffset + d]) lastMatch = false;
        }

        return { firstMatch, lastMatch, maxError: 0 };
      });

      expect(result.firstMatch).toBe(true);
      expect(result.lastMatch).toBe(true);
    });

    test('should handle single token', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<GatherResult> => {
        const vocabSize = 32000;
        const embedDim = 4096;

        const embeddings = new Float32Array(vocabSize * embedDim);
        for (let i = 0; i < embeddings.length; i++) {
          embeddings[i] = Math.random();
        }

        const indices = new Uint32Array([42]);

        const gpu = await window.testHarness.getGPU();
        const actual = await window.testHarness.runGather(
          gpu.device, embeddings, indices, vocabSize, embedDim
        );

        // Verify the gathered embedding matches
        let match = true;
        const offset = 42 * embedDim;
        for (let d = 0; d < embedDim; d++) {
          if (actual[d] !== embeddings[offset + d]) match = false;
        }

        return { match, length: actual.length, expectedLength: embedDim, maxError: 0 };
      });

      expect(result.match).toBe(true);
      expect(result.length).toBe(result.expectedLength);
    });
  });
});
