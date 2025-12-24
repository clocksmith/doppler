/**
 * MoE Gather Kernel Correctness Tests
 */

import { test, expect } from './setup.js';

interface MoEGatherResult {
  countsMatch?: boolean;
  totalAssignments?: number;
  expectedTotal?: number;
  refCounts?: number[];
  gpuCounts?: number[];
  counts?: number[];
  expectedPerExpert?: number;
}

interface MoEGatherConfig {
  tokens: number;
  hidden: number;
  experts: number;
  topK: number;
}

test.describe('MoE Gather Kernel', () => {
  test.describe('Basic functionality', () => {
    test('should gather tokens by expert assignment', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<MoEGatherResult> => {
        const { moeGatherRef, moeComputeAssignmentsRef } = window.testHarness.references;

        const numTokens = 8;
        const hiddenSize = 32;
        const numExperts = 4;
        const topK = 2;

        // Create tokens
        const tokens = new Float32Array(numTokens * hiddenSize);
        for (let i = 0; i < tokens.length; i++) {
          tokens[i] = Math.random() * 2 - 1;
        }

        // Create expert indices (each token routed to topK experts)
        const expertIndices = new Uint32Array(numTokens * topK);
        for (let t = 0; t < numTokens; t++) {
          // Pick unique experts for this token
          const used = new Set<number>();
          for (let k = 0; k < topK; k++) {
            let e: number;
            do {
              e = Math.floor(Math.random() * numExperts);
            } while (used.has(e));
            used.add(e);
            expertIndices[t * topK + k] = e;
          }
        }

        // Reference implementation
        const refResult = moeGatherRef(tokens, expertIndices, numTokens, hiddenSize, numExperts, topK);
        const assignments = moeComputeAssignmentsRef(expertIndices, numTokens, numExperts, topK);

        // GPU implementation
        const gpu = await window.testHarness.getGPU();
        const gpuResult = await window.testHarness.runMoEGather(
          gpu.device, tokens, expertIndices, numTokens, hiddenSize, numExperts, topK
        );

        // Compare token counts
        let countsMatch = true;
        for (let e = 0; e < numExperts; e++) {
          if (gpuResult.tokenCounts[e] !== refResult.tokenCounts[e]) {
            countsMatch = false;
            break;
          }
        }

        return {
          countsMatch,
          totalAssignments: assignments.totalAssignments,
          expectedTotal: numTokens * topK,
          refCounts: Array.from(refResult.tokenCounts),
          gpuCounts: Array.from(gpuResult.tokenCounts),
        };
      });

      expect(result.countsMatch).toBe(true);
      expect(result.totalAssignments).toBe(result.expectedTotal);
    });

    test('should compute correct token counts', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<MoEGatherResult> => {
        const { moeComputeAssignmentsRef } = window.testHarness.references;

        const numTokens = 16;
        const numExperts = 4;
        const topK = 2;

        // Create deterministic routing: token t -> experts [t % 4, (t+1) % 4]
        const expertIndices = new Uint32Array(numTokens * topK);
        for (let t = 0; t < numTokens; t++) {
          expertIndices[t * topK + 0] = t % numExperts;
          expertIndices[t * topK + 1] = (t + 1) % numExperts;
        }

        const { tokenCounts, totalAssignments } = moeComputeAssignmentsRef(
          expertIndices, numTokens, numExperts, topK
        );

        // Each expert should get numTokens * topK / numExperts assignments on average
        // With our pattern: each expert gets exactly 8 assignments (16 * 2 / 4)
        const expectedPerExpert = (numTokens * topK) / numExperts;

        return {
          counts: Array.from(tokenCounts),
          totalAssignments,
          expectedTotal: numTokens * topK,
          expectedPerExpert,
        };
      });

      expect(result.totalAssignments).toBe(result.expectedTotal);
      // Each expert should get equal assignments with our deterministic pattern
      for (const count of result.counts!) {
        expect(count).toBe(result.expectedPerExpert);
      }
    });
  });

  test.describe('Edge cases', () => {
    test('should handle all tokens to same expert', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<MoEGatherResult> => {
        const { moeComputeAssignmentsRef } = window.testHarness.references;

        const numTokens = 8;
        const numExperts = 4;
        const topK = 1;

        // All tokens go to expert 0
        const expertIndices = new Uint32Array(numTokens * topK).fill(0);

        const { tokenCounts, totalAssignments } = moeComputeAssignmentsRef(
          expertIndices, numTokens, numExperts, topK
        );

        return {
          counts: Array.from(tokenCounts),
          totalAssignments,
        };
      });

      expect(result.counts![0]).toBe(8);
      expect(result.counts![1]).toBe(0);
      expect(result.counts![2]).toBe(0);
      expect(result.counts![3]).toBe(0);
      expect(result.totalAssignments).toBe(8);
    });

    test('should handle topK=1', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async (): Promise<MoEGatherResult> => {
        const { moeGatherRef } = window.testHarness.references;

        const numTokens = 8;
        const hiddenSize = 16;
        const numExperts = 4;
        const topK = 1;

        const tokens = new Float32Array(numTokens * hiddenSize);
        for (let i = 0; i < tokens.length; i++) {
          tokens[i] = Math.random();
        }

        const expertIndices = new Uint32Array(numTokens);
        for (let t = 0; t < numTokens; t++) {
          expertIndices[t] = t % numExperts;
        }

        const refResult = moeGatherRef(tokens, expertIndices, numTokens, hiddenSize, numExperts, topK);

        // Each expert should have 2 tokens (8 / 4)
        const expectedPerExpert = numTokens / numExperts;

        return {
          counts: Array.from(refResult.tokenCounts),
          expectedPerExpert,
        };
      });

      for (const count of result.counts!) {
        expect(count).toBe(result.expectedPerExpert);
      }
    });
  });

  test.describe('Size variations', () => {
    const configs = [
      { tokens: 16, hidden: 64, experts: 4, topK: 2 },
      { tokens: 64, hidden: 256, experts: 8, topK: 2 },
      { tokens: 128, hidden: 512, experts: 8, topK: 2 },
      { tokens: 256, hidden: 256, experts: 16, topK: 4 },
    ];

    for (const cfg of configs) {
      test(`should handle ${cfg.tokens}t x ${cfg.hidden}h x ${cfg.experts}e x top${cfg.topK}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (c: MoEGatherConfig): Promise<MoEGatherResult> => {
          const { moeComputeAssignmentsRef } = window.testHarness.references;

          const { tokens, hidden, experts, topK } = c;

          const expertIndices = new Uint32Array(tokens * topK);
          for (let t = 0; t < tokens; t++) {
            const used = new Set<number>();
            for (let k = 0; k < topK; k++) {
              let e: number;
              do {
                e = Math.floor(Math.random() * experts);
              } while (used.has(e));
              used.add(e);
              expertIndices[t * topK + k] = e;
            }
          }

          const { tokenCounts, totalAssignments } = moeComputeAssignmentsRef(
            expertIndices, tokens, experts, topK
          );

          // Verify total
          let sumCounts = 0;
          for (const c of tokenCounts) {
            sumCounts += c;
          }

          return {
            totalAssignments,
            counts: [sumCounts],
            expectedTotal: tokens * topK,
          };
        }, cfg);

        expect(result.totalAssignments).toBe(result.expectedTotal);
        expect(result.counts![0]).toBe(result.expectedTotal);
      });
    }
  });
});
