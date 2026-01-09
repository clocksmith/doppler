/**
 * Check-Stop Kernel Correctness Tests
 *
 * Validates the check-stop GPU kernel for early stopping detection.
 * Tests EOS token detection and max tokens limit.
 */

import { test, expect } from './setup.js';

test.describe('Check-Stop Kernel', () => {
  test.describe('EOS token detection', () => {
    test('should stop when EOS token is sampled', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        const eosTokenId = 2;
        const sampledToken = 2; // EOS
        const maxTokens = 100;
        const currentPos = 10;

        const actual = await window.testHarness.runCheckStop(
          gpu.device,
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        const expected = window.testHarness.checkStopRef(
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        return { actual, expected };
      });

      expect(result.actual).toBe(result.expected);
      expect(result.actual).toBe(true);
    });

    test('should not stop when non-EOS token is sampled', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        const eosTokenId = 2;
        const sampledToken = 100; // Not EOS
        const maxTokens = 1000;
        const currentPos = 10;

        const actual = await window.testHarness.runCheckStop(
          gpu.device,
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        const expected = window.testHarness.checkStopRef(
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        return { actual, expected };
      });

      expect(result.actual).toBe(result.expected);
      expect(result.actual).toBe(false);
    });

    test('should handle various EOS token IDs', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();
        const results = [];

        // Common EOS token IDs
        const eosTokenIds = [0, 1, 2, 128000, 128001, 50256];

        for (const eosTokenId of eosTokenIds) {
          // Test when token matches EOS
          const actualStop = await window.testHarness.runCheckStop(
            gpu.device,
            eosTokenId, // sampled = EOS
            eosTokenId,
            1000,
            10
          );

          // Test when token doesn't match EOS
          const actualContinue = await window.testHarness.runCheckStop(
            gpu.device,
            eosTokenId + 1, // sampled != EOS
            eosTokenId,
            1000,
            10
          );

          results.push({
            eosTokenId,
            shouldStop: actualStop,
            shouldContinue: !actualContinue,
          });
        }

        return { results };
      });

      for (const r of result.results) {
        expect(r.shouldStop).toBe(true);
        expect(r.shouldContinue).toBe(true);
      }
    });
  });

  test.describe('Max tokens detection', () => {
    test('should stop when max tokens reached (equal)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        const eosTokenId = 2;
        const sampledToken = 100; // Not EOS
        const maxTokens = 50;
        const currentPos = 50; // Equal to max

        const actual = await window.testHarness.runCheckStop(
          gpu.device,
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        const expected = window.testHarness.checkStopRef(
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        return { actual, expected };
      });

      expect(result.actual).toBe(result.expected);
      expect(result.actual).toBe(true);
    });

    test('should stop when max tokens exceeded', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        const eosTokenId = 2;
        const sampledToken = 100; // Not EOS
        const maxTokens = 50;
        const currentPos = 51; // Exceeds max

        const actual = await window.testHarness.runCheckStop(
          gpu.device,
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        const expected = window.testHarness.checkStopRef(
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        return { actual, expected };
      });

      expect(result.actual).toBe(result.expected);
      expect(result.actual).toBe(true);
    });

    test('should not stop when below max tokens', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        const eosTokenId = 2;
        const sampledToken = 100; // Not EOS
        const maxTokens = 100;
        const currentPos = 50; // Below max

        const actual = await window.testHarness.runCheckStop(
          gpu.device,
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        const expected = window.testHarness.checkStopRef(
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        return { actual, expected };
      });

      expect(result.actual).toBe(result.expected);
      expect(result.actual).toBe(false);
    });

    test('should handle boundary condition (currentPos = maxTokens - 1)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        const eosTokenId = 2;
        const sampledToken = 100; // Not EOS
        const maxTokens = 100;
        const currentPos = 99; // One before max

        const actual = await window.testHarness.runCheckStop(
          gpu.device,
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        const expected = window.testHarness.checkStopRef(
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        return { actual, expected };
      });

      expect(result.actual).toBe(result.expected);
      expect(result.actual).toBe(false);
    });
  });

  test.describe('Combined conditions', () => {
    test('should stop when both EOS and max reached', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        const eosTokenId = 2;
        const sampledToken = 2; // EOS
        const maxTokens = 50;
        const currentPos = 50; // At max

        const actual = await window.testHarness.runCheckStop(
          gpu.device,
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        return { actual };
      });

      expect(result.actual).toBe(true);
    });

    test('should not stop when neither condition met', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        const eosTokenId = 2;
        const sampledToken = 1000; // Not EOS
        const maxTokens = 1000;
        const currentPos = 500; // Below max

        const actual = await window.testHarness.runCheckStop(
          gpu.device,
          sampledToken,
          eosTokenId,
          maxTokens,
          currentPos
        );

        return { actual };
      });

      expect(result.actual).toBe(false);
    });
  });

  test.describe('Position variations', () => {
    const positions = [0, 1, 10, 100, 500, 999, 1000, 2000, 4096, 8192];

    for (const pos of positions) {
      test(`should handle currentPos=${pos}`, async ({ gpuPage }) => {
        const result = await gpuPage.evaluate(async (currentPos) => {
          const gpu = await window.testHarness.getGPU();

          const eosTokenId = 2;
          const sampledToken = 100; // Not EOS
          const maxTokens = 2048;

          const actual = await window.testHarness.runCheckStop(
            gpu.device,
            sampledToken,
            eosTokenId,
            maxTokens,
            currentPos
          );

          const expected = window.testHarness.checkStopRef(
            sampledToken,
            eosTokenId,
            maxTokens,
            currentPos
          );

          return { actual, expected };
        }, pos);

        expect(result.actual).toBe(result.expected);
      });
    }
  });

  test.describe('Typical generation scenarios', () => {
    test('should handle Gemma-style generation (EOS=1)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        // Gemma typically uses EOS token ID 1
        const eosTokenId = 1;
        const maxTokens = 256;

        // Simulate generation sequence
        const tokensToGenerate = [100, 200, 300, 400, 1]; // Ends with EOS
        const results = [];

        for (let i = 0; i < tokensToGenerate.length; i++) {
          const shouldStop = await window.testHarness.runCheckStop(
            gpu.device,
            tokensToGenerate[i],
            eosTokenId,
            maxTokens,
            i
          );
          results.push({ pos: i, token: tokensToGenerate[i], shouldStop });

          if (shouldStop) break;
        }

        return { results };
      });

      // Should continue for first 4 tokens, stop at 5th (EOS)
      expect(result.results.length).toBe(5);
      expect(result.results[0].shouldStop).toBe(false);
      expect(result.results[1].shouldStop).toBe(false);
      expect(result.results[2].shouldStop).toBe(false);
      expect(result.results[3].shouldStop).toBe(false);
      expect(result.results[4].shouldStop).toBe(true);
    });

    test('should handle LLaMA-style generation (EOS=2)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        // LLaMA typically uses EOS token ID 2
        const eosTokenId = 2;
        const maxTokens = 10; // Short for test

        // Simulate generation that hits max tokens
        const results = [];

        for (let i = 0; i < 15; i++) {
          const shouldStop = await window.testHarness.runCheckStop(
            gpu.device,
            1000, // Regular token, not EOS
            eosTokenId,
            maxTokens,
            i
          );
          results.push({ pos: i, shouldStop });

          if (shouldStop) break;
        }

        return { results };
      });

      // Should stop at position 10 (max tokens)
      expect(result.results.length).toBe(11); // 0-10 inclusive
      expect(result.results[9].shouldStop).toBe(false);
      expect(result.results[10].shouldStop).toBe(true);
    });

    test('should handle GPT-style generation (EOS=50256)', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        // GPT-2 uses EOS token ID 50256
        const eosTokenId = 50256;
        const maxTokens = 100;

        // Test that high EOS token ID works
        const shouldStopOnEOS = await window.testHarness.runCheckStop(
          gpu.device,
          50256, // EOS
          eosTokenId,
          maxTokens,
          5
        );

        const shouldContinue = await window.testHarness.runCheckStop(
          gpu.device,
          50255, // Not EOS
          eosTokenId,
          maxTokens,
          5
        );

        return { shouldStopOnEOS, shouldContinue };
      });

      expect(result.shouldStopOnEOS).toBe(true);
      expect(result.shouldContinue).toBe(false);
    });
  });

  test.describe('Edge cases', () => {
    test('should handle position 0', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        // First token generation
        const shouldStop = await window.testHarness.runCheckStop(
          gpu.device,
          100, // Not EOS
          2,   // EOS
          100, // Max
          0    // First position
        );

        return { shouldStop };
      });

      expect(result.shouldStop).toBe(false);
    });

    test('should handle maxTokens = 0', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        // Edge case: max tokens is 0
        const shouldStop = await window.testHarness.runCheckStop(
          gpu.device,
          100, // Not EOS
          2,   // EOS
          0,   // Max tokens = 0
          0    // Position 0
        );

        const expected = window.testHarness.checkStopRef(100, 2, 0, 0);

        return { shouldStop, expected };
      });

      expect(result.shouldStop).toBe(result.expected);
      // Position 0 >= maxTokens 0, so should stop
      expect(result.shouldStop).toBe(true);
    });

    test('should handle maxTokens = 1', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        // First position should not stop
        const pos0 = await window.testHarness.runCheckStop(
          gpu.device,
          100, // Not EOS
          2,   // EOS
          1,   // Max tokens = 1
          0    // Position 0
        );

        // Second position should stop
        const pos1 = await window.testHarness.runCheckStop(
          gpu.device,
          100, // Not EOS
          2,   // EOS
          1,   // Max tokens = 1
          1    // Position 1
        );

        return { pos0, pos1 };
      });

      expect(result.pos0).toBe(false);
      expect(result.pos1).toBe(true);
    });

    test('should handle token ID 0', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        // Token 0 as EOS
        const shouldStopToken0 = await window.testHarness.runCheckStop(
          gpu.device,
          0,   // Sampled token 0
          0,   // EOS = 0
          100, // Max
          10   // Position
        );

        // Token 0 but EOS is different
        const shouldContinue = await window.testHarness.runCheckStop(
          gpu.device,
          0,   // Sampled token 0
          1,   // EOS = 1
          100, // Max
          10   // Position
        );

        return { shouldStopToken0, shouldContinue };
      });

      expect(result.shouldStopToken0).toBe(true);
      expect(result.shouldContinue).toBe(false);
    });

    test('should handle large token IDs', async ({ gpuPage }) => {
      const result = await gpuPage.evaluate(async () => {
        const gpu = await window.testHarness.getGPU();

        // Large token ID (typical vocab size)
        const largeTokenId = 128255;

        const shouldStop = await window.testHarness.runCheckStop(
          gpu.device,
          largeTokenId, // Sampled
          largeTokenId, // EOS
          100,          // Max
          10            // Position
        );

        const shouldContinue = await window.testHarness.runCheckStop(
          gpu.device,
          largeTokenId - 1, // Sampled
          largeTokenId,     // EOS
          100,              // Max
          10                // Position
        );

        return { shouldStop, shouldContinue };
      });

      expect(result.shouldStop).toBe(true);
      expect(result.shouldContinue).toBe(false);
    });
  });
});
