/**
 * Logit Reference Comparison Test
 *
 * Validates that specific logit values match HuggingFace reference values.
 * This catches bugs where model produces output but with wrong logits.
 *
 * See: docs/postmortems/2026-01-19-gemma3-postfeedforwardnorm.md
 */

import { test, expect } from '@playwright/test';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join } from 'path';

const FIXTURES_PATH = join(process.cwd(), 'tests', 'fixtures', 'logit-references.json');

// Default references if fixture doesn't exist - these are from HuggingFace transformers
const DEFAULT_REFERENCES = {
  'gemma-3-1b-it': {
    prompt: 'The color of the sky is',
    // Reference logits from HuggingFace for last token position
    // Format: { tokenId: expectedLogit }
    referenceLogits: {
      // Token 138 (double-space) should have NEGATIVE logit, not positive
      138: { expected: -5.5, tolerance: 0.5 },
      // Common next tokens should have reasonable logits
      235: { expected: 10.5, tolerance: 1.0 }, // "blue" related
    },
    // Expected first generated token (greedy with temp=0)
    expectedFirstToken: { notEquals: 138 }, // Should NOT be double-space
  },
};

function loadReferences() {
  if (existsSync(FIXTURES_PATH)) {
    return JSON.parse(readFileSync(FIXTURES_PATH, 'utf-8'));
  }
  return DEFAULT_REFERENCES;
}

test.describe('logit reference comparison', () => {
  const REFERENCES = loadReferences();

  for (const [modelId, entry] of Object.entries(REFERENCES)) {
    test(`${modelId} logits match reference`, async ({ page, baseURL }) => {
      const origin = baseURL || 'http://localhost:8080';

      // Build config with logit probes
      const runtimeConfig = {
        shared: {
          debug: {
            perfGuards: { allowGPUReadback: true },
            probes: [
              {
                stage: 'logits',
                tokens: [-1], // Last token only
                dims: Object.keys(entry.referenceLogits).map(Number),
              },
            ],
          },
        },
        inference: {
          sampling: { temperature: 0, topK: 1 },
          batching: { maxTokens: 1 },
        },
        harness: {
          mode: 'inference',
          autorun: true,
          modelId,
        },
      };

      const configParam = encodeURIComponent(JSON.stringify(runtimeConfig));
      await page.goto(`${origin}/doppler/tests/harness.html?runtimeConfig=${configParam}`);
      await page.waitForLoadState('domcontentloaded');

      // Wait for inference to complete and capture probe results
      const result = await page.evaluate(async ({ modelId, prompt, referenceLogits }) => {
        // Wait for DOPPLER to be ready
        const maxWait = 120000;
        const startTime = Date.now();
        while (!window.DOPPLER?.probeResults && Date.now() - startTime < maxWait) {
          await new Promise(r => setTimeout(r, 500));
        }

        if (!window.DOPPLER?.probeResults) {
          return { error: 'Probe results not available - inference may have failed' };
        }

        const probes = window.DOPPLER.probeResults;
        const logitsProbe = probes.find(p => p.stage === 'logits');
        if (!logitsProbe) {
          return { error: 'Logits probe not found in results' };
        }

        // Extract logit values for comparison
        const actualLogits = {};
        for (const tokenIdStr of Object.keys(referenceLogits)) {
          const tokenId = Number(tokenIdStr);
          const idx = logitsProbe.dims.indexOf(tokenId);
          if (idx !== -1 && logitsProbe.values[idx] !== undefined) {
            actualLogits[tokenId] = logitsProbe.values[idx];
          }
        }

        // Get first generated token
        const firstToken = window.DOPPLER?.lastGeneratedTokens?.[0] ?? null;

        return { actualLogits, firstToken };
      }, { modelId, prompt: entry.prompt, referenceLogits: entry.referenceLogits });

      // Check for errors
      if (result.error) {
        // Skip if model not available (common in CI without models)
        if (result.error.includes('not available')) {
          test.skip();
          return;
        }
        throw new Error(result.error);
      }

      // Validate logits
      for (const [tokenIdStr, ref] of Object.entries(entry.referenceLogits)) {
        const tokenId = Number(tokenIdStr);
        const actual = result.actualLogits[tokenId];

        if (actual === undefined) {
          console.warn(`Logit for token ${tokenId} not captured`);
          continue;
        }

        const diff = Math.abs(actual - ref.expected);
        expect(
          diff,
          `Token ${tokenId} logit: expected ${ref.expected} ± ${ref.tolerance}, got ${actual}`
        ).toBeLessThanOrEqual(ref.tolerance);
      }

      // Validate first token constraint
      if (entry.expectedFirstToken?.notEquals !== undefined) {
        expect(
          result.firstToken,
          `First token should not be ${entry.expectedFirstToken.notEquals}`
        ).not.toBe(entry.expectedFirstToken.notEquals);
      }
    });
  }
});

test.describe('logit sanity checks', () => {
  test('token 138 (double-space) is not selected for standard prompts', async ({ page, baseURL }) => {
    const origin = baseURL || 'http://localhost:8080';

    // This test catches the postFeedforwardNorm bug pattern
    const result = await page.evaluate(async (origin) => {
      try {
        const { initDevice, getDevice } = await import('/doppler/dist/gpu/device.js');
        await initDevice();
        const device = getDevice();

        // Try to load any available Gemma 3 model
        const modelIds = ['gemma-3-1b-it-wf16', 'gemma-3-1b-it-q4'];
        let manifest = null;
        let modelUrl = null;

        for (const modelId of modelIds) {
          try {
            const url = `${origin}/doppler/models/${modelId}`;
            const resp = await fetch(`${url}/manifest.json`);
            if (resp.ok) {
              manifest = await resp.json();
              modelUrl = url;
              break;
            }
          } catch {
            continue;
          }
        }

        if (!manifest) {
          return { skipped: true, reason: 'No Gemma 3 model available' };
        }

        const { createPipeline } = await import('/doppler/dist/inference/pipeline.js');
        const loadShard = async (idx) => {
          const shard = manifest.shards[idx];
          const resp = await fetch(`${modelUrl}/${shard.filename}`);
          return new Uint8Array(await resp.arrayBuffer());
        };

        const pipeline = await createPipeline(manifest, {
          storage: { loadShard },
          gpu: { device },
          baseUrl: modelUrl,
        });

        // Generate with greedy decoding
        const tokens = [];
        for await (const token of pipeline.generateTokens('The color of the sky is', {
          maxTokens: 5,
          temperature: 0,
        })) {
          tokens.push(token);
        }

        // Token 138 is double-space - should not appear in first position for coherent output
        const hasDoubleSpaceFirst = tokens[0] === 138;
        const allDoubleSpace = tokens.every(t => t === 138);

        return {
          tokens,
          hasDoubleSpaceFirst,
          allDoubleSpace,
          manifestHasPostFFNNorm: manifest.inference?.normalization?.postFeedforwardNorm,
        };
      } catch (err) {
        return { error: err.message };
      }
    }, origin);

    if (result.skipped) {
      test.skip();
      return;
    }

    if (result.error) {
      throw new Error(result.error);
    }

    // The key assertion: token 138 should NOT dominate output
    expect(
      result.allDoubleSpace,
      `Model produced all token-138 (double-space) - likely missing postFeedforwardNorm. ` +
      `Manifest postFeedforwardNorm=${result.manifestHasPostFFNNorm}`
    ).toBe(false);

    // First token should not be double-space for this prompt
    expect(
      result.hasDoubleSpaceFirst,
      `First token was 138 (double-space) - model may have configuration issue`
    ).toBe(false);
  });
});
