/**
 * Batch Generation Correctness Test
 *
 * Verifies that generate() produces identical output regardless of batchSize.
 * This is a pipeline-level correctness test that requires a loaded model.
 *
 * Run with: npx playwright test tests/correctness/batch-generate.spec.js
 */

import { test, expect } from '@playwright/test';

// Test configuration
const TEST_CONFIG = {
  model: process.env.TEST_MODEL || 'gemma3-1b-q4',
  prompt: 'the color of the sky is',
  maxTokens: 20,
  batchSize: 4,
  timeout: 120000, // 2 minutes for model loading
};

test.describe('Batch Generation Correctness', () => {
  test.setTimeout(TEST_CONFIG.timeout);

  test('batchSize=1 vs batchSize=N produces identical output at temperature=0', async ({ page }) => {
    // Navigate to inference test page
    const testUrl = `/doppler/tests/test-inference.html?model=${TEST_CONFIG.model}`;
    await page.goto(testUrl);

    // Wait for page to be ready
    await page.waitForFunction(() => window.testState?.ready === true, { timeout: 10000 });

    // Run batch compare test via the button click
    await page.click('#batch-compare-btn');

    // Wait for test to complete (model load + 2 generations)
    await page.waitForFunction(
      () => window.testState?.done === true,
      { timeout: TEST_CONFIG.timeout }
    );

    // Check results
    const testState = await page.evaluate(() => window.testState);

    // Log details for debugging
    console.log('Test state:', JSON.stringify(testState, null, 2));

    // Verify no errors
    expect(testState.errors).toHaveLength(0);

    // Verify batch compare passed
    expect(testState.batchCompare).toBeDefined();
    expect(testState.batchCompare.passed).toBe(true);
  });

  test('onBatch callback fires correct number of times', async ({ page }) => {
    const testUrl = `/doppler/tests/test-inference.html?model=${TEST_CONFIG.model}`;
    await page.goto(testUrl);

    await page.waitForFunction(() => window.testState?.ready === true, { timeout: 10000 });

    // Inject a custom test that tracks onBatch calls
    const result = await page.evaluate(async (config) => {
      const { initDevice, getDevice } = await import('/doppler/dist/gpu/device.js');
      await initDevice();
      const device = getDevice();

      const MODEL_URL = `http://localhost:8080/doppler/models/${config.model}`;

      // Load model
      const manifestResp = await fetch(`${MODEL_URL}/manifest.json`);
      const manifest = await manifestResp.json();

      const { parseManifest } = await import('/doppler/dist/storage/rdrr-format.js');
      const modelInfo = parseManifest(JSON.stringify(manifest));

      const { createPipeline } = await import('/doppler/dist/inference/pipeline.js');

      const loadShard = async (idx) => {
        const shard = manifest.shards[idx];
        const resp = await fetch(`${MODEL_URL}/${shard.fileName}`);
        return new Uint8Array(await resp.arrayBuffer());
      };

      const pipeline = await createPipeline(modelInfo, {
        storage: { loadShard },
        gpu: { device },
        baseUrl: MODEL_URL,
      });

      // Generate with batchSize and track callbacks
      const batchCalls = [];
      const tokens = [];

      for await (const text of pipeline.generate(config.prompt, {
        maxTokens: config.maxTokens,
        temperature: 0,
        batchSize: config.batchSize,
        onBatch: (batch) => {
          batchCalls.push(batch.length);
        },
      })) {
        tokens.push(text);
      }

      return {
        tokenCount: tokens.length,
        batchCalls,
        totalBatchTokens: batchCalls.reduce((a, b) => a + b, 0),
      };
    }, TEST_CONFIG);

    console.log('onBatch result:', result);

    // Verify onBatch was called
    expect(result.batchCalls.length).toBeGreaterThan(0);

    // Verify total tokens from batches matches generated tokens
    expect(result.totalBatchTokens).toBe(result.tokenCount);

    // Verify batch sizes are correct (last batch may be smaller)
    for (let i = 0; i < result.batchCalls.length - 1; i++) {
      expect(result.batchCalls[i]).toBe(TEST_CONFIG.batchSize);
    }
    // Last batch should be <= batchSize
    expect(result.batchCalls[result.batchCalls.length - 1]).toBeLessThanOrEqual(TEST_CONFIG.batchSize);
  });
});
