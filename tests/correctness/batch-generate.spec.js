

import { test, expect } from '@playwright/test';

// Test configuration
const TEST_CONFIG = {
  model: process.env.TEST_MODEL || 'gemma3-1b-q4',
  prompt: 'the color of the sky is',
  maxTokens: 20,
  batchSize: 4,
  timeout: 240000, // 4 minutes for model loading
};

test.describe('Batch Generation Correctness', () => {
  test.setTimeout(TEST_CONFIG.timeout);

  test('batchSize=1 vs batchSize=N produces identical output at temperature=0', async ({ page }) => {
    const testUrl = `/doppler/tests/harness.html`;
    await page.goto(testUrl);
    await page.waitForLoadState('domcontentloaded');

    const result = await page.evaluate(async (config) => {
      const { initDevice, getDevice } = await import('/doppler/dist/gpu/device.js');
      await initDevice();
      const device = getDevice();

      const MODEL_URL = `http://localhost:8080/doppler/models/${config.model}`;
      const manifestResp = await fetch(`${MODEL_URL}/manifest.json`);
      const manifest = await manifestResp.json();

      const { createPipeline } = await import('/doppler/dist/inference/pipeline.js');

      const loadShard = async (idx) => {
        const shard = manifest.shards[idx];
        const resp = await fetch(`${MODEL_URL}/${shard.filename}`);
        return new Uint8Array(await resp.arrayBuffer());
      };

      async function generateWithBatchSize(batchSize) {
        const pipeline = await createPipeline(manifest, {
          storage: { loadShard },
          gpu: { device },
          baseUrl: MODEL_URL,
          runtimeConfig: {
            inference: {
              batching: { batchSize },
            },
          },
        });

        const tokens = [];
        for await (const text of pipeline.generate(config.prompt, {
          maxTokens: config.maxTokens,
          temperature: 0,
        })) {
          tokens.push(text);
        }

        return tokens.join('');
      }

      const outputBatch1 = await generateWithBatchSize(1);
      const outputBatchN = await generateWithBatchSize(config.batchSize);

      return {
        outputBatch1,
        outputBatchN,
        passed: outputBatch1 === outputBatchN,
      };
    }, TEST_CONFIG);

    console.log('Batch compare prompt:', TEST_CONFIG.prompt);
    console.log('Batch compare output:', result.outputBatch1);

    expect(result.passed).toBe(true);
  });

  test('onBatch callback fires correct number of times', async ({ page }) => {
    const testUrl = `/doppler/tests/harness.html`;
    await page.goto(testUrl);

    // Inject a custom test that tracks onBatch calls
    const result = await page.evaluate(async (config) => {
      const { initDevice, getDevice } = await import('/doppler/dist/gpu/device.js');
      await initDevice();
      const device = getDevice();

      const MODEL_URL = `http://localhost:8080/doppler/models/${config.model}`;

      // Load model
      const manifestResp = await fetch(`${MODEL_URL}/manifest.json`);
      const manifest = await manifestResp.json();

      const { createPipeline } = await import('/doppler/dist/inference/pipeline.js');

      const loadShard = async (idx) => {
        const shard = manifest.shards[idx];
        const resp = await fetch(`${MODEL_URL}/${shard.filename}`);
        return new Uint8Array(await resp.arrayBuffer());
      };

      const pipeline = await createPipeline(manifest, {
        storage: { loadShard },
        gpu: { device },
        baseUrl: MODEL_URL,
        runtimeConfig: {
          inference: {
            batching: { batchSize: config.batchSize },
          },
        },
      });

      // Generate with batchSize and track callbacks
      const batchCalls = [];
      const tokens = [];

      for await (const text of pipeline.generate(config.prompt, {
        maxTokens: config.maxTokens,
        temperature: 0,
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

    // onBatch only covers batched decode tokens (first token can be sampled outside batch path)
    expect(result.totalBatchTokens).toBeLessThanOrEqual(result.tokenCount);
    expect(result.tokenCount - result.totalBatchTokens).toBeLessThanOrEqual(1);

    // Verify batch sizes are correct (last batch may be smaller)
    for (let i = 0; i < result.batchCalls.length - 1; i++) {
      expect(result.batchCalls[i]).toBe(TEST_CONFIG.batchSize);
    }
    // Last batch should be <= batchSize
    expect(result.batchCalls[result.batchCalls.length - 1]).toBeLessThanOrEqual(TEST_CONFIG.batchSize);
  });
});
