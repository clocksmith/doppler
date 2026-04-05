import assert from 'node:assert/strict';
import { runBrowserSuite } from '../../src/inference/browser-harness.js';

function createHarnessOverride(calls) {
  return {
    pipeline: {
      async *generate() {
        throw new Error('generate() should not be used for image transcription.');
      },
      async transcribeImage(args) {
        calls.push(args);
        return {
          text: 'A white square.',
          tokens: [101, 102, 103],
        };
      },
      getStats() {
        return {
          totalTimeMs: 25,
          ttftMs: 10,
          prefillTimeMs: 10,
          decodeTimeMs: 15,
          prefillTokens: 6,
          decodeTokens: 3,
          decodeMode: 'replay_prefill',
          kernelPathId: 'vision-smoke',
          kernelPathSource: 'model',
        };
      },
      tokenizer: {
        decode([tokenId]) {
          if (tokenId === 101) return 'A';
          if (tokenId === 102) return ' white';
          if (tokenId === 103) return ' square.';
          return '';
        },
      },
      async unload() {
        calls.push('unload');
      },
    },
    manifest: {
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      modelType: 'transformer',
    },
    modelLoadMs: 1,
  };
}

{
  const calls = [];
  const result = await runBrowserSuite({
    command: 'verify',
    workload: 'inference',
    surface: 'browser',
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    harnessOverride: createHarnessOverride(calls),
    inferenceInput: {
      prompt: 'Describe the image precisely.',
      maxTokens: 8,
      softTokenBudget: 70,
      image: {
        width: 1,
        height: 1,
        pixels: [255, 255, 255, 255],
      },
    },
  });

  assert.equal(result.output, 'A white square.');
  assert.equal(result.results[0].passed, true);
  assert.equal(result.metrics.prompt, 'image 1x1: Describe the image precisely.');
  assert.equal(result.metrics.maxTokens, 8);
  assert.equal(result.metrics.tokensGenerated, 3);
  assert.equal(calls.length, 2);
  assert.ok(calls[0].imageBytes instanceof Uint8Array);
  assert.equal(calls[0].imageBytes.length, 4);
  assert.equal(calls[0].width, 1);
  assert.equal(calls[0].height, 1);
  assert.equal(calls[0].prompt, 'Describe the image precisely.');
  assert.equal(calls[0].maxTokens, 8);
  assert.equal(calls[0].softTokenBudget, 70);
  assert.equal(calls[1], 'unload');
}

{
  const calls = [];
  await runBrowserSuite({
    command: 'verify',
    workload: 'inference',
    surface: 'browser',
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    harnessOverride: createHarnessOverride(calls),
    inferenceInput: {
      prompt: 'Describe the image precisely.',
      maxTokens: 8,
      image: {
        width: 1,
        height: 1,
        pixels: [255, 255, 255, 255],
      },
    },
  });

  assert.equal(calls[0].softTokenBudget, 70);
}

console.log('browser-harness-image-transcription.test: ok');
