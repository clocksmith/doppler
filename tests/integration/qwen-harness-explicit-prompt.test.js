import assert from 'node:assert/strict';

const { runBrowserSuite } = await import('../../src/inference/browser-harness.js');

const QWEN_PROMPT = {
  messages: [
    {
      role: 'user',
      content: 'Answer in one short sentence: What color is the sky on a clear day?',
    },
  ],
};

function explicitRuntimeConfig(prompt = QWEN_PROMPT) {
  return {
    inference: {
      prompt: structuredClone(prompt),
      generation: {
        maxTokens: 1,
      },
      sampling: {
        temperature: 0,
        topK: 1,
        topP: 1,
      },
    },
  };
}

function createHarnessOverride() {
  const prompts = [];
  const pipeline = {
    modelConfig: {
      chatTemplateType: 'qwen',
    },
    async *generate(promptInput, options = {}) {
      prompts.push(promptInput);
      options.onToken?.(42);
      yield 'The sky is blue.';
    },
    getStats() {
      return {
        prefillTimeMs: 1,
        ttftMs: 1,
        decodeTimeMs: 1,
        prefillTokens: 1,
        decodeTokens: 1,
        decodeProfileSteps: [],
      };
    },
    reset() {},
    async unload() {},
  };

  return {
    prompts,
    harnessOverride: {
      modelLoadMs: 1,
      manifest: {
        modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
        modelType: 'transformer',
        architecture: {
          numLayers: 24,
          hiddenSize: 1024,
          intermediateSize: 3584,
          numAttentionHeads: 8,
          numKeyValueHeads: 2,
          headDim: 256,
          vocabSize: 248320,
          maxSeqLen: 262144,
        },
        inference: {
          attention: {
            queryPreAttnScalar: 256,
          },
          chatTemplate: {
            type: 'qwen',
            enabled: true,
          },
        },
      },
      pipeline,
    },
  };
}

{
  const { prompts, harnessOverride } = createHarnessOverride();
  const result = await runBrowserSuite({
    suite: 'debug',
    command: 'debug',
    surface: 'node',
    runtime: {
      runtimeConfig: explicitRuntimeConfig(),
    },
    harnessOverride,
  });

  assert.equal(prompts.length, 1);
  assert.deepEqual(prompts[0], QWEN_PROMPT);
  assert.equal(
    result.metrics.prompt,
    'user: Answer in one short sentence: What color is the sky on a clear day?'
  );
}

{
  const { prompts, harnessOverride } = createHarnessOverride();
  await runBrowserSuite({
    runtime: {
      runtimeConfig: explicitRuntimeConfig(),
    },
    suite: 'inference',
    command: 'verify',
    surface: 'node',
    harnessOverride,
  });

  assert.equal(prompts.length, 1);
  assert.deepEqual(prompts[0], QWEN_PROMPT);
}

console.log('qwen-harness-explicit-prompt.test: ok');
