import assert from 'node:assert/strict';

const { runBrowserSuite } = await import('../../src/inference/browser-harness.js');

const EXPECTED_TRANSLATEGEMMA_PROMPT = {
  messages: [
    {
      role: 'user',
      content: [
        {
          type: 'text',
          source_lang_code: 'en',
          target_lang_code: 'fr',
          text: 'Hello world.',
        },
      ],
    },
  ],
};

function createHarnessOverride() {
  const prompts = [];
  const pipeline = {
    modelConfig: {
      chatTemplateType: 'translategemma',
    },
    async *generate(promptInput, options = {}) {
      prompts.push(promptInput);
      options.onToken?.(42);
      yield 'Bonjour le monde.';
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
        modelId: 'translategemma-4b-it-wq4k-ef16-hf16',
        modelType: 'transformer',
        inference: {
          chatTemplate: {
            type: 'translategemma',
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
    harnessOverride,
  });

  assert.equal(prompts.length, 1);
  assert.deepEqual(prompts[0], EXPECTED_TRANSLATEGEMMA_PROMPT);
  assert.equal(result.metrics.prompt, 'en -> fr: Hello world.');
}

{
  const { prompts, harnessOverride } = createHarnessOverride();
  const result = await runBrowserSuite({
    suite: 'bench',
    command: 'bench',
    surface: 'node',
    harnessOverride,
  });

  assert.ok(prompts.length > 0);
  for (const promptInput of prompts) {
    assert.deepEqual(promptInput, EXPECTED_TRANSLATEGEMMA_PROMPT);
  }
  assert.equal(result.metrics.prompt, 'en -> fr: Hello world.');
}

console.log('translategemma-harness-default-prompt.test: ok');
