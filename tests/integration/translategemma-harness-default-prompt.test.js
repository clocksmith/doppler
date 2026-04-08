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
        modelId: 'translategemma-4b-it-q4k-ehf16-af32',
        modelType: 'transformer',
        architecture: {
          numLayers: 34,
          hiddenSize: 2560,
          intermediateSize: 10240,
          numAttentionHeads: 8,
          numKeyValueHeads: 4,
          headDim: 256,
          vocabSize: 262208,
          maxSeqLen: 131072,
        },
        inference: {
          session: {
            kvcache: {
              layout: 'contiguous',
            },
            decodeLoop: {
              batchSize: 1,
              stopCheckMode: 'batch',
              readbackInterval: 1,
              disableCommandBatching: false,
            },
          },
          execution: {
            steps: [
              {
                id: 'prefill_attention',
                phase: 'prefill',
                op: 'attention',
              },
              {
                id: 'decode_attention',
                phase: 'decode',
                op: 'attention',
              },
            ],
          },
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
    runtime: {
      runtimeConfig: {
        inference: {
          prompt: 'Hello from Doppler.',
        },
      },
    },
    suite: 'inference',
    command: 'verify',
    surface: 'node',
    harnessOverride,
  });

  assert.equal(prompts.length, 1);
  assert.deepEqual(prompts[0], EXPECTED_TRANSLATEGEMMA_PROMPT);
  assert.equal(result.metrics.prompt, 'en -> fr: Hello world.');
  assert.equal(result.metrics.schemaVersion, 1);
  assert.equal(result.metrics.source, 'doppler');
  assert.equal(result.metrics.suite, 'inference');
  assert.equal(result.metrics.executionContractArtifact?.ok, true);

  assert.equal(result.metrics.layerPatternContractArtifact?.ok, true);
  assert.equal(result.metrics.requiredInferenceFieldsArtifact, null);
  assert.equal(result.metrics.executionContractArtifact?.session?.layout, 'contiguous');
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
  assert.equal(result.metrics.schemaVersion, 1);
  assert.equal(result.metrics.source, 'doppler');
  assert.equal(result.metrics.suite, 'debug');
  assert.equal(result.metrics.executionContractArtifact?.ok, true);

  assert.equal(result.metrics.layerPatternContractArtifact?.ok, true);
  assert.equal(result.metrics.requiredInferenceFieldsArtifact, null);
}

await assert.rejects(
  () => runBrowserSuite({
    runtime: {
      runtimeConfig: {
        inference: {
          prompt: 'Translate from English to French: Hello world.',
        },
      },
    },
    suite: 'debug',
    command: 'debug',
    surface: 'node',
    harnessOverride: createHarnessOverride().harnessOverride,
  }),
  /TranslateGemma harness prompt contract violation/
);

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
  assert.equal(result.metrics.schemaVersion, 1);
  assert.equal(result.metrics.source, 'doppler');
  assert.equal(result.metrics.suite, 'bench');
  assert.equal(result.metrics.executionContractArtifact?.ok, true);

  assert.equal(result.metrics.layerPatternContractArtifact?.ok, true);
  assert.equal(result.metrics.requiredInferenceFieldsArtifact, null);
}

console.log('translategemma-harness-default-prompt.test: ok');
