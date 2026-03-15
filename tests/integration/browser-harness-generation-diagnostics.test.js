import assert from 'node:assert/strict';

const { runBrowserSuite } = await import('../../src/inference/browser-harness.js');

function createHarnessOverride(records) {
  let recordIndex = 0;
  const pipeline = {
    tokenizer: {
      decode(ids, skipSpecialTokens = true) {
        const id = Array.isArray(ids) ? ids[0] : null;
        if (skipSpecialTokens) {
          return '';
        }
        return `<unused${String(id ?? 0)}>`;
      },
    },
    async *generate(_promptInput, options = {}) {
      for (const record of records) {
        options.onToken?.(record.id, record.text);
        yield record.text;
        recordIndex += 1;
      }
    },
    getStats() {
      return {
        prefillTimeMs: 1,
        ttftMs: 1,
        decodeTimeMs: 1,
        prefillTokens: 1,
        decodeTokens: Math.max(0, recordIndex),
        decodeProfileSteps: [],
      };
    },
    reset() {},
    async unload() {},
  };

  return {
    modelLoadMs: 1,
    manifest: {
      modelId: 'gemma-3-1b-it-f16-af32',
      modelType: 'transformer',
      architecture: {
        numLayers: 26,
        hiddenSize: 1152,
        intermediateSize: 6912,
        numAttentionHeads: 4,
        numKeyValueHeads: 1,
        headDim: 256,
        vocabSize: 262144,
        maxSeqLen: 32768,
      },
      inference: {
        attention: {
          queryPreAttnScalar: 256,
        },
        chatTemplate: {
          type: 'gemma',
          enabled: true,
        },
      },
    },
    pipeline,
  };
}

{
  const result = await runBrowserSuite({
    suite: 'debug',
    command: 'debug',
    surface: 'node',
    harnessOverride: createHarnessOverride([
      { id: 262140, text: '' },
      { id: 262141, text: '' },
      { id: 262142, text: '' },
    ]),
  });

  assert.equal(result.results[0]?.passed, false);
  assert.equal(result.results[0]?.error, 'Output dominated by padding or special tokens');
  assert.equal(result.metrics.generationDiagnostics.total, 3);
  assert.equal(result.metrics.generationDiagnostics.emptyTextCount, 3);
  assert.equal(result.metrics.generationDiagnostics.specialLikeFallbackCount, 3);
  assert.deepEqual(
    result.metrics.generationDiagnostics.preview.map((entry) => entry.id),
    [262140, 262141, 262142]
  );
  assert.deepEqual(
    result.metrics.generationDiagnostics.preview.map((entry) => entry.fallbackText),
    ['<unused262140>', '<unused262141>', '<unused262142>']
  );
}

{
  const result = await runBrowserSuite({
    suite: 'debug',
    command: 'debug',
    surface: 'node',
    harnessOverride: createHarnessOverride([
      { id: 1, text: 'Web' },
      { id: 2, text: 'GPU' },
    ]),
  });

  assert.equal(result.results[0]?.passed, true);
  assert.equal(result.output, 'WebGPU');
  assert.equal(result.metrics.generationDiagnostics.total, 2);
  assert.equal(result.metrics.generationDiagnostics.emptyTextCount, 0);
}

{
  const observed = { useChatTemplate: null };
  const pipeline = {
    async *generate(_promptInput, options = {}) {
      observed.useChatTemplate = options.useChatTemplate;
      options.onToken?.(1, 'Blue');
      yield 'Blue';
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

  await runBrowserSuite({
    suite: 'debug',
    command: 'debug',
    surface: 'node',
    harnessOverride: {
      modelLoadMs: 1,
      manifest: {
        modelId: 'qwen-3-5-0-8b-wf16-ef16-hf16-f16',
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
  });

  assert.equal(observed.useChatTemplate, undefined);
}

console.log('browser-harness-generation-diagnostics.test: ok');
