import assert from 'node:assert/strict';

const { runBrowserSuite } = await import('../../src/inference/browser-harness.js');
const { runGeneration } = await import('../../src/inference/browser-harness-text-helpers.js');

{
  const pipeline = {
    async *generate(_promptInput, options = {}) {
      options.onToken?.(1, 'Blue');
      yield 'Blue';
    },
    getStats() {
      return {
        prefillTimeMs: 2,
        ttftMs: 3,
        decodeTimeMs: 4,
        prefillTokens: 5,
        decodeTokens: 6,
        decodeMode: 'batched_gpu_stepwise_ple',
        batchGuardReason: null,
        singleTokenReadbackWaitMs: 7,
        singleTokenOrchestrationMs: 8,
        decodeRecordMs: 1,
        decodeSubmitWaitMs: 1,
        decodeReadbackWaitMs: 1,
        batching: {
          batchedForwardCalls: 9,
          unbatchedForwardCalls: 10,
          totalBatchedTimeMs: 11,
          totalUnbatchedTimeMs: 12,
          gpuSubmissions: 13,
        },
        decodeProfileSteps: [],
      };
    },
  };

  const result = await runGeneration(pipeline, {
    inference: {
      prompt: 'The sky is',
      generation: {
        maxTokens: 1,
      },
      sampling: {
        temperature: 0,
        topK: 1,
        topP: 1,
      },
    },
    shared: {
      tooling: {
        intent: 'calibrate',
      },
    },
  });

  assert.equal(result.phase.decodeMode, 'batched_gpu_stepwise_ple');
  assert.equal(result.phase.batching?.batchedForwardCalls, 9);
  assert.equal(result.phase.batching?.unbatchedForwardCalls, 10);
  assert.equal(result.phase.batching?.totalBatchedTimeMs, 11);
  assert.equal(result.phase.batching?.totalUnbatchedTimeMs, 12);
  assert.equal(result.phase.batching?.gpuSubmissions, 13);
  assert.equal(result.phase.gpu?.singleTokenReadbackWaitMs, 7);
  assert.equal(result.phase.gpu?.singleTokenOrchestrationMs, 8);
  assert.equal(result.phase.gpu?.decodeOrchestrationMs, 1);
}

{
  const result = await runBrowserSuite({
    suite: 'bench',
    command: 'bench',
    surface: 'node',
    harnessOverride: {
      modelLoadMs: 1,
      manifest: {
        modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
        modelType: 'transformer',
        architecture: {
          numLayers: 42,
          hiddenSize: 2048,
          intermediateSize: 16384,
          numAttentionHeads: 8,
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
      pipeline: {
        async *generate(_promptInput, options = {}) {
          options.onToken?.(1, 'Blue');
          yield 'Blue';
        },
        getStats() {
          return {
            totalTimeMs: 15,
            prefillTimeMs: 5,
            ttftMs: 6,
            decodeTimeMs: 10,
            prefillTokens: 7,
            decodeTokens: 1,
            decodeMode: 'batched_gpu_stepwise_ple',
            batchGuardReason: null,
            singleTokenReadbackWaitMs: 4,
            singleTokenOrchestrationMs: 2,
            decodeRecordMs: 3,
            decodeSubmitWaitMs: 2,
            decodeReadbackWaitMs: 2,
            batching: {
              batchedForwardCalls: 3,
              unbatchedForwardCalls: 0,
              totalBatchedTimeMs: 10,
              totalUnbatchedTimeMs: 0,
              gpuSubmissions: 1,
            },
            decodeProfileSteps: [],
          };
        },
        reset() {},
        async unload() {},
      },
    },
  });

  assert.equal(result.results[0]?.passed, true);
  assert.equal(result.metrics.decodeMode, 'batched_gpu_stepwise_ple');
  assert.equal(result.metrics.batching?.batchedForwardCalls?.median, 3);
  assert.equal(result.metrics.batching?.totalBatchedTimeMs?.median, 10);
  assert.equal(result.metrics.batching?.gpuSubmissions?.median, 1);
  assert.equal(result.metrics.gpu?.singleTokenReadbackWaitMs?.median, 4);
  assert.equal(result.metrics.gpu?.singleTokenOrchestrationMs?.median, 2);
  assert.equal(result.metrics.gpu?.decodeOrchestrationMs?.median, 3);
}

{
  // decodeOrchestrationMs is a derived residual: when component timings sum exceeds
  // decodeMs (scope drift), the residual is negative. It must surface as-is, not clamp to 0.
  const pipeline = {
    async *generate(_promptInput, options = {}) {
      options.onToken?.(1, 'Blue');
      yield 'Blue';
    },
    getStats() {
      return {
        prefillTimeMs: 2,
        ttftMs: 3,
        decodeTimeMs: 5,
        prefillTokens: 1,
        decodeTokens: 1,
        decodeRecordMs: 3,
        decodeSubmitWaitMs: 2,
        decodeReadbackWaitMs: 2,
        decodeProfileSteps: [],
      };
    },
  };

  const result = await runGeneration(pipeline, {
    inference: {
      prompt: 'The sky is',
      generation: { maxTokens: 1 },
      sampling: { temperature: 0, topK: 1, topP: 1 },
    },
    shared: { tooling: { intent: 'calibrate' } },
  });

  // 5 - 3 - 2 - 2 = -2: scope drift must be visible, not hidden as 0
  assert.equal(result.phase.gpu?.decodeOrchestrationMs, -2);
}

console.log('browser-harness-batching-metrics.test: ok');
