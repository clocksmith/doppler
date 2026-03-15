import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const {
  assertTokenIdsInRange,
  assertTokenIdInRange,
  resolveGenerateOptions,
  resolveStepOptions,
  resolvePrefillEmbeddingOptions,
  resolveAdvanceEmbeddingMode,
  extractEmbeddingFromHidden,
  resolveFloatDtypeFromByteSize,
} = await import('../../src/inference/pipelines/text/generator-runtime.js');

// === assertTokenIdsInRange ===

{
  const state = { modelConfig: { vocabSize: 100 }, tokenizer: null };
  assertTokenIdsInRange(state, [0, 50, 99]);
}

{
  const state = { modelConfig: { vocabSize: 100 }, tokenizer: null };
  assert.throws(
    () => assertTokenIdsInRange(state, [0, 100]),
    /token id out of range/
  );
}

{
  const state = { modelConfig: { vocabSize: 100 }, tokenizer: null };
  assert.throws(
    () => assertTokenIdsInRange(state, [0, -1]),
    /token id out of range/
  );
}

{
  const state = { modelConfig: { vocabSize: 100 }, tokenizer: null };
  assert.throws(
    () => assertTokenIdsInRange(state, [0, NaN]),
    /token id out of range/
  );
}

assert.throws(
  () => assertTokenIdsInRange({ modelConfig: { vocabSize: 100 } }, 'not-an-array'),
  /expected tokenIds array/
);

assert.throws(
  () => assertTokenIdsInRange({ modelConfig: { vocabSize: NaN } }, [0]),
  /invalid model vocabSize/
);

assert.throws(
  () => assertTokenIdsInRange({ modelConfig: { vocabSize: 0 } }, [0]),
  /invalid model vocabSize/
);

// === assertTokenIdInRange ===

{
  const state = { modelConfig: { vocabSize: 100 } };
  assertTokenIdInRange(state, 0);
  assertTokenIdInRange(state, 99);
}

assert.throws(
  () => assertTokenIdInRange({ modelConfig: { vocabSize: 100 } }, 100),
  /tokenId=100 out of range/
);

assert.throws(
  () => assertTokenIdInRange({ modelConfig: { vocabSize: 100 } }, -1),
  /tokenId=-1 out of range/
);

assert.throws(
  () => assertTokenIdInRange({ modelConfig: { vocabSize: NaN } }, 0),
  /invalid model vocabSize/
);

// === resolveGenerateOptions ===

// Minimal execution plan for testing option resolution
function createTestPlanState(maxTokens = 128) {
  return {
    executionPlanState: {
      activePlanId: 'primary',
      primaryPlan: {
        id: 'primary',
        source: 'test',
        kernelPath: null,
        kernelPathId: null,
        activationDtype: 'f32',
        finitenessGuardEnabled: false,
        finitenessAbsThreshold: 65504,
        finitenessIncludeNonFinite: false,
        deferredRoundingWindowTokens: 0,
        defaultDisableCommandBatching: false,
        defaultDisableMultiTokenDecode: false,
        defaultBatchSize: 1,
        defaultStopCheckMode: 'batch',
        defaultMaxTokens: maxTokens,
        readbackInterval: 1,
        ringTokens: 0,
        ringStop: false,
        ringStaging: false,
      },
      fallbackPlan: null,
    },
  };
}

{
  const state = {
    debug: false,
    runtimeConfig: {
      inference: {
        sampling: { temperature: 0.7, topP: 0.95, topK: 40, repetitionPenalty: 1.0 },
        generation: { maxTokens: 128, useSpeculative: false, profile: false, benchmark: false, embeddingMode: 'last' },
        chatTemplate: { enabled: false },
        session: {},
      },
    },
    modelConfig: { chatTemplateEnabled: false },
    manifest: { modelType: 'transformer' },
    ...createTestPlanState(),
  };

  const opts = resolveGenerateOptions(state, {});
  assert.equal(opts.temperature, 0.7);
  assert.equal(opts.topP, 0.95);
  assert.equal(opts.topK, 40);
  assert.equal(opts.repetitionPenalty, 1.0);
  assert.equal(opts.useChatTemplate, false);
  assert.equal(opts.debug, false);
  assert.deepEqual(opts.stopSequences, []);
  assert.equal(opts.maxTokens, 128);
  assert.equal(opts.executionPlan.planId, 'primary');
}

{
  const state = {
    debug: false,
    runtimeConfig: {
      inference: {
        sampling: { temperature: 0.7, topP: 0.95, topK: 40, repetitionPenalty: 1.0 },
        generation: { maxTokens: 128, useSpeculative: false, profile: false, benchmark: false, embeddingMode: 'last' },
        chatTemplate: { enabled: undefined },
        session: {},
      },
    },
    modelConfig: { chatTemplateEnabled: true },
    manifest: { modelType: 'transformer' },
    ...createTestPlanState(),
  };

  const opts = resolveGenerateOptions(state, {});
  assert.equal(opts.useChatTemplate, true);
}

// Explicit override from options
{
  const state = {
    debug: false,
    runtimeConfig: {
      inference: {
        sampling: { temperature: 0.7, topP: 0.95, topK: 40, repetitionPenalty: 1.0 },
        generation: { maxTokens: 128, useSpeculative: false, profile: false, benchmark: false, embeddingMode: 'last' },
        chatTemplate: { enabled: false },
        session: {},
      },
    },
    modelConfig: { chatTemplateEnabled: false },
    manifest: { modelType: 'transformer' },
    ...createTestPlanState(),
  };

  const opts = resolveGenerateOptions(state, {
    temperature: 0,
    topK: 1,
    topP: 1.0,
  });
  assert.equal(opts.temperature, 0);
  assert.equal(opts.topK, 1);
  assert.equal(opts.topP, 1.0);
}

{
  const state = {
    debug: false,
    runtimeConfig: {
      inference: {
        sampling: { temperature: 0.7, topP: 0.95, topK: 40, repetitionPenalty: 1.0 },
        generation: { maxTokens: 128, useSpeculative: false, profile: false, benchmark: false, embeddingMode: 'last' },
        chatTemplate: { enabled: undefined },
        session: {},
      },
    },
    modelConfig: { chatTemplateEnabled: true },
    manifest: { modelType: 'transformer' },
    ...createTestPlanState(),
  };

  const opts = resolveGenerateOptions(state, {
    useChatTemplate: false,
  });
  assert.equal(opts.useChatTemplate, false);
}

// null value for configured option should throw
{
  const state = {
    debug: false,
    runtimeConfig: {
      inference: {
        sampling: { temperature: 0.7, topP: 0.95, topK: 40, repetitionPenalty: 1.0 },
        generation: { maxTokens: 128, useSpeculative: false, profile: false, benchmark: false, embeddingMode: 'last' },
        chatTemplate: { enabled: false },
        session: {},
      },
    },
    modelConfig: { chatTemplateEnabled: false },
    manifest: { modelType: 'transformer' },
    ...createTestPlanState(),
  };

  assert.throws(
    () => resolveGenerateOptions(state, { temperature: null }),
    /null is unsupported/
  );
}

// === resolveStepOptions ===

{
  const state = {
    debug: true,
    runtimeConfig: {
      inference: {
        sampling: { temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1.0 },
        generation: { profile: false, embeddingMode: 'last' },
        session: {},
      },
    },
    modelConfig: {},
    manifest: { modelType: 'transformer' },
    ...createTestPlanState(),
  };

  const opts = resolveStepOptions(state);
  assert.equal(opts.temperature, 0);
  assert.equal(opts.topK, 1);
  assert.equal(opts.debug, true);
}

// === resolveAdvanceEmbeddingMode ===

{
  const state = {
    runtimeConfig: { inference: { generation: { embeddingMode: 'last' } } },
    manifest: { modelType: 'transformer' },
  };
  assert.equal(resolveAdvanceEmbeddingMode(state), 'last');
}

{
  const state = {
    runtimeConfig: { inference: { generation: { embeddingMode: 'last' } } },
    manifest: { modelType: 'embedding' },
  };
  assert.equal(resolveAdvanceEmbeddingMode(state), 'mean');
}

{
  const state = {
    runtimeConfig: { inference: { generation: { embeddingMode: 'last' } } },
    manifest: { modelType: 'embedding' },
  };
  assert.equal(resolveAdvanceEmbeddingMode(state, { embeddingMode: 'last' }), 'last');
}

assert.throws(
  () => resolveAdvanceEmbeddingMode(
    { runtimeConfig: { inference: { generation: { embeddingMode: 'last' } } }, manifest: {} },
    { embeddingMode: 'invalid' }
  ),
  /invalid value/
);

// === resolveFloatDtypeFromByteSize ===

assert.equal(resolveFloatDtypeFromByteSize(0, 10), 'f32');
assert.equal(resolveFloatDtypeFromByteSize(NaN, 10), 'f32');
assert.equal(resolveFloatDtypeFromByteSize(-1, 10), 'f32');
assert.equal(resolveFloatDtypeFromByteSize(10, 0), 'f32');

// === extractEmbeddingFromHidden ===

{
  const hiddenSize = 4;
  const numTokens = 2;
  const hidden = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const normWeights = new Float32Array([1, 1, 1, 1]);
  const config = { rmsNormEps: 1e-6, rmsNormWeightOffset: false };

  const lastEmbed = extractEmbeddingFromHidden(hidden, numTokens, hiddenSize, 'last', normWeights, config);
  assert.equal(lastEmbed.length, hiddenSize);
  assert.ok(lastEmbed instanceof Float32Array);

  const meanEmbed = extractEmbeddingFromHidden(hidden, numTokens, hiddenSize, 'mean', normWeights, config);
  assert.equal(meanEmbed.length, hiddenSize);
  assert.ok(meanEmbed instanceof Float32Array);
}

// Length mismatch
assert.throws(
  () => extractEmbeddingFromHidden(new Float32Array([1, 2]), 2, 4, 'last', new Float32Array(4), {}),
  /Hidden state length mismatch/
);

// Invalid embedding mode
assert.throws(
  () => extractEmbeddingFromHidden(new Float32Array([1, 2, 3, 4]), 1, 4, 'bad', new Float32Array(4), {}),
  /unsupported embeddingMode/
);

console.log('generator-runtime.test: ok');
