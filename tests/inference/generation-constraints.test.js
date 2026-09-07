import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { createDopplerConfig } from '../../src/config/schema/index.js';

installNodeFileFetchShim();
const { PipelineGenerator } = await import('../../src/inference/pipelines/text/generator.js');
const { resolveGenerateOptions } = await import('../../src/inference/pipelines/text/generator-runtime.js');

const state = {
  debug: false, modelConfig: { vocabSize: 3, chatTemplateEnabled: false },
  manifest: { modelType: 'transformer' }, runtimeConfig: createDopplerConfig().runtime,
  tokenizer: { getSpecialTokens: () => ({ pad: 0 }) },
  executionPlanState: { activePlanId: 'primary', fallbackPlan: null, primaryPlan: {
    id: 'primary', source: 'test', kernelPath: null, kernelPathId: null, activationDtype: 'f32',
    finitenessGuardEnabled: false, finitenessAbsThreshold: 65504, finitenessIncludeNonFinite: false,
    deferredRoundingWindowTokens: 0, defaultDisableCommandBatching: false,
    defaultDisableMultiTokenDecode: false, defaultBatchSize: 4, defaultStopCheckMode: 'batch',
    defaultMaxTokens: 8, readbackInterval: 1, ringTokens: 0, ringStop: false, ringStaging: false,
  } },
};
const sample = (options) => PipelineGenerator.prototype._sampleNextTokenFromLogits.call(
  { _state: state }, new Float32Array([-Infinity, 9, 2]), [1, 2, 1],
  { temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1, promptTokenCount: 2, ...options }
);
const mask = (logits, context) => { assert.deepEqual(context.generatedIds, [1]); logits[1] = -Infinity; };
assert.equal(resolveGenerateOptions(state, { logitMaskFn: mask }).logitMaskFn, mask);
assert.throws(() => resolveGenerateOptions(state, { logitMaskFn: 'not a function' }), /logitMaskFn/);
assert.equal(sample({ logitMaskFn: mask }), 2);
const original = new Error('constraint rejected state');
assert.throws(() => sample({ logitMaskFn: () => { throw original; } }), error => error === original);
assert.throws(() => sample({ logitMaskFn: async () => {} }), /synchronous/);
assert.throws(() => sample({ logitMaskFn: logits => { logits.fill(-Infinity); } }), /no finite candidate/);
assert.equal(sample({}), 1);
console.log('generation-constraints.test: ok');
