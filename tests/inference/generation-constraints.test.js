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

// Exercise the real decode dispatcher with a batch-capable GPU test double.
// A constraint must visit every sampled token, even when batching is enabled.
const { setDevice } = await import('../../src/gpu/device.js');
const { _runDecodeLoop } = await import('../../src/inference/pipelines/text/generator/decode-runtime.js');
setDevice({ features: new Set(), limits: { maxStorageBufferBindingSize: 1048576,
  maxBufferSize: 1048576, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupStorageSize: 16384 },
queue: { submit() {} }, createBuffer() { return { destroy() {} }; }, createBindGroup() { return {}; } }, { platformConfig: null });
try {
  const generatedIds = [1];
  const visits = [];
  const decodeState = { ...state, useGPU: true,
    modelConfig: { ...state.modelConfig, hiddenSize: 4, numLayers: 1, layerTypes: ['full_attention'] },
    weights: new Map(), kvCache: { layout: 'contiguous' }, currentSeqLen: 1,
    batchingStats: {}, stats: {}, tokenizer: { getSpecialTokens: () => ({ eos: 2, pad: 0 }) } };
  const pipeline = { _state: decodeState, _hasFinitenessFallbackWindow: () => false,
    _consumeFinitenessFallbackToken() {}, _shouldUseFinitenessFallback: () => false,
    _recordStopReason(reason, tokenId) { decodeState.stats.stopReason = reason; decodeState.stats.stopTokenId = tokenId; },
    async _generateNTokensGPU() { throw new Error('Unconstrained batch path must not execute'); },
    async _decodeNextTokenViaLogits(ids, options) {
      return PipelineGenerator.prototype._sampleNextTokenFromLogits.call(this, new Float32Array([-Infinity, 9, 2]), ids, options);
    } };
  const opts = { maxTokens: 6, stopSequences: [], suppressTokenIds: [], speculation: null,
    temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1, promptTokenCount: 1,
    logitMaskFn(logits, context) { visits.push([...context.generatedIds]); logits[1] = -Infinity; },
    executionPlan: { batchSize: 4, readbackInterval: 1, disableMultiTokenDecode: false,
      disableCommandBatching: false, maxBatchDecodeTokens: null, activationDtype: 'f32' } };
  const emitted = [];
  for await (const token of _runDecodeLoop.call(pipeline, generatedIds, opts, {},
    { stopTokenIds: [], eosToken: 2, stopSequenceStart: 0, decodeToken: String, emitMode: 'token' })) emitted.push(token);
  assert.deepEqual(visits, [[]]);
  assert.deepEqual(emitted, [2]);
  assert.equal(decodeState.stats.stopReason, 'stop-token');
} finally { setDevice(null); }
console.log('generation-constraints decode dispatch: ok');
