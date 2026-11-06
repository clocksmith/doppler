import assert from 'node:assert/strict';
import { PipelineGenerator } from '../../src/inference/pipelines/text/generator.js';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { buildGenerationBackendIdentity } from '../../src/client/model-host/model-evidence.js';

// Run both real generation entry points with controlled prefill/sampling, without
// a GPU. Receipts must describe this invocation, including its final active plan.
for (const surface of ['generateTokenIds', 'generateTokens']) {
  const plan = id => ({ id, kernelPathId: `${id}-path`, kernelPathSource: 'execution-v1',
    activationDtype: 'f32', defaultMaxTokens: 1, defaultBatchSize: 1,
    defaultStopCheckMode: 'batch', defaultDisableCommandBatching: false,
    defaultDisableMultiTokenDecode: false, readbackInterval: 1,
    ringTokens: 1, ringStop: 1, ringStaging: 1 });
  const state = {
    modelConfig: { vocabSize: 3, stopTokenIds: [2], reasoningOutput: null },
    tokenizer: { getSpecialTokens: () => ({ eos: 2 }) },
    isLoaded: true, isGenerating: false, currentSeqLen: 0, stats: {},
    runtimeConfig: createDopplerConfig().runtime,
    executionPlanState: { activePlanId: 'primary', primaryPlan: plan('primary'),
      fallbackPlan: plan('finiteness_fallback') },
  };
  const generator = new PipelineGenerator(state);
  generator._sampleNextTokenFromLogits = () => 2;
  let fallback = false;
  let failure = null;
  generator._prefillPromptToLogits = async () => {
    if (fallback) {
      state.executionPlanState.activePlanId = 'finiteness_fallback';
      state.stats.kernelPathId = 'finiteness_fallback-path';
    }
    if (failure) throw failure;
    return { inputIds: [0], logits: new Float32Array([0, 1, 2]) };
  };
  const run = async () => {
    const options = { maxTokens: 1, useChatTemplate: false };
    if (surface === 'generateTokenIds') {
      const result = await generator.generateTokenIds('', options);
      assert.deepEqual(result.tokenIds, [2]);
      return result.stats;
    }
    const tokens = [];
    for await (const token of generator.generateTokens('', options)) tokens.push(token);
    assert.deepEqual(tokens, [2]);
    return state.stats;
  };
  const check = (stats, finalId) => {
    assert.equal(stats.executionPlan?.activePlanIdAtStart, 'primary', surface);
    assert.equal(stats.executionPlan.finalActivePlanId, finalId, surface);
    const backend = buildGenerationBackendIdentity({ stats });
    assert.equal(backend.executionPlanId, finalId);
    assert.equal(backend.activationDtype, 'f32');
    assert.equal(backend.kernelPathId, `${finalId}-path`);
    assert.equal(state.executionPlanState.activePlanId, 'primary', 'cleanup restores the primary plan');
    assert.equal(state.isGenerating, false);
  };
  check(await run(), 'primary');
  fallback = true;
  check(await run(), 'finiteness_fallback');
  failure = new Error('controlled prefill failure');
  await assert.rejects(run, error => error === failure);
  check(state.stats, 'finiteness_fallback');
  failure = null;
  fallback = false;
  check(await run(), 'primary');
}
console.log('generation-plan-evidence.test: passed (controlled execution, no GPU)');
