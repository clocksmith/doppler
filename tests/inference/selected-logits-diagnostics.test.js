import assert from 'node:assert/strict';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { prefillWithEmbedding, prefillWithLogits, prefillWithTokenLogits, prefillWithTokenLogitsFromKV } from '../../src/inference/pipelines/text/generator/training.js';

const diagnostics = { enabled: true, captureConfig: { defaultLevel: 'none', targetOpIds: ['layer.0.attn.q_proj'], targetLevel: 'full' } };
const failure = new Error('projection failed');
function context({ reject = false, enabled = true, closeError = null } = {}) {
  const state = {
    isLoaded: true, isGenerating: false, useGPU: false, currentSeqLen: 2,
    modelConfig: { vocabSize: 32, layerTypes: [], chatTemplateEnabled: false },
    manifest: { modelId: 'selected-logits-diagnostic-fixture' },
    runtimeConfig: createDopplerConfig().runtime,
    executionPlanState: { activePlanId: 'primary', primaryPlan: { id: 'primary', defaultDisableCommandBatching: false, defaultDisableMultiTokenDecode: false } },
    kvCache: { clone: () => ({ destroy() {} }) },
    stats: {}, operatorDiagnostics: null,
  };
  return {
    _state: state,
    _resetDecodeRuntimeState() {},
    closeCount: 0,
    _closeFinitenessFallbackWindow() {
      this.closeCount += 1;
      if (closeError) throw closeError;
    },
    async _prefillPromptToLogits(prompt, options) {
      assert.equal(state.operatorDiagnostics?.enabled === true, enabled, 'requested captures must be active during selected-token execution');
      if (enabled) {
        assert.equal(state.operatorDiagnostics.captureConfig.targetLevel, 'full');
        state.operatorDiagnostics.emitter.emitRecord('attn.q_proj', { layerIdx: 0, capture: { level: 'full', data: [1, 2] } });
      }
      if (reject) throw failure;
      return { inputIds: [1, 2], logits: new Float32Array([3, 4]), phase: {} };
    },
  };
}

const invocations = [
  (ctx, options) => prefillWithTokenLogits.call(ctx, 'prompt', [3, 4], options),
  (ctx, options) => prefillWithTokenLogitsFromKV.call(ctx, { cache: ctx._state.kvCache, seqLen: 0, tokens: [] }, 'prompt', [3, 4], options),
  (ctx, options) => prefillWithLogits.call(ctx, 'prompt', options),
];
for (const invoke of invocations) {
  for (const reject of [false, true]) {
    const ctx = context({ reject });
    const before = JSON.stringify(ctx._state.executionPlanState);
    if (reject) await assert.rejects(invoke(ctx, { diagnostics }), (error) => error === failure);
    else await invoke(ctx, { diagnostics });
    assert.equal(ctx._state.operatorDiagnostics, null, 'capture state must close on success and failure');
    assert.equal(ctx._state.stats.operatorDiagnostics.recordCount, 1);
    assert.deepEqual(ctx._state.stats.operatorDiagnostics.timeline[0].capture.data, [1, 2]);
    assert.equal(ctx._state.stats.operatorDiagnostics.timeline[0].executionPlanHash, 'primary');
    assert.equal(ctx.closeCount, 1);
    assert.equal(JSON.stringify(ctx._state.executionPlanState), before, 'observation cannot rewrite the execution plan');
  }
  const disabled = context({ enabled: false });
  disabled._state.stats.operatorDiagnostics = { stale: true };
  await invoke(disabled, {});
  assert.equal(disabled._state.stats.operatorDiagnostics, null, 'a later ordinary run cannot inherit prior captures');
  const invalid = context();
  await assert.rejects(invoke(invalid, { diagnostics: { enabled: true, captureConfig: { defaultLevel: 'invalid' } } }), /CapturePolicy/);
  assert.equal(invalid._state.operatorDiagnostics, null);
  const closeError = new Error('fallback cleanup failed');
  const closing = context({ closeError });
  await assert.rejects(invoke(closing, { diagnostics }), (error) => error === closeError);
  assert.equal(closing._state.operatorDiagnostics, null);
  assert.equal(closing._state.stats.operatorDiagnostics.recordCount, 1);
}

// Embedding input validation runs inside the same observation lifetime.
const embedding = context();
embedding._resolvePromptOrInputIds = () => { throw failure; };
await assert.rejects(prefillWithEmbedding.call(embedding, 'prompt', { diagnostics }), (error) => error === failure);
assert.equal(embedding._state.operatorDiagnostics, null);
assert.equal(embedding._state.stats.operatorDiagnostics.recordCount, 0);
assert.equal(embedding.closeCount, 1);
console.log('selected-logits-diagnostics.test: ok (synthetic execution; no physical inference claim)');
