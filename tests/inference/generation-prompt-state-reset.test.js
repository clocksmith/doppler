import assert from 'node:assert/strict';
import { PipelineGenerator } from '../../src/inference/pipelines/text/generator.js';

// Exercise the actual generation entry points and reset owner, stopping before
// GPU work. Physical reset/no-reset controls are retained separately.
for (const surface of ['generate', 'generateTokenIds']) {
  for (const condition of ['ready', 'busy', 'unloaded']) {
    let clears = 0;
    const layer = { seqLen: 19 };
    const state = {
      isLoaded: condition !== 'unloaded',
      isGenerating: condition === 'busy',
      currentSeqLen: 19,
      kvCache: { clear() { clears += 1; } },
      linearAttentionRuntime: { schemaVersion: 2, layers: new Map([[0, layer]]) },
    };
    const generator = new PipelineGenerator(state);
    const reachedDecodeReset = new Error('reached decode reset');
    generator._resetDecodeRuntimeState = () => { throw reachedDecodeReset; };
    const run = () => surface === 'generate'
      ? generator.generate('a new complete prompt', {}).next()
      : generator.generateTokenIds('a new complete prompt', {});
    if (condition === 'ready') {
      await assert.rejects(run, error => error === reachedDecodeReset);
      assert.equal(clears, 1, `${surface}: new prompts must clear the preceding KV context`);
      assert.equal(state.currentSeqLen, 0);
      assert.equal(state.linearAttentionRuntime.layers.size, 0);
    } else {
      await assert.rejects(run, condition === 'busy' ? /Generation already in progress/ : /Model not loaded/);
      assert.equal(clears, 0, 'Rejected concurrent/unloaded calls must not reset owned state');
      assert.equal(state.currentSeqLen, 19);
      assert.equal(state.linearAttentionRuntime.layers.get(0), layer);
    }
  }
}
console.log('generation-prompt-state-reset.test: ok');
