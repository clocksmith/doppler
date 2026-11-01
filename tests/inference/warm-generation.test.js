import assert from 'node:assert/strict';

const { runWarmGeneration } = await import('../../src/inference/pipelines/text/warm-generation.js');

// A pipeline whose generator yields `count` tokens (or throws after them) and records
// the options it was given and every reset.
function fakePipeline({ count, stopReason = 'max-tokens', fail = null, ready = true }) {
  const pipeline = {
    resets: 0, options: null, stats: { stopReason: null },
    wholeTokenProgram: { program: { ready } },
    reset() { this.resets++; },
    generator: {
      async *generateTokens(prompt, options) {
        pipeline.prompt = prompt; pipeline.options = options;
        for (let i = 0; i < count; i++) yield 100 + i;
        if (fail) throw fail;
        pipeline.stats.stopReason = stopReason;
      },
    },
  };
  return pipeline;
}

const full = fakePipeline({ count: 8 });
assert.deepEqual(await runWarmGeneration(full, { prompt: '1, 2, 3,', maxTokens: 8 }),
  { generatedTokens: 8, maxTokens: 8, stopReason: 'max-tokens', wholeTokenProgramReady: true });
assert.equal(full.resets, 1, 'sequence state is reset after the warm generation');
assert.equal(full.prompt, '1, 2, 3,');
assert.deepEqual(full.options, { maxTokens: 8, useChatTemplate: false, temperature: 0, topK: 1, topP: 1,
  repetitionPenalty: 1, presencePenalty: 0, suppressTokenIds: [] }, 'greedy, raw prompt');

const early = fakePipeline({ count: 3, stopReason: 'stop-token', ready: false });
assert.deepEqual(await runWarmGeneration(early, { prompt: 'x', maxTokens: 8 }),
  { generatedTokens: 3, maxTokens: 8, stopReason: 'stop-token', wholeTokenProgramReady: false });

const failing = fakePipeline({ count: 2, fail: new Error('device lost') });
await assert.rejects(runWarmGeneration(failing, { prompt: 'x', maxTokens: 4 }), /device lost/);
assert.equal(failing.resets, 1, 'a failed warm generation still resets sequence state');

for (const [policy, message] of [
  [{ prompt: '', maxTokens: 4 }, /prompt must be a non-empty string/],
  [{ prompt: 'x', maxTokens: 1 }, /maxTokens must be an integer of at least 2/],
  [{ prompt: 'x', maxTokens: 2.5 }, /maxTokens must be an integer of at least 2/],
]) {
  const untouched = fakePipeline({ count: 4 });
  await assert.rejects(runWarmGeneration(untouched, policy), message);
  assert.equal(untouched.options, null, 'an invalid policy never generates');
}
console.log('warm-generation.test: ok');
