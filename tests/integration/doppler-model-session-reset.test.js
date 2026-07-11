import assert from 'node:assert/strict';

import { createModelHandle } from '../../src/client/runtime/model-session.js';

function makePipeline(overrides = {}) {
  return {
    manifest: {},
    isLoaded: true,
    generate() {},
    embed() {},
    embedBatch() {},
    embedImage() {},
    embedAudio() {},
    transcribeImage() {},
    transcribeAudio() {},
    transcribeVideo() {},
    unload() {},
    ...overrides,
  };
}

{
  let resetCount = 0;
  const handle = createModelHandle(makePipeline({
    resetGenerationState() {
      resetCount += 1;
    },
  }), { modelId: 'unit-model' });

  handle.resetGenerationState();
  assert.equal(resetCount, 1);
  assert.equal(handle.manifestHash, null);
}

{
  let seqLen = null;
  const handle = createModelHandle(makePipeline({
    resetToSeqLen(value) {
      seqLen = value;
    },
  }), { modelId: 'unit-model' });

  handle.resetGenerationState();
  assert.equal(seqLen, 0);
}

{
  const calls = [];
  const prefix = { cache: {}, seqLen: 2, tokens: [4, 5] };
  const handle = createModelHandle(makePipeline({
    tokenizer: {
      encode(text) {
        assert.equal(text, 'rank this');
        return Uint32Array.from([4, 5, 6]);
      },
    },
    prefillKVOnly(prompt, options) {
      calls.push(['prefillKVOnly', prompt, options]);
      return prefix;
    },
    prefillWithTokenLogits(prompt, tokenIds, options) {
      calls.push(['prefillWithTokenLogits', prompt, tokenIds, options]);
      return { logits: Float32Array.from([1, -1]) };
    },
    prefillWithTokenLogitsFromKV(snapshot, prompt, tokenIds, options) {
      calls.push(['prefillWithTokenLogitsFromKV', snapshot, prompt, tokenIds, options]);
      return { logits: Float32Array.from([2, -2]) };
    },
  }), { modelId: 'unit-model' });

  assert.deepEqual(handle.advanced.tokenizeText('rank this'), [4, 5, 6]);
  assert.equal(await handle.advanced.prefillKV('', { inputIds: [4, 5] }), prefix);
  await handle.advanced.prefillWithTokenLogits('rank this', [1, 0], { useChatTemplate: false });
  await handle.advanced.prefillWithTokenLogitsFromKV(prefix, '', [1, 0], { inputIds: [6] });
  assert.deepEqual(calls, [
    ['prefillKVOnly', '', { inputIds: [4, 5] }],
    ['prefillWithTokenLogits', 'rank this', [1, 0], { useChatTemplate: false }],
    ['prefillWithTokenLogitsFromKV', prefix, '', [1, 0], { inputIds: [6] }],
  ]);
}

{
  const handle = createModelHandle(makePipeline(), {
    modelId: 'unit-model',
    manifestHash: '0123456789abcdef',
  });
  assert.equal(handle.manifestHash, '0123456789abcdef');
}

console.log('doppler-model-session-reset.test: ok');
