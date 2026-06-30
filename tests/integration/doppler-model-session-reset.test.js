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

console.log('doppler-model-session-reset.test: ok');
