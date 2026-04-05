import assert from 'node:assert/strict';

const { buildConservativeMultimodalGenerationOptions } = await import('../../src/inference/pipelines/text.js');

{
  const options = buildConservativeMultimodalGenerationOptions({
    maxTokens: 32,
    temperature: 0,
    topK: 1,
    topP: 1,
    repetitionPenalty: 1,
    disableCommandBatching: false,
    disableMultiTokenDecode: false,
    stopCheckMode: 'batch',
  });

  assert.equal(options.maxTokens, 32);
  assert.equal(options.temperature, 0);
  assert.equal(options.topK, 1);
  assert.equal(options.topP, 1);
  assert.equal(options.repetitionPenalty, 1);
  assert.equal(options.disableCommandBatching, true);
  assert.equal(options.disableMultiTokenDecode, true);
  assert.equal(options.stopCheckMode, 'per-token');
}

{
  const first = buildConservativeMultimodalGenerationOptions();
  const second = buildConservativeMultimodalGenerationOptions();

  assert.notEqual(first, second);
  assert.deepEqual(first, {
    disableCommandBatching: true,
    disableMultiTokenDecode: true,
    stopCheckMode: 'per-token',
  });
}

console.log('multimodal-generation-options.test: ok');
