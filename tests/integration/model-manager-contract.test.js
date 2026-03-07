import assert from 'node:assert/strict';

const { extractTextModelConfig } = await import('../../src/client/doppler-provider/model-manager.js');

{
  const config = extractTextModelConfig({
    architecture: {
      numLayers: 2,
      hiddenSize: 128,
      intermediateSize: 256,
      numAttentionHeads: 4,
      numKeyValueHeads: 4,
      headDim: 32,
      vocabSize: 32000,
      maxSeqLen: 2048,
    },
    quantization: 'Q4_K_M',
  });

  assert.equal(config.quantization, 'Q4_K_M');
}

{
  assert.throws(
    () => extractTextModelConfig({
      architecture: {
        numLayers: 2,
        hiddenSize: 128,
        intermediateSize: 256,
        numAttentionHeads: 4,
        numKeyValueHeads: 4,
        headDim: 32,
        vocabSize: 32000,
        maxSeqLen: 2048,
      },
    }),
    /Manifest is missing quantization; re-convert the model\./
  );
}

console.log('model-manager-contract.test: ok');
