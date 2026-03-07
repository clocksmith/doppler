import assert from 'node:assert/strict';

const {
  extractTextModelConfig,
  shouldAutoTuneKernels,
  verifyExplicitModelUrlMatch,
} = await import('../../src/client/doppler-provider/model-manager.js');

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

{
  assert.equal(shouldAutoTuneKernels(null), false);
  assert.equal(shouldAutoTuneKernels({ shared: {} }), false);
  assert.equal(shouldAutoTuneKernels({
    shared: {
      kernelWarmup: {
        autoTune: true,
      },
    },
  }), true);
}

{
  const manifest = {
    modelId: 'explicit-source',
    quantization: 'Q4_K_M',
    hashAlgorithm: 'sha256',
    totalSize: 12,
    shards: [
      {
        filename: 'model-00000.bin',
        size: 12,
        hash: 'abc',
      },
    ],
  };

  await verifyExplicitModelUrlMatch(
    manifest,
    'https://example.test/model',
    async () => ({
      ...manifest,
    })
  );

  await assert.rejects(
    () => verifyExplicitModelUrlMatch(
      manifest,
      'https://example.test/model',
      async () => ({
        ...manifest,
        totalSize: 99,
      })
    ),
    /does not match the cached manifest/
  );

  await assert.rejects(
    () => verifyExplicitModelUrlMatch(
      manifest,
      'https://example.test/model',
      async () => {
        throw new Error('fetch failed');
      }
    ),
    /Could not compare cached manifest with explicit modelUrl/
  );
}

console.log('model-manager-contract.test: ok');
