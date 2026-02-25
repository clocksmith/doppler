import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { createManifest } = await import('../../src/converter/core.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');
const { convertModel } = await import('../../src/converter/core.js');
const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');

const model = {
  modelId: 'manifest-time-test',
  modelType: 'transformer',
  quantization: 'F16',
  architecture: {
    numLayers: 1,
    hiddenSize: 2,
    intermediateSize: 8,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 2,
    vocabSize: 16,
    maxSeqLen: 8,
  },
  config: {
    model_type: 'gemma3_text',
    architectures: ['Gemma3ForCausalLM'],
    num_hidden_layers: 1,
    hidden_size: 2,
    intermediate_size: 8,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 2,
    vocab_size: 16,
    max_position_embeddings: 8,
  },
  tensors: [],
};

const shards = [
  {
    index: 0,
    filename: 'shard_00000.bin',
    size: 16,
    hash: 'hash',
    offset: 0,
  },
];

const tensorLocations = {
  'model.embed_tokens.weight': {
    shard: 0,
    offset: 0,
    size: 16,
    shape: [2, 4],
    dtype: 'F16',
    role: 'embedding',
  },
};

{
  const manifest = createManifest(
    'manifest-time-test',
    model,
    shards,
    tensorLocations,
    {
      source: 'unit-test',
      modelType: 'transformer',
      quantization: 'F16',
      hashAlgorithm: 'blake3',
      inference: { ...DEFAULT_MANIFEST_INFERENCE, presetId: 'gemma3' },
      eosTokenId: 1,
      convertedAt: '2026-01-05T10:20:30.000Z',
      conversionInfo: {
        source: 'unit-test',
        convertedAt: '2026-01-05T10:20:30.000Z',
        tool: 'tests',
        version: '1.0.0',
      },
    }
  );

  assert.equal(manifest.metadata.convertedAt, '2026-01-05T10:20:30.000Z');
  assert.equal(manifest.conversion?.tool, 'tests');
}

{
  assert.throws(
    () => createManifest(
      'manifest-time-test',
      model,
      shards,
      tensorLocations,
      {
        source: 'unit-test',
        modelType: 'transformer',
        quantization: 'F16',
        hashAlgorithm: 'blake3',
        inference: { ...DEFAULT_MANIFEST_INFERENCE, presetId: 'gemma3' },
        eosTokenId: 1,
        convertedAt: 'not-a-date',
      }
    ),
    /Invalid manifest convertedAt timestamp/
  );
}

{
  const io = {
    async readTensorData() {
      return new Uint8Array(16).buffer;
    },
    async writeShard() {
      return 'hash';
    },
    async writeManifest() {
      return undefined;
    },
  };

  const converted = await convertModel({
    ...model,
    tensors: [
      {
        name: 'model.embed_tokens.weight',
        shape: [2, 4],
        dtype: 'F16',
        size: 16,
        offset: 0,
      },
    ],
  }, io, {
    modelId: 'manifest-time-test',
    modelType: 'transformer',
    quantization: 'F16',
    quantizationInfo: {
      weights: 'f16',
      embeddings: 'f16',
      compute: 'f16',
      variantTag: 'wf16',
    },
    eosTokenId: 1,
    architecture: model.architecture,
    inference: { ...DEFAULT_MANIFEST_INFERENCE, presetId: 'gemma3' },
    converterConfig: createConverterConfig({
      manifest: {
        hashAlgorithm: 'blake3',
        conversion: {
          source: 'unit-test',
          convertedAt: '2026-01-06T11:22:33.000Z',
          tool: 'tests',
          version: '1.0.0',
        },
      },
    }),
  });

  assert.equal(converted.manifest.metadata.convertedAt, '2026-01-06T11:22:33.000Z');
}

console.log('core-manifest-converted-at.test: ok');
