import assert from 'node:assert/strict';

import { createManifest } from '../../src/converter/core.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';

const model = {
  modelId: 'translategemma-eos-test',
  modelType: 'transformer',
  quantization: 'Q4K',
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
    model_type: 'translategemma',
    architectures: ['Gemma3ForConditionalGeneration'],
    num_hidden_layers: 1,
    hidden_size: 2,
    intermediate_size: 8,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 2,
    vocab_size: 16,
    max_position_embeddings: 8,
    eos_token_id: 1,
  },
  generationConfig: {
    eos_token_id: [1, 106],
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
    dtype: 'Q4K',
    role: 'embedding',
  },
};

const manifest = createManifest(
  'translategemma-eos-test',
  model,
  shards,
  tensorLocations,
  {
    source: 'unit-test',
    modelType: 'transformer',
    quantization: 'Q4K',
    hashAlgorithm: 'sha256',
    inference: { ...DEFAULT_MANIFEST_INFERENCE, },
  }
);

assert.deepEqual(manifest.eos_token_id, [1, 106]);

console.log('core-manifest-generation-config-eos.test: ok');
