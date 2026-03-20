import assert from 'node:assert/strict';

const { createManifest } = await import('../../src/converter/core.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

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
  'model.layers.0.feed_forward.w1.weight': {
    shard: 0,
    offset: 0,
    size: 4,
    shape: [8192, 2048],
    dtype: 'Q4_K_M',
    role: 'matmul',
  },
  'model.layers.0.feed_forward.w2.weight': {
    shard: 0,
    offset: 4,
    size: 4,
    shape: [2048, 8192],
    dtype: 'Q4_K_M',
    role: 'matmul',
  },
  'model.layers.0.feed_forward.w3.weight': {
    shard: 0,
    offset: 8,
    size: 4,
    shape: [8192, 2048],
    dtype: 'Q4_K_M',
    role: 'matmul',
  },
};

const model = {
  modelId: 'lfm2-intermediate-test',
  modelType: 'transformer',
  quantization: 'Q4_K_M',
  architecture: {
    numLayers: 16,
    hiddenSize: 2048,
    intermediateSize: 12288,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    headDim: 64,
    vocabSize: 65536,
    maxSeqLen: 32768,
  },
  config: {
    model_type: 'lfm2',
    block_auto_adjust_ff_dim: true,
    architectures: ['Lfm2ForCausalLM'],
  },
  tensors: [
    { name: 'model.layers.0.feed_forward.w1.weight', shape: [8192, 2048] },
    { name: 'model.layers.0.feed_forward.w2.weight', shape: [2048, 8192] },
    { name: 'model.layers.0.feed_forward.w3.weight', shape: [8192, 2048] },
  ],
};

const manifest = createManifest(
  'lfm2-intermediate-test',
  model,
  shards,
  tensorLocations,
  {
    source: 'unit-test',
    modelType: 'transformer',
    quantization: 'Q4_K_M',
    hashAlgorithm: 'blake3',
    inference: {
      ...DEFAULT_MANIFEST_INFERENCE,
    },
    eosTokenId: 7,
  }
);

assert.equal(manifest.architecture.intermediateSize, 8192);

console.log('core-manifest-lfm2-intermediate.test: ok');
