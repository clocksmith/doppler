import assert from 'node:assert/strict';

const { createManifest } = await import('../../src/converter/core.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

const model = {
  modelId: 'bundled-tokenizer-flags-test',
  modelType: 'transformer',
  quantization: 'F16',
  architecture: {
    numLayers: 1,
    hiddenSize: 2,
    intermediateSize: 8,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 2,
    vocabSize: 4,
    maxSeqLen: 8,
  },
  config: {
    model_type: 'qwen2',
    num_hidden_layers: 1,
    hidden_size: 2,
    intermediate_size: 8,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 2,
    vocab_size: 4,
    max_position_embeddings: 8,
    eos_token_id: 1,
  },
  tokenizerConfig: {
    add_bos_token: false,
    add_eos_token: true,
  },
  tokenizerJson: {
    model: {
      vocab: {
        a: 0,
        b: 1,
        c: 2,
        d: 3,
      },
    },
  },
  tensors: [],
};

const manifest = createManifest(
  'bundled-tokenizer-flags-test',
  model,
  [],
  {},
  {
    source: 'unit-test',
    modelType: 'transformer',
    quantization: 'F16',
    hashAlgorithm: 'blake3',
    inference: { ...DEFAULT_MANIFEST_INFERENCE },
  }
);

assert.equal(manifest.tokenizer?.type, 'bundled');
assert.equal(manifest.tokenizer?.file, 'tokenizer.json');
assert.equal(manifest.tokenizer?.addBosToken, false);
assert.equal(manifest.tokenizer?.addEosToken, true);

console.log('core-bundled-tokenizer-flags.test: ok');
