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
    add_eos_token: true,
  },
  tokenizerJson: {
    post_processor: {
      type: 'TemplateProcessing',
      single: [
        { SpecialToken: { id: '<bos>', type_id: 0 } },
        { Sequence: { id: 'A', type_id: 0 } },
      ],
      special_tokens: {
        '<bos>': { id: '<bos>', ids: [7], tokens: ['<bos>'] },
      },
    },
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
assert.equal(manifest.tokenizer?.addBosToken, true);
assert.equal(manifest.tokenizer?.addEosToken, true);

const sequencePostProcessorModel = {
  ...model,
  tokenizerConfig: {},
  tokenizerJson: {
    post_processor: {
      type: 'Sequence',
      processors: [
        {
          type: 'ByteLevel',
          add_prefix_space: false,
          trim_offsets: false,
          use_regex: false,
        },
        {
          type: 'TemplateProcessing',
          single: [
            { Sequence: { id: 'A', type_id: 0 } },
            { SpecialToken: { id: '<|endoftext|>', type_id: 0 } },
          ],
          special_tokens: {
            '<|endoftext|>': {
              id: '<|endoftext|>',
              ids: [1],
              tokens: ['<|endoftext|>'],
            },
          },
        },
      ],
    },
    model: {
      vocab: {
        a: 0,
        '<|endoftext|>': 1,
      },
    },
  },
};

const sequencePostProcessorManifest = createManifest(
  'bundled-tokenizer-sequence-post-processor-test',
  sequencePostProcessorModel,
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

assert.equal(sequencePostProcessorManifest.tokenizer?.addBosToken, undefined);
assert.equal(sequencePostProcessorManifest.tokenizer?.addEosToken, true);

console.log('core-bundled-tokenizer-flags.test: ok');
