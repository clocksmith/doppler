import assert from 'node:assert/strict';

import { createManifest } from '../../src/converter/core.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';

const manifest = createManifest(
  'gemma4-multimodal-token-ids',
  {
    modelId: 'gemma4-multimodal-token-ids',
    modelType: 'gemma4',
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
      model_type: 'gemma4',
      image_token_id: 258880,
      audio_token_id: 258881,
      video_token_id: 258884,
      vision_config: {
        hidden_size: 768,
      },
      audio_config: {
        hidden_size: 1024,
      },
      text_config: {
        model_type: 'gemma4_text',
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
    },
    tensors: [],
  },
  [
    {
      index: 0,
      filename: 'shard_00000.bin',
      size: 16,
      hash: 'hash',
      offset: 0,
    },
  ],
  {
    'model.language_model.embed_tokens.weight': {
      shard: 0,
      offset: 0,
      size: 16,
      shape: [2, 4],
      dtype: 'F16',
      role: 'embedding',
    },
  },
  {
    source: 'unit-test',
    modelType: 'gemma4',
    quantization: 'F16',
    hashAlgorithm: 'sha256',
    inference: { ...DEFAULT_MANIFEST_INFERENCE },
    manifestConfig: {
      visionConfig: {
        vision_architecture: 'gemma4',
        hidden_size: 768,
      },
      audioConfig: {
        audio_architecture: 'gemma4',
        hidden_size: 1024,
      },
    },
  }
);

assert.equal(manifest.image_token_id, 258880);
assert.equal(manifest.audio_token_id, 258881);
assert.equal(manifest.video_token_id, 258884);
assert.deepEqual(manifest.config, {
  vision_config: {
    vision_architecture: 'gemma4',
    hidden_size: 768,
  },
  audio_config: {
    audio_architecture: 'gemma4',
    hidden_size: 1024,
  },
});

console.log('core-manifest-multimodal-token-ids.test: ok');
