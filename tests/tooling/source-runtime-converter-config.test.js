import assert from 'node:assert/strict';

import {
  createSourceRuntimeConverterConfig,
  createSourceRuntimeInference,
} from '../../src/tooling/source-runtime-converter-config.js';

const inference = createSourceRuntimeInference({
  model_type: 'gemma4',
  hidden_activation: null,
  sliding_window: null,
  final_logit_softcapping: null,
  use_double_wide_mlp: null,
  attention_bias: null,
  tie_word_embeddings: true,
  scale_embeddings: null,
  text_config: {
    model_type: 'gemma4_text',
    hidden_activation: 'gelu_pytorch_tanh',
    sliding_window: 512,
    final_logit_softcapping: 30,
    use_double_wide_mlp: true,
    attention_bias: false,
    layer_types: [
      'sliding_attention',
      'sliding_attention',
      'full_attention',
      'sliding_attention',
      'full_attention',
    ],
    global_head_dim: 512,
    head_dim: 256,
    rope_parameters: {
      full_attention: {
        rope_theta: 1000000,
        rope_type: 'proportional',
        partial_rotary_factor: 0.25,
      },
      sliding_attention: {
        rope_theta: 10000,
        rope_type: 'default',
      },
    },
  },
});

assert.equal(inference.attention.queryPreAttnScalar, 1);
assert.equal(inference.attention.queryKeyNorm, true);
assert.equal(inference.attention.valueNorm, true);
assert.equal(inference.attention.slidingWindow, 512);
assert.equal(inference.attention.attentionBias, false);

assert.equal(inference.normalization.rmsNormWeightOffset, false);
assert.equal(inference.normalization.postAttentionNorm, true);
assert.equal(inference.normalization.preFeedforwardNorm, true);
assert.equal(inference.normalization.postFeedforwardNorm, true);

assert.equal(inference.ffn.activation, 'gelu');
assert.equal(inference.ffn.useDoubleWideMlp, true);

assert.equal(inference.rope.ropeTheta, 1000000);
assert.equal(inference.rope.ropeLocalTheta, 10000);
assert.equal(inference.rope.partialRotaryFactor, 0.25);
assert.equal(inference.rope.ropeFrequencyBaseDim, 512);
assert.equal(inference.rope.ropeLocalFrequencyBaseDim, null);

assert.equal(inference.output.tieWordEmbeddings, true);
assert.equal(inference.output.scaleEmbeddings, true);
assert.equal(inference.output.finalLogitSoftcapping, 30);

assert.equal(inference.layerPattern.type, 'custom');
assert.deepEqual(inference.layerPattern.layerTypes, [
  'sliding_attention',
  'sliding_attention',
  'full_attention',
  'sliding_attention',
  'full_attention',
]);

assert.equal(inference.chatTemplate.type, 'gemma4');
assert.equal(inference.chatTemplate.enabled, true);

const converterConfig = createSourceRuntimeConverterConfig({
  modelId: 'gemma4-source-runtime-fixture',
  rawConfig: {
    model_type: 'gemma4',
    text_config: {
      model_type: 'gemma4_text',
    },
    vision_config: {
      model_type: 'gemma4_vision',
      hidden_size: 768,
    },
    audio_config: {
      model_type: 'gemma4_audio',
      hidden_size: 1024,
    },
  },
});

assert.equal(converterConfig.manifest.visionConfig?.vision_architecture, 'gemma4');
assert.equal(converterConfig.manifest.audioConfig?.audio_architecture, 'gemma4');

console.log('source-runtime-converter-config.test: ok');
