import assert from 'node:assert/strict';

import { parseModelConfigFromManifest } from '../../src/inference/pipelines/text/config.js';

function createBaseManifest() {
  return {
    modelId: 'multimodal-config-contract',
    modelType: 'transformer',
    quantization: 'q4k',
    eos_token_id: 1,
    architecture: {
      numLayers: 2,
      hiddenSize: 256,
      intermediateSize: 512,
      numAttentionHeads: 4,
      numKeyValueHeads: 4,
      headDim: 64,
      vocabSize: 1024,
      maxSeqLen: 512,
    },
    inference: {
      attention: {
        queryPreAttnScalar: 64,
        attnLogitSoftcapping: null,
        slidingWindow: null,
        queryKeyNorm: false,
        valueNorm: false,
        causal: true,
        attentionBias: false,
        attentionOutputGate: false,
      },
      normalization: {
        rmsNormEps: 1e-6,
        rmsNormWeightOffset: false,
        postAttentionNorm: false,
        preFeedforwardNorm: false,
        postFeedforwardNorm: false,
      },
      ffn: {
        activation: 'silu',
        gatedActivation: true,
        useDoubleWideMlp: false,
        swigluLimit: null,
      },
      rope: {
        ropeTheta: 10000,
        ropeLocalTheta: null,
        ropeInterleaved: false,
        mropeInterleaved: false,
        mropeSection: null,
        partialRotaryFactor: null,
        ropeLocalPartialRotaryFactor: null,
        ropeFrequencyBaseDim: null,
        ropeLocalFrequencyBaseDim: null,
        ropeScalingType: null,
        ropeScalingFactor: 1,
        ropeLocalScalingType: null,
        ropeLocalScalingFactor: 1,
        yarnBetaFast: null,
        yarnBetaSlow: null,
        yarnOriginalMaxPos: null,
        ropeLocalYarnBetaFast: null,
        ropeLocalYarnBetaSlow: null,
        ropeLocalYarnOriginalMaxPos: null,
      },
      output: {
        finalLogitSoftcapping: null,
        tieWordEmbeddings: false,
        scaleEmbeddings: false,
        embeddingTranspose: false,
        embeddingVocabSize: 1024,
        embeddingPostprocessor: null,
      },
      layerPattern: {
        type: 'global',
        globalPattern: 'attention',
        period: null,
        offset: null,
        layerTypes: null,
      },
      chatTemplate: {
        type: null,
        enabled: false,
      },
      pipeline: null,
    },
  };
}

assert.throws(
  () => parseModelConfigFromManifest({
    ...createBaseManifest(),
    image_token_id: 99,
    config: {
      vision_config: {
        depth: 12,
        hidden_size: 768,
        intermediate_size: 3072,
        num_heads: 12,
        num_key_value_heads: 12,
        head_dim: 64,
        out_hidden_size: 1024,
        patch_size: 16,
        pooling_kernel_size: 2,
        spatial_merge_size: 2,
        temporal_patch_size: 2,
        eps: 1e-6,
        hidden_activation: 'gelu_pytorch_tanh',
        min_pixels: 65536,
        max_pixels: 16777216,
        normalization: {
          mean: [0.5, 0.5, 0.5],
          std: [0.5, 0.5, 0.5],
        },
      },
    },
  }),
  /vision_config\.vision_architecture/
);

assert.throws(
  () => parseModelConfigFromManifest({
    ...createBaseManifest(),
    image_token_id: 99,
    config: {
      vision_config: {
        vision_architecture: 'qwen3vl',
        depth: 12,
        hidden_size: 768,
        intermediate_size: 3072,
        num_heads: 12,
        num_key_value_heads: 12,
        head_dim: 64,
        out_hidden_size: 1024,
        patch_size: 16,
        pooling_kernel_size: 2,
        spatial_merge_size: 2,
        temporal_patch_size: 2,
        eps: 1e-6,
        hidden_activation: 'gelu_pytorch_tanh',
        min_pixels: 65536,
        max_pixels: 16777216,
      },
    },
  }),
  /vision_config\.normalization/
);

assert.throws(
  () => parseModelConfigFromManifest({
    ...createBaseManifest(),
    audio_token_id: 77,
    config: {
      audio_config: {
        num_hidden_layers: 12,
        hidden_size: 1024,
        num_attention_heads: 8,
        conv_kernel_size: 5,
        subsampling_conv_channels: [128, 32],
        output_proj_dims: 1536,
        attention_context_left: 13,
        attention_context_right: 0,
        attention_chunk_size: 12,
        attention_logit_cap: 50,
        attention_invalid_logits_value: -1e9,
        residual_weight: 0.5,
        rms_norm_eps: 1e-6,
        hidden_act: 'silu',
        use_clipped_linears: true,
      },
    },
  }),
  /audio_config\.audio_architecture/
);

console.log('multimodal-config-contract.test: ok');
