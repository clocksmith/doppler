import assert from 'node:assert/strict';

import { parseModelConfigFromManifest } from '../../src/inference/pipelines/text/config.js';
import { createExecutionV1Session } from '../helpers/execution-v1-fixtures.js';

function createManifest(visionConfig) {
  return {
    modelId: 'zai-glm-ocr-browser-candidate',
    modelType: 'glm_ocr',
    quantization: 'q4k',
    eos_token_id: [59246, 59253],
    image_token_id: 59280,
    architecture: {
      numLayers: 16,
      hiddenSize: 1536,
      intermediateSize: 4608,
      numAttentionHeads: 16,
      numKeyValueHeads: 8,
      headDim: 128,
      vocabSize: 59392,
      maxSeqLen: 131072,
    },
    inference: {
      attention: {
        queryPreAttnScalar: 128,
        attnLogitSoftcapping: null,
        slidingWindow: null,
        queryKeyNorm: false,
        valueNorm: false,
        causal: true,
        attentionBias: false,
        attentionOutputGate: false,
      },
      normalization: {
        rmsNormEps: 1e-5,
        rmsNormWeightOffset: false,
        postAttentionNorm: true,
        preFeedforwardNorm: true,
        postFeedforwardNorm: true,
      },
      ffn: {
        activation: 'silu',
        gatedActivation: true,
        branchMode: 'auto',
        useDoubleWideMlp: false,
        swigluLimit: null,
      },
      rope: {
        ropeTheta: 10000,
        ropeLocalTheta: null,
        ropeInterleaved: true,
        mropeInterleaved: true,
        mropeSection: [16, 24, 24],
        partialRotaryFactor: 1,
        ropeLocalPartialRotaryFactor: null,
        ropeFrequencyBaseDim: 128,
        ropeLocalFrequencyBaseDim: null,
        ropeScalingType: null,
        ropeScalingFactor: 1,
        ropeLocalScalingType: null,
        ropeLocalScalingFactor: 1,
        yarnBetaFast: null,
        yarnBetaSlow: null,
        yarnOriginalMaxPos: null,
        longropeShortFactor: null,
        longropeLongFactor: null,
        longropeOriginalMaxPos: null,
        ropeLocalYarnBetaFast: null,
        ropeLocalYarnBetaSlow: null,
        ropeLocalYarnOriginalMaxPos: null,
      },
      output: {
        finalLogitSoftcapping: null,
        tieWordEmbeddings: false,
        scaleEmbeddings: false,
        embeddingScale: null,
        logitInputScale: 1,
        embeddingTranspose: false,
        embeddingVocabSize: 59392,
        embeddingPostprocessor: null,
      },
      layerPattern: {
        type: 'global',
        globalPattern: 'attention',
        period: null,
        offset: null,
        layerTypes: null,
        residualBranchScale: 1,
      },
      chatTemplate: { type: 'glmocr', enabled: true, thinking: null },
      pipeline: null,
      session: createExecutionV1Session(),
    },
    config: { vision_config: visionConfig },
  };
}

const pinnedVisionConfig = {
  vision_architecture: 'glmocr',
  depth: 24,
  hidden_size: 1024,
  intermediate_size: 4096,
  num_heads: 16,
  out_hidden_size: 1536,
  patch_size: 14,
  spatial_merge_size: 2,
  temporal_patch_size: 2,
  rms_norm_eps: 1e-5,
  hidden_act: 'silu',
  rope_theta: 10000,
  min_pixels: 12544,
  max_pixels: 9633792,
  default_output_length: 6144,
  in_channels: 3,
  merger_intermediate_size: 4608,
  downsample_kernel_size: 2,
  normalization: {
    mean: [0.48145466, 0.4578275, 0.40821073],
    std: [0.26862954, 0.26130258, 0.27577711],
  },
};

const parsed = parseModelConfigFromManifest(createManifest(pinnedVisionConfig));
assert.equal(parsed.visionConfig.visionArchitecture, 'glmocr');
assert.equal(parsed.visionConfig.headDim, 64);
assert.equal(parsed.visionConfig.outHiddenSize, 1536);
assert.equal(parsed.visionConfig.mergerIntermediateSize, 4608);
assert.equal(parsed.visionConfig.downsampleKernelSize, 2);
assert.equal(parsed.visionConfig.defaultOutputLength, 6144);
assert.equal(parsed.chatTemplateType, 'glmocr');
assert.equal(parsed.chatTemplateThinking, null);

assert.throws(
  () => parseModelConfigFromManifest(createManifest({
    ...pinnedVisionConfig,
    rope_theta: undefined,
  })),
  /vision_config\.rope_theta/
);

assert.throws(
  () => parseModelConfigFromManifest(createManifest({
    ...pinnedVisionConfig,
    hidden_act: 'gelu',
  })),
  /requires "silu"/
);

console.log('glmocr-vision-config-contract.test: ok');
