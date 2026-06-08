import assert from 'node:assert/strict';
import { parseModelConfigFromManifest } from '../../src/inference/pipelines/text/config.js';
import { createExecutionV1Session } from '../helpers/execution-v1-fixtures.js';

function createBaseManifest() {
  return {
    modelId: 'gemma4-12b-encoder-free-contract',
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
      session: createExecutionV1Session(),
    },
  };
}

// 1. Validate vision config parsing with depth 0
{
  const manifest = {
    ...createBaseManifest(),
    image_token_id: 99,
    config: {
      vision_config: {
        vision_architecture: 'gemma4',
        depth: 0,
        hidden_size: 4096,
        patch_size: 16,
        pooling_kernel_size: 1,
        position_embedding_size: 1024,
        default_output_length: 280,
        eps: 1e-6,
        hidden_activation: 'gelu_pytorch_tanh',
        use_clipped_linears: true,
      },
    },
  };

  const parsed = parseModelConfigFromManifest(manifest);
  assert.ok(parsed.visionConfig);
  assert.equal(parsed.visionConfig.depth, 0);
  assert.equal(parsed.visionConfig.hiddenSize, 4096);
  assert.equal(parsed.visionConfig.ropeTheta, 10000); // Check default fallback
}

// 2. Validate audio config parsing with depth 0
{
  const manifest = {
    ...createBaseManifest(),
    audio_token_id: 77,
    config: {
      audio_config: {
        audio_architecture: 'gemma4',
        num_hidden_layers: 0,
        hidden_size: 4096,
        output_proj_dims: 640,
        use_clipped_linears: true,
      },
    },
  };

  const parsed = parseModelConfigFromManifest(manifest);
  assert.ok(parsed.audioConfig);
  assert.equal(parsed.audioConfig.depth, 0);
  assert.equal(parsed.audioConfig.hiddenSize, 4096);
  assert.equal(parsed.audioConfig.outputProjDims, 640);
  assert.equal(parsed.audioConfig.convKernelSize, 1); // Check default fallback
}

// 3. Validate manifest-owned prefill token chunk size is parsed into session settings
{
  const manifest = createBaseManifest();
  manifest.inference.session.prefillTokenChunkSize = 64;

  const parsed = parseModelConfigFromManifest(manifest);
  assert.equal(parsed.sessionSettings.prefillTokenChunkSize, 64);
}

// 4. Validate invalid manifest-owned prefill token chunk size fails fast
{
  const manifest = createBaseManifest();
  manifest.inference.session.prefillTokenChunkSize = 0;

  assert.throws(
    () => parseModelConfigFromManifest(manifest),
    /prefillTokenChunkSize/
  );
}

console.log('gemma4-12b-encoder-free-contract.test: ok');
