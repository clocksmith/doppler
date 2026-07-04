import assert from 'node:assert/strict';
import test from 'node:test';
import { createExecutionV1Session } from '../helpers/execution-v1-fixtures.js';
import diffusionGemmaConfig from '../../src/config/conversion/diffusiongemma/diffusiongemma-26b-a4b-it-q4k-ehf16-af16.json' with { type: 'json' };

const {
  parseModelConfigFromManifest,
  validateRequiredInferenceFields,
} = await import('../../src/inference/pipelines/text/config.js');
const {
  expandExecutionV1,
} = await import('../../src/config/schema/execution-v1.schema.js');
const {
  resolvePrefixEmbeddingOverrideTransitionDeclaredBy,
} = await import('../../src/inference/pipelines/text/generator-prefill-helpers.js');

function createDiffusionGemmaManifest(branchMode = 'dense_plus_moe') {
  return {
    modelId: 'diffusiongemma-config-contract',
    modelType: 'diffusion_gemma',
    quantization: 'Q4_K_M',
    architecture: {
      numLayers: 30,
      hiddenSize: 2816,
      intermediateSize: 2112,
      numAttentionHeads: 16,
      numKeyValueHeads: 8,
      numGlobalKeyValueHeads: 2,
      headDim: 256,
      globalHeadDim: 512,
      vocabSize: 262144,
      maxSeqLen: 262144,
      ropeTheta: 1000000,
    },
    eos_token_id: [1, 106, 50],
    moeConfig: {
      numExperts: 128,
      numExpertsPerToken: 8,
      expertFormat: 'gemma4',
      expertIntermediateSize: 704,
    },
    inference: {
      attention: {
        queryPreAttnScalar: 1,
        queryKeyNorm: true,
        valueNorm: true,
        attentionBias: false,
        causal: true,
        slidingWindow: 1024,
        attnLogitSoftcapping: null,
        attentionOutputGate: false,
      },
      normalization: {
        rmsNormWeightOffset: false,
        rmsNormEps: 1e-6,
        postAttentionNorm: true,
        preFeedforwardNorm: true,
        postFeedforwardNorm: true,
      },
      ffn: {
        activation: 'gelu',
        gatedActivation: true,
        branchMode,
        useDoubleWideMlp: false,
        swigluLimit: null,
      },
      rope: {
        ropeTheta: 1000000,
        ropeLocalTheta: 10000,
        ropeInterleaved: false,
        mropeInterleaved: false,
        mropeSection: null,
        partialRotaryFactor: 0.25,
        ropeLocalPartialRotaryFactor: null,
        ropeFrequencyBaseDim: 512,
        ropeLocalFrequencyBaseDim: null,
        ropeScalingType: 'proportional',
        ropeScalingFactor: 1,
        ropeLocalScalingType: null,
        ropeLocalScalingFactor: 1,
        yarnBetaFast: null,
        yarnBetaSlow: null,
        yarnOriginalMaxPos: null,
        ropeLocalYarnBetaFast: null,
        ropeLocalYarnBetaSlow: null,
        ropeLocalYarnOriginalMaxPos: null,
        longropeShortFactor: null,
        longropeLongFactor: null,
        longropeOriginalMaxPos: null,
      },
      output: {
        tieWordEmbeddings: true,
        scaleEmbeddings: true,
        embeddingTranspose: false,
        finalLogitSoftcapping: 30,
        embeddingVocabSize: null,
        embeddingPostprocessor: null,
        embeddingScale: null,
        logitInputScale: 1,
      },
      layerPattern: {
        type: 'custom',
        globalPattern: null,
        period: null,
        offset: null,
        layerTypes: Array.from({ length: 30 }, (_, index) => (index + 1) % 6 === 0 ? 'full_attention' : 'sliding_attention'),
        residualBranchScale: 1,
      },
      chatTemplate: {
        type: 'gemma4',
        enabled: true,
      },
      pipeline: null,
      session: createExecutionV1Session(),
      execution: null,
    },
  };
}

test('DiffusionGemma dense_plus_moe branch keeps dense intermediate width unchanged', () => {
  const manifest = createDiffusionGemmaManifest();
  manifest.inference.diffusionGemma = {
    decoderCacheMode: 'encoder_kv_readonly_canvas_concat',
  };
  const parsed = parseModelConfigFromManifest(manifest);

  assert.equal(parsed.ffnBranchMode, 'dense_plus_moe');
  assert.equal(parsed.useDoubleWideMlp, false);
  assert.equal(parsed.intermediateSize, 2112);
  assert.equal(parsed.maxIntermediateSize, 2112);
  assert.equal(parsed.moeExpertIntermediateSize, 704);
  assert.equal(parsed.numExperts, 128);
  assert.equal(parsed.moeTopK, 8);
  assert.equal(parsed.diffusionGemma?.decoderCacheMode, 'encoder_kv_readonly_canvas_concat');
});

test('invalid FFN branch mode fails manifest inference validation', () => {
  const manifest = createDiffusionGemmaManifest('dense-and-moe');

  assert.throws(
    () => validateRequiredInferenceFields(manifest.inference, manifest.modelId),
    /ffn\.branchMode must be one of: auto, dense, moe, dense_plus_moe/
  );
});

test('DiffusionGemma declares explicit prefill override f32 to f16 cast', () => {
  const expanded = expandExecutionV1(diffusionGemmaConfig.execution);
  const castStep = expanded.find((step) => step.section === 'preLayer' && step.op === 'cast');

  assert.ok(castStep, 'expected pre-layer cast step');
  assert.equal(castStep.kernel, 'cast_f32_to_f16.wgsl');
  assert.equal(castStep.fromDtype, 'f32');
  assert.equal(castStep.toDtype, 'f16');
  assert.equal(
    resolvePrefixEmbeddingOverrideTransitionDeclaredBy({
      resolvedSteps: { all: expanded },
    }),
    'explicit_cast_step'
  );
});
