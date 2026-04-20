import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const {
  inferSourceWeightQuantization,
  resolveConversionPlan,
  resolveConvertedModelId,
} = await import('../../src/converter/conversion-plan.js');
const { resolveEffectiveQuantizationInfo } = await import('../../src/converter/quantization-info.js');

const converterConfig = createConverterConfig();

{
  const reconciled = resolveEffectiveQuantizationInfo(
    {
      weights: 'f16',
      embeddings: 'f32',
      compute: 'f16',
      variantTag: 'f16-ehf32',
    },
    [
      { name: 'embed_tokens.weight', role: 'embedding', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.q_proj.weight', role: 'matmul', dtype: 'F16' },
    ]
  );
  assert.equal(reconciled.weights, 'f16');
  assert.equal(reconciled.embeddings, 'f16');
  assert.equal(reconciled.variantTag, 'f16');
}

{
  // Legacy (non-v1) configs are rejected with an actionable error.
  assert.throws(
    () => resolveConversionPlan({
      rawConfig: { diffusion: { layout: 'flux' } },
      tensors: [
        { name: 'transformer.block.weight', dtype: 'F16' },
        { name: 'text_encoder.embed.weight', dtype: 'F16' },
      ],
      converterConfig,
      modelKind: 'diffusion',
    }),
    /v1 format/
  );
}

{
  // Bare converterConfig (no execution.kernels) is also rejected.
  assert.throws(
    () => resolveConversionPlan({
      rawConfig: {},
      tensors: [],
      converterConfig: createConverterConfig(),
    }),
    /v1 format/
  );
}

{
  assert.throws(
    () => inferSourceWeightQuantization([
      { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.k_proj.weight', dtype: 'F32' },
    ]),
    /Ambiguous source weight dtypes/
  );
}

{
  // Legacy configs with model-kind hints but no execution.kernels are also rejected.
  assert.throws(
    () => resolveConversionPlan({
      rawConfig: {
        model_type: 'gemma3',
        architectures: ['Gemma3ForCausalLM'],
        hidden_size: 3072,
        num_attention_heads: 16,
        num_hidden_layers: 28,
      },
      tensors: [
        { name: 'model.embed_tokens.weight', dtype: 'F16' },
        { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F16' },
      ],
      converterConfig,
      modelKind: 'transformer',
      architectureHint: 'Gemma3ForCausalLM',
      architectureConfig: { headDim: 192 },
    }),
    /v1 format/
  );
}

{
  const modelId = resolveConvertedModelId({
    explicitModelId: null,
    converterConfig,
    detectedModelId: 'Flux.2-Klein-4B',
    quantizationInfo: { variantTag: 'f16' },
  });
  assert.equal(typeof modelId, 'string');
  assert.ok(modelId.includes('flux-2-klein-4b'));
}

{
  const qwenConfig = {
    output: {
      modelBaseId: 'qwen-3-5-0-8b-q4k-ehaf16',
    },
  };
  const modelId = resolveConvertedModelId({
    explicitModelId: null,
    converterConfig: qwenConfig,
    detectedModelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    quantizationInfo: {
      variantTag: 'q4k-ehf16-af32',
    },
  });
  assert.equal(modelId, 'qwen-3-5-0-8b-q4k-ehaf16');
}

{
  const gemma4Int4PleConfig = {
    output: {
      modelBaseId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    },
  };
  const modelId = resolveConvertedModelId({
    explicitModelId: null,
    converterConfig: gemma4Int4PleConfig,
    detectedModelId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    quantizationInfo: {
      variantTag: 'q4k-ehf16-af32-vf16-audiof16-pf16',
    },
  });
  assert.equal(modelId, 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple');
}

{
  const gemma4Config = createConverterConfig({
    quantization: {
      weights: 'f16',
      embeddings: 'f16',
      projector: 'f16',
      computePrecision: 'f16',
      q4kLayout: 'row',
    },
    output: {
      modelBaseId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      textOnly: false,
    },
    inference: {
      attention: {
        queryPreAttnScalar: 256,
        attnLogitSoftcapping: null,
        slidingWindow: 512,
        queryKeyNorm: true,
        valueNorm: false,
        causal: true,
        attentionBias: false,
        attentionOutputGate: false,
      },
      normalization: {
        rmsNormEps: 1e-6,
        rmsNormWeightOffset: false,
        postAttentionNorm: true,
        preFeedforwardNorm: true,
        postFeedforwardNorm: true,
      },
      ffn: {
        activation: 'gelu',
        gatedActivation: true,
        useDoubleWideMlp: true,
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
        finalLogitSoftcapping: 30,
        tieWordEmbeddings: true,
        scaleEmbeddings: true,
        embeddingTranspose: false,
        embeddingVocabSize: null,
        embeddingPostprocessor: null,
      },
      layerPattern: {
        type: 'every_n',
        globalPattern: null,
        period: 5,
        offset: 4,
        layerTypes: null,
      },
      chatTemplate: {
        type: 'gemma4',
        enabled: true,
      },
    },
    session: {
      compute: {
        defaults: {
          activationDtype: 'f16',
          mathDtype: 'f16',
          accumDtype: 'f16',
          outputDtype: 'f16',
        },
      },
      kvcache: null,
      decodeLoop: null,
    },
    execution: {
      kernels: {
        embed: { kernel: 'gather_f16.wgsl', entry: 'main', digest: 'sha256:0000000000000000000000000000000000000000000000000000000000000000' },
      },
      preLayer: [],
      decode: [],
      prefill: [],
      postLayer: [],
      policies: {
        unsupportedPrecision: 'error',
        dtypeTransition: 'require_cast_step',
        unresolvedKernel: 'error',
      },
    },
  });

  const plan = resolveConversionPlan({
    rawConfig: { model_type: 'gemma4' },
    tensors: [
      { name: 'model.language_model.embed_tokens.weight', dtype: 'F16' },
      { name: 'model.embed_vision.embedding_projection.weight', dtype: 'F16' },
    ],
    converterConfig: gemma4Config,
  });
  assert.equal(plan.quantizationInfo?.projector, 'f16');

  const invalidGemma4Config = structuredClone(gemma4Config);
  delete invalidGemma4Config.inference.attention.valueNorm;
  assert.throws(
    () => resolveConversionPlan({
      rawConfig: { model_type: 'gemma4' },
      tensors: [
        { name: 'model.language_model.embed_tokens.weight', dtype: 'F16' },
        { name: 'model.embed_vision.embedding_projection.weight', dtype: 'F16' },
      ],
      converterConfig: invalidGemma4Config,
    }),
    /attention\.valueNorm is required/
  );
}

console.log('conversion-plan.test: ok');
