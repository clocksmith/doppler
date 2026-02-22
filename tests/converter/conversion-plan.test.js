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
      variantTag: 'wf16-ef32',
    },
    [
      { name: 'embed_tokens.weight', role: 'embedding', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.q_proj.weight', role: 'matmul', dtype: 'F16' },
    ]
  );
  assert.equal(reconciled.weights, 'f16');
  assert.equal(reconciled.embeddings, 'f16');
  assert.equal(reconciled.variantTag, 'wf16');
}

{
  const plan = resolveConversionPlan({
    rawConfig: { diffusion: { layout: 'flux' } },
    tensors: [
      { name: 'transformer.block.weight', dtype: 'F16' },
      { name: 'text_encoder.embed.weight', dtype: 'F16' },
    ],
    converterConfig,
    modelKind: 'diffusion',
  });
  assert.equal(plan.modelType, 'diffusion');
  assert.equal(plan.presetId, 'diffusion');
  assert.equal(plan.manifestInference?.presetId, 'diffusion');
}

{
  const overrideConfig = createConverterConfig({
    inference: {
      defaultKernelPath: 'gemma3-f16-fused-f32a-online',
    },
  });
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3_text',
      architectures: ['Gemma3ForCausalLM'],
      hidden_size: 640,
      num_attention_heads: 4,
      num_hidden_layers: 18,
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F16' },
    ],
    converterConfig: overrideConfig,
    modelKind: 'transformer',
    architectureHint: 'Gemma3ForCausalLM',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.manifestInference?.defaultKernelPath, 'gemma3-f16-fused-f32a-online');
}

{
  const invalidOverrideConfig = createConverterConfig({
    inference: {
      defaultKernelPath: 42,
    },
  });
  assert.throws(
    () => resolveConversionPlan({
      rawConfig: {
        model_type: 'gemma3_text',
        architectures: ['Gemma3ForCausalLM'],
        hidden_size: 640,
        num_attention_heads: 4,
        num_hidden_layers: 18,
      },
      tensors: [
        { name: 'model.embed_tokens.weight', dtype: 'F16' },
        { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F16' },
      ],
      converterConfig: invalidOverrideConfig,
      modelKind: 'transformer',
      architectureHint: 'Gemma3ForCausalLM',
      architectureConfig: { headDim: 256 },
    }),
    /converterConfig\.inference\.defaultKernelPath must be a string/
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
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3',
      architectures: ['Gemma3ForCausalLM'],
      vocab_size: 32000,
      hidden_size: 3072,
      intermediate_size: 24576,
      num_hidden_layers: 28,
      num_attention_heads: 16,
      max_position_embeddings: 8192,
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'F16' },
      { name: 'lm_head.weight', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F16' },
    ],
    converterConfig,
    modelKind: 'transformer',
    architectureHint: 'Gemma3ForCausalLM',
    architectureConfig: { headDim: 192 },
  });
  assert.equal(plan.modelType, 'transformer');
  assert.equal(typeof plan.presetId, 'string');
  assert.equal(typeof plan.manifestInference?.defaultKernelPath, 'string');
}

{
  // Gemma 3 270M(-it) style config: model_type=gemma3_text, BF16 source,
  // no lm_head tensor (tied embeddings).
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3_text',
      architectures: ['Gemma3ForCausalLM'],
      vocab_size: 262144,
      hidden_size: 640,
      intermediate_size: 2048,
      num_hidden_layers: 18,
      num_attention_heads: 4,
      num_key_value_heads: 1,
      max_position_embeddings: 32768,
      rope_theta: 1000000.0,
      rope_local_base_freq: 10000.0,
      use_bidirectional_attention: false,
      layer_types: [
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'full_attention',
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'full_attention',
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'sliding_attention',
        'full_attention',
      ],
      attn_logit_softcapping: null,
      final_logit_softcapping: null,
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'BF16' },
      { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'BF16' },
      { name: 'model.layers.0.self_attn.k_proj.weight', dtype: 'BF16' },
      { name: 'model.layers.0.self_attn.v_proj.weight', dtype: 'BF16' },
      { name: 'model.layers.0.self_attn.o_proj.weight', dtype: 'BF16' },
      { name: 'model.layers.0.self_attn.q_norm.weight', dtype: 'BF16' },
      { name: 'model.layers.0.self_attn.k_norm.weight', dtype: 'BF16' },
      { name: 'model.layers.0.pre_feedforward_layernorm.weight', dtype: 'BF16' },
      { name: 'model.layers.0.post_feedforward_layernorm.weight', dtype: 'BF16' },
      { name: 'model.norm.weight', dtype: 'BF16' },
    ],
    converterConfig,
    modelKind: 'transformer',
    architectureHint: 'Gemma3ForCausalLM',
    architectureConfig: { headDim: 256 },
  });

  assert.equal(plan.presetId, 'gemma3');
  assert.equal(plan.sourceQuantization, 'f16');
  assert.equal(plan.quantizationInfo.weights, 'f16');
  assert.equal(plan.quantizationInfo.embeddings, 'f16');
  assert.equal(plan.quantizationInfo.compute, 'f16');
  assert.equal(plan.quantizationInfo.variantTag, 'wf16');
  assert.equal(plan.manifestInference?.defaultKernelPath, 'gemma3-f16-f32a');
  assert.equal(plan.manifestInference?.output?.tieWordEmbeddings, true);
  assert.equal(plan.manifestInference?.output?.scaleEmbeddings, true);
  assert.equal(plan.manifestInference?.attention?.queryKeyNorm, true);
  assert.equal(plan.manifestInference?.rope?.ropeTheta, 1000000);
  assert.equal(plan.manifestInference?.rope?.ropeLocalTheta, 10000);
  assert.equal(plan.manifestInference?.attention?.causal, true);
  assert.equal(plan.manifestInference?.layerPattern?.type, 'every_n');
  assert.equal(plan.manifestInference?.layerPattern?.period, 6);
  assert.equal(plan.manifestInference?.layerPattern?.offset, 5);
}

{
  const q4kConverterConfig = createConverterConfig({
    quantization: {
      weights: 'q4k',
      q4kLayout: 'row',
    },
  });
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3_text',
      architectures: ['Gemma3ForCausalLM'],
      hidden_size: 640,
      num_attention_heads: 4,
      num_hidden_layers: 18,
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'BF16' },
      { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'BF16' },
    ],
    converterConfig: q4kConverterConfig,
    modelKind: 'transformer',
    architectureHint: 'Gemma3ForCausalLM',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.quantizationInfo.weights, 'q4k');
  assert.equal(plan.quantizationInfo.layout, 'row');
  assert.equal(plan.quantizationInfo.variantTag, 'wq4k-ef16');
  assert.equal(plan.manifestInference?.defaultKernelPath, 'gemma3-q4k-dequant-f32a');
}

{
  const q4kF32ComputeConfig = createConverterConfig({
    quantization: {
      weights: 'q4k',
      q4kLayout: 'row',
      computePrecision: 'f32',
    },
  });
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3_text',
      architectures: ['Gemma3ForCausalLM'],
      hidden_size: 640,
      num_attention_heads: 4,
      num_hidden_layers: 18,
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'BF16' },
      { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'BF16' },
    ],
    converterConfig: q4kF32ComputeConfig,
    modelKind: 'transformer',
    architectureHint: 'Gemma3ForCausalLM',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.quantizationInfo.weights, 'q4k');
  assert.equal(plan.quantizationInfo.compute, 'f32');
  assert.equal(plan.quantizationInfo.layout, 'row');
  assert.equal(plan.manifestInference?.defaultKernelPath, 'gemma3-q4k-dequant-f32a');
}

{
  // Weak architecture hints: gemma3_text + bidirectional attention should still
  // resolve to embeddinggemma preset/config.
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3_text',
      use_bidirectional_attention: true,
      hidden_size: 768,
      num_attention_heads: 3,
      num_hidden_layers: 24,
      vocab_size: 262144,
      max_position_embeddings: 8192,
      rope_theta: 1000000.0,
      rope_local_base_freq: 10000.0,
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.k_proj.weight', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.v_proj.weight', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.o_proj.weight', dtype: 'F16' },
      { name: 'model.layers.0.mlp.gate_proj.weight', dtype: 'F16' },
      { name: 'model.layers.0.mlp.up_proj.weight', dtype: 'F16' },
      { name: 'model.layers.0.mlp.down_proj.weight', dtype: 'F16' },
      { name: 'model.norm.weight', dtype: 'F16' },
    ],
    converterConfig,
    modelKind: 'transformer',
    architectureHint: 'gemma3_text',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.presetId, 'embeddinggemma');
  assert.equal(plan.modelType, 'embedding');
  assert.equal(plan.manifestInference?.attention?.causal, false);
  assert.equal(plan.manifestInference?.defaultKernelPath, 'embeddinggemma-f16-f32a');
}

{
  // Gemma3TextModel architecture alone should not force embeddinggemma unless
  // bidirectional/model_type evidence is present.
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3_text',
      use_bidirectional_attention: false,
      hidden_size: 768,
      num_attention_heads: 3,
      num_hidden_layers: 24,
      vocab_size: 262144,
      max_position_embeddings: 8192,
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F16' },
      { name: 'lm_head.weight', dtype: 'F16' },
    ],
    converterConfig,
    modelKind: 'transformer',
    architectureHint: 'Gemma3TextModel',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.presetId, 'gemma3');
  assert.equal(plan.modelType, 'transformer');
}

{
  const q4kEmbeddingConfig = createConverterConfig({
    quantization: {
      weights: 'q4k',
      q4kLayout: 'row',
    },
  });
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3_text',
      use_bidirectional_attention: true,
      hidden_size: 768,
      num_attention_heads: 3,
      num_hidden_layers: 24,
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'F16' },
      { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F16' },
    ],
    converterConfig: q4kEmbeddingConfig,
    modelKind: 'transformer',
    architectureHint: 'gemma3_text',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.presetId, 'embeddinggemma');
  assert.equal(plan.quantizationInfo.weights, 'q4k');
  assert.equal(plan.quantizationInfo.layout, 'row');
  assert.equal(plan.manifestInference?.defaultKernelPath, 'embeddinggemma-q4k-dequant-f32a');
}

{
  // EmbeddingGemma with F32 source weights should still resolve an explicit
  // F32-activation kernel path (never fall back to runtime F16 defaults).
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3_text',
      use_bidirectional_attention: true,
      hidden_size: 768,
      num_attention_heads: 3,
      num_hidden_layers: 24,
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'F32' },
      { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F32' },
    ],
    converterConfig,
    modelKind: 'transformer',
    architectureHint: 'gemma3_text',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.presetId, 'embeddinggemma');
  assert.equal(plan.quantizationInfo.weights, 'f32');
  assert.equal(plan.quantizationInfo.embeddings, 'f32');
  assert.equal(plan.manifestInference?.defaultKernelPath, 'embeddinggemma-f32-f32a');
}

{
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'llama',
      architectures: ['LlamaForCausalLM'],
    },
    tensors: [
      { name: 'token_embd.weight', dtype: 'F16' },
      { name: 'output.weight', dtype: 'F16' },
      { name: 'blk.0.attn_q.weight', dtype: 'Q4_K' },
    ],
    converterConfig,
    sourceQuantization: 'q4k',
    modelKind: 'transformer',
    architectureHint: 'LlamaForCausalLM',
    architectureConfig: { headDim: 128 },
    presetOverride: 'llama3',
  });
  assert.equal(plan.quantizationInfo.embeddings, 'f16');
  assert.equal(plan.quantizationInfo.variantTag, 'wq4k-ef16');
}

{
  const modelId = resolveConvertedModelId({
    explicitModelId: null,
    converterConfig,
    detectedModelId: 'Flux.2-Klein-4B',
    quantizationInfo: { variantTag: 'wf16' },
  });
  assert.equal(typeof modelId, 'string');
  assert.ok(modelId.includes('flux-2-klein-4b'));
}

console.log('conversion-plan.test: ok');
