import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const {
  inferSourceWeightQuantization,
  resolveConversionPlan,
  resolveConvertedModelId,
} = await import('../../src/converter/conversion-plan.js');

const converterConfig = createConverterConfig();

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
