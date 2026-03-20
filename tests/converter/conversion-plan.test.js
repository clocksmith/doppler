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

console.log('conversion-plan.test: ok');
