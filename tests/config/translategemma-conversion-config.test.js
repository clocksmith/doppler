import assert from 'node:assert/strict';
import fs from 'node:fs';

const { resolveConversionPlan } = await import('../../src/converter/conversion-plan.js');

const converterConfig = JSON.parse(
  fs.readFileSync(
    'tools/configs/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json',
    'utf8'
  )
);

assert.equal(converterConfig.quantization?.computePrecision, 'f32');
assert.equal(converterConfig.inference?.defaultKernelPath, 'gemma3-q4k-dequant-f32a-online');
assert.equal(converterConfig.inference?.schema, 'doppler.execution/v0');
assert.equal(converterConfig.inference?.sessionDefaults?.kvcache?.layout, 'contiguous');
assert.equal(converterConfig.inference?.sessionDefaults?.decodeLoop, null);

const plan = resolveConversionPlan({
  rawConfig: {
    model_type: 'translategemma',
    architectures: ['Gemma3ForConditionalGeneration'],
    hidden_size: 2560,
    intermediate_size: 10240,
    num_hidden_layers: 34,
    num_attention_heads: 8,
    num_key_value_heads: 4,
    rope_parameters: {
      full_attention: {
        rope_type: 'linear',
        factor: 8.0,
        rope_theta: 1000000,
      },
      sliding_attention: {
        rope_type: 'default',
        rope_theta: 10000,
      },
    },
  },
  tensors: [
    { name: 'model.embed_tokens.weight', dtype: 'F16' },
    { name: 'model.layers.0.self_attn.q_proj.weight', dtype: 'F16' },
  ],
  converterConfig,
  modelKind: 'transformer',
  architectureHint: 'Gemma3ForConditionalGeneration',
  architectureConfig: { headDim: 256 },
});

assert.equal(plan.quantizationInfo?.variantTag, 'q4k-ehf16-af32');
assert.equal(plan.manifestInference?.defaultKernelPath, 'gemma3-q4k-dequant-f32a-online');
assert.equal(plan.manifestInference?.sessionDefaults?.compute?.defaults?.activationDtype, 'f32');
assert.equal(plan.manifestInference?.sessionDefaults?.compute?.defaults?.mathDtype, 'f32');
assert.equal(plan.manifestInference?.sessionDefaults?.compute?.defaults?.accumDtype, 'f32');
assert.equal(plan.manifestInference?.sessionDefaults?.compute?.defaults?.outputDtype, 'f32');
assert.equal(plan.manifestInference?.sessionDefaults?.kvcache?.layout, 'contiguous');
assert.equal(plan.manifestInference?.sessionDefaults?.decodeLoop, null);
assert.ok(Array.isArray(plan.manifestInference?.execution?.steps));
assert.ok(plan.manifestInference.execution.steps.length > 0);

console.log('translategemma-conversion-config.test: ok');
