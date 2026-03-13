import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const {
  inferSourceWeightQuantization,
  resolveConversionPlan,
  resolveConvertedModelId,
  validateDefaultKernelPath,
} = await import('../../src/converter/conversion-plan.js');
const { buildManifestInference } = await import('../../src/converter/manifest-inference.js');
const { resolvePreset } = await import('../../src/config/loader.js');
const { resolveEffectiveQuantizationInfo } = await import('../../src/converter/quantization-info.js');
const { buildKernelRefFromKernelEntry } = await import('../../src/config/kernels/kernel-ref.js');

const converterConfig = createConverterConfig();
const embeddingComputeF32Config = createConverterConfig({
  quantization: {
    computePrecision: 'f32',
  },
});

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
  assert.equal(plan.manifestInference?.schema, null);
  assert.equal(plan.manifestInference?.execution, null);
}

{
  const overrideConfig = createConverterConfig({
    inference: {
      defaultKernelPath: 'gemma3-f16-fused-f16a-online',
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
  assert.equal(plan.manifestInference?.defaultKernelPath, 'gemma3-f16-fused-f16a-online');
  assert.equal(plan.manifestInference?.schema, 'doppler.execution/v0');
  assert.ok(Array.isArray(plan.manifestInference?.execution?.steps));
  assert.ok(plan.manifestInference.execution.steps.length > 0);
  assert.ok(plan.manifestInference.execution.steps[0].src);
  assert.ok(plan.manifestInference.execution.steps[0].dst);
  assert.ok(plan.manifestInference.execution.steps[0].kernelRef);
  assert.ok((plan.manifestInference.sessionDefaults?.compute?.kernelProfiles?.length ?? 0) > 0);
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
    () => validateDefaultKernelPath(
      { defaultKernelPath: 'gemma3-q4k-dequant-f16a-online' },
      {
        presetId: 'lfm2',
        quantizationInfo: {
          weights: 'q4k',
          compute: 'f32',
          layout: 'row',
        },
      }
    ),
    /kernel activation dtype "f16" is incompatible with compute "f32"/
  );
}

{
  const preset = resolvePreset('lfm2');
  assert.throws(
    () => buildManifestInference(
      preset,
      {
        model_type: 'lfm2',
        layer_types: ['conv', 'conv', 'full_attention', 'conv'],
      },
      64,
      { weights: 'q4k', layout: 'row' }
    ),
    /requires quantizationInfo\.compute to resolve a compute-specific defaultKernelPath/
  );
}

{
  const preset = resolvePreset('qwen3');
  const manifestInference = buildManifestInference(
    preset,
    {
      model_type: 'qwen3_5',
      rms_norm_eps: 1e-6,
      attn_output_gate: true,
      layer_types: [
        'linear_attention',
        'linear_attention',
        'linear_attention',
        'full_attention',
      ],
    },
    256
  );
  assert.equal(manifestInference.normalization.rmsNormWeightOffset, false);
  assert.equal(manifestInference.attention.attentionOutputGate, true);
}

{
  const preset = resolvePreset('qwen3');
  const manifestInference = buildManifestInference(
    preset,
    {
      model_type: 'qwen2',
      rms_norm_eps: 1e-6,
      layer_types: [
        'linear_attention',
        'full_attention',
        'linear_attention',
        'full_attention',
      ],
    },
    256
  );
  assert.equal(manifestInference.normalization.rmsNormWeightOffset, false);
  assert.equal(manifestInference.attention.attentionOutputGate, true);
}

{
  const preset = resolvePreset('qwen3');
  const manifestInference = buildManifestInference(
    preset,
    {
      model_type: 'qwen2',
      rms_norm_eps: 1e-6,
      layer_types: [
        'full_attention',
        'full_attention',
        'full_attention',
        'full_attention',
      ],
    },
    256
  );
  assert.equal(manifestInference.normalization.rmsNormWeightOffset, false);
}

{
  assert.throws(
    () => buildManifestInference(
      {
        id: 'unit-missing-compute-map',
        inference: {
          kernelPaths: {
            q4k: {
              f16: 'unit-q4k-f16a',
            },
          },
        },
      },
      {
        model_type: 'unit_missing_compute_map',
      },
      64,
      { weights: 'q4k', compute: 'f32', layout: 'row' }
    ),
    /is missing compute "f32". Add an explicit compute-specific mapping or default/
  );
}

{
  assert.throws(
    () => buildManifestInference(
      {
        id: 'unit-q4k-layout',
        inference: {
          kernelPaths: {
            q4k: {
              f16: 'unit-q4k-fused-f16a',
            },
          },
        },
      },
      {
        model_type: 'unit_q4k_layout',
      },
      64,
      { weights: 'q4k', compute: 'f16', layout: 'col' }
    ),
    /Add an explicit dequant kernel path mapping to the preset instead of relying on JS rewrites/
  );
}

{
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
      converterConfig: createConverterConfig({
        quantization: {
          weights: 'q4k',
          q4kLayout: 'diagonal',
        },
      }),
      modelKind: 'transformer',
      architectureHint: 'Gemma3ForCausalLM',
      architectureConfig: { headDim: 256 },
    }),
    /converter\.quantization\.q4kLayout must be "row" or "col"/
  );
}

{
  const attentionKernelRef = buildKernelRefFromKernelEntry('attention_streaming_f16.wgsl', 'main');
  const executionOverrideConfig = createConverterConfig({
    inference: {
      sessionDefaults: {
        compute: {
          defaults: {
            activationDtype: 'f16',
            mathDtype: 'f16',
            accumDtype: 'f32',
            outputDtype: 'f16',
          },
          kernelProfiles: [
            {
              kernelRef: attentionKernelRef,
            },
          ],
        },
        kvcache: {
          layout: 'bdpa',
          kvDtype: 'f16',
          pageSize: 128,
          windowSize: 1024,
          bdpaVocabSize: 4096,
        },
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 4,
          ringTokens: 4,
          ringStop: 4,
          ringStaging: 4,
        },
      },
      execution: {
        steps: [
          {
            id: 'attn_prefill',
            phase: 'prefill',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16.wgsl',
            entry: 'main',
            kernelRef: attentionKernelRef,
          },
        ],
        policies: {
          precisionPrecedence: 'step_then_kernel_profile_then_session_default',
          unsupportedPrecision: 'error',
          dtypeTransition: 'require_cast_step',
          unresolvedKernel: 'error',
        },
      },
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
    converterConfig: executionOverrideConfig,
    modelKind: 'transformer',
    architectureHint: 'Gemma3ForCausalLM',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.manifestInference?.sessionDefaults?.decodeLoop?.batchSize, 8);
  assert.equal(plan.manifestInference?.execution?.steps?.length, 1);
  assert.equal(plan.manifestInference?.execution?.steps?.[0]?.id, 'attn_prefill');
  assert.equal(plan.manifestInference?.schema, 'doppler.execution/v0');
}

{
  const generatedExecutionConfig = createConverterConfig({
    inference: {
      defaultKernelPath: 'gemma3-f16-fused-f16a-online',
      sessionDefaults: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 2,
        },
      },
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
    converterConfig: generatedExecutionConfig,
    modelKind: 'transformer',
    architectureHint: 'Gemma3ForCausalLM',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.manifestInference?.schema, 'doppler.execution/v0');
  assert.ok(Array.isArray(plan.manifestInference?.execution?.steps));
  assert.equal(plan.manifestInference?.sessionDefaults?.compute?.defaults?.activationDtype, 'f16');
  assert.equal(plan.manifestInference?.sessionDefaults?.kvcache?.kvDtype, 'f16');
  assert.equal(plan.manifestInference?.sessionDefaults?.decodeLoop?.batchSize, 8);
  assert.equal(plan.manifestInference?.sessionDefaults?.decodeLoop?.readbackInterval, 2);
}

{
  const invalidExecutionOnlyConfig = createConverterConfig({
    inference: {
      execution: {
        steps: [
          {
            id: 'attn_prefill',
            phase: 'prefill',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16.wgsl',
            entry: 'main',
            kernelRef: buildKernelRefFromKernelEntry('attention_streaming_f16.wgsl', 'main'),
          },
        ],
      },
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
      converterConfig: invalidExecutionOnlyConfig,
      modelKind: 'transformer',
      architectureHint: 'Gemma3ForCausalLM',
      architectureConfig: { headDim: 256 },
    }),
    /converterConfig\.inference\.execution requires converterConfig\.inference\.sessionDefaults/
  );
}

{
  const sessionDefaultsOnlyConfig = createConverterConfig({
    quantization: {
      weights: 'q4k',
      embeddings: 'f16',
      lmHead: 'f16',
      computePrecision: 'f32',
      q4kLayout: 'row',
    },
    inference: {
      defaultKernelPath: 'lfm2-q4k-dequant-f32a-online',
      sessionDefaults: {
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
        },
      },
    },
  });
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'lfm2',
      architectures: ['Lfm2ForCausalLM'],
      hidden_size: 256,
      intermediate_size: 512,
      num_hidden_layers: 4,
      num_attention_heads: 4,
      num_key_value_heads: 4,
      layer_types: ['conv', 'conv', 'full_attention', 'conv'],
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'BF16' },
      { name: 'model.layers.0.conv.in_proj.weight', dtype: 'BF16' },
      { name: 'model.layers.2.self_attn.q_proj.weight', dtype: 'BF16' },
    ],
    converterConfig: sessionDefaultsOnlyConfig,
    modelKind: 'transformer',
    architectureHint: 'Lfm2ForCausalLM',
    architectureConfig: { headDim: 64 },
  });
  assert.equal(plan.manifestInference?.schema, null);
  assert.equal(plan.manifestInference?.execution, null);
  assert.equal(plan.manifestInference?.defaultKernelPath, 'lfm2-q4k-dequant-f32a-online');
  assert.equal(plan.manifestInference?.sessionDefaults?.decodeLoop?.batchSize, 8);
  assert.equal(plan.manifestInference?.sessionDefaults?.compute, undefined);
}

{
  const invalidNonExecutionSessionDefaultsConfig = createConverterConfig({
    quantization: {
      weights: 'q4k',
      embeddings: 'f16',
      lmHead: 'f16',
      computePrecision: 'f32',
      q4kLayout: 'row',
    },
    inference: {
      sessionDefaults: {
        compute: {
          defaults: {
            activationDtype: 'f16',
          },
        },
      },
    },
  });
  assert.throws(
    () => resolveConversionPlan({
      rawConfig: {
        model_type: 'lfm2',
        architectures: ['Lfm2ForCausalLM'],
        hidden_size: 256,
        intermediate_size: 512,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        layer_types: ['conv', 'conv', 'full_attention', 'conv'],
      },
      tensors: [
        { name: 'model.embed_tokens.weight', dtype: 'BF16' },
        { name: 'model.layers.0.conv.in_proj.weight', dtype: 'BF16' },
        { name: 'model.layers.2.self_attn.q_proj.weight', dtype: 'BF16' },
      ],
      converterConfig: invalidNonExecutionSessionDefaultsConfig,
      modelKind: 'transformer',
      architectureHint: 'Lfm2ForCausalLM',
      architectureConfig: { headDim: 64 },
    }),
    /sessionDefaults may only set decodeLoop unless converterConfig\.inference\.execution is present/
  );
}

{
  const invalidSessionDefaultsConfig = createConverterConfig({
    inference: {
      sessionDefaults: 'bad',
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
      converterConfig: invalidSessionDefaultsConfig,
      modelKind: 'transformer',
      architectureHint: 'Gemma3ForCausalLM',
      architectureConfig: { headDim: 256 },
    }),
    /converterConfig\.inference\.sessionDefaults must be an object/
  );
}

{
  const invalidExecutionConfig = createConverterConfig({
    inference: {
      execution: {
        policies: {
          dtypeTransition: 'require_cast_step',
        },
      },
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
      converterConfig: invalidExecutionConfig,
      modelKind: 'transformer',
      architectureHint: 'Gemma3ForCausalLM',
      architectureConfig: { headDim: 256 },
    }),
    /converterConfig\.inference\.execution\.steps must be an array/
  );
}

{
  const invalidPinnedKernelConfig = createConverterConfig({
    inference: {
      execution: {
        steps: [
          {
            id: 'attn_prefill',
            phase: 'prefill',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16.wgsl',
            entry: 'main',
            kernelRef: buildKernelRefFromKernelEntry('matmul_f16.wgsl', 'main'),
          },
        ],
      },
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
      converterConfig: invalidPinnedKernelConfig,
      modelKind: 'transformer',
      architectureHint: 'Gemma3ForCausalLM',
      architectureConfig: { headDim: 256 },
    }),
    /kernelRef must match kernel binding/
  );
}

{
  const missingKernelRefConfig = createConverterConfig({
    inference: {
      execution: {
        steps: [
          {
            id: 'attn_prefill',
            phase: 'prefill',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16.wgsl',
            entry: 'main',
          },
        ],
      },
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
      converterConfig: missingKernelRefConfig,
      modelKind: 'transformer',
      architectureHint: 'Gemma3ForCausalLM',
      architectureConfig: { headDim: 256 },
    }),
    /kernelRef \{id, version, digest\} is required/
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
  assert.equal(plan.manifestInference?.schema, 'doppler.execution/v0');
  assert.ok(Array.isArray(plan.manifestInference?.execution?.steps));
  assert.ok(plan.manifestInference.execution.steps.length > 0);
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
  assert.equal(plan.quantizationInfo.variantTag, 'f16');
  assert.equal(plan.manifestInference?.defaultKernelPath, 'gemma3-f16-fused-f16a-online');
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
  // LFM2 hybrid configs must preserve explicit custom layer schedule.
  const layerTypes = [
    'conv',
    'conv',
    'full_attention',
    'conv',
  ];
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'lfm2',
      architectures: ['Lfm2ForCausalLM'],
      vocab_size: 65536,
      hidden_size: 2048,
      intermediate_size: 12288,
      num_hidden_layers: 4,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      max_position_embeddings: 32768,
      layer_types: layerTypes,
      tie_embedding: true,
    },
    tensors: [
      { name: 'model.embed_tokens.weight', dtype: 'BF16' },
      { name: 'model.layers.0.conv.in_proj.weight', dtype: 'BF16' },
      { name: 'model.layers.0.conv.conv.weight', dtype: 'BF16' },
      { name: 'model.layers.0.conv.out_proj.weight', dtype: 'BF16' },
      { name: 'model.layers.0.feed_forward.w1.weight', dtype: 'BF16' },
      { name: 'model.layers.2.self_attn.q_proj.weight', dtype: 'BF16' },
      { name: 'model.layers.2.self_attn.k_proj.weight', dtype: 'BF16' },
      { name: 'model.layers.2.self_attn.v_proj.weight', dtype: 'BF16' },
      { name: 'model.layers.2.self_attn.out_proj.weight', dtype: 'BF16' },
      { name: 'model.norm.weight', dtype: 'BF16' },
    ],
    converterConfig,
    modelKind: 'transformer',
    architectureHint: 'Lfm2ForCausalLM',
    architectureConfig: { headDim: 64 },
  });

  assert.equal(plan.presetId, 'lfm2');
  assert.equal(plan.modelType, 'transformer');
  assert.equal(plan.manifestInference?.layerPattern?.type, 'custom');
  assert.deepEqual(plan.manifestInference?.layerPattern?.layerTypes, layerTypes);
  assert.equal(plan.manifestInference?.layerPattern?.period, null);
  assert.equal(plan.manifestInference?.layerPattern?.globalPattern, null);
}

{
  const rawLfm2Config = {
    model_type: 'lfm2',
    architectures: ['Lfm2ForCausalLM'],
    vocab_size: 65536,
    hidden_size: 2048,
    intermediate_size: 12288,
    num_hidden_layers: 4,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    max_position_embeddings: 32768,
    tie_embedding: true,
    layer_types: ['conv', 'conv', 'full_attention', 'conv'],
  };
  const lfm2Tensors = [
    { name: 'model.embed_tokens.weight', dtype: 'BF16' },
    { name: 'model.layers.0.conv.in_proj.weight', dtype: 'BF16' },
    { name: 'model.layers.0.conv.conv.weight', dtype: 'BF16' },
    { name: 'model.layers.0.conv.out_proj.weight', dtype: 'BF16' },
    { name: 'model.layers.0.feed_forward.w1.weight', dtype: 'BF16' },
    { name: 'model.layers.2.self_attn.q_proj.weight', dtype: 'BF16' },
    { name: 'model.layers.2.self_attn.k_proj.weight', dtype: 'BF16' },
    { name: 'model.layers.2.self_attn.v_proj.weight', dtype: 'BF16' },
    { name: 'model.layers.2.self_attn.out_proj.weight', dtype: 'BF16' },
    { name: 'model.norm.weight', dtype: 'BF16' },
  ];
  const lfm2Q4kF16Config = createConverterConfig({
    quantization: {
      weights: 'q4k',
      embeddings: 'f16',
      lmHead: 'f16',
      computePrecision: 'f16',
      q4kLayout: 'row',
    },
  });
  const lfm2Q4kF32Config = createConverterConfig({
    quantization: {
      weights: 'q4k',
      embeddings: 'f16',
      lmHead: 'f16',
      computePrecision: 'f32',
      q4kLayout: 'row',
    },
  });

  const f16Plan = resolveConversionPlan({
    rawConfig: rawLfm2Config,
    tensors: lfm2Tensors,
    converterConfig: lfm2Q4kF16Config,
    modelKind: 'transformer',
    architectureHint: 'Lfm2ForCausalLM',
    architectureConfig: { headDim: 64 },
  });
  assert.equal(f16Plan.manifestInference?.defaultKernelPath, 'gemma3-q4k-dequant-f16a-online');
  assert.equal(f16Plan.manifestInference?.schema, null);
  assert.equal(f16Plan.manifestInference?.execution, null);
  assert.equal(f16Plan.manifestInference?.sessionDefaults, null);

  const f32Plan = resolveConversionPlan({
    rawConfig: rawLfm2Config,
    tensors: lfm2Tensors,
    converterConfig: lfm2Q4kF32Config,
    modelKind: 'transformer',
    architectureHint: 'Lfm2ForCausalLM',
    architectureConfig: { headDim: 64 },
  });
  assert.equal(f32Plan.manifestInference?.defaultKernelPath, 'lfm2-q4k-dequant-f32a-online');
  assert.equal(f32Plan.manifestInference?.schema, null);
  assert.equal(f32Plan.manifestInference?.execution, null);
  assert.equal(f32Plan.manifestInference?.sessionDefaults, null);
}

{
  // TranslateGemma style rope_parameters should map to manifest rope fields.
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3_text',
      architectures: ['Gemma3ForCausalLM'],
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
    architectureHint: 'Gemma3ForCausalLM',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.manifestInference?.rope?.ropeTheta, 1000000);
  assert.equal(plan.manifestInference?.rope?.ropeLocalTheta, 10000);
  assert.equal(plan.manifestInference?.rope?.ropeScalingType, 'linear');
  assert.equal(plan.manifestInference?.rope?.ropeScalingFactor, 8);
  assert.equal(plan.manifestInference?.rope?.ropeLocalScalingType, null);
  assert.equal(plan.manifestInference?.rope?.ropeLocalScalingFactor, 1);
}

{
  // Per-layer scaling should map to global/local RoPE scaling fields.
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'gemma3_text',
      architectures: ['Gemma3ForCausalLM'],
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
          rope_type: 'linear',
          factor: 4.0,
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
    architectureHint: 'Gemma3ForCausalLM',
    architectureConfig: { headDim: 256 },
  });
  assert.equal(plan.manifestInference?.rope?.ropeScalingType, 'linear');
  assert.equal(plan.manifestInference?.rope?.ropeScalingFactor, 8);
  assert.equal(plan.manifestInference?.rope?.ropeLocalScalingType, 'linear');
  assert.equal(plan.manifestInference?.rope?.ropeLocalScalingFactor, 4);
}

{
  // TranslateGemma architecture should resolve to translategemma preset.
  const plan = resolveConversionPlan({
    rawConfig: {
      model_type: 'translategemma',
      architectures: ['Gemma3ForConditionalGeneration'],
      hidden_size: 2560,
      intermediate_size: 10240,
      num_hidden_layers: 34,
      num_attention_heads: 8,
      num_key_value_heads: 4,
      text_config: {
        sliding_window: 1024,
      },
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
  assert.equal(plan.presetId, 'translategemma');
  assert.equal(plan.modelType, 'transformer');
  assert.equal(plan.manifestInference?.chatTemplate?.type, 'translategemma');
  assert.equal(plan.manifestInference?.chatTemplate?.enabled, true);
  assert.equal(plan.manifestInference?.attention?.slidingWindow, 1024);
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
  assert.equal(plan.quantizationInfo.variantTag, 'q4k-ehaf16');
  assert.equal(plan.manifestInference?.defaultKernelPath, 'gemma3-q4k-dequant-f16a-online');
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
  assert.equal(plan.manifestInference?.defaultKernelPath, 'gemma3-q4k-dequant-f32a-online');
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
    converterConfig: embeddingComputeF32Config,
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
      computePrecision: 'f32',
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
    converterConfig: embeddingComputeF32Config,
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
  assert.equal(plan.quantizationInfo.variantTag, 'q4k-ehaf16');
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
