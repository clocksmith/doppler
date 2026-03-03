import assert from 'node:assert/strict';

import { ERROR_CODES } from '../../src/errors/index.js';
import {
  PRESET_DETECTION_ORDER,
  detectPreset,
  getPreset,
  listPresets,
  resolveConfig,
  resolvePreset,
} from '../../src/config/loader.js';

assert.ok(PRESET_DETECTION_ORDER.length > 0);
assert.ok(PRESET_DETECTION_ORDER.includes('transformer'));
assert.ok(PRESET_DETECTION_ORDER.includes('gemma2'));
assert.ok(PRESET_DETECTION_ORDER.includes('lfm2'));

{
  const presets = listPresets();
  assert.ok(Array.isArray(presets));
  assert.ok(presets.includes('transformer'));
  assert.ok(presets.includes('gemma2'));
  assert.ok(presets.includes('lfm2'));
  assert.ok(getPreset('transformer'));
  assert.equal(getPreset('missing-preset'), null);
}

{
  const resolved = resolvePreset('qwen3');
  assert.equal(resolved.id, 'qwen3');
  assert.equal(resolved.extends, undefined);
  assert.equal(resolved.modelType, 'transformer');
  assert.ok(resolved.inference);
}

{
  const resolved = resolvePreset('lfm2');
  assert.equal(resolved.id, 'lfm2');
  assert.equal(resolved.inference?.kernelPaths?.q4k?.f16, 'gemma3-q4k-dequant-f16a-online');
  assert.equal(resolved.inference?.kernelPaths?.q4k?.f32, 'lfm2-q4k-dequant-f32a-online');
}

{
  assert.throws(
    () => resolvePreset('definitely-not-a-preset'),
    (error) => {
      assert.equal(error?.code, ERROR_CODES.CONFIG_PRESET_UNKNOWN);
      assert.match(String(error?.message), /Unknown preset/);
      return true;
    }
  );
}

{
  assert.equal(
    detectPreset({ model_type: 'unknown' }, 'DeepSeekV2ForCausalLM'),
    'deepseek'
  );
  assert.equal(
    detectPreset({ model_type: 'lfm2' }, 'Lfm2ForCausalLM'),
    'lfm2'
  );
  assert.equal(
    detectPreset({ model_type: 'gemma2' }, 'irrelevant'),
    'gemma2'
  );
}

{
  // Weak hints (arch/model_type equal) allow config-pattern fallback.
  const detected = detectPreset(
    {
      model_type: 'qwen2',
    },
    'qwen2'
  );
  assert.equal(detected, 'qwen3');
}

{
  // Strong hints prevent config-pattern fallback from hijacking detection.
  const detected = detectPreset(
    {
      model_type: 'custom_model',
      n_shared_experts: 2,
    },
    'custom_architecture'
  );
  assert.equal(detected, 'transformer');
}

{
  const resolved = resolveConfig({
    modelId: 'manifest-with-config',
    modelType: 'transformer',
    config: {
      num_hidden_layers: 2,
      hidden_size: 8,
      intermediate_size: 16,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      head_dim: 4,
      vocab_size: 128,
      max_position_embeddings: 64,
      rope_theta: 750000,
      rms_norm_eps: 1e-6,
      sliding_window: 32,
      attn_logit_softcapping: 7.5,
      final_logit_softcapping: 9.0,
      tie_word_embeddings: true,
      scale_embeddings: true,
      pipeline: 'decode-only',
      rope_scaling_type: 'linear',
      rope_scaling_factor: 2.0,
    },
    tokenizer: {
      type: 'bundled',
      vocabSize: 128,
      bosTokenId: 1,
      eosTokenId: 2,
      unknownField: 'ignored',
    },
  });

  assert.equal(resolved.preset, 'transformer');
  assert.equal(resolved.modelType, 'transformer');
  assert.equal(resolved.architecture.numLayers, 2);
  assert.equal(resolved.architecture.hiddenSize, 8);
  assert.equal(resolved.architecture.numKeyValueHeads, 1);
  assert.equal(resolved.inference.attention.slidingWindow, 32);
  assert.equal(resolved.inference.attention.attnLogitSoftcapping, 7.5);
  assert.equal(resolved.inference.output.finalLogitSoftcapping, 9.0);
  assert.equal(resolved.inference.output.tieWordEmbeddings, true);
  assert.equal(resolved.inference.pipeline, 'decode-only');
  assert.equal(resolved.inference.rope.ropeScalingType, 'linear');
  assert.equal(resolved.inference.rope.ropeScalingFactor, 2.0);
  assert.equal(resolved.tokenizer.type, 'bundled');
  assert.equal(resolved.tokenizer.vocabSize, 128);
  assert.equal(resolved.tokenizer.bosTokenId, 1);
  assert.equal(resolved.tokenizer.unknownField, undefined);
  assert.ok(resolved.sampling);
  assert.ok(resolved.loading);
}

{
  const resolved = resolveConfig({
    modelId: 'explicit-preset-resolve',
    modelType: null,
    config: {
      num_hidden_layers: 3,
      hidden_size: 12,
      intermediate_size: 24,
      num_attention_heads: 3,
      num_key_value_heads: 3,
      head_dim: 4,
      vocab_size: 256,
      max_position_embeddings: 128,
    },
  }, 'gemma2');

  assert.equal(resolved.preset, 'gemma2');
  assert.equal(resolved.modelType, 'transformer');
  assert.equal(resolved.architecture.numLayers, 3);
  assert.equal(resolved.architecture.hiddenSize, 12);
  assert.equal(resolved.architecture.maxSeqLen, 128);
}

{
  assert.throws(
    () => resolveConfig({
      modelId: 'bad-manifest',
      modelType: 'transformer',
      config: {},
    }),
    /Missing or invalid architecture/
  );
}

console.log('model-preset-loader.test: ok');
