import assert from 'node:assert/strict';

const { extractArchitecture } = await import('../../src/converter/core.js');

{
  const architecture = extractArchitecture({
    num_hidden_layers: 2,
    hidden_size: 256,
    intermediate_size: 512,
    num_attention_heads: 4,
    num_key_value_heads: 2,
    head_dim: 64,
    vocab_size: 32000,
    max_position_embeddings: 8192,
    rope_theta: 1000000,
  });
  assert.equal(architecture.numLayers, 2);
  assert.equal(architecture.hiddenSize, 256);
  assert.equal(architecture.intermediateSize, 512);
  assert.equal(architecture.numAttentionHeads, 4);
  assert.equal(architecture.numKeyValueHeads, 2);
  assert.equal(architecture.headDim, 64);
  assert.equal(architecture.vocabSize, 32000);
  assert.equal(architecture.maxSeqLen, 8192);
  assert.equal(architecture.ropeTheta, 1000000);
}

{
  const architecture = extractArchitecture({
    model_type: 'multi_modality',
    language_config: {
      model_type: 'janus_text',
      num_hidden_layers: 24,
      hidden_size: 2048,
      intermediate_size: 5632,
      num_attention_heads: 16,
      num_key_value_heads: 16,
      head_dim: 128,
      vocab_size: 102400,
      max_position_embeddings: 16384,
      rope_theta: 10000,
    },
  });
  assert.equal(architecture.numLayers, 24);
  assert.equal(architecture.hiddenSize, 2048);
  assert.equal(architecture.intermediateSize, 5632);
  assert.equal(architecture.numAttentionHeads, 16);
  assert.equal(architecture.numKeyValueHeads, 16);
  assert.equal(architecture.headDim, 128);
  assert.equal(architecture.vocabSize, 102400);
  assert.equal(architecture.maxSeqLen, 16384);
  assert.equal(architecture.ropeTheta, 10000);
}

{
  const architecture = extractArchitecture({
    model_type: 'gemma3',
    text_config: {
      model_type: 'gemma3_text',
      num_hidden_layers: 34,
      hidden_size: 2560,
      intermediate_size: 10240,
      num_attention_heads: 8,
      num_key_value_heads: 4,
      head_dim: 256,
      vocab_size: 262208,
      max_position_embeddings: 131072,
      rope_theta: 1000000,
    },
  });
  assert.equal(architecture.numLayers, 34);
  assert.equal(architecture.hiddenSize, 2560);
  assert.equal(architecture.intermediateSize, 10240);
  assert.equal(architecture.numAttentionHeads, 8);
  assert.equal(architecture.numKeyValueHeads, 4);
  assert.equal(architecture.headDim, 256);
  assert.equal(architecture.vocabSize, 262208);
  assert.equal(architecture.maxSeqLen, 131072);
  assert.equal(architecture.ropeTheta, 1000000);
}

{
  const architecture = extractArchitecture({
    model_type: 'qwen2',
    num_hidden_layers: 4,
    hidden_size: 512,
    intermediate_size: 2048,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    head_dim: 64,
    vocab_size: 151936,
    max_position_embeddings: 32768,
    linear_norm_mode: 'per_head',
    layer_types: ['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
  });
  assert.equal(architecture.linearNormMode, 'per_head');
}

{
  const architecture = extractArchitecture({
    model_type: 'qwen2',
    num_hidden_layers: 4,
    hidden_size: 512,
    intermediate_size: 2048,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    head_dim: 64,
    vocab_size: 151936,
    max_position_embeddings: 32768,
    layer_types: ['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
  });
  assert.equal(architecture.linearNormMode, 'shared');
}

{
  assert.throws(
    () => extractArchitecture({
      model_type: 'qwen2',
      num_hidden_layers: 4,
      hidden_size: 512,
      intermediate_size: 2048,
      num_attention_heads: 8,
      num_key_value_heads: 2,
      head_dim: 64,
      vocab_size: 151936,
      max_position_embeddings: 32768,
      linear_norm_mode: 'invalid-mode',
    }),
    /Unsupported linear_norm_mode/
  );
}

console.log('core-extract-architecture.test: ok');
