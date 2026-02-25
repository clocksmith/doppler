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

console.log('core-extract-architecture.test: ok');
