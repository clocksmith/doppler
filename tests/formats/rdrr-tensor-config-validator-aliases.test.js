import assert from 'node:assert/strict';

import { validateTensorConfigConsistency } from '../../src/formats/rdrr/tensor-config-validator.js';

function manifestWith(tensorNames, normalization) {
  return {
    inference: {
      normalization,
      attention: { queryKeyNorm: false },
      output: { tieWordEmbeddings: false },
    },
    tensors: Object.fromEntries(tensorNames.map(name => [name, {}])),
  };
}

const glmOcr = manifestWith([
  'model.language_model.layers.0.input_layernorm.weight',
  'model.language_model.layers.0.post_self_attn_layernorm.weight',
  'model.language_model.layers.0.post_attention_layernorm.weight',
  'model.language_model.layers.0.post_mlp_layernorm.weight',
], {
  postAttentionNorm: true,
  preFeedforwardNorm: true,
  postFeedforwardNorm: true,
});
assert.deepEqual(validateTensorConfigConsistency(glmOcr).errors, []);

const qwenStyle = manifestWith([
  'model.layers.0.input_layernorm.weight',
  'model.layers.0.post_attention_layernorm.weight',
], {
  postAttentionNorm: true,
  preFeedforwardNorm: false,
  postFeedforwardNorm: false,
});
assert.deepEqual(validateTensorConfigConsistency(qwenStyle).errors, []);

const missingGlmPostMlp = manifestWith([
  'model.language_model.layers.0.post_self_attn_layernorm.weight',
  'model.language_model.layers.0.post_attention_layernorm.weight',
], {
  postAttentionNorm: true,
  preFeedforwardNorm: true,
  postFeedforwardNorm: true,
});
const missingResult = validateTensorConfigConsistency(missingGlmPostMlp);
assert.equal(missingResult.valid, false);
assert.equal(
  missingResult.errors[0]?.code,
  'TENSOR_CONFIG_MISMATCH_INFERENCE_NORMALIZATION_POSTFEEDFORWARDNORM'
);

const hyperconnectionNorms = [
  'blk.0.hc_attn_norm.weight',
  'blk.0.hc_ffn_norm.weight',
];
const disabledNorms = {
  postAttentionNorm: false,
  preFeedforwardNorm: false,
  postFeedforwardNorm: false,
};
assert.deepEqual(validateTensorConfigConsistency(
  manifestWith(hyperconnectionNorms, disabledNorms)
).errors, [], 'Hyperconnection norms are distinct from ordinary post-attention norms');
for (const ordinary of ['ffn_norm.weight', 'blk.0.ffn_norm.weight', 'model.layers.0.post_attention_layernorm.weight']) {
  const manifest = manifestWith([...hyperconnectionNorms, ordinary], { ...disabledNorms });
  assert.equal(validateTensorConfigConsistency(manifest).valid, false,
    'An ordinary norm must still reject a disabled normalization flag');
  manifest.inference.normalization.postAttentionNorm = true;
  assert.equal(validateTensorConfigConsistency(manifest).valid, true);
}
assert.equal(validateTensorConfigConsistency(manifestWith(hyperconnectionNorms,
  { ...disabledNorms, postAttentionNorm: true })).valid, false,
  'Hyperconnection norms must not satisfy a required ordinary normalization tensor');

console.log('rdrr-tensor-config-validator-aliases.test: ok');
