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

console.log('rdrr-tensor-config-validator-aliases.test: ok');
