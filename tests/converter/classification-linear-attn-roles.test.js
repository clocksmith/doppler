import assert from 'node:assert/strict';
import { classifyTensorRole } from '../../src/formats/rdrr/classification.js';

assert.equal(
  classifyTensorRole('model.language_model.layers.0.linear_attn.in_proj_qkv.weight'),
  'matmul'
);
assert.equal(
  classifyTensorRole('model.language_model.layers.0.linear_attn.in_proj_z.weight'),
  'matmul'
);
assert.equal(
  classifyTensorRole('model.language_model.layers.0.linear_attn.in_proj_a.weight'),
  'matmul'
);
assert.equal(
  classifyTensorRole('model.language_model.layers.0.linear_attn.in_proj_b.weight'),
  'matmul'
);

console.log('classification-linear-attn-roles.test: ok');
