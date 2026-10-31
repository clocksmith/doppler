import assert from 'node:assert/strict';
import { classifyTensor, classifyTensorRole } from '../../../src/formats/rdrr/classification.js';

// GGUF head tensors: output_norm is the final norm the loader selects by role "norm" in group "head".
assert.equal(classifyTensor('output_norm.weight', 'transformer'), 'head');
assert.equal(classifyTensorRole('output_norm.weight'), 'norm');
assert.equal(classifyTensor('output.weight', 'transformer'), 'head');
assert.equal(classifyTensor('token_embd.weight', 'transformer'), 'embed');
// Per-layer norms stay out of the head group.
assert.notEqual(classifyTensor('blk.3.attn_norm.weight', 'transformer'), 'head');
assert.notEqual(classifyTensor('blk.3.post_attention_norm.weight', 'transformer'), 'head');
console.log('classification-gguf-final-norm.test: ok');
