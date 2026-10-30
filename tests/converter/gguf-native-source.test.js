import assert from 'node:assert/strict';
import { normalizeQwenGGUFSource } from '../../src/formats/gguf/qwen-source.js';
import { buildGGUFTokenizer } from '../../src/formats/gguf/tokenizer.js';
import { parseGGUFModel } from '../../src/converter/parsers/gguf.js';
const metadata = {
 'qwen35moe.block_count': 5, 'qwen35moe.nextn_predict_layers': 1,
 'qwen35moe.full_attention_interval': 4, 'qwen35moe.embedding_length': 32,
 'qwen35moe.attention.head_count': 2, 'qwen35moe.attention.head_count_kv': 1,
 'qwen35moe.attention.key_length': 16, 'qwen35moe.attention.value_length': 16,
 'qwen35moe.attention.layer_norm_rms_epsilon': 1e-6,
 'qwen35moe.rope.freq_base': 10000000, 'qwen35moe.rope.dimension_count': 8,
 'qwen35moe.rope.dimension_sections': [2, 1, 1, 0],
 'qwen35moe.ssm.time_step_rank': 4, 'qwen35moe.ssm.inner_size': 32,
 'qwen35moe.ssm.group_count': 2, 'qwen35moe.ssm.state_size': 8, 'qwen35moe.ssm.conv_kernel': 4,
 'qwen35moe.context_length': 512, 'qwen35moe.expert_feed_forward_length': 32,
 'qwen35moe.expert_count': 8, 'qwen35moe.expert_used_count': 2,
 'qwen35moe.expert_shared_feed_forward_length': 32,
 'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen35',
 'tokenizer.ggml.tokens': ['a', 'b', '<|im_end|>'], 'tokenizer.ggml.token_type': [1, 1, 3],
 'tokenizer.ggml.merges': ['a b'], 'tokenizer.ggml.eos_token_id': 2,
};
const native = normalizeQwenGGUFSource('qwen35moe', metadata);
assert.equal(native.textLayers, 4);
assert.equal(native.config.linear_attention_source_layout, 'gguf-v1');
assert.equal(native.config.linear_num_value_heads, 4);
assert.deepEqual(native.config.layer_types, ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']);
assert.throws(() => normalizeQwenGGUFSource('qwen35moe', { ...metadata, 'qwen35moe.ssm.inner_size': 33 }), /geometry/);
const assets = buildGGUFTokenizer(metadata);
assert.equal(assets.tokenizerJson.model.vocab['<|im_end|>'], 2);
assert.equal(assets.tokenizerJson.added_tokens[0].id, 2);
assert.throws(() => buildGGUFTokenizer({ ...metadata, 'tokenizer.ggml.pre': 'unsupported' }), /Unsupported/);
assert.throws(() => buildGGUFTokenizer({ ...metadata, 'tokenizer.ggml.eos_token_id': 3 }), /outside/);
const source = { size: 4096, async readRange() { return new ArrayBuffer(0); } };
const parsed = await parseGGUFModel({ file: source, normalizeTensorSource: value => value,
 parseGGUFHeaderFromSource: async () => ({ architecture: 'qwen35moe', metadata, config: {}, tensors: [
  { name: 'blk.0.ffn_gate_inp_shexp.weight', dtype: 'F32', shape: [32], offset: 0, size: 128 },
  { name: 'blk.4.ssm_a', dtype: 'F32', shape: [4], offset: 128, size: 16 },
 ], tensorDataOffset: 0, quantization: 'Q4_0' }) });
assert.equal(parsed.normalizedArchitecture.numLayers, 4);
assert.deepEqual(parsed.tensors[0].shape, [1, 32]);
assert.equal(parsed.tensors[0].offset, 0);
assert.equal(parsed.excludedTensors[0].name, 'blk.4.ssm_a');
assert.ok(parsed.tokenizerJson);
console.log('gguf-native-source.test: ok');
