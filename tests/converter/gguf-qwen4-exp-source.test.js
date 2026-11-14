import assert from 'node:assert/strict';
import metadata from '../fixtures/gguf/qwen4exp-flash-metadata.json' with { type: 'json' };
import { normalizeQwenGGUFSource } from '../../src/formats/gguf/qwen-source.js';
import { bindQwen4ExpGGUFTable } from '../../src/formats/gguf/qwen4-exp-source.js';
import { resolveQwen4ExpSourceContracts } from '../../src/converter/qwen4-exp-source.js';
import { extractArchitecture } from '../../src/converter/artifact-identity.js';
import { createSourceRuntimeManifestInference } from '../../src/tooling/source-runtime-converter-config.js';
import { parseGGUFModel } from '../../src/converter/parsers/gguf.js';

// Exact scalar/array metadata from the three-part Flash-Next Q4_0 artifact.
// Synthetic vocabulary avoids retaining the model tokenizer in this fixture.
const m = { ...metadata, 'tokenizer.ggml.tokens': Array.from({ length: 248320 }, (_, i) => `t${i}`),
  'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen35',
  'tokenizer.ggml.token_type': Array(248320).fill(1), 'tokenizer.ggml.merges': [] };
const native = normalizeQwenGGUFSource('qwen4exp', m);
assert.equal(native.config.model_type, 'qwen4_exp');
assert.equal(native.config.intermediate_size, 640);
assert.equal(native.config.moe_intermediate_size, 640);
assert.equal(native.config.output_gate_type, 'sigmoid');
assert.equal(native.config.eos_token_id, 248046);
assert.equal(native.config.ple_eos_token_id, 248044);
assert.deepEqual(native.config.ple_layer_ids, [2]);
assert.equal(native.config.ple_embed_dim, 2560);
assert.equal(extractArchitecture(native.config, null).numLayers, 48);
const table = { name: 'per_layer_token_embd.weight', shape: [160, 320001536], dtype: 'Q4_0', size: 28800138240, offset: 128 };
const binding = bindQwen4ExpGGUFTable(native.config, [table]);
assert.equal(binding.name, 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight');
const contracts = resolveQwen4ExpSourceContracts(native.config);
assert.equal(contracts.ngramEmbedding.tableDtype, 'q4_0');
assert.equal(contracts.ngramEmbedding.boundaryTokenId, 248044);
assert.equal(contracts.indexedAttention.tokenBudget, 2048);
assert.equal(contracts.indexedAttention.blockTopK, 512);
assert.equal(contracts.hyperConnections.streamCount, 4);
for (const [key, replacement] of [
  ['qwen4exp.ple.layer_multipliers', [1, 2, 3]],
  ['qwen4exp.ple.head_offsets', metadata['qwen4exp.ple.head_offsets'].map((n, i) => n + (i === 1 ? 1 : 0))],
  ['qwen4exp.ple.head_vocab_sizes', metadata['qwen4exp.ple.head_vocab_sizes'].map(n => n + 1)],
  ['qwen4exp.ple.layers', [1, 2]],
  ['qwen4exp.embedding_length_per_layer_input', 161],
  ['qwen4exp.attention.compress_ratios', Array(48).fill(4)],
  ['qwen4exp.hyper_connection.count', undefined],
  ['qwen4exp.ple.eos_token_id', 248320],
]) assert.throws(() => normalizeQwenGGUFSource('qwen4exp', { ...m, [key]: replacement }), /GGUF qwen4exp/);
assert.throws(() => bindQwen4ExpGGUFTable(native.config, []), /missing/);
assert.throws(() => bindQwen4ExpGGUFTable(native.config, [{ ...table, size: table.size - 1 }]), /byte extent/);
assert.throws(() => bindQwen4ExpGGUFTable(native.config, [{ ...table, shape: [320, 160000768] }]), /shape/);
assert.throws(() => bindQwen4ExpGGUFTable(native.config, [{ ...table, dtype: 'Q8_0' }]), /unsupported/);
const source = { size: table.offset + table.size, async readRange() { throw new Error('header-only test must not read tensor bytes'); } };
const parsed = await parseGGUFModel({ file: source, normalizeTensorSource: x => x,
  parseGGUFHeaderFromSource: async () => ({ architecture: 'qwen4exp', metadata: m, config: {}, tensors: [table], quantization: 'Q4_0' }) });
assert.equal(parsed.normalizedArchitecture.intermediateSize, 640);
assert.equal(parsed.config.ngram_table_dtype, 'q4_0');
assert.equal(parsed.tokenizerConfig.eos_token_id, 248046);
assert.equal(parsed.tensors[0].role, 'ngram_table');
assert.equal(parsed.tensors[0].name, binding.name);
assert.equal(parsed.tensors[0].originalName, table.name);
assert.deepEqual(parsed.tensors[0].shape, [320001536, 160]);
assert.equal(parsed.tensors[0].offset, 128);
assert.equal(parsed.tensors[0].source, source);
const inference = createSourceRuntimeManifestInference(parsed.config);
assert.deepEqual(inference.hyperConnections, contracts.hyperConnections);
assert.deepEqual(inference.indexedAttention, contracts.indexedAttention);
assert.deepEqual(inference.ngramEmbedding, contracts.ngramEmbedding);
assert.equal(inference.normalization.rmsNormWeightOffset, false);
assert.equal(inference.normalization.linearNormGateActivation, 'sigmoid');
assert.equal(inference.session.compute.defaults.activationDtype, 'f32');
assert.equal(inference.session.kvcache.kvDtype, 'f16');
assert.equal(inference.session.ngramEmbedding.reads.maxConcurrent, 16);
inference.session.ngramEmbedding.reads.maxConcurrent = 1;
assert.equal(createSourceRuntimeManifestInference(parsed.config).session.ngramEmbedding.reads.maxConcurrent, 16);
console.log('gguf-qwen4-exp-source.test: ok');
