import sourceDefaults from '../qwen4-exp-source-defaults.json' with { type: 'json' };
import { resolveQwen4ExpSourceContracts } from '../qwen4-exp-source-contracts.js';

// GGUF qwen4exp uses zero-based PLE layer indices and per-head PLE width.
// HF Qwen4ExpTextConfig uses one-based indices and the concatenated width.
// Source semantics: artifacts/qwen38-flash-next-20261013/source/config.json
// and modeling_qwen4_exp.py. Header hash arrays must match those semantics;
// recognizing the architecture name alone never admits different hash rules.
/** @type {typeof import('./qwen4-exp-source.js').normalizeQwen4ExpGGUFConfig} */
export function normalizeQwen4ExpGGUFConfig(config, metadata, family) {
  /** @param {string} key */
  const value = key => metadata[`qwen4exp.${key}`];
  /** @param {string} key */
  const integer = key => {
    const n = value(key);
    if (typeof n !== 'number' || !Number.isSafeInteger(n) || n <= 0) throw new Error(`GGUF qwen4exp requires positive integer ${key}.`);
    return n;
  };
  const layers = value('ple.layers');
  if (!Array.isArray(layers) || layers.length !== 1 || !Number.isSafeInteger(layers[0])
      || layers[0] < 0 || layers[0] >= config.num_hidden_layers
      || config.layer_types[layers[0]] !== 'linear_attention') {
    throw new Error('GGUF qwen4exp supports one explicitly declared linear PLE layer.');
  }
  const recurrent = value('attention.recurrent_layers');
  if (recurrent !== undefined) {
    const flags = Array.isArray(recurrent) ? recurrent : Array(config.num_hidden_layers).fill(recurrent);
    if (flags.length !== config.num_hidden_layers || flags.some((flag, i) =>
      typeof flag !== 'boolean' || flag !== (config.layer_types[i] === 'linear_attention'))) {
      throw new Error('GGUF qwen4exp recurrent layer flags conflict with full_attention_interval.');
    }
  }
  const rawRatios = value('attention.compress_ratios');
  const ratios = Array.isArray(rawRatios) ? rawRatios : Array(config.num_hidden_layers).fill(rawRatios);
  const fullRatios = ratios.filter((_, i) => config.layer_types[i] === 'full_attention');
  const ratio = fullRatios[0];
  if (ratios.length !== config.num_hidden_layers || !Number.isSafeInteger(ratio) || ratio < 2
      || ratios.some((r, i) => r !== (config.layer_types[i] === 'full_attention' ? ratio : 0))) {
    throw new Error('GGUF qwen4exp requires consistent indexed-attention compress ratios.');
  }
  const ngramSize = integer('ple.ngram_size');
  const headsPerNgram = integer('ple.heads_per_ngram');
  const embedDim = integer('embedding_length_per_layer_input') * (ngramSize - 1) * headsPerNgram;
  if (embedDim !== config.hidden_size) throw new Error('GGUF qwen4exp PLE head widths do not match hidden size.');
  const boundaryToken = integer('ple.eos_token_id');
  if (boundaryToken >= config.vocab_size) throw new Error('GGUF qwen4exp PLE boundary token is outside vocabulary.');
  Object.assign(config, {
    ple_eos_token_id: boundaryToken,
    output_gate_type: family.outputGateType,
    hc_count: integer('hyper_connection.count'),
    hc_lowrank: integer('hyper_connection.low_rank'),
    indexer_n_heads: integer('attention.indexer.head_count'),
    indexer_kv_heads: family.indexerKvHeads,
    indexer_head_dim: integer('attention.indexer.key_length'),
    indexer_budget: integer('attention.indexer.top_k'),
    indexer_compress_ratio: ratio,
    ple_layer_ids: layers.map(layer => layer + 1),
    ple_embed_dim: embedDim,
    ple_conv_kernel_size: integer('ple.conv_kernel'),
    ngram_size: ngramSize,
    heads_per_ngram: headsPerNgram,
    ngram_vocab_size_base: sourceDefaults.defaults.ngram_vocab_size_base,
    make_ngram_vocab_size_divisible_by: sourceDefaults.defaults.make_ngram_vocab_size_divisible_by,
    seed: sourceDefaults.defaults.seed,
    norm_topk_prob: sourceDefaults.defaults.norm_topk_prob,
  });
  // Dtype here is only for deriving hash geometry; the parser binds the actual
  // table dtype from its tensor descriptor, never from general.file_type.
  const contract = resolveQwen4ExpSourceContracts({ ...config, ngram_table_dtype: 'bf16' })?.ngramEmbedding;
  if (!contract) throw new Error('GGUF qwen4exp requires a PLE contract.');
  /** @type {Record<string, 'multipliers' | 'headOffsets' | 'headVocabSizes'>} */
  const fields = { 'ple.layer_multipliers': 'multipliers', 'ple.head_offsets': 'headOffsets', 'ple.head_vocab_sizes': 'headVocabSizes' };
  for (const [source, target] of Object.entries(fields)) {
    const actual = value(source);
    const expected = contract.layers[0][target];
    if (!Array.isArray(actual) || actual.length !== expected.length
        || actual.some((n, i) => !Number.isSafeInteger(n) || String(n) !== String(expected[i]))) {
      throw new Error(`GGUF qwen4exp ${source} differs from the pinned source hash contract.`);
    }
  }
}

/** @type {typeof import('./qwen4-exp-source.js').bindQwen4ExpGGUFTable} */
export function bindQwen4ExpGGUFTable(config, tensors) {
  const table = tensors.find(tensor => tensor.name === 'per_layer_token_embd.weight');
  if (!table) throw new Error('GGUF qwen4exp is missing per_layer_token_embd.weight.');
  const dtype = String(table.dtype).toLowerCase();
  if (dtype !== 'q4_0' && dtype !== 'bf16') throw new Error(`GGUF qwen4exp unsupported PLE table dtype ${dtype}.`);
  config.ngram_table_dtype = dtype;
  const contract = resolveQwen4ExpSourceContracts(config)?.ngramEmbedding;
  if (!contract) throw new Error('GGUF qwen4exp requires a PLE contract.');
  const layer = contract.layers[0];
  if (table.shape.length !== 2 || table.shape[0] !== contract.headDim || table.shape[1] !== layer.rows) {
    throw new Error('GGUF qwen4exp PLE table shape differs from the authenticated header geometry.');
  }
  const bytes = dtype === 'q4_0' ? contract.headDim / 32 * 18 : contract.headDim * 2;
  if ((dtype === 'q4_0' && contract.headDim % 32) || table.size !== layer.rows * bytes) {
    throw new Error('GGUF qwen4exp PLE table byte extent differs from its stored dtype.');
  }
  return { table, name: `model.language_model.layers.${layer.decoderLayerIndex}.ple.ple_embedding.ngram_embedding.shard_0.weight` };
}
