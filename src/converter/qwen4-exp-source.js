import sourceDefaults from './qwen4-exp-source-defaults.json' with { type: 'json' };
import { sha256BytesHex } from '../formats/sha256.js';
import {
  deriveNgramLayerTable,
  deriveNgramMultipliers,
} from '../config/schema/ngram-embedding.schema.js';

// Lowers a qwen4_exp (Qwen3.8 Flash-Next) source config into Doppler's
// hyper-connection decoder contracts. Omitted source fields resolve from the
// pinned Qwen4ExpTextConfig class defaults (qwen4-exp-source-defaults.json),
// never from another family. Upstream validate_architecture rules are applied
// here so a malformed source fails before conversion.

/** @type {Readonly<Record<string, unknown>>} */
const DEFAULTS = sourceDefaults.defaults;
const QWEN4_EXP_FAMILIES = new Set(sourceDefaults.sourceModelTypes);
const QSA_FIELDS = Object.freeze([
  'indexer_n_heads', 'indexer_kv_heads', 'indexer_head_dim', 'indexer_budget', 'indexer_compress_ratio',
]);
const CONTRACT_FIELDS = Object.freeze(['hyperConnections', 'indexedAttention', 'ngramEmbedding']);
/** @type {Readonly<Record<string, 'bf16'>>} */
const SOURCE_TABLE_DTYPES = Object.freeze({ bfloat16: 'bf16' });

/** @param {Record<string, any> | null | undefined} rawConfig @returns {Record<string, any> | null} */
function textConfigOf(rawConfig) {
  if (!rawConfig || typeof rawConfig !== 'object') return null;
  const text = rawConfig.text_config;
  return text && typeof text === 'object' ? text : rawConfig;
}

/** Whether the source config names the qwen4_exp family. @type {typeof import('./qwen4-exp-source.js').isQwen4ExpSource} */
export function isQwen4ExpSource(rawConfig) {
  const top = String(rawConfig?.model_type ?? '').trim().toLowerCase();
  const text = String(textConfigOf(rawConfig)?.model_type ?? '').trim().toLowerCase();
  return QWEN4_EXP_FAMILIES.has(top) || QWEN4_EXP_FAMILIES.has(text);
}

/** @param {Record<string, any>} text @param {string} key @returns {any} */
function sourceField(text, key) {
  if (text[key] !== undefined) return text[key];
  if (!Object.hasOwn(DEFAULTS, key)) {
    throw new Error(`qwen4_exp source omits "${key}", which has no pinned Qwen4ExpTextConfig default.`);
  }
  return DEFAULTS[key];
}

/** @param {unknown} value @param {string} label @returns {number} */
function requireInteger(value, label) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value)) {
    throw new Error(`qwen4_exp source ${label} must be an integer; got ${JSON.stringify(value)}.`);
  }
  return value;
}

/** @param {Record<string, any>} text @returns {string[]} */
function sourceLayerTypes(text) {
  if (Array.isArray(text.layer_types)) return text.layer_types.map(String);
  const numLayers = requireInteger(sourceField(text, 'num_hidden_layers'), 'num_hidden_layers');
  const interval = requireInteger(text.full_attention_interval ?? 4, 'full_attention_interval');
  return Array.from({ length: numLayers }, (_, i) => ((i + 1) % interval ? 'linear_attention' : 'indexed_attention'));
}

/** @param {Record<string, any>} text @returns {number} */
function resolveRotaryDim(text) {
  const factor = text.rope_parameters?.partial_rotary_factor ?? 1.0;
  return Math.trunc(requireInteger(sourceField(text, 'head_dim'), 'head_dim') * factor);
}

/** @param {Record<string, any>} text @returns {number} */
function resolveBoundaryToken(text) {
  const eos = sourceField(text, 'eos_token_id');
  const value = Array.isArray(eos) ? eos[0] : eos;
  if (!Number.isSafeInteger(value)) {
    throw new Error('qwen4_exp PLE requires text_config.eos_token_id (the n-gram boundary token).');
  }
  return value;
}

/**
 * @param {Record<string, any>} text @param {string[]} layerTypes
 * @returns {import('../config/schema/ngram-embedding.schema.js').NgramEmbeddingSchema | null}
 */
function lowerNgramEmbedding(text, layerTypes) {
  const rawIds = sourceField(text, 'ple_layer_ids') ?? [];
  const sourceLayerIds = [...new Set(rawIds.map((/** @type {unknown} */ id) => requireInteger(id, 'ple_layer_ids[]')))].sort((a, b) => a - b);
  if (sourceLayerIds.length === 0) return null;
  const hiddenSize = requireInteger(sourceField(text, 'hidden_size'), 'hidden_size');
  const embedDim = requireInteger(sourceField(text, 'ple_embed_dim') ?? hiddenSize, 'ple_embed_dim');
  const ngramSize = requireInteger(sourceField(text, 'ngram_size'), 'ngram_size');
  const headsPerNgram = requireInteger(sourceField(text, 'heads_per_ngram'), 'heads_per_ngram');
  const headCount = (ngramSize - 1) * headsPerNgram;
  if (headCount <= 0 || embedDim <= 0 || embedDim % headCount !== 0) {
    throw new Error(`qwen4_exp ple_embed_dim ${embedDim} must be positive and divisible by ${headCount} n-gram heads.`);
  }
  const unigramVocabSize = requireInteger(sourceField(text, 'vocab_size'), 'vocab_size');
  const vocabSizeBase = requireInteger(sourceField(text, 'ngram_vocab_size_base'), 'ngram_vocab_size_base');
  const vocabPadMultiple = requireInteger(sourceField(text, 'make_ngram_vocab_size_divisible_by'), 'make_ngram_vocab_size_divisible_by');
  const seed = requireInteger(sourceField(text, 'seed'), 'seed');
  const convKernelSize = requireInteger(sourceField(text, 'ple_conv_kernel_size'), 'ple_conv_kernel_size');
  const sourceDtype = String(text.dtype ?? text.torch_dtype ?? '');
  const tableDtype = SOURCE_TABLE_DTYPES[sourceDtype];
  if (!tableDtype) throw new Error(`qwen4_exp n-gram table dtype ${JSON.stringify(sourceDtype)} is not a supported stored dtype.`);
  const layers = sourceLayerIds.map((sourceLayerId, pleLayerIndex) => {
    if (sourceLayerId < 1 || sourceLayerId > layerTypes.length) {
      throw new Error(`qwen4_exp ple_layer_ids must be one-indexed ids in [1, ${layerTypes.length}]; got ${sourceLayerId}.`);
    }
    if (layerTypes[sourceLayerId - 1] !== 'linear_attention') {
      throw new Error(`qwen4_exp PLE is only supported on linear_attention layers; got layer id ${sourceLayerId}.`);
    }
    const table = deriveNgramLayerTable({ vocabSizeBase, vocabPadMultiple, headCount, pleLayerIndex });
    return {
      sourceLayerId,
      decoderLayerIndex: sourceLayerId - 1,
      pleLayerIndex,
      multipliers: deriveNgramMultipliers(unigramVocabSize, ngramSize, pleLayerIndex, seed).map(String),
      headVocabSizes: table.headVocabSizes,
      headOffsets: table.headOffsets,
      unpaddedRows: table.unpaddedRows,
      rows: table.rows,
    };
  });
  return {
    semantics: 'qwen4_exp_ple',
    unigramVocabSize,
    ngramSize,
    headsPerNgram,
    headCount,
    embedDim,
    headDim: embedDim / headCount,
    vocabSizeBase,
    vocabPadMultiple,
    seed,
    boundaryTokenId: resolveBoundaryToken(text),
    convKernelSize,
    convDilation: ngramSize,
    convStateLength: (convKernelSize - 1) * ngramSize,
    normEps: sourceField(text, 'rms_norm_eps'),
    tableDtype,
    layers,
  };
}

/**
 * @param {Record<string, any>} text @param {string[]} layerTypes
 * @returns {import('../config/schema/hyper-connection-decoder.schema.js').IndexedAttentionSchema | null}
 */
function lowerIndexedAttention(text, layerTypes) {
  const values = Object.fromEntries(QSA_FIELDS.map((field) => [field, sourceField(text, field)]));
  if (QSA_FIELDS.every((field) => values[field] === null)) return null;
  const missing = QSA_FIELDS.filter((field) => values[field] === null);
  if (missing.length > 0) throw new Error(`qwen4_exp QSA config is missing required fields: ${missing.join(', ')}.`);
  QSA_FIELDS.forEach((field) => {
    if (requireInteger(values[field], field) <= 0) throw new Error(`qwen4_exp ${field} must be positive.`);
  });
  if (values.indexer_kv_heads !== 1) throw new Error('qwen4_exp QSA requires indexer_kv_heads=1.');
  if (values.indexer_budget % values.indexer_compress_ratio !== 0) {
    throw new Error('qwen4_exp indexer_budget must be divisible by indexer_compress_ratio.');
  }
  const rotaryDim = resolveRotaryDim(text);
  if (rotaryDim > values.indexer_head_dim) {
    throw new Error(`qwen4_exp attention rotary_dim ${rotaryDim} must fit indexer_head_dim ${values.indexer_head_dim}.`);
  }
  return {
    semantics: 'qwen4_exp_qsa',
    layers: layerTypes.flatMap((type, index) => (type === 'full_attention' || type === 'indexed_attention' ? [index] : [])),
    queryHeads: values.indexer_n_heads,
    keyHeads: 1,
    headDim: values.indexer_head_dim,
    rotaryDim,
    tokenBudget: values.indexer_budget,
    compressRatio: values.indexer_compress_ratio,
    blockTopK: values.indexer_budget / values.indexer_compress_ratio,
    denseVisibleLimit: values.indexer_budget + values.indexer_compress_ratio - 1,
    normEps: sourceField(text, 'rms_norm_eps'),
    tieOrder: 'lower_block_index',
    forceSelfToken: false,
  };
}

/**
 * Source-derived contracts for a qwen4_exp config, or null for other families.
 * Layer types use the checkpoint spelling ("full_attention" layers are indexed).
 * @type {typeof import('./qwen4-exp-source.js').resolveQwen4ExpSourceContracts}
 */
export function resolveQwen4ExpSourceContracts(rawConfig) {
  if (!isQwen4ExpSource(rawConfig)) return null;
  const text = /** @type {Record<string, any>} */ (textConfigOf(rawConfig));
  const layerTypes = sourceLayerTypes(text);
  const unsupported = [...new Set(layerTypes)].filter((type) => !['linear_attention', 'full_attention', 'indexed_attention'].includes(type));
  if (unsupported.length > 0) throw new Error(`Unsupported qwen4_exp layer types: ${unsupported.join(', ')}.`);
  const streamCount = requireInteger(sourceField(text, 'hc_count'), 'hc_count');
  if (streamCount <= 1) throw new Error(`qwen4_exp requires hc_count > 1; got ${streamCount}.`);
  return {
    hyperConnections: {
      semantics: 'qwen4_exp_gated_residual',
      streamCount,
      streamWidth: requireInteger(sourceField(text, 'hidden_size'), 'hidden_size'),
      projectionRank: requireInteger(sourceField(text, 'hc_lowrank'), 'hc_lowrank'),
      normEps: sourceField(text, 'rms_norm_eps'),
    },
    indexedAttention: lowerIndexedAttention(text, layerTypes),
    ngramEmbedding: lowerNgramEmbedding(text, layerTypes),
  };
}

/** @param {unknown} value @returns {string} */
function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    const record = /** @type {Record<string, unknown>} */ (value);
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

/**
 * Conversion contract: a qwen4_exp source requires every authored contract to
 * equal the source-derived one (null where the source has none); any other
 * source may not author them, because no other source mapping exists.
 * @type {typeof import('./qwen4-exp-source.js').assertConvertedQwen4ExpContracts}
 */
export function assertConvertedQwen4ExpContracts(inference, rawConfig, modelId) {
  /** @type {Record<string, unknown> | null} */
  const derived = /** @type {any} */ (resolveQwen4ExpSourceContracts(rawConfig));
  for (const field of CONTRACT_FIELDS) {
    const declared = inference?.[field];
    if (derived === null) {
      if (declared !== undefined && declared !== null) {
        throw new Error(
          `Config for "${modelId}" declares inference.${field}, but its source is not qwen4_exp; ` +
          'no source mapping establishes these semantics.'
        );
      }
      continue;
    }
    if (declared === undefined) {
      throw new Error(
        `Config for "${modelId}" converts a qwen4_exp source and must author inference.${field} ` +
        `(null only when the source has none); source derives ${canonicalJson(derived[field])}.`
      );
    }
    if (canonicalJson(declared) !== canonicalJson(derived[field])) {
      throw new Error(
        `Config for "${modelId}" declares inference.${field}=${canonicalJson(declared)}, ` +
        `but the qwen4_exp source derives ${canonicalJson(derived[field])}.`
      );
    }
  }
}

/** @type {ReadonlyArray<[string, 'multipliers' | 'headOffsets' | 'headVocabSizes']>} */
const SOURCE_BUFFER_FIELDS = Object.freeze([
  ['layer_multipliers', 'multipliers'],
  ['ngram_heads_offsets', 'headOffsets'],
  ['ngram_heads_vocab_sizes', 'headVocabSizes'],
]);

/**
 * Authenticate the checkpoint's retained n-gram buffers (I64 layer_multipliers,
 * ngram_heads_offsets, ngram_heads_vocab_sizes of every PLE layer) against the
 * authored inference.ngramEmbedding contract and record them, with their
 * SHA-256, in manifest.conversion.sourceBuffers. No-op without a contract.
 * @type {typeof import('./qwen4-exp-source.js').authenticateQwen4ExpSourceBuffers}
 */
export async function authenticateQwen4ExpSourceBuffers(tensors, converterConfig, readTensor) {
  const contract = converterConfig?.inference?.ngramEmbedding;
  if (!contract) return null;
  const byName = new Map(tensors.map((tensor) => [tensor.name, tensor]));
  /** @type {import('./qwen4-exp-source.js').Qwen4ExpSourceBufferReceipt[]} */
  const receipt = [];
  for (const layer of contract.layers) {
    for (const [field, key] of SOURCE_BUFFER_FIELDS) {
      const name = `model.language_model.layers.${layer.decoderLayerIndex}.ple.ple_embedding.${field}`;
      const tensor = byName.get(name);
      if (!tensor) throw new Error(`qwen4_exp source lacks the retained n-gram buffer ${name}.`);
      if (String(tensor.dtype).toUpperCase() !== 'I64') throw new Error(`${name} must be I64; got ${String(tensor.dtype)}.`);
      const raw = await readTensor(tensor);
      const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const values = Array.from({ length: bytes.byteLength / 8 }, (_, i) => view.getBigInt64(i * 8, true));
      const expected = /** @type {ReadonlyArray<string | number>} */ (layer[key]).map((value) => BigInt(value));
      if (values.length !== expected.length || values.some((value, i) => value !== expected[i])) {
        throw new Error(`${name} differs from the authored ngramEmbedding ${key}; re-derive the contract from the source.`);
      }
      receipt.push({ tensor: name, dtype: 'I64', sha256: sha256BytesHex(bytes), values: values.map(String) });
    }
  }
  converterConfig.manifest = {
    ...converterConfig.manifest,
    conversion: { ...converterConfig.manifest?.conversion, sourceBuffers: receipt },
  };
  return receipt;
}
