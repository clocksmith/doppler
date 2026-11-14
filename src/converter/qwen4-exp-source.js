import { sha256BytesHex } from '../formats/sha256.js';
import { resolveQwen4ExpSourceContracts } from '../formats/qwen4-exp-source-contracts.js';

const CONTRACT_FIELDS = Object.freeze(['hyperConnections', 'indexedAttention', 'ngramEmbedding']);

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
