// Hashed n-gram per-layer embedding (qwen4_exp "PLE") contract.
//
// Source semantics are pinned in artifacts/qwen38-flash-next-20261013/source
// (Transformers e0eb2810, modeling_qwen4_exp.py Qwen4ExpTextNGramEmbedding and
// Qwen4ExpTextPLELayer). The manifest declares every derived value; validation
// recomputes them with exact integer arithmetic. Multipliers reach 63-bit
// products with token ids, so they are BigInt values serialized as decimal
// strings; JavaScript Number cannot represent them.

export const NGRAM_EMBEDDING_SEMANTICS = Object.freeze(['qwen4_exp_ple']);
export const NGRAM_EMBEDDING_TABLE_DTYPES = Object.freeze(['bf16']);
export const NGRAM_ROW_CACHE_EVICTIONS = Object.freeze(['lru']);

const MASK64 = (1n << 64n) - 1n;
const MAX_INT64 = (1n << 63n) - 1n;
const SPLITMIX_GAMMA = 0x9E3779B97F4A7C15n;
const SPLITMIX_M1 = 0xBF58476D1CE4E5B9n;
const SPLITMIX_M2 = 0x94D049BB133111EBn;
const PRIME_1 = 10007n;
// Trial-division prime search stays bounded: sqrt(2^32) = 65536 divisors.
const MAX_VOCAB_SIZE_BASE = 2 ** 32;

const CONTRACT_KEYS = Object.freeze([
  'semantics', 'unigramVocabSize', 'ngramSize', 'headsPerNgram', 'headCount', 'embedDim', 'headDim',
  'vocabSizeBase', 'vocabPadMultiple', 'seed', 'boundaryTokenId', 'convKernelSize', 'convDilation',
  'convStateLength', 'normEps', 'tableDtype', 'layers',
]);
const LAYER_KEYS = Object.freeze([
  'sourceLayerId', 'decoderLayerIndex', 'pleLayerIndex', 'multipliers', 'headVocabSizes', 'headOffsets',
  'unpaddedRows', 'rows',
]);
const SESSION_KEYS = Object.freeze(['rowCache', 'staging', 'reads']);

/** @param {string} label @param {string} message @returns {never} */
function fail(label, message) {
  throw new Error(`${label}: ${message}`);
}

/** @param {unknown} value @returns {value is Record<string, any>} */
function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** @param {unknown} value @param {readonly string[]} keys @param {string} label @returns {asserts value is Record<string, any>} */
function assertExactKeys(value, keys, label) {
  if (!isPlainObject(value)) fail(label, 'must be an object.');
  const unknown = Object.keys(value).filter((key) => !keys.includes(key));
  if (unknown.length > 0) fail(label, `unknown field(s) ${unknown.join(', ')}.`);
  const missing = keys.filter((key) => value[key] === undefined);
  if (missing.length > 0) fail(label, `missing required field(s) ${missing.join(', ')}.`);
}

/** @param {unknown} value @param {string} label @returns {number} */
function positiveInteger(value, label) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) fail(label, `must be a positive safe integer; got ${JSON.stringify(value)}.`);
  return value;
}

/** @param {unknown} value @param {string} label @returns {number} */
function nonNegativeInteger(value, label) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) fail(label, `must be a non-negative safe integer; got ${JSON.stringify(value)}.`);
  return value;
}

/** @param {bigint} value @returns {bigint} */
function splitmix64(value) {
  let v = (value + SPLITMIX_GAMMA) & MASK64;
  v = ((v ^ (v >> 30n)) * SPLITMIX_M1) & MASK64;
  v = ((v ^ (v >> 27n)) * SPLITMIX_M2) & MASK64;
  return (v ^ (v >> 31n)) & MASK64;
}

/** @param {number} value @returns {boolean} */
function isPrime(value) {
  if (value < 2) return false;
  if (value % 2 === 0) return value === 2;
  const limit = Math.floor(Math.sqrt(value));
  for (let divisor = 3; divisor <= limit; divisor += 2) {
    if (value % divisor === 0) return false;
  }
  return true;
}

/** Upstream `_build_layer_multipliers`, exact. @type {typeof import('./ngram-embedding.schema.js').deriveNgramMultipliers} */
export function deriveNgramMultipliers(unigramVocabSize, ngramSize, pleLayerIndex, seed) {
  const multiplierMax = MAX_INT64 / BigInt(Math.max(unigramVocabSize, 1));
  const halved = multiplierMax / 2n;
  const halfBound = halved > 1n ? halved : 1n;
  const baseSeed = BigInt(seed) + PRIME_1 * BigInt(pleLayerIndex);
  const multipliers = [];
  for (let index = 0; index < ngramSize; index++) {
    const value = (baseSeed + SPLITMIX_GAMMA * BigInt(index + 1)) & MASK64;
    multipliers.push(2n * (splitmix64(value) % halfBound) + 1n);
  }
  return multipliers;
}

/**
 * Upstream head vocabulary sizes and offsets for one PLE layer: head h uses the
 * (pleLayerIndex * headCount + h + 1)-th prime after vocabSizeBase - 1; rows are
 * padded to a multiple of vocabPadMultiple.
 * @type {typeof import('./ngram-embedding.schema.js').deriveNgramLayerTable}
 */
export function deriveNgramLayerTable({ vocabSizeBase, vocabPadMultiple, headCount, pleLayerIndex }) {
  if (!Number.isSafeInteger(vocabSizeBase) || vocabSizeBase < 2 || vocabSizeBase > MAX_VOCAB_SIZE_BASE) {
    throw new Error(`ngram vocabSizeBase must be an integer in [2, ${MAX_VOCAB_SIZE_BASE}]; got ${vocabSizeBase}.`);
  }
  const first = pleLayerIndex * headCount;
  const headVocabSizes = [];
  let prime = vocabSizeBase - 1;
  for (let count = 0; count < first + headCount; count++) {
    prime += 1;
    while (!isPrime(prime)) prime += 1;
    if (count >= first) headVocabSizes.push(prime);
  }
  const headOffsets = [];
  let unpaddedRows = 0;
  for (const size of headVocabSizes) {
    headOffsets.push(unpaddedRows);
    unpaddedRows += size;
  }
  const rows = Math.ceil(unpaddedRows / vocabPadMultiple) * vocabPadMultiple;
  if (!Number.isSafeInteger(rows) || rows > 2 ** 32) {
    throw new Error(`ngram table rows ${rows} exceed the 32-bit row index range.`);
  }
  return { headVocabSizes, headOffsets, unpaddedRows, rows };
}

/** @param {unknown} value @param {string} label @returns {bigint} */
function parseMultiplier(value, label) {
  if (typeof value !== 'string' || !/^[1-9][0-9]*$/.test(value)) {
    fail(label, `must be a positive decimal integer string; got ${JSON.stringify(value)}.`);
  }
  return BigInt(value);
}

/**
 * @param {unknown} layer @param {Record<string, any>} contract @param {number} numLayers
 * @param {readonly string[] | null} layerTypes @param {string} label
 * @returns {import('./ngram-embedding.schema.js').ResolvedNgramEmbeddingLayer}
 */
function validateLayer(layer, contract, numLayers, layerTypes, label) {
  assertExactKeys(layer, LAYER_KEYS, label);
  const sourceLayerId = positiveInteger(layer.sourceLayerId, `${label}.sourceLayerId`);
  const decoderLayerIndex = nonNegativeInteger(layer.decoderLayerIndex, `${label}.decoderLayerIndex`);
  // Source ple_layer_ids are one-based decoder ids; source [2] selects index 1.
  if (decoderLayerIndex !== sourceLayerId - 1) {
    fail(label, `decoderLayerIndex ${decoderLayerIndex} must equal one-based sourceLayerId ${sourceLayerId} minus one.`);
  }
  if (decoderLayerIndex >= numLayers) fail(label, `decoderLayerIndex ${decoderLayerIndex} is outside ${numLayers} layers.`);
  if (layerTypes && layerTypes[decoderLayerIndex] !== 'linear_attention') {
    fail(label, `qwen4_exp PLE runs only on linear_attention layers; layer ${decoderLayerIndex} is ${JSON.stringify(layerTypes[decoderLayerIndex])}.`);
  }
  const pleLayerIndex = nonNegativeInteger(layer.pleLayerIndex, `${label}.pleLayerIndex`);
  /** @type {Record<string, any>} */
  const expectedTable = deriveNgramLayerTable({
    vocabSizeBase: contract.vocabSizeBase, vocabPadMultiple: contract.vocabPadMultiple, headCount: contract.headCount, pleLayerIndex,
  });
  const expectedMultipliers = deriveNgramMultipliers(contract.unigramVocabSize, contract.ngramSize, pleLayerIndex, contract.seed);
  if (!Array.isArray(layer.multipliers) || layer.multipliers.length !== contract.ngramSize) {
    fail(`${label}.multipliers`, `must list ${contract.ngramSize} multipliers.`);
  }
  const multipliers = layer.multipliers.map((value, index) => parseMultiplier(value, `${label}.multipliers[${index}]`));
  multipliers.forEach((value, index) => {
    if (value !== expectedMultipliers[index]) {
      fail(`${label}.multipliers[${index}]`, `declares ${value}, but seed ${contract.seed} derives ${expectedMultipliers[index]}.`);
    }
  });
  for (const field of ['headVocabSizes', 'headOffsets']) {
    const declared = layer[field];
    if (!Array.isArray(declared) || declared.length !== contract.headCount
      || declared.some((value, index) => value !== expectedTable[field][index])) {
      fail(`${label}.${field}`, `must equal the derived ${JSON.stringify(expectedTable[field])}.`);
    }
  }
  for (const field of ['unpaddedRows', 'rows']) {
    if (layer[field] !== expectedTable[field]) {
      fail(`${label}.${field}`, `declares ${layer[field]}, derived ${expectedTable[field]}.`);
    }
  }
  return Object.freeze({
    sourceLayerId, decoderLayerIndex, pleLayerIndex,
    multipliers: Object.freeze(multipliers),
    headVocabSizes: Object.freeze([...expectedTable.headVocabSizes]),
    headOffsets: Object.freeze([...expectedTable.headOffsets]),
    unpaddedRows: expectedTable.unpaddedRows,
    rows: expectedTable.rows,
  });
}

/**
 * Validate a declared ngramEmbedding contract against model geometry. Returns a
 * frozen copy with BigInt multipliers, or null when the model declares none.
 * @type {typeof import('./ngram-embedding.schema.js').resolveNgramEmbeddingContract}
 */
export function resolveNgramEmbeddingContract(contract, geometry, label = 'inference.ngramEmbedding') {
  if (contract === null || contract === undefined) return null;
  assertExactKeys(contract, CONTRACT_KEYS, label);
  /** @type {Record<string, any>} */
  const record = contract;
  const { hiddenSize, numLayers, vocabSize, layerTypes = null } = geometry;
  if (!NGRAM_EMBEDDING_SEMANTICS.includes(contract.semantics)) {
    fail(`${label}.semantics`, `must be one of ${NGRAM_EMBEDDING_SEMANTICS.join(', ')}; got ${JSON.stringify(contract.semantics)}.`);
  }
  for (const field of ['unigramVocabSize', 'ngramSize', 'headsPerNgram', 'headCount', 'embedDim', 'headDim',
    'vocabSizeBase', 'vocabPadMultiple', 'convKernelSize', 'convDilation', 'convStateLength']) {
    positiveInteger(record[field], `${label}.${field}`);
  }
  if (!Number.isSafeInteger(contract.seed)) fail(`${label}.seed`, 'must be a safe integer.');
  nonNegativeInteger(contract.boundaryTokenId, `${label}.boundaryTokenId`);
  if (!(typeof contract.normEps === 'number' && Number.isFinite(contract.normEps) && contract.normEps > 0)) {
    fail(`${label}.normEps`, 'must be a positive finite number.');
  }
  if (!NGRAM_EMBEDDING_TABLE_DTYPES.includes(contract.tableDtype)) {
    fail(`${label}.tableDtype`, `must be one of ${NGRAM_EMBEDDING_TABLE_DTYPES.join(', ')}.`);
  }
  if (contract.ngramSize < 2) fail(`${label}.ngramSize`, 'must be at least 2.');
  if (contract.unigramVocabSize !== vocabSize) {
    fail(`${label}.unigramVocabSize`, `${contract.unigramVocabSize} must equal architecture.vocabSize ${vocabSize}.`);
  }
  if (contract.boundaryTokenId >= contract.unigramVocabSize) {
    fail(`${label}.boundaryTokenId`, `${contract.boundaryTokenId} is outside the ${contract.unigramVocabSize}-token vocabulary.`);
  }
  if (contract.headCount !== (contract.ngramSize - 1) * contract.headsPerNgram) {
    fail(`${label}.headCount`, `must equal (ngramSize - 1) * headsPerNgram = ${(contract.ngramSize - 1) * contract.headsPerNgram}.`);
  }
  if (contract.embedDim % contract.headCount !== 0 || contract.headDim !== contract.embedDim / contract.headCount) {
    fail(`${label}.headDim`, `must equal embedDim / headCount (${contract.embedDim} / ${contract.headCount}).`);
  }
  if (contract.convDilation !== contract.ngramSize) {
    fail(`${label}.convDilation`, `qwen4_exp dilates the PLE convolution by ngramSize ${contract.ngramSize}.`);
  }
  if (contract.convStateLength !== (contract.convKernelSize - 1) * contract.convDilation) {
    fail(`${label}.convStateLength`, 'must equal (convKernelSize - 1) * convDilation.');
  }
  if (!Array.isArray(contract.layers) || contract.layers.length === 0) {
    fail(`${label}.layers`, 'must list at least one PLE layer (use null for models without PLE).');
  }
  const layers = contract.layers.map((layer, index) => validateLayer(layer, record, numLayers, layerTypes, `${label}.layers[${index}]`));
  layers.forEach((layer, index) => {
    if (layer.pleLayerIndex !== index) fail(`${label}.layers[${index}].pleLayerIndex`, 'must equal the position in sorted source ple_layer_ids.');
    if (index > 0 && layer.decoderLayerIndex <= layers[index - 1].decoderLayerIndex) {
      fail(`${label}.layers`, 'must be sorted by decoderLayerIndex without duplicates.');
    }
  });
  if (!Number.isSafeInteger(hiddenSize) || hiddenSize <= 0) fail(label, 'requires architecture.hiddenSize.');
  return Object.freeze({
    semantics: contract.semantics,
    unigramVocabSize: contract.unigramVocabSize,
    ngramSize: contract.ngramSize,
    headsPerNgram: contract.headsPerNgram,
    headCount: contract.headCount,
    embedDim: contract.embedDim,
    headDim: contract.headDim,
    vocabSizeBase: contract.vocabSizeBase,
    vocabPadMultiple: contract.vocabPadMultiple,
    seed: contract.seed,
    boundaryTokenId: contract.boundaryTokenId,
    convKernelSize: contract.convKernelSize,
    convDilation: contract.convDilation,
    convStateLength: contract.convStateLength,
    normEps: contract.normEps,
    tableDtype: contract.tableDtype,
    layers: Object.freeze(layers),
  });
}

/** Serialize a resolved contract back to its manifest JSON form. @type {typeof import('./ngram-embedding.schema.js').serializeNgramEmbeddingContract} */
export function serializeNgramEmbeddingContract(resolved) {
  return {
    ...resolved,
    layers: resolved.layers.map((layer) => ({
      ...layer,
      multipliers: layer.multipliers.map((value) => value.toString()),
      headVocabSizes: [...layer.headVocabSizes],
      headOffsets: [...layer.headOffsets],
    })),
  };
}

/**
 * Validate the n-gram table session policy. It is required when the model
 * declares an ngramEmbedding contract and must be absent (null) otherwise.
 * @type {typeof import('./ngram-embedding.schema.js').resolveNgramEmbeddingSession}
 */
export function resolveNgramEmbeddingSession(session, contract, label = 'inference.session.ngramEmbedding') {
  if (contract === null) {
    if (session !== null && session !== undefined) fail(label, 'is set but the model declares no ngramEmbedding contract.');
    return null;
  }
  if (session === null || session === undefined) {
    fail(label, 'is required for models with an ngramEmbedding contract (row cache, staging and read bounds).');
  }
  assertExactKeys(session, SESSION_KEYS, label);
  assertExactKeys(session.rowCache, ['maxRows', 'maxBytes', 'eviction'], `${label}.rowCache`);
  assertExactKeys(session.staging, ['maxRows'], `${label}.staging`);
  assertExactKeys(session.reads, ['maxConcurrent'], `${label}.reads`);
  const maxRows = nonNegativeInteger(session.rowCache.maxRows, `${label}.rowCache.maxRows`);
  const maxBytes = nonNegativeInteger(session.rowCache.maxBytes, `${label}.rowCache.maxBytes`);
  if (!NGRAM_ROW_CACHE_EVICTIONS.includes(session.rowCache.eviction)) {
    fail(`${label}.rowCache.eviction`, `must be one of ${NGRAM_ROW_CACHE_EVICTIONS.join(', ')}.`);
  }
  const stagingRows = positiveInteger(session.staging.maxRows, `${label}.staging.maxRows`);
  if (stagingRows < contract.headCount) {
    fail(`${label}.staging.maxRows`, `must hold at least one token's ${contract.headCount} rows.`);
  }
  return Object.freeze({
    rowCache: Object.freeze({ maxRows, maxBytes, eviction: session.rowCache.eviction }),
    staging: Object.freeze({ maxRows: stagingRows }),
    reads: Object.freeze({ maxConcurrent: positiveInteger(session.reads.maxConcurrent, `${label}.reads.maxConcurrent`) }),
  });
}
