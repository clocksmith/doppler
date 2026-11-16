// Source-owned contracts for hyper-connection decoders (qwen4_exp, Qwen3.8
// Flash-Next): multi-stream gated residuals, the QSA sparse-attention indexer,
// and the hashed n-gram per-layer embedding. Pinned semantics live in
// artifacts/qwen38-flash-next-20261013/source (Transformers e0eb2810).
//
// Each contract is null for every other family. A manifest whose tensors carry
// one of these mechanisms without its contract fails here, before allocation,
// instead of executing as an ordinary pre-norm decoder.

import {
  resolveNgramEmbeddingContract,
  resolveNgramEmbeddingSession,
} from './ngram-embedding.schema.js';

export const HYPER_CONNECTION_SEMANTICS = Object.freeze(['qwen4_exp_gated_residual']);
export const INDEXED_ATTENTION_SEMANTICS = Object.freeze(['qwen4_exp_qsa']);
// Doppler's declared order among equal block scores. Upstream torch.topk leaves
// tie order unspecified, so reference comparisons check tie-invariant parts.
export const INDEXED_ATTENTION_TIE_ORDERS = Object.freeze(['lower_block_index']);

const HYPER_CONNECTION_KEYS = Object.freeze(['semantics', 'streamCount', 'streamWidth', 'projectionRank', 'normEps']);
const INDEXED_ATTENTION_KEYS = Object.freeze([
  'semantics', 'layers', 'queryHeads', 'keyHeads', 'headDim', 'rotaryDim', 'tokenBudget', 'compressRatio',
  'blockTopK', 'denseVisibleLimit', 'normEps', 'tieOrder', 'forceSelfToken',
]);

// Tensor name fragments owned by each contract (source names are kept in RDRR).
const MECHANISM_TENSOR_MARKERS = Object.freeze({
  hyperConnections: ['.attn_hyper_connection.', '.mlp_hyper_connection.', 'hyper_connection_mixer.'],
  indexedAttention: ['.self_attn.indexer.'],
  ngramEmbedding: ['.ple.'],
});

/** @param {string} label @param {string} message @returns {never} */
function fail(label, message) {
  throw new Error(`${label}: ${message}`);
}

/** @param {unknown} value @param {readonly string[]} keys @param {string} label @returns {asserts value is Record<string, any>} */
function assertExactKeys(value, keys, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) fail(label, 'must be an object or null.');
  const unknown = Object.keys(value).filter((key) => !keys.includes(key));
  if (unknown.length > 0) fail(label, `unknown field(s) ${unknown.join(', ')}.`);
  const record = /** @type {Record<string, unknown>} */ (value);
  const missing = keys.filter((key) => record[key] === undefined);
  if (missing.length > 0) fail(label, `missing required field(s) ${missing.join(', ')}.`);
}

/** @param {unknown} value @param {string} label @returns {number} */
function positiveInteger(value, label) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) fail(label, `must be a positive safe integer; got ${JSON.stringify(value)}.`);
  return value;
}

/** @param {unknown} value @param {string} label @returns {number} */
function positiveFinite(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) fail(label, 'must be a positive finite number.');
  return value;
}

/**
 * qwen4_exp_gated_residual: grouped RMSNorm over streamCount groups of
 * streamWidth with (1 + weight); mix = sigmoid(up(silu(down(n) / streamCount)));
 * branch input = mean over streams of mix * n; injection = 2 * sigmoid(inject(n)
 * / streamCount) scales the branch output added to the un-normalized streams.
 * The embedding enters replicated on every stream; the root mixer reduces
 * without injection and replaces the final norm.
 * @type {typeof import('./hyper-connection-decoder.schema.js').resolveHyperConnectionsContract}
 */
export function resolveHyperConnectionsContract(contract, geometry, label = 'inference.hyperConnections') {
  if (contract === null || contract === undefined) return null;
  assertExactKeys(contract, HYPER_CONNECTION_KEYS, label);
  if (!HYPER_CONNECTION_SEMANTICS.includes(contract.semantics)) {
    fail(`${label}.semantics`, `must be one of ${HYPER_CONNECTION_SEMANTICS.join(', ')}; got ${JSON.stringify(contract.semantics)}.`);
  }
  const streamCount = positiveInteger(contract.streamCount, `${label}.streamCount`);
  if (streamCount < 2) fail(`${label}.streamCount`, 'qwen4_exp requires more than one residual stream.');
  if (contract.streamWidth !== geometry.hiddenSize) {
    fail(`${label}.streamWidth`, `${contract.streamWidth} must equal architecture.hiddenSize ${geometry.hiddenSize}.`);
  }
  return Object.freeze({
    semantics: contract.semantics,
    streamCount,
    streamWidth: contract.streamWidth,
    projectionRank: positiveInteger(contract.projectionRank, `${label}.projectionRank`),
    normEps: positiveFinite(contract.normEps, `${label}.normEps`),
  });
}

/**
 * qwen4_exp_qsa: per query, visible tokens come from its causal (and padding)
 * mask. With at most denseVisibleLimit = tokenBudget + compressRatio - 1
 * visible tokens every visible token is selected. Otherwise visible tokens form
 * consecutive blocks of compressRatio from the first visible token; a block key
 * is RMSNorm(mean of raw keys) rotated at the block's first position; the score
 * is the sum over query heads of relu(q . k) / sqrt(headDim); blockTopK blocks
 * are kept and the trailing incomplete block is always kept. Main attention
 * consumes the original per-token keys and values of selected tokens.
 * @type {typeof import('./hyper-connection-decoder.schema.js').resolveIndexedAttentionContract}
 */
export function resolveIndexedAttentionContract(contract, geometry, label = 'inference.indexedAttention') {
  if (contract === null || contract === undefined) return null;
  assertExactKeys(contract, INDEXED_ATTENTION_KEYS, label);
  if (!INDEXED_ATTENTION_SEMANTICS.includes(contract.semantics)) {
    fail(`${label}.semantics`, `must be one of ${INDEXED_ATTENTION_SEMANTICS.join(', ')}; got ${JSON.stringify(contract.semantics)}.`);
  }
  const { numLayers, layerTypes = null, attentionRotaryDim = null } = geometry;
  if (!Array.isArray(contract.layers) || contract.layers.length === 0) {
    fail(`${label}.layers`, 'must list the indexed full-attention layers (use null for models without QSA).');
  }
  contract.layers.forEach((layer, index) => {
    if (!Number.isSafeInteger(layer) || layer < 0 || layer >= numLayers) fail(`${label}.layers[${index}]`, `must be a layer index below ${numLayers}.`);
    if (index > 0 && layer <= contract.layers[index - 1]) fail(`${label}.layers`, 'must be sorted without duplicates.');
    if (layerTypes && layerTypes[layer] !== 'full_attention') {
      fail(`${label}.layers[${index}]`, `layer ${layer} is ${JSON.stringify(layerTypes[layer])}, not full_attention.`);
    }
  });
  if (layerTypes) {
    const fullLayers = layerTypes.flatMap((type, index) => (type === 'full_attention' ? [index] : []));
    if (fullLayers.length !== contract.layers.length) {
      fail(`${label}.layers`, `qwen4_exp indexes every full-attention layer ${JSON.stringify(fullLayers)}.`);
    }
  }
  const queryHeads = positiveInteger(contract.queryHeads, `${label}.queryHeads`);
  if (contract.keyHeads !== 1) fail(`${label}.keyHeads`, 'qwen4_exp QSA requires exactly one indexer key head.');
  const headDim = positiveInteger(contract.headDim, `${label}.headDim`);
  const rotaryDim = positiveInteger(contract.rotaryDim, `${label}.rotaryDim`);
  if (rotaryDim > headDim || rotaryDim % 2 !== 0) fail(`${label}.rotaryDim`, `must be even and at most headDim ${headDim}.`);
  if (attentionRotaryDim !== null && rotaryDim !== attentionRotaryDim) {
    fail(`${label}.rotaryDim`, `${rotaryDim} must equal the attention rotary dimension ${attentionRotaryDim}; the indexer shares its RoPE.`);
  }
  const tokenBudget = positiveInteger(contract.tokenBudget, `${label}.tokenBudget`);
  const compressRatio = positiveInteger(contract.compressRatio, `${label}.compressRatio`);
  if (tokenBudget % compressRatio !== 0) fail(`${label}.tokenBudget`, 'must be divisible by compressRatio.');
  if (contract.blockTopK !== tokenBudget / compressRatio) {
    fail(`${label}.blockTopK`, `must equal tokenBudget / compressRatio = ${tokenBudget / compressRatio}.`);
  }
  if (contract.denseVisibleLimit !== tokenBudget + compressRatio - 1) {
    fail(`${label}.denseVisibleLimit`, `must equal tokenBudget + compressRatio - 1 = ${tokenBudget + compressRatio - 1}.`);
  }
  if (!INDEXED_ATTENTION_TIE_ORDERS.includes(contract.tieOrder)) {
    fail(`${label}.tieOrder`, `must be one of ${INDEXED_ATTENTION_TIE_ORDERS.join(', ')}.`);
  }
  if (contract.forceSelfToken !== false) {
    fail(`${label}.forceSelfToken`, 'must be false: the reference indexer does not force a query\'s own token into selected blocks.');
  }
  return Object.freeze({
    semantics: contract.semantics,
    layers: Object.freeze([...contract.layers]),
    queryHeads,
    keyHeads: 1,
    headDim,
    rotaryDim,
    tokenBudget,
    compressRatio,
    blockTopK: contract.blockTopK,
    denseVisibleLimit: contract.denseVisibleLimit,
    normEps: positiveFinite(contract.normEps, `${label}.normEps`),
    tieOrder: contract.tieOrder,
    forceSelfToken: false,
  });
}

/**
 * @param {readonly string[] | null} tensorNames @param {Record<string, unknown>} contracts @param {string} modelId
 */
function assertNoUndeclaredMechanismTensors(tensorNames, contracts, modelId) {
  if (!tensorNames) return;
  for (const [field, markers] of Object.entries(MECHANISM_TENSOR_MARKERS)) {
    if (contracts[field] !== null) continue;
    const owned = tensorNames.find((name) => markers.some((marker) => name.includes(marker)));
    if (owned !== undefined) {
      throw new Error(
        `Manifest "${modelId}" carries ${JSON.stringify(owned)} but declares no inference.${field} contract. ` +
        'Re-convert with an explicit source-owned contract; Doppler will not run these tensors as another family.'
      );
    }
  }
}

/**
 * Resolve the hyper-connection decoder contracts of a merged inference config.
 * Missing contracts mean the mechanism is absent; contracts that depend on each
 * other or contradict pre-norm residual fields fail before any allocation.
 * @type {typeof import('./hyper-connection-decoder.schema.js').resolveHyperConnectionDecoderContracts}
 */
export function resolveHyperConnectionDecoderContracts(inference, geometry) {
  const { modelId = 'unknown', tensorNames = null } = geometry;
  const hyperConnections = resolveHyperConnectionsContract(inference.hyperConnections, geometry);
  const indexedAttention = resolveIndexedAttentionContract(inference.indexedAttention, geometry);
  const ngramEmbedding = resolveNgramEmbeddingContract(inference.ngramEmbedding, geometry);
  const ngramEmbeddingSession = resolveNgramEmbeddingSession(inference.session?.ngramEmbedding, ngramEmbedding);
  assertNoUndeclaredMechanismTensors(tensorNames, { hyperConnections, indexedAttention, ngramEmbedding }, modelId);
  if ((indexedAttention !== null || ngramEmbedding !== null) && hyperConnections === null) {
    throw new Error(
      `Manifest "${modelId}" declares qwen4_exp indexedAttention or ngramEmbedding without hyperConnections; ` +
      'their qwen4_exp semantics are defined only inside the hyper-connection decoder.'
    );
  }
  if (hyperConnections !== null) {
    /** @type {Record<string, unknown>} */
    const normalization = inference.normalization ?? {};
    const contradictions = ['postAttentionNorm', 'preFeedforwardNorm', 'postFeedforwardNorm']
      .filter((field) => normalization[field] !== false);
    if (contradictions.length > 0 || inference.layerPattern?.residualBranchScale !== 1) {
      throw new Error(
        `Manifest "${modelId}" declares hyperConnections, which replace pre-norm residuals; ` +
        `normalization.{${contradictions.join(', ')}} must be false and layerPattern.residualBranchScale must be 1.`
      );
    }
  }
  return { hyperConnections, indexedAttention, ngramEmbedding, ngramEmbeddingSession };
}
