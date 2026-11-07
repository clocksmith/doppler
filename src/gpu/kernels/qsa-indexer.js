import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';
import { selectRuleValue } from '../../rules/rule-registry.js';

// qwen4_exp QSA indexer operators (Qwen4ExpTextQSAIndexer) and attention over
// the selected cache rows. Key caches are caller-owned buffers; every operator
// validates geometry before dispatch and supports immediate and recorded use.

const MAX_U32 = 0xffffffff;
const NORM_WORKGROUP = 128;
const WIDE_WORKGROUP = 256;

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[QSA] ${message}`);
}

/** @param {unknown} value @param {string} name @param {number} [minimum] */
function integer(value, name, minimum = 1) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum) {
    fail(`${name} must be an integer >= ${minimum}; got ${String(value)}.`);
  }
  return value;
}

/** @param {unknown} value @param {string} name */
function positive(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) fail(`${name} must be a positive finite number.`);
  return value;
}

/** @param {import('../tensor.js').Tensor} tensor @param {readonly number[]} shape @param {string} label @param {string} [dtype] */
function assertTensor(tensor, shape, label, dtype = 'f32') {
  const elements = shape.reduce((product, dim) => product * dim, 1);
  if (tensor?.dtype !== dtype || !Array.isArray(tensor.shape) || tensor.shape.length !== shape.length
    || tensor.shape.some((dim, index) => dim !== shape[index])) {
    fail(`${label} must be a ${dtype} tensor of shape [${shape.join(', ')}]; got ${tensor?.dtype} [${tensor?.shape?.join(', ')}].`);
  }
  assertBuffer(tensor.buffer, elements * (dtype === 'f16' ? 2 : 4), label);
}

/** @param {GPUBuffer} buffer @param {number} bytes @param {string} label */
function assertBuffer(buffer, bytes, label) {
  if (!buffer || !Number.isSafeInteger(buffer.size) || buffer.size < bytes) fail(`${label} buffer must hold ${bytes} bytes.`);
}

/** @param {import('./qsa-indexer.js').QsaRopeTables} rope @param {number} positions @param {number} rotaryDim */
function assertRope(rope, positions, rotaryDim) {
  const bytes = positions * (rotaryDim / 2) * 4;
  assertBuffer(rope?.cos, bytes, 'RoPE cos table');
  assertBuffer(rope?.sin, bytes, 'RoPE sin table');
}

/** @param {import('../command-recorder.js').CommandRecorder | null} target @param {number} groups */
function groupGrid(target, groups) {
  const device = target?.device ?? getDevice();
  const maxGroups = device?.limits?.maxComputeWorkgroupsPerDimension;
  if (typeof maxGroups !== 'number' || maxGroups < 1) fail('active device dispatch limits are required.');
  const columns = Math.max(1, Math.min(groups, maxGroups));
  const rows = Math.ceil(groups / columns);
  if (rows > maxGroups) fail('geometry exceeds device dispatch capacity.');
  return { columns, grid: [columns, rows, 1] };
}

/** @param {import('./qsa-indexer.js').QsaIndexerGeometry} geometry */
function validateIndexerGeometry({ queryHeads, headDim, rotaryDim }) {
  integer(queryHeads, 'queryHeads');
  integer(headDim, 'headDim');
  integer(rotaryDim, 'rotaryDim', 2);
  if (rotaryDim % 2 !== 0 || rotaryDim > headDim) fail('rotaryDim must be even and at most headDim.');
  if (headDim > 512 || queryHeads * headDim > 2048) fail('indexer heads exceed the kernel workgroup storage.');
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} projection
 * @param {import('../tensor.js').Tensor} qNormWeight
 * @param {import('./qsa-indexer.js').QsaRopeTables} rope
 * @param {GPUBuffer} rawKeys
 * @param {import('./qsa-indexer.js').QsaPrepareOptions} options
 */
async function executePrepare(target, projection, qNormWeight, rope, rawKeys, options) {
  const { numTokens, startPos, queryHeads, headDim, rotaryDim, eps, capacity, outputBuffer = null } = options;
  validateIndexerGeometry({ queryHeads, headDim, rotaryDim });
  integer(numTokens, 'numTokens');
  integer(startPos, 'startPos', 0);
  integer(capacity, 'capacity');
  positive(eps, 'eps');
  if (startPos + numTokens > capacity) fail(`append of ${numTokens} rows at ${startPos} exceeds cache capacity ${capacity}.`);
  assertTensor(projection, [numTokens, (queryHeads + 1) * headDim], 'index_qk projection');
  assertTensor(qNormWeight, [headDim], 'q_layernorm weight');
  assertBuffer(rawKeys, capacity * headDim * 4, 'raw key cache');
  assertRope(rope, startPos + numTokens, rotaryDim);
  const bytes = numTokens * queryHeads * headDim * 4;
  if (outputBuffer && (outputBuffer.size < bytes || outputBuffer === projection.buffer || outputBuffer === rawKeys)) {
    fail('query outputBuffer must be large enough and distinct.');
  }
  const { columns, grid } = groupGrid(target, numTokens * (queryHeads + 1));
  return withKernelOutput(target, outputBuffer, bytes, 'qsa_query', async (output) => {
    await unifiedKernelWrapper('qsa_prepare', target, 'f32',
      [projection, qNormWeight, rope.cos, rope.sin, output, rawKeys],
      { num_tokens: numTokens, start_pos: startPos, group_stride: columns, eps }, grid,
      { WORKGROUP_SIZE: NORM_WORKGROUP, QUERY_HEADS: queryHeads, HEAD_DIM: headDim, ROTARY_DIM: rotaryDim });
    return createTensor(output, 'f32', [numTokens, queryHeads * headDim], 'qsa_query');
  });
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {GPUBuffer} rawKeys
 * @param {import('../tensor.js').Tensor} kNormWeight
 * @param {import('./qsa-indexer.js').QsaRopeTables} rope
 * @param {GPUBuffer} blockKeys
 * @param {import('./qsa-indexer.js').QsaBlockKeyOptions} options
 */
async function executeBlockKeys(target, rawKeys, kNormWeight, rope, blockKeys, options) {
  const { firstBlock, blockCount, headDim, rotaryDim, compressRatio, eps, capacity } = options;
  validateIndexerGeometry({ queryHeads: 1, headDim, rotaryDim });
  integer(firstBlock, 'firstBlock', 0);
  integer(blockCount, 'blockCount', 0);
  integer(compressRatio, 'compressRatio');
  integer(capacity, 'capacity');
  positive(eps, 'eps');
  if (blockCount === 0) return;
  const lastToken = (firstBlock + blockCount) * compressRatio;
  if (lastToken > capacity) fail(`blocks through ${firstBlock + blockCount} exceed cache capacity ${capacity}.`);
  assertBuffer(rawKeys, capacity * headDim * 4, 'raw key cache');
  assertBuffer(blockKeys, Math.floor(capacity / compressRatio) * headDim * 4, 'block key cache');
  assertTensor(kNormWeight, [headDim], 'k_layernorm weight');
  assertRope(rope, lastToken, rotaryDim);
  const { columns, grid } = groupGrid(target, blockCount);
  await unifiedKernelWrapper('qsa_block_keys', target, 'f32',
    [rawKeys, kNormWeight, rope.cos, rope.sin, blockKeys],
    { first_block: firstBlock, block_count: blockCount, group_stride: columns, eps }, grid,
    { WORKGROUP_SIZE: NORM_WORKGROUP, HEAD_DIM: headDim, ROTARY_DIM: rotaryDim, COMPRESS_RATIO: compressRatio });
}

/** @param {import('./qsa-indexer.js').QsaSelectionGeometry} options */
function validateSelectionGeometry(options) {
  const { numTokens, startPos, compressRatio, blockTopK, denseVisibleLimit } = options;
  integer(numTokens, 'numTokens');
  integer(startPos, 'startPos', 0);
  integer(compressRatio, 'compressRatio');
  integer(blockTopK, 'blockTopK');
  if (denseVisibleLimit !== blockTopK * compressRatio + compressRatio - 1) {
    fail('denseVisibleLimit must equal blockTopK * compressRatio + compressRatio - 1.');
  }
  const blockStride = Math.floor((startPos + numTokens) / compressRatio);
  if (numTokens * blockStride > MAX_U32 / 4) fail('score scratch exceeds u32 addressing; split the query rows.');
  return { blockStride, maxSelected: denseVisibleLimit };
}

/** @type {typeof import('./qsa-indexer.js').qsaHasSparseQueries} */
export function qsaHasSparseQueries({ numTokens, startPos, denseVisibleLimit }) {
  return startPos + numTokens > denseVisibleLimit;
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} query
 * @param {GPUBuffer} blockKeys
 * @param {import('./qsa-indexer.js').QsaScoreOptions} options
 */
async function executeScores(target, query, blockKeys, options) {
  const { queryHeads, headDim, scoreDivisor } = options;
  validateIndexerGeometry({ queryHeads, headDim, rotaryDim: 2 });
  positive(scoreDivisor, 'scoreDivisor');
  const { blockStride } = validateSelectionGeometry(options);
  assertTensor(query, [options.numTokens, queryHeads * headDim], 'QSA query');
  if (!qsaHasSparseQueries(options)) return null;
  assertBuffer(blockKeys, blockStride * headDim * 4, 'block key cache');
  const bytes = options.numTokens * blockStride * 4;
  const groups = [Math.ceil(blockStride / WIDE_WORKGROUP), options.numTokens, 1];
  return withKernelOutput(target, null, bytes, 'qsa_scores', async (output) => {
    await unifiedKernelWrapper('qsa_block_scores', target, 'f32', [query, blockKeys, output],
      { num_tokens: options.numTokens, start_pos: options.startPos, block_stride: blockStride,
        dense_visible_limit: options.denseVisibleLimit, score_divisor: scoreDivisor, _pad0: 0, _pad1: 0, _pad2: 0 },
      groups, { WORKGROUP_SIZE: WIDE_WORKGROUP, QUERY_HEADS: queryHeads, HEAD_DIM: headDim, COMPRESS_RATIO: options.compressRatio });
    return createTensor(output, 'f32', [options.numTokens, blockStride], 'qsa_scores');
  });
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor | null} scores
 * @param {import('./qsa-indexer.js').QsaSelectionGeometry} options
 */
async function executeSelect(target, scores, options) {
  const { blockStride, maxSelected } = validateSelectionGeometry(options);
  const sparse = qsaHasSparseQueries(options);
  if (sparse) {
    if (!scores) fail('scores are required when a query is sparse.');
    assertTensor(scores, [options.numTokens, blockStride], 'QSA scores');
  }
  else if (scores !== null) fail('scores must be null when every query takes the dense shortcut.');
  const numTokens = options.numTokens;
  const release = (/** @type {GPUBuffer} */ buffer) => {
    if (target && typeof target.beginComputePass === 'function') target.trackTemporaryBuffer(buffer);
    else releaseBuffer(buffer);
  };
  /** @type {GPUBuffer | null} */
  let selection = null;
  /** @type {GPUBuffer | null} */
  let meta = null;
  /** @type {GPUBuffer | null} */
  let placeholder = null;
  try {
    selection = acquireBuffer(numTokens * maxSelected * 4, undefined, 'qsa_selection');
    meta = acquireBuffer(numTokens * 2 * 4, undefined, 'qsa_selection_meta');
    // All-dense calls read no scores; a minimal placeholder satisfies the binding.
    if (!sparse) placeholder = acquireBuffer(16, undefined, 'qsa_scores_placeholder');
    await unifiedKernelWrapper('qsa_select', target, 'u32',
      [scores?.buffer ?? /** @type {GPUBuffer} */ (placeholder), selection, meta],
      { num_tokens: numTokens, start_pos: options.startPos, block_stride: Math.max(blockStride, 1),
        dense_visible_limit: options.denseVisibleLimit, max_selected: maxSelected, _pad0: 0, _pad1: 0, _pad2: 0 },
      [numTokens, 1, 1], { WORKGROUP_SIZE: WIDE_WORKGROUP, COMPRESS_RATIO: options.compressRatio, BLOCK_TOP_K: options.blockTopK });
  } catch (error) {
    if (selection) release(selection);
    if (meta) release(meta);
    throw error;
  } finally {
    if (placeholder) release(placeholder);
  }
  return { selection, meta, maxSelected };
}

/**
 * @param {import('../command-recorder.js').CommandRecorder | null} target
 * @param {import('../tensor.js').Tensor} query
 * @param {GPUBuffer} keys
 * @param {GPUBuffer} values
 * @param {import('./qsa-indexer.js').QsaSelection} selection
 * @param {import('./qsa-indexer.js').IndexedAttentionOptions} options
 */
async function executeIndexedAttention(target, query, keys, values, selection, options) {
  const { numTokens, numHeads, numKVHeads, headDim, scale, kvRows } = options;
  integer(numTokens, 'numTokens');
  integer(numHeads, 'numHeads');
  integer(numKVHeads, 'numKVHeads');
  integer(headDim, 'headDim');
  integer(kvRows, 'kvRows');
  positive(scale, 'scale');
  if (numHeads % numKVHeads !== 0) fail('numHeads must be a multiple of numKVHeads.');
  if (headDim > 2 * WIDE_WORKGROUP) fail('headDim exceeds the indexed attention workgroup.');
  if (options.kvDtype !== 'f16' && options.kvDtype !== 'f32') fail('indexed attention requires an explicit f16 or f32 kvDtype.');
  assertTensor(query, [numTokens, numHeads * headDim], 'attention query');
  const storage = selectRuleValue('kernels', 'attention', 'indexedStorage', { kvDtype: options.kvDtype });
  assertBuffer(keys, kvRows * numKVHeads * headDim * storage.bytesPerElement, `${options.kvDtype} key cache`);
  assertBuffer(values, kvRows * numKVHeads * headDim * storage.bytesPerElement, `${options.kvDtype} value cache`);
  assertBuffer(selection.selection, numTokens * selection.maxSelected * 4, 'selection list');
  assertBuffer(selection.meta, numTokens * 8, 'selection meta');
  const bytes = numTokens * numHeads * headDim * 4;
  return withKernelOutput(target, options.outputBuffer ?? null, bytes, 'attention_indexed_output', async (output) => {
    await unifiedKernelWrapper('attention_indexed', target, storage.variant,
      [query, keys, values, selection.selection, selection.meta, output],
      { num_tokens: numTokens, num_heads: numHeads, num_kv_heads: numKVHeads, head_dim: headDim,
        max_selected: selection.maxSelected, scale, _pad0: 0, _pad1: 0 },
      [numHeads, numTokens, 1], { WORKGROUP_SIZE: WIDE_WORKGROUP });
    return createTensor(output, 'f32', [numTokens, numHeads * headDim], 'attention_indexed_output');
  });
}

/** @type {typeof import('./qsa-indexer.js').runQsaPrepare} */
export function runQsaPrepare(projection, qNormWeight, rope, rawKeys, options) {
  return executePrepare(null, projection, qNormWeight, rope, rawKeys, options);
}
/** @type {typeof import('./qsa-indexer.js').recordQsaPrepare} */
export function recordQsaPrepare(recorder, projection, qNormWeight, rope, rawKeys, options) {
  return executePrepare(recorder, projection, qNormWeight, rope, rawKeys, options);
}
/** @type {typeof import('./qsa-indexer.js').runQsaBlockKeys} */
export function runQsaBlockKeys(rawKeys, kNormWeight, rope, blockKeys, options) {
  return executeBlockKeys(null, rawKeys, kNormWeight, rope, blockKeys, options);
}
/** @type {typeof import('./qsa-indexer.js').recordQsaBlockKeys} */
export function recordQsaBlockKeys(recorder, rawKeys, kNormWeight, rope, blockKeys, options) {
  return executeBlockKeys(recorder, rawKeys, kNormWeight, rope, blockKeys, options);
}
/** @type {typeof import('./qsa-indexer.js').runQsaBlockScores} */
export function runQsaBlockScores(query, blockKeys, options) {
  return executeScores(null, query, blockKeys, options);
}
/** @type {typeof import('./qsa-indexer.js').recordQsaBlockScores} */
export function recordQsaBlockScores(recorder, query, blockKeys, options) {
  return executeScores(recorder, query, blockKeys, options);
}
/** @type {typeof import('./qsa-indexer.js').runQsaSelect} */
export function runQsaSelect(scores, options) {
  return executeSelect(null, scores, options);
}
/** @type {typeof import('./qsa-indexer.js').recordQsaSelect} */
export function recordQsaSelect(recorder, scores, options) {
  return executeSelect(recorder, scores, options);
}
/** @type {typeof import('./qsa-indexer.js').runIndexedAttention} */
export function runIndexedAttention(query, keys, values, selection, options) {
  return executeIndexedAttention(null, query, keys, values, selection, options);
}
/** @type {typeof import('./qsa-indexer.js').recordIndexedAttention} */
export function recordIndexedAttention(recorder, query, keys, values, selection, options) {
  return executeIndexedAttention(recorder, query, keys, values, selection, options);
}
