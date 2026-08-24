import { readBuffer } from '../../memory/buffer-pool.js';
import { isPlainObject } from '../../utils/plain-object.js';
import { sha256BytesHex } from '../../utils/sha256.js';
import { resolvePromptInput } from '../pipelines/text/generator/prompt-input.js';
import { isStructuredPromptInput } from './text-input.js';

const EMBEDDING_PREVIEW_LENGTH = 16;
const GENERATION_TOKEN_DIAGNOSTIC_LIMIT = 32;
const DECODE_RECORD_TOP_OP_LIMIT = 20;

export function normalizeDecodeRecordOpLabels(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const entries = [];
  for (const [label, rawCount] of Object.entries(value)) {
    const count = Number(rawCount);
    if (typeof label !== 'string' || label.length === 0 || !Number.isFinite(count) || count <= 0) {
      continue;
    }
    entries.push([label, count]);
  }
  if (entries.length === 0) {
    return null;
  }
  entries.sort((a, b) => {
    const countDelta = b[1] - a[1];
    return countDelta !== 0 ? countDelta : a[0].localeCompare(b[0]);
  });
  return Object.fromEntries(entries);
}

export function buildDecodeRecordTopOps(labelCounts, totalOps = null, limit = DECODE_RECORD_TOP_OP_LIMIT) {
  const normalized = normalizeDecodeRecordOpLabels(labelCounts);
  if (!normalized) {
    return [];
  }
  const entries = Object.entries(normalized);
  const denominator = Number.isFinite(totalOps) && totalOps > 0
    ? totalOps
    : entries.reduce((sum, [, count]) => sum + count, 0);
  const maxEntries = Number.isFinite(limit) && limit > 0
    ? Math.floor(limit)
    : DECODE_RECORD_TOP_OP_LIMIT;
  return entries.slice(0, maxEntries).map(([label, count]) => ({
    label,
    count,
    shareOfOps: denominator > 0 ? count / denominator : null,
  }));
}

function normalizeDecodeRecordOpGroupLabel(label) {
  const grouped = label
    .replace(/^L\d+[.:]/, '')
    .replace(/:L\d+(?=:|$)/g, '');
  return grouped.length > 0 ? grouped : label;
}

export function groupDecodeRecordOpLabels(labelCounts) {
  const normalized = normalizeDecodeRecordOpLabels(labelCounts);
  if (!normalized) {
    return null;
  }
  const groups = {};
  for (const [label, count] of Object.entries(normalized)) {
    const groupLabel = normalizeDecodeRecordOpGroupLabel(label);
    groups[groupLabel] = (groups[groupLabel] ?? 0) + count;
  }
  return normalizeDecodeRecordOpLabels(groups);
}

export function buildDecodeRecordTopOpGroups(labelCounts, totalOps = null, limit = DECODE_RECORD_TOP_OP_LIMIT) {
  return buildDecodeRecordTopOps(
    groupDecodeRecordOpLabels(labelCounts),
    totalOps,
    limit
  );
}

function nonNegativeFiniteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

export function normalizeUniformCacheStats(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const hits = nonNegativeFiniteNumber(value.hits);
  const misses = nonNegativeFiniteNumber(value.misses);
  const evictions = nonNegativeFiniteNumber(value.evictions);
  const currentSize = nonNegativeFiniteNumber(value.currentSize);
  const pendingDestruction = nonNegativeFiniteNumber(value.pendingDestruction);
  const totalLookups = Number.isFinite(hits) && Number.isFinite(misses)
    ? hits + misses
    : null;
  const stats = {};
  if (Number.isFinite(hits)) stats.hits = hits;
  if (Number.isFinite(misses)) stats.misses = misses;
  if (Number.isFinite(totalLookups)) stats.totalLookups = totalLookups;
  if (Number.isFinite(totalLookups) && totalLookups > 0) stats.hitRateRatio = hits / totalLookups;
  if (typeof value.hitRate === 'string' && value.hitRate.length > 0) stats.hitRate = value.hitRate;
  if (Number.isFinite(evictions)) stats.evictions = evictions;
  if (Number.isFinite(currentSize)) stats.currentSize = currentSize;
  if (Number.isFinite(pendingDestruction)) stats.pendingDestruction = pendingDestruction;
  return Object.keys(stats).length > 0 ? stats : null;
}

export function normalizeTokenIdArray(value, label) {
  const raw = ArrayBuffer.isView(value) ? Array.from(value) : value;
  if (!Array.isArray(raw)) {
    throw new Error(`${label} must be an array or typed array of token IDs.`);
  }
  return raw.map((entry) => {
    const tokenId = Number(entry);
    if (!Number.isInteger(tokenId) || tokenId < 0) {
      throw new Error(`${label} contains invalid token ID ${entry}.`);
    }
    return tokenId;
  });
}

export function resolveGenerationUseChatTemplate(pipeline, runtimeConfig, runOverrides, promptInput) {
  if (typeof runOverrides?.useChatTemplate === 'boolean') {
    return runOverrides.useChatTemplate;
  }
  if (typeof runtimeConfig?.inference?.chatTemplate?.enabled === 'boolean') {
    return runtimeConfig.inference.chatTemplate.enabled;
  }
  if (isStructuredPromptInput(promptInput)) {
    return true;
  }
  if (typeof pipeline?.modelConfig?.chatTemplateEnabled === 'boolean') {
    return pipeline.modelConfig.chatTemplateEnabled;
  }
  return false;
}

export function resolvePromptTokenIdsForTranscript(pipeline, promptInput, useChatTemplate) {
  if (!pipeline?.tokenizer || typeof pipeline.tokenizer.encode !== 'function') {
    return null;
  }
  const processedPrompt = resolvePromptInput(
    { modelConfig: pipeline.modelConfig ?? {} },
    promptInput,
    useChatTemplate,
    'browserHarness.referenceTranscript'
  );
  return normalizeTokenIdArray(
    pipeline.tokenizer.encode(processedPrompt),
    'browserHarness.referenceTranscript.promptTokenIds'
  );
}

function bytesFromArrayBufferView(view) {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

function digestBytes(bytes) {
  return `sha256:${sha256BytesHex(bytes)}`;
}

function selectTopLogits(logits, limit, decodeToken) {
  const top = [];
  const count = Math.max(1, Math.floor(limit));
  for (let tokenId = 0; tokenId < logits.length; tokenId++) {
    const logit = logits[tokenId];
    if (!Number.isFinite(logit)) continue;
    const candidate = { tokenId, logit };
    let insertAt = top.length;
    while (
      insertAt > 0
      && (
        candidate.logit > top[insertAt - 1].logit
        || (candidate.logit === top[insertAt - 1].logit && candidate.tokenId < top[insertAt - 1].tokenId)
      )
    ) {
      insertAt -= 1;
    }
    top.splice(insertAt, 0, candidate);
    if (top.length > count) {
      top.pop();
    }
  }
  return top.map((entry) => ({
    tokenId: entry.tokenId,
    logit: entry.logit,
    text: typeof decodeToken === 'function' ? decodeToken(entry.tokenId) : null,
  }));
}

function getReferenceTranscriptRuntimeConfig(runtimeConfig) {
  const config = runtimeConfig?.shared?.harness?.referenceTranscript;
  return isPlainObject(config) ? config : null;
}

export function shouldCaptureReferenceLogits(runOverrides, runtimeConfig) {
  const referenceConfig = getReferenceTranscriptRuntimeConfig(runtimeConfig);
  return runOverrides?.diagnostics?.referenceTranscript?.captureLogits === true
    || runOverrides?.diagnostics?.captureLogits === true
    || referenceConfig?.captureLogits === true;
}

export function shouldCaptureReferenceKvBytes(runOverrides, runtimeConfig) {
  const referenceConfig = getReferenceTranscriptRuntimeConfig(runtimeConfig);
  return runOverrides?.diagnostics?.referenceTranscript?.captureKvBytes === true
    || runOverrides?.diagnostics?.captureKvBytes === true
    || referenceConfig?.captureKvBytes === true;
}

export function shouldEnableReferenceTranscriptDiagnostics(runOverrides, runtimeConfig) {
  const referenceConfig = getReferenceTranscriptRuntimeConfig(runtimeConfig);
  return runOverrides?.diagnostics?.enabled === true
    || referenceConfig?.enabled === true
    || referenceConfig?.captureLogits === true
    || referenceConfig?.captureKvBytes === true;
}

export function digestLogitsForTranscript(logits, context) {
  if (!(logits instanceof Float32Array)) {
    throw new Error('reference transcript logits capture requires Float32Array logits.');
  }
  const digest = digestBytes(bytesFromArrayBufferView(logits));
  const topK = Number.isInteger(context?.topK) ? Math.max(1, context.topK) : 8;
  const decodeToken = typeof context?.decodeToken === 'function' ? context.decodeToken : null;
  return {
    index: Number.isInteger(context?.index) ? context.index : null,
    tokenId: Number.isInteger(context?.tokenId) ? context.tokenId : null,
    inputTokenCount: Number.isInteger(context?.inputTokenCount) ? context.inputTokenCount : null,
    dtype: 'f32',
    elementCount: logits.length,
    digest,
    top: selectTopLogits(logits, topK, decodeToken),
  };
}

async function digestKvLayerBytes(layer, layerIdx, kvCache) {
  const seqLen = Number.isFinite(layer?.seqLen) ? Math.max(0, Math.floor(layer.seqLen)) : 0;
  const byteLength = seqLen * kvCache.kvSize * kvCache.bytesPerElem;
  if (byteLength < 1) {
    return {
      layer: layerIdx,
      seqLen,
      keyBytes: 0,
      valueBytes: 0,
      keyDigest: digestBytes(new Uint8Array()),
      valueDigest: digestBytes(new Uint8Array()),
    };
  }

  if (layer?.keysGPU && layer?.valuesGPU) {
    const [keyBuffer, valueBuffer] = await Promise.all([
      readBuffer(layer.keysGPU, byteLength),
      readBuffer(layer.valuesGPU, byteLength),
    ]);
    return {
      layer: layerIdx,
      seqLen,
      keyBytes: byteLength,
      valueBytes: byteLength,
      keyDigest: digestBytes(new Uint8Array(keyBuffer)),
      valueDigest: digestBytes(new Uint8Array(valueBuffer)),
    };
  }

  const elementCount = seqLen * kvCache.kvSize;
  if (layer?.keys instanceof Float32Array && layer?.values instanceof Float32Array) {
    const keys = layer.keys.subarray(0, elementCount);
    const values = layer.values.subarray(0, elementCount);
    return {
      layer: layerIdx,
      seqLen,
      keyBytes: keys.byteLength,
      valueBytes: values.byteLength,
      keyDigest: digestBytes(bytesFromArrayBufferView(keys)),
      valueDigest: digestBytes(bytesFromArrayBufferView(values)),
    };
  }

  throw new Error(`reference transcript KV byte capture unsupported for layer ${layerIdx}.`);
}

export async function captureKvCacheByteProof(pipeline, enabled) {
  if (!enabled) return null;
  const kvCache = pipeline?.kvCache ?? null;
  if (!kvCache || !Array.isArray(kvCache.layers)) {
    return null;
  }
  if (kvCache.layout !== 'contiguous') {
    throw new Error(
      `reference transcript KV byte capture only supports contiguous KV cache layout; got ${kvCache.layout}.`
    );
  }
  const layers = [];
  for (let layerIdx = 0; layerIdx < kvCache.layers.length; layerIdx += 1) {
    layers.push(await digestKvLayerBytes(kvCache.layers[layerIdx], layerIdx, kvCache));
  }
  const canonicalBytes = new TextEncoder().encode(JSON.stringify({
    mode: 'sha256-layer-kv-bytes',
    layout: kvCache.layout,
    kvDtype: kvCache.kvDtype ?? null,
    kvSize: kvCache.kvSize,
    bytesPerElem: kvCache.bytesPerElem,
    layers,
  }));
  return {
    mode: 'sha256-layer-kv-bytes',
    layout: kvCache.layout,
    kvDtype: kvCache.kvDtype ?? null,
    layerCount: layers.length,
    digest: digestBytes(canonicalBytes),
    layers,
  };
}

export function summarizeRerankScores(scores) {
  const sorted = [...scores].sort((a, b) => {
    const scoreDelta = b.score - a.score;
    return scoreDelta !== 0 ? scoreDelta : a.index - b.index;
  });
  return {
    ranking: sorted.map((entry, rank) => ({
      rank: rank + 1,
      index: entry.index,
      document: entry.document,
      score: Number(entry.score.toFixed(6)),
      probability: Number(entry.probability.toFixed(6)),
      trueLogit: Number(entry.trueLogit.toFixed(6)),
      falseLogit: Number(entry.falseLogit.toFixed(6)),
      tokenCount: entry.tokenCount,
      scoringPath: entry.scoringPath,
    })),
    top: sorted[0] ?? null,
  };
}

export function summarizeEmbeddingValues(embedding) {
  const values = ArrayBuffer.isView(embedding) || Array.isArray(embedding) ? embedding : null;
  const embeddingDim = Number.isFinite(values?.length) ? values.length : 0;
  const preview = [];

  let nonFiniteCount = 0;
  let finiteCount = 0;
  let min = Infinity;
  let max = -Infinity;
  let maxAbs = 0;
  let sum = 0;
  let sumSq = 0;

  for (let i = 0; i < embeddingDim; i++) {
    const value = Number(values[i]);
    if (preview.length < EMBEDDING_PREVIEW_LENGTH) {
      preview.push(Number.isFinite(value) ? Number(value.toFixed(6)) : null);
    }
    if (!Number.isFinite(value)) {
      nonFiniteCount++;
      continue;
    }
    finiteCount++;
    if (value < min) min = value;
    if (value > max) max = value;
    const abs = Math.abs(value);
    if (abs > maxAbs) maxAbs = abs;
    sum += value;
    sumSq += value * value;
  }

  const mean = finiteCount > 0 ? (sum / finiteCount) : null;
  const variance = finiteCount > 0 ? Math.max(0, (sumSq / finiteCount) - ((mean || 0) * (mean || 0))) : null;
  const stdDev = variance == null ? null : Math.sqrt(variance);
  const l2Norm = finiteCount > 0 ? Math.sqrt(sumSq) : null;
  const finiteRatio = embeddingDim > 0 ? finiteCount / embeddingDim : 0;

  return {
    embeddingDim,
    nonFiniteCount,
    finiteCount,
    finiteRatio,
    min: finiteCount > 0 ? min : null,
    max: finiteCount > 0 ? max : null,
    maxAbs: finiteCount > 0 ? maxAbs : null,
    mean,
    stdDev,
    l2Norm,
    preview,
  };
}

export function cosineSimilarity(a, b) {
  if (!a || !b || !Number.isFinite(a.length) || !Number.isFinite(b.length)) return NaN;
  if (a.length !== b.length || a.length === 0) return NaN;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const av = Number(a[i]);
    const bv = Number(b[i]);
    if (!Number.isFinite(av) || !Number.isFinite(bv)) return NaN;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA <= 0 || normB <= 0) return NaN;
  return dot / Math.sqrt(normA * normB);
}

export function top1Index(values) {
  let best = -1;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const value = Number(values[i]);
    if (!Number.isFinite(value)) continue;
    if (value > bestValue) {
      bestValue = value;
      best = i;
    }
  }
  return best;
}

const SPECIAL_TOKEN_RE = /^(<pad>|<unused\d*>|<eos>|<bos>|<s>|<\/s>|\[PAD\]|\[UNK\]|\[SEP\]|\[CLS\]|<[^>]{1,32}>)$/i;
const PAD_DOMINANCE_THRESHOLD = 0.5;

function isSpecialLikeTokenText(value) {
  if (typeof value !== 'string') return false;
  return SPECIAL_TOKEN_RE.test(value.trim());
}

export function summarizeGenerationTokens(tokenRecords) {
  const records = Array.isArray(tokenRecords) ? tokenRecords : [];
  const preview = records.slice(0, GENERATION_TOKEN_DIAGNOSTIC_LIMIT).map((record) => ({
    id: record.id,
    text: record.text,
    fallbackText: record.fallbackText,
  }));
  let emptyTextCount = 0;
  let specialLikeTextCount = 0;
  let specialLikeFallbackCount = 0;
  for (const record of records) {
    if (!record || typeof record !== 'object') continue;
    if (typeof record.text === 'string' && record.text.length === 0) {
      emptyTextCount += 1;
    }
    if (isSpecialLikeTokenText(record.text)) {
      specialLikeTextCount += 1;
    }
    if (isSpecialLikeTokenText(record.fallbackText)) {
      specialLikeFallbackCount += 1;
    }
  }
  return {
    preview,
    total: records.length,
    omitted: Math.max(0, records.length - preview.length),
    emptyTextCount,
    specialLikeTextCount,
    specialLikeFallbackCount,
  };
}

export function buildGenerationPhaseFromStats(pipeline, durationMs, tokenCount) {
  const stats = typeof pipeline?.getStats === 'function'
    ? (pipeline.getStats() || {})
    : {};
  const memoryStats = typeof pipeline?.getMemoryStats === 'function'
    ? (pipeline.getMemoryStats() || {})
    : {};
  const kvMemory = memoryStats.kvCache && typeof memoryStats.kvCache === 'object'
    ? memoryStats.kvCache
    : null;
  const prefillMs = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : 0;
  const ttftMs = Number.isFinite(stats.ttftMs) ? stats.ttftMs : prefillMs;
  const decodeMs = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : 0;
  const prefillTokens = Number.isFinite(stats.prefillTokens) ? stats.prefillTokens : 0;
  const decodeTokens = Number.isFinite(stats.decodeTokens)
    ? stats.decodeTokens
    : Math.max(0, tokenCount - 1);
  const decodeTokensPerSec = decodeMs > 0
    ? (decodeTokens / decodeMs) * 1000
    : 0;
  const prefillTokensPerSec = prefillMs > 0
    ? (prefillTokens / prefillMs) * 1000
    : 0;
  const prefillTokensPerSecTtft = ttftMs > 0
    ? (prefillTokens / ttftMs) * 1000
    : 0;
  const gpu = {};
  if (Number.isFinite(stats.gpuTimePrefillMs)) gpu.prefillMs = stats.gpuTimePrefillMs;
  if (Number.isFinite(stats.gpuTimeDecodeMs)) gpu.decodeMs = stats.gpuTimeDecodeMs;
  if (Number.isFinite(stats.decodeRecordMs)) gpu.decodeRecordMs = stats.decodeRecordMs;
  if (Number.isFinite(stats.decodeRecordOps)) gpu.decodeRecordOps = stats.decodeRecordOps;
  if (Number.isFinite(stats.decodeRecordPasses)) gpu.decodeRecordPasses = stats.decodeRecordPasses;
  const decodeRecordOpLabels = normalizeDecodeRecordOpLabels(stats.decodeRecordOpLabels);
  if (decodeRecordOpLabels) {
    const decodeRecordTopOps = buildDecodeRecordTopOps(decodeRecordOpLabels, stats.decodeRecordOps);
    const decodeRecordTopOpGroups = buildDecodeRecordTopOpGroups(decodeRecordOpLabels, stats.decodeRecordOps);
    gpu.decodeRecordOpLabels = decodeRecordOpLabels;
    gpu.decodeRecordUniqueOpLabels = Object.keys(decodeRecordOpLabels).length;
    gpu.decodeRecordTopOps = decodeRecordTopOps;
    gpu.decodeRecordTopOpGroups = decodeRecordTopOpGroups;
  }
  if (
    Number.isFinite(stats.decodeRecordMs) &&
    Number.isFinite(stats.decodeRecordOps) &&
    stats.decodeRecordOps > 0
  ) {
    gpu.decodeRecordMsPerOp = stats.decodeRecordMs / stats.decodeRecordOps;
  }
  if (
    Number.isFinite(stats.decodeRecordMs) &&
    Number.isFinite(stats.decodeRecordPasses) &&
    stats.decodeRecordPasses > 0
  ) {
    gpu.decodeRecordMsPerPass = stats.decodeRecordMs / stats.decodeRecordPasses;
  }
  if (
    Number.isFinite(stats.decodeRecordOps) &&
    Number.isFinite(stats.decodeRecordPasses) &&
    stats.decodeRecordOps > 0
  ) {
    gpu.decodeRecordPassesPerOp = stats.decodeRecordPasses / stats.decodeRecordOps;
  }
  if (
    Number.isFinite(stats.decodeRecordPasses) &&
    Number.isFinite(stats.batching?.executedBatchTokens) &&
    stats.batching.executedBatchTokens > 0
  ) {
    gpu.decodeRecordPassesPerExecutedBatchToken = stats.decodeRecordPasses / stats.batching.executedBatchTokens;
  }
  if (
    Number.isFinite(stats.decodeRecordMs) &&
    Number.isFinite(stats.batching?.executedBatchTokens) &&
    stats.batching.executedBatchTokens > 0
  ) {
    gpu.decodeRecordMsPerExecutedBatchToken = stats.decodeRecordMs / stats.batching.executedBatchTokens;
  }
  if (
    Number.isFinite(stats.decodeRecordOps) &&
    Number.isFinite(stats.batching?.executedBatchTokens) &&
    stats.batching.executedBatchTokens > 0
  ) {
    gpu.decodeRecordOpsPerExecutedBatchToken = stats.decodeRecordOps / stats.batching.executedBatchTokens;
  }
  const uniformCacheStats = normalizeUniformCacheStats(stats.uniformCache);
  if (uniformCacheStats) {
    gpu.uniformCache = uniformCacheStats;
  }
  if (Number.isFinite(stats.decodeSubmitWaitMs)) gpu.decodeSubmitWaitMs = stats.decodeSubmitWaitMs;
  if (Number.isFinite(stats.decodeReadbackWaitMs)) gpu.decodeReadbackWaitMs = stats.decodeReadbackWaitMs;
  if (Number.isFinite(stats.decodeReadbackMapWaitMs)) gpu.decodeReadbackMapWaitMs = stats.decodeReadbackMapWaitMs;
  if (Number.isFinite(stats.decodeReadbackCleanupMs)) gpu.decodeReadbackCleanupMs = stats.decodeReadbackCleanupMs;
  if (Number.isFinite(stats.decodeReadbackCopyMs)) gpu.decodeReadbackCopyMs = stats.decodeReadbackCopyMs;
  if (Number.isFinite(stats.prefillRecordMs)) gpu.prefillRecordMs = stats.prefillRecordMs;
  if (Number.isFinite(stats.prefillRecordOps)) gpu.prefillRecordOps = stats.prefillRecordOps;
  if (Number.isFinite(stats.prefillRecordPasses)) gpu.prefillRecordPasses = stats.prefillRecordPasses;
  const prefillRecordOpLabels = normalizeDecodeRecordOpLabels(stats.prefillRecordOpLabels);
  if (prefillRecordOpLabels) {
    const prefillRecordTopOps = buildDecodeRecordTopOps(prefillRecordOpLabels, stats.prefillRecordOps);
    const prefillRecordTopOpGroups = buildDecodeRecordTopOpGroups(prefillRecordOpLabels, stats.prefillRecordOps);
    gpu.prefillRecordOpLabels = prefillRecordOpLabels;
    gpu.prefillRecordUniqueOpLabels = Object.keys(prefillRecordOpLabels).length;
    gpu.prefillRecordTopOps = prefillRecordTopOps;
    gpu.prefillRecordTopOpGroups = prefillRecordTopOpGroups;
  }
  if (Number.isFinite(stats.prefillSubmitWaitMs)) gpu.prefillSubmitWaitMs = stats.prefillSubmitWaitMs;
  if (
    Number.isFinite(decodeMs) &&
    Number.isFinite(stats.decodeRecordMs) &&
    Number.isFinite(stats.decodeSubmitWaitMs) &&
    Number.isFinite(stats.decodeReadbackWaitMs)
  ) {
    const decodeGpuWaitMs = Math.max(stats.decodeSubmitWaitMs, stats.decodeReadbackWaitMs);
    gpu.decodeOrchestrationMs = decodeMs - stats.decodeRecordMs - decodeGpuWaitMs;
  }
  if (Number.isFinite(stats.singleTokenSubmitWaitMs)) gpu.singleTokenSubmitWaitMs = stats.singleTokenSubmitWaitMs;
  if (Number.isFinite(stats.singleTokenReadbackWaitMs)) gpu.singleTokenReadbackWaitMs = stats.singleTokenReadbackWaitMs;
  if (Number.isFinite(stats.singleTokenReadbackMapWaitMs)) gpu.singleTokenReadbackMapWaitMs = stats.singleTokenReadbackMapWaitMs;
  if (Number.isFinite(stats.singleTokenReadbackCleanupMs)) gpu.singleTokenReadbackCleanupMs = stats.singleTokenReadbackCleanupMs;
  if (Number.isFinite(stats.singleTokenReadbackCopyMs)) gpu.singleTokenReadbackCopyMs = stats.singleTokenReadbackCopyMs;
  if (Number.isFinite(stats.singleTokenOrchestrationMs)) gpu.singleTokenOrchestrationMs = stats.singleTokenOrchestrationMs;
  const gpuPhase = Object.keys(gpu).length > 0 ? gpu : null;
  const batching = {};
  if (Number.isFinite(stats.batching?.batchedForwardCalls)) {
    batching.batchedForwardCalls = stats.batching.batchedForwardCalls;
  }
  if (Number.isFinite(stats.batching?.unbatchedForwardCalls)) {
    batching.unbatchedForwardCalls = stats.batching.unbatchedForwardCalls;
  }
  if (Number.isFinite(stats.batching?.totalBatchedTimeMs)) {
    batching.totalBatchedTimeMs = stats.batching.totalBatchedTimeMs;
  }
  if (Number.isFinite(stats.batching?.totalUnbatchedTimeMs)) {
    batching.totalUnbatchedTimeMs = stats.batching.totalUnbatchedTimeMs;
  }
  if (Number.isFinite(stats.batching?.gpuSubmissions)) {
    batching.gpuSubmissions = stats.batching.gpuSubmissions;
  }
  if (Number.isFinite(stats.batching?.requestedBatchTokens)) {
    batching.requestedBatchTokens = stats.batching.requestedBatchTokens;
  }
  if (Number.isFinite(stats.batching?.effectiveBatchTokens)) {
    batching.effectiveBatchTokens = stats.batching.effectiveBatchTokens;
  }
  if (Number.isFinite(stats.batching?.executedBatchTokens)) {
    batching.executedBatchTokens = stats.batching.executedBatchTokens;
  }
  if (Number.isFinite(stats.batching?.resolvedBatchTokens)) {
    batching.resolvedBatchTokens = stats.batching.resolvedBatchTokens;
  }
  if (Number.isFinite(stats.batching?.maxBatchTokenCap)) {
    batching.maxBatchTokenCap = stats.batching.maxBatchTokenCap;
  }
  if (Number.isFinite(stats.batching?.batchClampCount)) {
    batching.batchClampCount = stats.batching.batchClampCount;
  }
  const batchingPhase = Object.keys(batching).length > 0 ? batching : null;
  const plePreparedTokenCache = {};
  if (Number.isFinite(stats.plePreparedTokenCacheHits)) {
    plePreparedTokenCache.hits = stats.plePreparedTokenCacheHits;
  }
  if (Number.isFinite(stats.plePreparedTokenCacheMisses)) {
    plePreparedTokenCache.misses = stats.plePreparedTokenCacheMisses;
  }
  if (Number.isFinite(stats.plePreparedTokenCacheEntries)) {
    plePreparedTokenCache.entries = stats.plePreparedTokenCacheEntries;
  }
  if (Number.isFinite(stats.plePreparedTokenCacheBytes)) {
    plePreparedTokenCache.bytes = stats.plePreparedTokenCacheBytes;
  }
  if (Number.isFinite(stats.pleWriteBufferCount)) {
    plePreparedTokenCache.writeBufferCount = stats.pleWriteBufferCount;
  }
  if (Number.isFinite(stats.pleWriteBufferBytes)) {
    plePreparedTokenCache.writeBufferBytes = stats.pleWriteBufferBytes;
  }
  const plePreparedTokenCachePhase = Object.keys(plePreparedTokenCache).length > 0
    ? plePreparedTokenCache
    : null;
  const wallMs = Number.isFinite(stats.totalTimeMs) ? stats.totalTimeMs : durationMs;

  return {
    phase: {
      totalMs: prefillMs + decodeMs,
      wallMs,
      ttftMs,
      prefillMs,
      decodeMs,
      prefillTokens,
      decodeTokens,
      prefillTokensPerSec,
      prefillTokensPerSecTtft,
      decodeTokensPerSec,
      gpu: gpuPhase,
      prefillProfileSteps: Array.isArray(stats.prefillProfileSteps)
        ? stats.prefillProfileSteps
        : null,
      decodeProfileSteps: Array.isArray(stats.decodeProfileSteps)
        ? stats.decodeProfileSteps
        : null,
      decodeMode: stats.decodeMode ?? null,
      batchGuardReason: stats.batchGuardReason ?? null,
      stopReason: stats.stopReason ?? null,
      stopTokenId: Number.isInteger(stats.stopTokenId) ? stats.stopTokenId : null,
      batching: batchingPhase,
      plePreparedTokenCache: plePreparedTokenCachePhase,
      kvCache: kvMemory
        ? {
          layout: kvMemory.layout ?? null,
          kvDtype: kvMemory.kvDtype ?? null,
          seqLen: Number.isFinite(kvMemory.seqLen) ? kvMemory.seqLen : null,
          maxSeqLen: Number.isFinite(kvMemory.maxSeqLen) ? kvMemory.maxSeqLen : null,
          usedBytes: Number.isFinite(kvMemory.used) ? kvMemory.used : null,
          allocatedBytes: Number.isFinite(kvMemory.allocated) ? kvMemory.allocated : null,
          counters: kvMemory.counters ?? null,
        }
        : null,
      executionPlan: stats.executionPlan ?? null,
      kernelPathId: stats.kernelPathId ?? null,
      operatorDiagnostics: stats.operatorDiagnostics ?? null,
      kernelPathSource: stats.kernelPathSource ?? null,
    },
  };
}

export function isCoherentOutput(tokens, output) {
  if (tokens.length === 0) return false;
  const specialTokenCount = tokens.filter((t) => SPECIAL_TOKEN_RE.test(String(t).trim())).length;
  if (specialTokenCount / tokens.length >= PAD_DOMINANCE_THRESHOLD) return false;
  const cleanedOutput = String(output || '')
    .replace(/<[^>\n]{1,80}>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return cleanedOutput.length > 0;
}
