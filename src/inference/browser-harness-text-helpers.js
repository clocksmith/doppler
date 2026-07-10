import { readBuffer } from '../memory/buffer-pool.js';
import { CAPTURE_LEVELS, createDefaultCaptureConfig } from '../debug/capture-policy.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { loadJson } from '../utils/load-json.js';
import { isPlainObject } from '../utils/plain-object.js';
import { cloneJsonValue } from '../utils/clone-json.js';
import { sha256BytesHex } from '../utils/sha256.js';
import { resolvePromptInput } from './pipelines/text/generator-prefill-helpers.js';

const DEFAULT_IMAGE_TRANSCRIPTION_PROMPT = 'Describe the image in one short sentence.';
const DEFAULT_IMAGE_TRANSCRIPTION_SOFT_TOKEN_BUDGET = 70;
const EMBEDDING_PREVIEW_LENGTH = 16;
const GENERATION_TOKEN_DIAGNOSTIC_LIMIT = 32;
const DECODE_RECORD_TOP_OP_LIMIT = 20;
const RERANK_SCORE_POLICIES = new Set(['logit_difference', 'true_logit']);

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

function normalizeTokenIdArray(value, label) {
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

function resolveGenerationUseChatTemplate(pipeline, runtimeConfig, runOverrides, promptInput) {
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

function resolvePromptTokenIdsForTranscript(pipeline, promptInput, useChatTemplate) {
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

function shouldCaptureReferenceLogits(runOverrides, runtimeConfig) {
  const referenceConfig = getReferenceTranscriptRuntimeConfig(runtimeConfig);
  return runOverrides?.diagnostics?.referenceTranscript?.captureLogits === true
    || runOverrides?.diagnostics?.captureLogits === true
    || referenceConfig?.captureLogits === true;
}

function shouldCaptureReferenceKvBytes(runOverrides, runtimeConfig) {
  const referenceConfig = getReferenceTranscriptRuntimeConfig(runtimeConfig);
  return runOverrides?.diagnostics?.referenceTranscript?.captureKvBytes === true
    || runOverrides?.diagnostics?.captureKvBytes === true
    || referenceConfig?.captureKvBytes === true;
}

function shouldEnableReferenceTranscriptDiagnostics(runOverrides, runtimeConfig) {
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

const embeddingSemanticFixtureAsset = await loadJson(
  './fixtures/embedding-semantic-fixtures.json',
  import.meta.url,
  'Failed to load embedding semantic fixtures'
);

const rerankSemanticFixtureAsset = await loadJson(
  './fixtures/rerank-semantic-fixtures.json',
  import.meta.url,
  'Failed to load rerank semantic fixtures'
);

function asText(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed || null;
}

function normalizeRetrievalFixtures(cases) {
  if (!Array.isArray(cases)) return null;
  const normalized = [];
  for (let i = 0; i < cases.length; i++) {
    const entry = cases[i];
    if (!entry || typeof entry !== 'object') continue;

    const query = asText(entry.query);
    const docs = Array.isArray(entry.docs) ? entry.docs.map(asText).filter(Boolean) : [];
    if (!query || docs.length === 0 || !Number.isFinite(entry.expectedDoc)) {
      continue;
    }
    const expectedDoc = Math.floor(entry.expectedDoc);
    normalized.push({
      id: asText(entry.id) ?? `case-${i + 1}`,
      query,
      docs,
      expectedDoc: Math.max(0, Math.min(expectedDoc, docs.length - 1)),
    });
  }
  return normalized.length > 0 ? normalized : null;
}

function normalizePairFixtures(cases) {
  if (!Array.isArray(cases)) return null;
  const normalized = [];
  for (let i = 0; i < cases.length; i++) {
    const entry = cases[i];
    if (!entry || typeof entry !== 'object') continue;

    const anchor = asText(entry.anchor);
    const positive = asText(entry.positive);
    const negative = asText(entry.negative);
    if (!anchor || !positive || !negative) {
      continue;
    }
    normalized.push({
      id: asText(entry.id) ?? `pair-${i + 1}`,
      anchor,
      positive,
      negative,
    });
  }
  return normalized.length > 0 ? normalized : null;
}

function normalizeLengthStabilityCases(cases) {
  if (!Array.isArray(cases)) return null;
  const normalized = [];
  for (let i = 0; i < cases.length; i++) {
    const entry = cases[i];
    if (!entry || typeof entry !== 'object') continue;
    const short_ = asText(entry.short);
    const medium = asText(entry.medium);
    const long_ = asText(entry.long);
    if (!short_ || !medium || !long_) continue;
    normalized.push({
      id: asText(entry.id) ?? `length-${i + 1}`,
      short: short_,
      medium,
      long: long_,
      maxCosineDrift: Number.isFinite(entry.maxCosineDrift) ? entry.maxCosineDrift : 0.25,
    });
  }
  return normalized.length > 0 ? normalized : null;
}

function normalizeThroughputCorpus(corpus) {
  if (!Array.isArray(corpus)) return null;
  const normalized = corpus.map(asText).filter(Boolean);
  return normalized.length > 0 ? normalized : null;
}

function normalizeRerankCases(cases) {
  if (!Array.isArray(cases)) return null;
  const normalized = [];
  for (let i = 0; i < cases.length; i++) {
    const entry = cases[i];
    if (!entry || typeof entry !== 'object') continue;

    const query = asText(entry.query);
    const positive = asText(entry.positive);
    const negative = asText(entry.negative);
    if (!query || !positive || !negative) {
      continue;
    }
    normalized.push({
      id: asText(entry.id) ?? `rerank-${i + 1}`,
      query,
      positive,
      negative,
    });
  }
  return normalized.length > 0 ? normalized : null;
}

function resolveDefaultRerankSemanticFixtures() {
  const defaults = rerankSemanticFixtureAsset?.defaults;
  if (!isPlainObject(defaults)) {
    throw new Error('Rerank semantic fixture asset must define defaults.');
  }

  const cases = normalizeRerankCases(defaults.cases);
  if (!cases) {
    throw new Error('Rerank semantic fixture asset must define cases.');
  }
  if (!Number.isFinite(defaults.minPairAcc)) {
    throw new Error('Rerank semantic fixture asset must define minPairAcc.');
  }
  if (!Number.isFinite(defaults.minScoreMargin)) {
    throw new Error('Rerank semantic fixture asset must define minScoreMargin.');
  }

  return {
    cases,
    minPairAcc: Math.max(0, Math.min(1, Number(defaults.minPairAcc))),
    minScoreMargin: Number(defaults.minScoreMargin),
  };
}

const DEFAULT_RERANK_SEMANTIC_FIXTURES = resolveDefaultRerankSemanticFixtures();

export function getDefaultRerankSemanticFixtures() {
  return cloneJsonValue(DEFAULT_RERANK_SEMANTIC_FIXTURES);
}

function resolveDefaultEmbeddingSemanticFixtures() {
  const defaults = embeddingSemanticFixtureAsset?.defaults;
  if (!isPlainObject(defaults)) {
    throw new Error('Embedding semantic fixture asset must define defaults.');
  }

  const retrievalCases = normalizeRetrievalFixtures(defaults.retrievalCases);
  if (!retrievalCases) {
    throw new Error('Embedding semantic fixture asset must define retrievalCases.');
  }

  const pairCases = normalizePairFixtures(defaults.pairCases);
  if (!pairCases) {
    throw new Error('Embedding semantic fixture asset must define pairCases.');
  }

  if (!Number.isFinite(defaults.minRetrievalTop1Acc)) {
    throw new Error('Embedding semantic fixture asset must define minRetrievalTop1Acc.');
  }
  if (!Number.isFinite(defaults.minPairAcc)) {
    throw new Error('Embedding semantic fixture asset must define minPairAcc.');
  }
  if (!Number.isFinite(defaults.pairMargin)) {
    throw new Error('Embedding semantic fixture asset must define pairMargin.');
  }

  return {
    retrievalCases,
    pairCases,
    lengthStabilityCases: normalizeLengthStabilityCases(defaults.lengthStabilityCases) ?? [],
    throughputCorpus: normalizeThroughputCorpus(defaults.throughputCorpus) ?? [],
    minRetrievalTop1Acc: Math.max(0, Math.min(1, Number(defaults.minRetrievalTop1Acc))),
    minPairAcc: Math.max(0, Math.min(1, Number(defaults.minPairAcc))),
    pairMargin: Number(defaults.pairMargin),
  };
}

const DEFAULT_EMBEDDING_SEMANTIC_FIXTURES = resolveDefaultEmbeddingSemanticFixtures();

export function getDefaultEmbeddingSemanticFixtures() {
  return cloneJsonValue(DEFAULT_EMBEDDING_SEMANTIC_FIXTURES);
}

function resolveEmbeddingSemanticFixtures(runtimeConfig, options = null) {
  const overrides = isPlainObject(options?.embeddingSemantic)
    ? options.embeddingSemantic
    : null;
  const runtimeOverrides = runtimeConfig?.shared?.benchmark?.run?.embeddingSemantic;
  const source = overrides ?? (isPlainObject(runtimeOverrides) ? runtimeOverrides : null);

  const retrievalCases = normalizeRetrievalFixtures(source?.retrievalCases)
    ?? DEFAULT_EMBEDDING_SEMANTIC_FIXTURES.retrievalCases;
  const pairCases = normalizePairFixtures(source?.pairCases)
    ?? DEFAULT_EMBEDDING_SEMANTIC_FIXTURES.pairCases;
  const minRetrievalTop1Acc = Number.isFinite(source?.minRetrievalTop1Acc)
    ? Math.max(0, Math.min(1, Number(source.minRetrievalTop1Acc)))
    : DEFAULT_EMBEDDING_SEMANTIC_FIXTURES.minRetrievalTop1Acc;
  const minPairAcc = Number.isFinite(source?.minPairAcc)
    ? Math.max(0, Math.min(1, Number(source.minPairAcc)))
    : DEFAULT_EMBEDDING_SEMANTIC_FIXTURES.minPairAcc;
  const pairMargin = Number.isFinite(source?.pairMargin)
    ? Number(source.pairMargin)
    : DEFAULT_EMBEDDING_SEMANTIC_FIXTURES.pairMargin;

  const lengthStabilityCases = normalizeLengthStabilityCases(source?.lengthStabilityCases)
    ?? DEFAULT_EMBEDDING_SEMANTIC_FIXTURES.lengthStabilityCases;
  const throughputCorpus = normalizeThroughputCorpus(source?.throughputCorpus)
    ?? DEFAULT_EMBEDDING_SEMANTIC_FIXTURES.throughputCorpus;

  return {
    retrievalCases,
    pairCases,
    lengthStabilityCases,
    throughputCorpus,
    minRetrievalTop1Acc,
    minPairAcc,
    pairMargin,
  };
}

function resolveEmbeddingSemanticStyle(pipeline) {
  const manifest = pipeline?.manifest ?? null;
  const style = selectRuleValue('inference', 'config', 'embeddingSemanticStyle', {
    modelType: String(manifest?.modelType ?? '').toLowerCase(),
    manifestModelType: String(
      manifest?.config?.model_type
      ?? manifest?.config?.text_config?.model_type
      ?? ''
    ).toLowerCase(),
    sourceCheckpointId: String(manifest?.artifactIdentity?.sourceCheckpointId ?? ''),
  });
  if (typeof style === 'string' && style.length > 0) {
    return style;
  }
  return 'default';
}

function formatEmbeddingSemanticText(text, kind, style) {
  if (style === 'embeddinggemma') {
    if (kind === 'query') {
      return `task: search result | query: ${text}`;
    }
    if (kind === 'document') {
      return `title: None | text: ${text}`;
    }
  }
  if (style === 'qwen3_embedding') {
    if (kind === 'query') {
      return `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ${text}`;
    }
    return text;
  }
  return text;
}

export function resolvePrompt(runtimeConfig) {
  const runtimePrompt = runtimeConfig?.inference?.prompt;
  if (typeof runtimePrompt === 'string' && runtimePrompt.trim()) {
    return runtimePrompt.trim();
  }
  throw new Error('Harness embedding requires explicit runtime.inference.prompt.');
}

function assertRerankTokenId(value, label) {
  const tokenId = Number(value);
  if (!Number.isInteger(tokenId) || tokenId < 0) {
    throw new Error(`Manifest rerank config requires non-negative integer ${label}.`);
  }
  return tokenId;
}

function assertRerankText(value, label, preserve = false) {
  if (typeof value !== 'string') {
    throw new Error(`Manifest rerank config requires non-empty ${label}.`);
  }
  const text = value.trim();
  if (!text) {
    throw new Error(`Manifest rerank config requires non-empty ${label}.`);
  }
  return preserve ? value : text;
}

export function resolveRerankScoringConfig(pipeline) {
  const config = pipeline?.manifest?.inference?.rerank;
  if (!isPlainObject(config)) {
    throw new Error('Rerank workload requires manifest.inference.rerank scoring config.');
  }
  const format = assertRerankText(config.format, 'format');
  if (format !== 'qwen3_yes_no_logit') {
    throw new Error(`Unsupported rerank scoring format "${format}".`);
  }
  const trueTokenId = assertRerankTokenId(config.trueTokenId, 'trueTokenId');
  const falseTokenId = assertRerankTokenId(config.falseTokenId, 'falseTokenId');
  if (trueTokenId === falseTokenId) {
    throw new Error('Manifest rerank config trueTokenId and falseTokenId must be distinct.');
  }
  const score = assertRerankText(config.score, 'score');
  if (!RERANK_SCORE_POLICIES.has(score)) {
    throw new Error(`Unsupported rerank score policy "${score}".`);
  }
  const probability = assertRerankText(config.probability, 'probability');
  if (probability !== 'sigmoid') {
    throw new Error(`Unsupported rerank probability policy "${probability}".`);
  }
  return {
    format,
    instruction: assertRerankText(config.instruction, 'instruction'),
    inputTemplate: assertRerankText(config.inputTemplate, 'inputTemplate', true),
    prefix: assertRerankText(config.prefix, 'prefix', true),
    suffix: assertRerankText(config.suffix, 'suffix', true),
    trueToken: assertRerankText(config.trueToken, 'trueToken'),
    trueTokenId,
    falseToken: assertRerankText(config.falseToken, 'falseToken'),
    falseTokenId,
    score,
    probability,
  };
}

function replaceRerankTemplate(template, values) {
  let output = template;
  for (const [key, value] of Object.entries(values)) {
    const placeholder = `{${key}}`;
    if (!output.includes(placeholder)) {
      throw new Error(`Manifest rerank inputTemplate is missing ${placeholder}.`);
    }
    output = output.split(placeholder).join(value);
  }
  return output;
}

export function formatRerankPrompt(query, document, scoringConfig) {
  const instruction = assertRerankText(scoringConfig?.instruction, 'instruction');
  const normalizedQuery = assertRerankText(query, 'query');
  const normalizedDocument = assertRerankText(document, 'document');
  const input = replaceRerankTemplate(
    assertRerankText(scoringConfig?.inputTemplate, 'inputTemplate', true),
    {
      instruction,
      query: normalizedQuery,
      document: normalizedDocument,
    }
  );
  return `${assertRerankText(scoringConfig?.prefix, 'prefix', true)}${input}${assertRerankText(scoringConfig?.suffix, 'suffix', true)}`;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function computeRerankScore(scoringConfig, trueLogit, falseLogit) {
  if (scoringConfig.score === 'logit_difference') {
    return trueLogit - falseLogit;
  }
  if (scoringConfig.score === 'true_logit') {
    return trueLogit;
  }
  throw new Error(`Unsupported rerank score policy "${scoringConfig.score}".`);
}

function assertLogitsVector(value) {
  if (!ArrayBuffer.isView(value) && !Array.isArray(value)) {
    throw new Error('Rerank prefillWithLogits result must include a logits vector.');
  }
  return value;
}

function buildRerankScoreRecord(query, document, prompt, tokenCount, trueLogit, falseLogit, config, scoringPath, phase = null) {
  if (!Number.isFinite(trueLogit) || !Number.isFinite(falseLogit)) {
    throw new Error(
      `Rerank logits missing finite yes/no scores at token IDs ${config.trueTokenId}/${config.falseTokenId}.`
    );
  }
  const score = computeRerankScore(config, trueLogit, falseLogit);
  const probability = sigmoid(score);
  return {
    query,
    document,
    prompt,
    tokenCount,
    score,
    probability,
    trueLogit,
    falseLogit,
    trueTokenId: config.trueTokenId,
    falseTokenId: config.falseTokenId,
    scoringPath,
    phase,
  };
}

export async function scoreRerankDocument(pipeline, query, document, scoringConfig = null, options = {}) {
  if (!pipeline || (typeof pipeline.prefillWithTokenLogits !== 'function' && typeof pipeline.prefillWithLogits !== 'function')) {
    throw new Error('Rerank workload requires pipeline.prefillWithTokenLogits() or pipeline.prefillWithLogits().');
  }
  const config = scoringConfig ?? resolveRerankScoringConfig(pipeline);
  const prompt = formatRerankPrompt(query, document, config);
  pipeline.reset?.();
  let trueLogit;
  let falseLogit;
  let tokenCount = 0;
  let scoringPath = 'full-logits';
  let resultPhase = null;
  const totalStart = performance.now();
  if (typeof pipeline.prefillWithTokenLogits === 'function') {
    const prefillCallStart = performance.now();
    const result = await pipeline.prefillWithTokenLogits(
      prompt,
      [config.trueTokenId, config.falseTokenId],
      {
        useChatTemplate: false,
        benchmark: options.benchmark === true,
      }
    );
    const prefillCallMs = performance.now() - prefillCallStart;
    const logits = assertLogitsVector(result?.logits);
    trueLogit = Number(logits[0]);
    falseLogit = Number(logits[1]);
    tokenCount = Number.isFinite(result?.tokens?.length) ? result.tokens.length : 0;
    scoringPath = 'selected-token-logits';
    resultPhase = {
      ...(isPlainObject(result?.phase) ? result.phase : {}),
      prefillCallMs,
    };
  } else {
    const prefillCallStart = performance.now();
    const result = await pipeline.prefillWithLogits(prompt, {
      useChatTemplate: false,
      benchmark: options.benchmark === true,
    });
    const prefillCallMs = performance.now() - prefillCallStart;
    const logits = assertLogitsVector(result?.logits);
    trueLogit = Number(logits[config.trueTokenId]);
    falseLogit = Number(logits[config.falseTokenId]);
    tokenCount = Number.isFinite(result?.tokens?.length) ? result.tokens.length : 0;
    resultPhase = {
      ...(isPlainObject(result?.phase) ? result.phase : {}),
      prefillCallMs,
    };
  }
  return buildRerankScoreRecord(
    query,
    document,
    prompt,
    tokenCount,
    trueLogit,
    falseLogit,
    config,
    scoringPath,
    {
      ...(resultPhase ?? {}),
      totalMs: performance.now() - totalStart,
      promptChars: prompt.length,
    }
  );
}

function resolveRerankInput(runtimeConfig, runOverrides = null) {
  const source = isPlainObject(runOverrides?.rerank)
    ? runOverrides.rerank
    : runtimeConfig?.inference?.rerank;
  if (!isPlainObject(source)) {
    throw new Error('Harness rerank requires explicit runtime.inference.rerank.');
  }
  const query = asText(source.query);
  if (!query) {
    throw new Error('Harness rerank requires non-empty runtime.inference.rerank.query.');
  }
  const documents = Array.isArray(source.documents)
    ? source.documents.map(asText).filter(Boolean)
    : [];
  if (documents.length === 0) {
    throw new Error('Harness rerank requires non-empty runtime.inference.rerank.documents.');
  }
  return { query, documents };
}

function summarizeRerankScores(scores) {
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

function getPipelineTokenizer(pipeline) {
  const tokenizer = pipeline?.tokenizer;
  return tokenizer && typeof tokenizer.encode === 'function' ? tokenizer : null;
}

function longestCommonTokenPrefixLength(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return 0;
  }
  const first = rows[0];
  let length = Array.isArray(first) ? first.length : 0;
  for (let rowIndex = 1; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    if (!Array.isArray(row)) {
      return 0;
    }
    length = Math.min(length, row.length);
    for (let index = 0; index < length; index += 1) {
      if (row[index] !== first[index]) {
        length = index;
        break;
      }
    }
  }
  return length;
}

async function createRerankPrefixContext(pipeline, query, documents, config, options = {}) {
  const totalStart = performance.now();
  if (
    typeof pipeline?.prefillKVOnly !== 'function'
    || typeof pipeline?.prefillWithTokenLogits !== 'function'
    || typeof pipeline?.resetToSeqLen !== 'function'
  ) {
    return null;
  }
  const tokenizer = getPipelineTokenizer(pipeline);
  if (!tokenizer) {
    return null;
  }
  const prompts = documents.map((document) => formatRerankPrompt(query, document, config));
  const tokenRows = prompts.map((prompt, index) => normalizeTokenIdArray(
    tokenizer.encode(prompt),
    `rerank.prompt[${index}].tokenIds`
  ));
  const prefixLength = longestCommonTokenPrefixLength(tokenRows);
  if (prefixLength <= 0 || tokenRows.some((row) => row.length <= prefixLength)) {
    return null;
  }
  const prefixTokens = tokenRows[0].slice(0, prefixLength);
  pipeline.reset?.();
  const prefillStart = performance.now();
  const prefix = await pipeline.prefillKVOnly('', {
    useChatTemplate: false,
    inputIds: prefixTokens,
    benchmark: options.benchmark === true,
  });
  const prefillMs = performance.now() - prefillStart;
  return {
    prefix,
    seqLen: prefix.seqLen,
    prefixTokens,
    prompts,
    tokenRows,
    suffixRows: tokenRows.map((row) => row.slice(prefixLength)),
    phase: {
      totalMs: performance.now() - totalStart,
      prefillMs,
      prefixTokens: prefixTokens.length,
      documentCount: documents.length,
    },
  };
}

async function scoreRerankDocumentFromPrefix(pipeline, query, document, config, prefixContext, index, options = {}) {
  const suffixTokens = prefixContext.suffixRows[index];
  const totalStart = performance.now();
  try {
    const prefillCallStart = performance.now();
    const result = await pipeline.prefillWithTokenLogits(
      '',
      [config.trueTokenId, config.falseTokenId],
      {
        useChatTemplate: false,
        inputIds: suffixTokens,
        benchmark: options.benchmark === true,
      }
    );
    const prefillCallMs = performance.now() - prefillCallStart;
    const logits = assertLogitsVector(result?.logits);
    const trueLogit = Number(logits[0]);
    const falseLogit = Number(logits[1]);
    return buildRerankScoreRecord(
      query,
      document,
      prefixContext.prompts[index],
      prefixContext.prefixTokens.length + suffixTokens.length,
      trueLogit,
      falseLogit,
      config,
      'prefix-selected-token-logits',
      {
        ...(isPlainObject(result?.phase) ? result.phase : {}),
        totalMs: performance.now() - totalStart,
        prefillCallMs,
        suffixTokens: suffixTokens.length,
      }
    );
  } finally {
    pipeline.resetToSeqLen(prefixContext.seqLen);
  }
}

export async function runRerank(pipeline, runtimeConfig, runOverrides = null) {
  const input = resolveRerankInput(runtimeConfig, runOverrides);
  const config = resolveRerankScoringConfig(pipeline);
  const start = performance.now();
  const scores = [];
  const prefixStart = performance.now();
  const prefixContext = await createRerankPrefixContext(pipeline, input.query, input.documents, config, {
    benchmark: runOverrides?.benchmark === true,
  });
  const prefixMs = performance.now() - prefixStart;
  for (let i = 0; i < input.documents.length; i++) {
    const scored = prefixContext
      ? await scoreRerankDocumentFromPrefix(
        pipeline,
        input.query,
        input.documents[i],
        config,
        prefixContext,
        i,
        {
          benchmark: runOverrides?.benchmark === true,
        }
      )
      : await scoreRerankDocument(
        pipeline,
        input.query,
        input.documents[i],
        config,
        {
          benchmark: runOverrides?.benchmark === true,
        }
      );
    scores.push({
      index: i,
      ...scored,
    });
  }
  const summary = summarizeRerankScores(scores);
  const durationMs = Math.max(1, performance.now() - start);
  const documentDurations = scores
    .map((entry) => Number(entry.phase?.totalMs))
    .filter((value) => Number.isFinite(value));
  const documentTotalMs = documentDurations.reduce((sum, value) => sum + value, 0);
  return {
    query: input.query,
    documents: input.documents,
    documentCount: input.documents.length,
    scores,
    ranking: summary.ranking,
    topDocument: summary.top
      ? {
        index: summary.top.index,
        document: summary.top.document,
        score: summary.top.score,
        probability: summary.top.probability,
      }
      : null,
    phase: {
      totalMs: durationMs,
      prefixMs,
      prefixApplied: prefixContext != null,
      prefixTokens: Number.isFinite(prefixContext?.prefixTokens?.length)
        ? prefixContext.prefixTokens.length
        : 0,
      prefix: prefixContext?.phase ?? null,
      documentCount: input.documents.length,
      documentTotalMs,
      maxDocumentMs: documentDurations.length > 0 ? Math.max(...documentDurations) : 0,
      avgDocumentMs: documentDurations.length > 0 ? documentTotalMs / documentDurations.length : 0,
      documents: scores.map((entry) => ({
        index: entry.index,
        scoringPath: entry.scoringPath,
        tokenCount: entry.tokenCount,
        phase: entry.phase ?? null,
      })),
    },
    durationMs,
  };
}

function resolveRerankSemanticFixtures(runtimeConfig, options = null) {
  const overrides = isPlainObject(options?.rerankSemantic)
    ? options.rerankSemantic
    : null;
  const runtimeOverrides = runtimeConfig?.shared?.benchmark?.run?.rerankSemantic;
  const source = overrides ?? (isPlainObject(runtimeOverrides) ? runtimeOverrides : null);
  const cases = normalizeRerankCases(source?.cases)
    ?? DEFAULT_RERANK_SEMANTIC_FIXTURES.cases;
  const minPairAcc = Number.isFinite(source?.minPairAcc)
    ? Math.max(0, Math.min(1, Number(source.minPairAcc)))
    : DEFAULT_RERANK_SEMANTIC_FIXTURES.minPairAcc;
  const minScoreMargin = Number.isFinite(source?.minScoreMargin)
    ? Number(source.minScoreMargin)
    : DEFAULT_RERANK_SEMANTIC_FIXTURES.minScoreMargin;
  return {
    cases,
    minPairAcc,
    minScoreMargin,
  };
}

export async function runRerankSemanticChecks(pipeline, options = null) {
  const fixture = resolveRerankSemanticFixtures(pipeline?.runtimeConfig ?? {}, options);
  const config = resolveRerankScoringConfig(pipeline);
  const start = performance.now();
  const pairs = [];
  let pairPassed = 0;
  for (const testCase of fixture.cases) {
    const positive = await scoreRerankDocument(
      pipeline,
      testCase.query,
      testCase.positive,
      config
    );
    const negative = await scoreRerankDocument(
      pipeline,
      testCase.query,
      testCase.negative,
      config
    );
    const margin = positive.score - negative.score;
    const passed = Number.isFinite(margin) && margin > fixture.minScoreMargin;
    if (passed) pairPassed++;
    pairs.push({
      id: testCase.id,
      query: testCase.query,
      positive: testCase.positive,
      negative: testCase.negative,
      passed,
      positiveScore: Number(positive.score.toFixed(6)),
      negativeScore: Number(negative.score.toFixed(6)),
      positiveProbability: Number(positive.probability.toFixed(6)),
      negativeProbability: Number(negative.probability.toFixed(6)),
      margin: Number.isFinite(margin) ? Number(margin.toFixed(6)) : null,
    });
  }
  const pairAcc = pairs.length > 0 ? pairPassed / pairs.length : 0;
  const passed = pairAcc >= fixture.minPairAcc;
  return {
    passed,
    pairAcc,
    pairPassed,
    pairTotal: pairs.length,
    minPairAcc: Number(fixture.minPairAcc.toFixed(4)),
    minScoreMargin: Number(fixture.minScoreMargin.toFixed(4)),
    failedCaseIds: pairs.filter((item) => !item.passed).map((item) => item.id),
    pairs,
    durationMs: Math.max(1, performance.now() - start),
  };
}

function isStructuredPromptInput(value) {
  return Array.isArray(value) || (value != null && typeof value === 'object');
}

function clonePromptInput(promptInput) {
  if (!isStructuredPromptInput(promptInput)) {
    return promptInput;
  }
  if (typeof structuredClone === 'function') {
    return structuredClone(promptInput);
  }
  return JSON.parse(JSON.stringify(promptInput));
}

function resolvePromptTemplateType(source) {
  const sourceTemplateType = asText(source?.chatTemplateType);
  if (sourceTemplateType) {
    return sourceTemplateType;
  }
  const modelConfigTemplateType = asText(source?.modelConfig?.chatTemplateType);
  if (modelConfigTemplateType) {
    return modelConfigTemplateType;
  }
  return asText(source?.manifest?.inference?.chatTemplate?.type);
}

function assertPromptContract(runtimePrompt, templateType, source = 'runtime.inference.prompt') {
  if (templateType !== 'translategemma') {
    return;
  }
  if (runtimePrompt === undefined || runtimePrompt === null) {
    return;
  }
  if (typeof runtimePrompt === 'string') {
    throw new Error(
      `TranslateGemma harness prompt contract violation: ${source} must be ` +
      '{ messages: [...] } with source_lang_code/target_lang_code blocks, not a plain string.'
    );
  }
  if (!isStructuredPromptInput(runtimePrompt)) {
    throw new Error(
      `TranslateGemma harness prompt contract violation: ${source} must be ` +
      '{ messages: [...] } with source_lang_code/target_lang_code blocks.'
    );
  }
}

function describePromptInput(promptInput) {
  if (typeof promptInput === 'string') {
    return promptInput.trim() || '[empty prompt]';
  }
  if (isPlainObject(promptInput?.image) && typeof promptInput?.prompt === 'string') {
    const width = Number.isFinite(promptInput.image.width) ? promptInput.image.width : '?';
    const height = Number.isFinite(promptInput.image.height) ? promptInput.image.height : '?';
    const source = asText(promptInput.image.source) ?? 'image';
    return `${source} ${width}x${height}: ${promptInput.prompt}`;
  }
  const firstMessage = Array.isArray(promptInput?.messages)
    ? promptInput.messages[0]
    : null;
  const firstContent = Array.isArray(firstMessage?.content)
    ? firstMessage.content[0]
    : null;
  const sourceLang = asText(firstContent?.source_lang_code);
  const targetLang = asText(firstContent?.target_lang_code);
  const text = asText(firstContent?.text);
  if (sourceLang && targetLang) {
    return `${sourceLang} -> ${targetLang}: ${text || '[non-text request]'}`;
  }
  const stringContent = asText(firstMessage?.content);
  if (stringContent) {
    const role = asText(firstMessage?.role) || 'user';
    return `${role}: ${stringContent}`;
  }
  try {
    return JSON.stringify(promptInput);
  } catch {
    return '[structured prompt]';
  }
}

function decodeBase64ToBytes(base64, label) {
  const normalized = asText(base64);
  if (!normalized) {
    throw new Error(`${label} must be a non-empty base64 string.`);
  }
  if (typeof Buffer !== 'undefined') {
    return new Uint8Array(Buffer.from(normalized, 'base64'));
  }
  if (typeof atob === 'function') {
    const binary = atob(normalized);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }
  throw new Error(`${label} requires a base64 decoder in this runtime.`);
}

function assertRawImageByteLength(bytes, width, height, label) {
  const expectedRgb = width * height * 3;
  const expectedRgba = width * height * 4;
  if (bytes.length !== expectedRgb && bytes.length !== expectedRgba) {
    throw new Error(
      `${label} must contain width*height*3 or width*height*4 bytes. ` +
      `Got ${bytes.length} for ${width}x${height}.`
    );
  }
}

function normalizeRawImageBytes(value, width, height, label) {
  let bytes = null;
  if (ArrayBuffer.isView(value)) {
    bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  } else if (Array.isArray(value)) {
    const normalized = new Uint8Array(value.length);
    for (let i = 0; i < value.length; i++) {
      const parsed = Number(value[i]);
      if (!Number.isInteger(parsed) || parsed < 0 || parsed > 255) {
        throw new Error(`${label}[${i}] must be an integer in [0, 255].`);
      }
      normalized[i] = parsed;
    }
    bytes = normalized;
  }
  if (!(bytes instanceof Uint8Array)) {
    throw new Error(`${label} must be an array or typed array.`);
  }
  assertRawImageByteLength(bytes, width, height, label);
  return new Uint8Array(bytes);
}

function createCanvasForImageDecode(width, height) {
  if (typeof OffscreenCanvas === 'function') {
    return new OffscreenCanvas(width, height);
  }
  if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  return null;
}

function isNodeRuntime() {
  return typeof process !== 'undefined' && !!process.versions?.node;
}

async function decodeImageUrlToPixelsOnNode(url) {
  let sharpModule = null;
  try {
    sharpModule = await import('sharp');
  } catch (error) {
    throw new Error(
      `URL-backed inferenceInput.image.url on the node surface requires the optional "sharp" decoder. ${error?.message || error}`
    );
  }
  const sharp = typeof sharpModule?.default === 'function'
    ? sharpModule.default
    : sharpModule;
  if (typeof sharp !== 'function') {
    throw new Error('Node image decode requires sharp to export a callable default.');
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`image fetch failed: HTTP ${response.status}`);
  }
  const sourceBytes = new Uint8Array(await response.arrayBuffer());
  const decoded = await sharp(sourceBytes)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return {
    imageBytes: new Uint8Array(decoded.data),
    width: decoded.info.width,
    height: decoded.info.height,
  };
}

async function decodeImageUrlToPixels(url) {
  if (isNodeRuntime()) {
    return decodeImageUrlToPixelsOnNode(url);
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`image fetch failed: HTTP ${response.status}`);
  }
  if (typeof createImageBitmap !== 'function') {
    throw new Error(
      'URL-backed inferenceInput.image.url requires createImageBitmap support. ' +
      'Use raw pixels or run on a browser-capable surface.'
    );
  }

  const imageBlob = await response.blob();
  const imageBitmap = await createImageBitmap(imageBlob);
  try {
    const canvas = createCanvasForImageDecode(imageBitmap.width, imageBitmap.height);
    if (!canvas) {
      throw new Error(
        'URL-backed inferenceInput.image.url requires OffscreenCanvas or a DOM canvas in this runtime. ' +
        'Use raw pixels or run on a browser-capable surface.'
      );
    }
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context || typeof context.getImageData !== 'function') {
      throw new Error('Image decode canvas did not provide a readable 2D context.');
    }
    context.drawImage(imageBitmap, 0, 0);
    const imageData = context.getImageData(0, 0, imageBitmap.width, imageBitmap.height);
    return {
      imageBytes: new Uint8Array(imageData.data),
      width: imageBitmap.width,
      height: imageBitmap.height,
    };
  } finally {
    imageBitmap.close?.();
  }
}

async function resolveInferenceImagePayload(imageInput) {
  if (!isPlainObject(imageInput)) {
    throw new Error('inference image input must be an object.');
  }

  if (typeof imageInput.url === 'string' && imageInput.url.trim()) {
    const decoded = await decodeImageUrlToPixels(imageInput.url.trim());
    return {
      imageBytes: decoded.imageBytes,
      width: decoded.width,
      height: decoded.height,
      descriptor: {
        source: 'url',
        width: decoded.width,
        height: decoded.height,
        url: imageInput.url.trim(),
      },
    };
  }

  const width = Math.max(1, Math.floor(Number(imageInput.width)));
  const height = Math.max(1, Math.floor(Number(imageInput.height)));
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    throw new Error('Raw inference image input requires positive integer width and height.');
  }

  if (typeof imageInput.pixelDataBase64 === 'string' && imageInput.pixelDataBase64.trim()) {
    const decodedBytes = decodeBase64ToBytes(
      imageInput.pixelDataBase64,
      'inferenceInput.image.pixelDataBase64'
    );
    assertRawImageByteLength(
      decodedBytes,
      width,
      height,
      'inferenceInput.image.pixelDataBase64'
    );
    return {
      imageBytes: decodedBytes,
      width,
      height,
      descriptor: {
        source: 'pixelDataBase64',
        width,
        height,
      },
    };
  }

  return {
    imageBytes: normalizeRawImageBytes(
      imageInput.pixels,
      width,
      height,
      'inferenceInput.image.pixels'
    ),
    width,
    height,
    descriptor: {
      source: 'pixels',
      width,
      height,
    },
  };
}

function resolveGenerationPromptInput(runtimeConfig, runOverrides = null, source = null) {
  const templateType = resolvePromptTemplateType(source);
  const overridePrompt = runOverrides?.prompt;
  assertPromptContract(overridePrompt, templateType, 'runOverrides.prompt');
  if (typeof overridePrompt === 'string' && overridePrompt.trim()) {
    return overridePrompt;
  }
  if (isStructuredPromptInput(overridePrompt)) {
    return clonePromptInput(overridePrompt);
  }

  const runtimePrompt = runtimeConfig?.inference?.prompt;
  assertPromptContract(runtimePrompt, templateType, 'runtimeConfig.inference.prompt');
  if (typeof runtimePrompt === 'string' && runtimePrompt.trim()) {
    return runtimePrompt;
  }
  if (isStructuredPromptInput(runtimePrompt)) {
    return clonePromptInput(runtimePrompt);
  }

  throw new Error('Harness generation requires explicit runOverrides.prompt or runtime.inference.prompt.');
}

function resolveMaxTokens(runtimeConfig) {
  const runtimeMax = runtimeConfig?.inference?.generation?.maxTokens;
  if (Number.isFinite(runtimeMax) && runtimeMax > 0) {
    return Math.floor(runtimeMax);
  }
  throw new Error('Harness generation requires explicit runtime.inference.generation.maxTokens.');
}

function resolveAutomaticGenerationDiagnostics(runtimeConfig, runOverrides = null) {
  const overrideDiagnostics = runOverrides?.diagnostics ?? null;
  if (overrideDiagnostics?.enabled === true) {
    return overrideDiagnostics;
  }

  const diagnosticsPolicy = runtimeConfig?.shared?.tooling?.diagnostics ?? 'off';
  if (diagnosticsPolicy !== 'always') {
    return overrideDiagnostics;
  }

  return {
    enabled: true,
    captureConfig: {
      ...createDefaultCaptureConfig(),
      enabled: true,
      defaultLevel: CAPTURE_LEVELS.NONE,
    },
  };
}

export function resolveBenchmarkRunSettings(runtimeConfig, source = null) {
  const benchConfig = runtimeConfig?.shared?.benchmark?.run || {};
  const runtimeSampling = isPlainObject(runtimeConfig?.inference?.sampling)
    ? runtimeConfig.inference.sampling
    : {};
  const benchSampling = isPlainObject(benchConfig?.sampling)
    ? benchConfig.sampling
    : {};
  const runSeed = Number.isFinite(benchConfig.seed)
    ? Math.max(0, Math.floor(benchConfig.seed))
    : null;
  const runtimeSeed = Number.isFinite(runtimeSampling.seed)
    ? Math.max(0, Math.floor(runtimeSampling.seed))
    : null;
  const benchSeed = Number.isFinite(benchSampling.seed)
    ? Math.max(0, Math.floor(benchSampling.seed))
    : null;
  const mergedSeed = runSeed != null
    ? runSeed
    : benchSeed != null
      ? benchSeed
      : runtimeSeed;
  const promptInput = typeof benchConfig.customPrompt === 'string' && benchConfig.customPrompt.trim()
    ? benchConfig.customPrompt
    : resolveGenerationPromptInput(runtimeConfig, null, source);
  const maxTokens = Number.isFinite(benchConfig.maxNewTokens)
    ? Math.max(1, Math.floor(benchConfig.maxNewTokens))
    : resolveMaxTokens(runtimeConfig);
  const sampling = {
    ...runtimeSampling,
    ...benchSampling,
  };
  if (Number.isFinite(mergedSeed)) {
    sampling.seed = mergedSeed;
  }

  return {
    warmupRuns: Math.max(0, Math.floor(benchConfig.warmupRuns ?? 0)),
    timedRuns: Math.max(1, Math.floor(benchConfig.timedRuns ?? 1)),
    ...(Number.isFinite(mergedSeed) ? { seed: mergedSeed } : {}),
    prompt: promptInput,
    promptLabel: describePromptInput(promptInput),
    maxTokens,
    sampling,
  };
}

function summarizeEmbeddingValues(embedding) {
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

function cosineSimilarity(a, b) {
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

function top1Index(values) {
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

async function embedStandaloneText(pipeline, text) {
  pipeline.reset?.();
  const result = await pipeline.embed(text);
  const embedding = result?.embedding;
  if (!embedding || !Number.isFinite(embedding.length) || embedding.length <= 0) {
    throw new Error('Semantic check embedding is missing.');
  }
  return embedding;
}

export async function runEmbeddingSemanticChecks(pipeline, options = null) {
  const config = resolveEmbeddingSemanticFixtures(
    pipeline?.runtimeConfig ?? {},
    options
  );
  const start = performance.now();
  const semanticStyle = resolveEmbeddingSemanticStyle(pipeline);
  const retrieval = [];
  let retrievalPassed = 0;

  for (const testCase of config.retrievalCases) {
    const formattedQuery = formatEmbeddingSemanticText(testCase.query, 'query', semanticStyle);
    const queryEmbedding = await embedStandaloneText(
      pipeline,
      formattedQuery
    );
    const docEmbeddings = [];
    const docs = [];
    for (const doc of testCase.docs) {
      const formattedDoc = formatEmbeddingSemanticText(doc, 'document', semanticStyle);
      docEmbeddings.push(await embedStandaloneText(
        pipeline,
        formattedDoc
      ));
      docs.push({
        text: doc,
        formattedText: formattedDoc,
      });
    }
    const sims = docEmbeddings.map((docEmbedding) => cosineSimilarity(queryEmbedding, docEmbedding));
    const topDoc = top1Index(sims);
    const passed = topDoc === testCase.expectedDoc;
    if (passed) retrievalPassed++;
    retrieval.push({
      id: testCase.id,
      query: testCase.query,
      formattedQuery,
      docs,
      passed,
      expectedDoc: testCase.expectedDoc,
      topDoc,
      sims: sims.map((v) => (Number.isFinite(v) ? Number(v.toFixed(6)) : null)),
    });
  }

  const pairs = [];
  let pairPassed = 0;
  for (const testCase of config.pairCases) {
    const formattedAnchor = formatEmbeddingSemanticText(testCase.anchor, 'query', semanticStyle);
    const anchor = await embedStandaloneText(
      pipeline,
      formattedAnchor
    );
    const formattedPositive = formatEmbeddingSemanticText(testCase.positive, 'query', semanticStyle);
    const positive = await embedStandaloneText(
      pipeline,
      formattedPositive
    );
    const formattedNegative = formatEmbeddingSemanticText(testCase.negative, 'query', semanticStyle);
    const negative = await embedStandaloneText(
      pipeline,
      formattedNegative
    );
    const simPos = cosineSimilarity(anchor, positive);
    const simNeg = cosineSimilarity(anchor, negative);
    const margin = simPos - simNeg;
    const passed = Number.isFinite(margin) && margin > config.pairMargin;
    if (passed) pairPassed++;
    pairs.push({
      id: testCase.id,
      anchor: testCase.anchor,
      formattedAnchor,
      positive: testCase.positive,
      formattedPositive,
      negative: testCase.negative,
      formattedNegative,
      passed,
      simPos: Number.isFinite(simPos) ? Number(simPos.toFixed(6)) : null,
      simNeg: Number.isFinite(simNeg) ? Number(simNeg.toFixed(6)) : null,
      margin: Number.isFinite(margin) ? Number(margin.toFixed(6)) : null,
    });
  }

  const lengthStability = [];
  let lengthStabilityPassed = 0;
  for (const testCase of config.lengthStabilityCases) {
    const shortEmb = await embedStandaloneText(
      pipeline,
      formatEmbeddingSemanticText(testCase.short, 'document', semanticStyle)
    );
    const mediumEmb = await embedStandaloneText(
      pipeline,
      formatEmbeddingSemanticText(testCase.medium, 'document', semanticStyle)
    );
    const longEmb = await embedStandaloneText(
      pipeline,
      formatEmbeddingSemanticText(testCase.long, 'document', semanticStyle)
    );
    const simShortMedium = cosineSimilarity(shortEmb, mediumEmb);
    const simShortLong = cosineSimilarity(shortEmb, longEmb);
    const simMediumLong = cosineSimilarity(mediumEmb, longEmb);
    const minSim = Math.min(
      Number.isFinite(simShortMedium) ? simShortMedium : -1,
      Number.isFinite(simShortLong) ? simShortLong : -1,
      Number.isFinite(simMediumLong) ? simMediumLong : -1
    );
    const maxDrift = 1 - minSim;
    const passed = Number.isFinite(maxDrift) && maxDrift <= testCase.maxCosineDrift;
    if (passed) lengthStabilityPassed++;
    lengthStability.push({
      id: testCase.id,
      passed,
      simShortMedium: Number.isFinite(simShortMedium) ? Number(simShortMedium.toFixed(6)) : null,
      simShortLong: Number.isFinite(simShortLong) ? Number(simShortLong.toFixed(6)) : null,
      simMediumLong: Number.isFinite(simMediumLong) ? Number(simMediumLong.toFixed(6)) : null,
      maxDrift: Number.isFinite(maxDrift) ? Number(maxDrift.toFixed(6)) : null,
      maxCosineDrift: testCase.maxCosineDrift,
    });
  }

  let throughput = null;
  if (config.throughputCorpus.length > 0) {
    const corpusStart = performance.now();
    for (const text of config.throughputCorpus) {
      await embedStandaloneText(
        pipeline,
        formatEmbeddingSemanticText(text, 'document', semanticStyle)
      );
    }
    const corpusDurationMs = Math.max(1, performance.now() - corpusStart);
    throughput = {
      corpusSize: config.throughputCorpus.length,
      durationMs: Number(corpusDurationMs.toFixed(1)),
      docsPerSecond: Number((config.throughputCorpus.length / (corpusDurationMs / 1000)).toFixed(2)),
    };
  }

  const retrievalTop1Acc = retrieval.length > 0 ? retrievalPassed / retrieval.length : 0;
  const pairAcc = pairs.length > 0 ? pairPassed / pairs.length : 0;
  const lengthStabilityAcc = lengthStability.length > 0
    ? lengthStabilityPassed / lengthStability.length : 1;
  const passed = retrievalTop1Acc >= config.minRetrievalTop1Acc
    && pairAcc >= config.minPairAcc;
  const failedCaseIds = [
    ...retrieval.filter((item) => !item.passed).map((item) => `retrieval:${item.id}`),
    ...pairs.filter((item) => !item.passed).map((item) => `pair:${item.id}`),
    ...lengthStability.filter((item) => !item.passed).map((item) => `length:${item.id}`),
  ];

  return {
    passed,
    style: semanticStyle,
    retrievalTop1Acc,
    pairAcc,
    lengthStabilityAcc,
    retrievalPassed,
    retrievalTotal: retrieval.length,
    pairPassed,
    pairTotal: pairs.length,
    lengthStabilityPassed,
    lengthStabilityTotal: lengthStability.length,
    minRetrievalTop1Acc: Number(config.minRetrievalTop1Acc.toFixed(4)),
    minPairAcc: Number(config.minPairAcc.toFixed(4)),
    pairMarginThreshold: Number(config.pairMargin.toFixed(4)),
    failedCaseIds,
    retrieval,
    pairs,
    lengthStability,
    throughput,
    durationMs: Math.max(1, performance.now() - start),
  };
}

const SPECIAL_TOKEN_RE = /^(<pad>|<unused\d*>|<eos>|<bos>|<s>|<\/s>|\[PAD\]|\[UNK\]|\[SEP\]|\[CLS\]|<[^>]{1,32}>)$/i;
const PAD_DOMINANCE_THRESHOLD = 0.5;

function isSpecialLikeTokenText(value) {
  if (typeof value !== 'string') return false;
  return SPECIAL_TOKEN_RE.test(value.trim());
}

function summarizeGenerationTokens(tokenRecords) {
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

function buildGenerationPhaseFromStats(pipeline, durationMs, tokenCount) {
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

export async function runGeneration(pipeline, runtimeConfig, runOverrides = null) {
  const tokens = [];
  const tokenIds = [];
  const tokenRecords = [];
  const logitsDigests = [];
  const promptInput = resolveGenerationPromptInput(runtimeConfig, runOverrides, pipeline);
  const promptLabel = describePromptInput(promptInput);
  const useChatTemplate = resolveGenerationUseChatTemplate(pipeline, runtimeConfig, runOverrides, promptInput);
  const promptTokenIds = resolvePromptTokenIdsForTranscript(pipeline, promptInput, useChatTemplate);
  const maxTokens = Number.isFinite(runOverrides?.maxTokens)
    ? Math.max(1, Math.floor(runOverrides.maxTokens))
    : resolveMaxTokens(runtimeConfig);
  const sampling = {
    ...(runtimeConfig.inference?.sampling || {}),
    ...(isPlainObject(runOverrides?.sampling) ? runOverrides.sampling : {}),
  };
  const seed = Number.isFinite(runOverrides?.seed)
    ? Math.max(0, Math.floor(runOverrides.seed))
    : Number.isFinite(sampling.seed)
      ? Math.max(0, Math.floor(sampling.seed))
      : null;
  const debugProbes = runtimeConfig.shared?.debug?.probes || [];
  const profile = runtimeConfig.shared?.debug?.profiler?.enabled === true;
  const explicitDiagnosticsEnabled = runtimeConfig.shared?.harness?.mode === 'diagnose'
    || shouldEnableReferenceTranscriptDiagnostics(runOverrides, runtimeConfig);
  const disableCommandBatchingForDiagnostics = explicitDiagnosticsEnabled
    || (Array.isArray(debugProbes) && debugProbes.length > 0);
  const start = performance.now();
  const diagnostics = resolveAutomaticGenerationDiagnostics(runtimeConfig, runOverrides);
  const captureLogits = shouldCaptureReferenceLogits(runOverrides, runtimeConfig);

  for await (const tokenText of pipeline.generate(promptInput, {
    maxTokens,
    ...(Number.isFinite(seed) ? { seed } : {}),
    temperature: sampling.temperature,
    topP: sampling.topP,
    topK: sampling.topK,
    repetitionPenalty: sampling.repetitionPenalty,
    greedyThreshold: sampling.greedyThreshold,
    useChatTemplate,
    benchmark: runOverrides?.benchmark === true,
    profile,
    ...(disableCommandBatchingForDiagnostics ? { disableCommandBatching: true } : {}),
    diagnostics,
    ...(captureLogits ? {
      onLogits: (logits, context) => {
        logitsDigests.push(digestLogitsForTranscript(logits, {
          ...context,
          index: logitsDigests.length,
          decodeToken: (tokenId) => pipeline?.tokenizer?.decode?.([tokenId], false, false) ?? null,
        }));
      },
    } : {}),
    onToken: (tokenId, tokenText) => {
      tokenIds.push(tokenId);
      tokenRecords.push({
        id: tokenId,
        text: typeof tokenText === 'string' ? tokenText : '',
        fallbackText: pipeline?.tokenizer?.decode?.([tokenId], false, false) ?? '',
      });
    },
  })) {
    if (typeof tokenText === 'string') {
      tokens.push(tokenText);
    }
  }

  const durationMs = Math.max(1, performance.now() - start);
  const tokensPerSec = (tokens.length / durationMs) * 1000;
  const { phase } = buildGenerationPhaseFromStats(pipeline, durationMs, tokenIds.length);
  const kvCacheByteProof = await captureKvCacheByteProof(
    pipeline,
    shouldCaptureReferenceKvBytes(runOverrides, runtimeConfig)
  );

  return {
    ...(Number.isFinite(seed) ? { seed } : {}),
    prompt: promptLabel,
    promptInput,
    promptTokenIds,
    maxTokens,
    tokens,
    tokenIds,
    tokenDiagnostics: summarizeGenerationTokens(tokenRecords),
    logitsDigests,
    kvCacheByteProof,
    output: tokens.join(''),
    durationMs,
    tokensPerSec,
    phase,
  };
}

export async function runImageTranscription(pipeline, runtimeConfig, runOverrides = null) {
  const imageInput = runOverrides?.image;
  if (!isPlainObject(imageInput)) {
    throw new Error('Image transcription requires inferenceInput.image.');
  }
  const prompt = typeof runOverrides?.prompt === 'string' && runOverrides.prompt.trim()
    ? runOverrides.prompt.trim()
    : DEFAULT_IMAGE_TRANSCRIPTION_PROMPT;
  const maxTokens = Number.isFinite(runOverrides?.maxTokens)
    ? Math.max(1, Math.floor(runOverrides.maxTokens))
    : resolveMaxTokens(runtimeConfig);
  const softTokenBudget = Number.isFinite(runOverrides?.softTokenBudget)
    ? Math.max(1, Math.floor(runOverrides.softTokenBudget))
    : DEFAULT_IMAGE_TRANSCRIPTION_SOFT_TOKEN_BUDGET;
  const {
    imageBytes,
    width,
    height,
    descriptor,
  } = await resolveInferenceImagePayload(imageInput);
  const start = performance.now();
  const result = await pipeline.transcribeImage({
    imageBytes,
    width,
    height,
    prompt,
    maxTokens,
    softTokenBudget,
  });
  const durationMs = Math.max(1, performance.now() - start);
  const tokenIds = Array.isArray(result?.tokens)
    ? result.tokens.map((value) => Number(value)).filter((value) => Number.isInteger(value))
    : [];
  const tokenRecords = tokenIds.map((tokenId) => {
    const decoded = pipeline?.tokenizer?.decode?.([tokenId], false, false) ?? '';
    return {
      id: tokenId,
      text: decoded,
      fallbackText: decoded,
    };
  });
  const { phase } = buildGenerationPhaseFromStats(pipeline, durationMs, tokenIds.length);
  return {
    inputMode: 'image_to_text',
    prompt: `image ${width}x${height}: ${prompt}`,
    promptInput: {
      prompt,
      image: descriptor,
    },
    maxTokens,
    tokens: tokenRecords.map((record) => record.text),
    tokenIds,
    tokenDiagnostics: summarizeGenerationTokens(tokenRecords),
    output: typeof result?.text === 'string' ? result.text : tokenRecords.map((record) => record.text).join(''),
    durationMs,
    tokensPerSec: tokenIds.length > 0 ? (tokenIds.length / durationMs) * 1000 : 0,
    phase,
  };
}

export async function runTextInference(pipeline, runtimeConfig, runOverrides = null) {
  if (isPlainObject(runOverrides?.image)) {
    return runImageTranscription(pipeline, runtimeConfig, runOverrides);
  }
  return runGeneration(pipeline, runtimeConfig, runOverrides);
}

export async function runEmbedding(pipeline, runtimeConfig, runOverrides = null) {
  const prompt = typeof runOverrides?.prompt === 'string' && runOverrides.prompt.trim()
    ? runOverrides.prompt.trim()
    : resolvePrompt(runtimeConfig);
  const start = performance.now();
  const result = await pipeline.embed(prompt, {
    benchmark: runOverrides?.benchmark === true,
  });
  const durationMs = Math.max(1, performance.now() - start);
  const tokenCount = Number.isFinite(result?.tokens?.length) ? result.tokens.length : 0;
  const stats = summarizeEmbeddingValues(result?.embedding);
  return {
    prompt,
    tokenCount,
    durationMs,
    phase: result?.phase ?? null,
    ...stats,
  };
}
