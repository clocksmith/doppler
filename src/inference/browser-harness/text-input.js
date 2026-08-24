import { CAPTURE_LEVELS, createDefaultCaptureConfig } from '../../debug/capture-policy.js';
import { selectRuleValue } from '../../rules/rule-registry.js';
import { loadJson } from '../../formats/load-json.js';
import { isPlainObject } from '../../formats/plain-object.js';
import { cloneJsonValue } from '../../formats/clone-json.js';

export const DEFAULT_IMAGE_TRANSCRIPTION_PROMPT = 'Describe the image in one short sentence.';
export const DEFAULT_IMAGE_TRANSCRIPTION_SOFT_TOKEN_BUDGET = 70;

const embeddingSemanticFixtureAsset = await loadJson(
  '../fixtures/embedding-semantic-fixtures.json',
  import.meta.url,
  'Failed to load embedding semantic fixtures'
);

const rerankSemanticFixtureAsset = await loadJson(
  '../fixtures/rerank-semantic-fixtures.json',
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

export function resolveEmbeddingSemanticFixtures(runtimeConfig, options = null) {
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

export function resolveEmbeddingSemanticStyle(pipeline) {
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

export function formatEmbeddingSemanticText(text, kind, style) {
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

export function resolveRerankInput(runtimeConfig, runOverrides = null) {
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

export function resolveRerankSemanticFixtures(runtimeConfig, options = null) {
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

export function isStructuredPromptInput(value) {
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

export function describePromptInput(promptInput) {
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

export async function resolveInferenceImagePayload(imageInput) {
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

export function resolveGenerationPromptInput(runtimeConfig, runOverrides = null, source = null) {
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

export function resolveMaxTokens(runtimeConfig) {
  const runtimeMax = runtimeConfig?.inference?.generation?.maxTokens;
  if (Number.isFinite(runtimeMax) && runtimeMax > 0) {
    return Math.floor(runtimeMax);
  }
  throw new Error('Harness generation requires explicit runtime.inference.generation.maxTokens.');
}

export function resolveAutomaticGenerationDiagnostics(runtimeConfig, runOverrides = null) {
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

export function resolveBenchmarkRunSettings(runtimeConfig, source = null, runOverrides = null) {
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
  const promptInput = runOverrides?.prompt != null
    ? resolveGenerationPromptInput(runtimeConfig, runOverrides, source)
    : typeof benchConfig.customPrompt === 'string' && benchConfig.customPrompt.trim()
      ? benchConfig.customPrompt
      : resolveGenerationPromptInput(runtimeConfig, null, source);
  const maxTokens = Number.isFinite(runOverrides?.maxTokens)
    ? Math.max(1, Math.floor(runOverrides.maxTokens))
    : Number.isFinite(benchConfig.maxNewTokens)
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
