import { selectRuleValue } from '../rules/rule-registry.js';
import { isPlainObject } from '../utils/plain-object.js';

const DEFAULT_HARNESS_PROMPT = 'Summarize this input in one sentence.';
const DEFAULT_RUNTIME_PLACEHOLDER_PROMPT = 'Hello from Doppler.';
const DEFAULT_QWEN_PROMPT = Object.freeze({
  messages: Object.freeze([
    Object.freeze({
      role: 'user',
      content: 'Answer in one short sentence: What color is the sky on a clear day?',
    }),
  ]),
});
const DEFAULT_TRANSLATEGEMMA_PROMPT = Object.freeze({
  messages: Object.freeze([
    Object.freeze({
      role: 'user',
      content: Object.freeze([
        Object.freeze({
          type: 'text',
          source_lang_code: 'en',
          target_lang_code: 'fr',
          text: 'Hello world.',
        }),
      ]),
    }),
  ]),
});
const DEFAULT_HARNESS_MAX_TOKENS = 32;
const EMBEDDING_PREVIEW_LENGTH = 16;
const GENERATION_TOKEN_DIAGNOSTIC_LIMIT = 32;
const EMBEDDING_SEMANTIC_MIN_RETRIEVAL_TOP1 = 0.67;
const EMBEDDING_SEMANTIC_MIN_PAIR_ACC = 0.67;
const EMBEDDING_SEMANTIC_PAIR_MARGIN = 0.01;

const EMBEDDING_SEMANTIC_RETRIEVAL_CASES = Object.freeze([
  Object.freeze({
    id: 'library_search',
    query: 'Where can I borrow books and study quietly?',
    docs: Object.freeze([
      'The city library lends books, provides study rooms, and offers free Wi-Fi.',
      'The cafe serves coffee, pastries, and sandwiches all day.',
      'The bike repair shop fixes flat tires and broken chains.',
    ]),
    expectedDoc: 0,
  }),
  Object.freeze({
    id: 'password_reset',
    query: 'How do I reset my account password?',
    docs: Object.freeze([
      'To reset your password, open account settings and choose the forgot-password flow.',
      'Our shipping policy explains delivery timelines and tracking updates.',
      'The recipe combines tomatoes, basil, and olive oil.',
    ]),
    expectedDoc: 0,
  }),
  Object.freeze({
    id: 'damaged_package',
    query: 'What should I do if my package arrives damaged?',
    docs: Object.freeze([
      'Contact support within seven days with photos to request a replacement for damaged items.',
      'The concert starts at 8 PM at the downtown arena.',
      'Plant roses in spring and water them twice a week.',
    ]),
    expectedDoc: 0,
  }),
  Object.freeze({
    id: 'flight_change_policy',
    query: 'Can I change my flight after booking?',
    docs: Object.freeze([
      'The museum opens daily at 10 AM and offers guided tours on weekends.',
      'You can change your flight in Manage Booking up to 24 hours before departure, with any fare difference applied.',
      'Our gym membership includes group classes and access to the pool.',
    ]),
    expectedDoc: 1,
  }),
  Object.freeze({
    id: 'wifi_troubleshoot',
    query: 'Why does my home Wi-Fi keep disconnecting?',
    docs: Object.freeze([
      'The dessert menu includes cheesecake, brownies, and fruit tart.',
      'You can review your recent orders in your account purchase history.',
      'Frequent Wi-Fi drops can be fixed by restarting the router, updating firmware, and changing the wireless channel.',
    ]),
    expectedDoc: 2,
  }),
  Object.freeze({
    id: 'refund_deadline',
    query: 'How long do I have to request a refund?',
    docs: Object.freeze([
      'Refund requests are accepted within 30 days of purchase when the item is in original condition.',
      'The conference keynote starts at 9 AM in the main hall.',
      'Use a medium grind when brewing coffee with a drip machine.',
    ]),
    expectedDoc: 0,
  }),
  Object.freeze({
    id: 'passport_renewal_docs',
    query: 'What documents do I need to renew a passport?',
    docs: Object.freeze([
      'To care for houseplants, water only when the top soil is dry.',
      'Passport renewal usually requires the application form, current passport, compliant photo, and payment.',
      'The train to downtown runs every 20 minutes during peak hours.',
    ]),
    expectedDoc: 1,
  }),
]);

const EMBEDDING_SEMANTIC_PAIR_CASES = Object.freeze([
  Object.freeze({
    id: 'bike_paraphrase',
    anchor: 'The child is riding a bicycle through the park.',
    positive: 'A kid bikes along a path in the park.',
    negative: 'The stock market closed lower after interest-rate news.',
  }),
  Object.freeze({
    id: 'cancel_subscription',
    anchor: 'Please cancel my subscription before renewal.',
    positive: 'I want to stop the plan so it does not renew.',
    negative: 'The mountain trail is closed after heavy snow.',
  }),
  Object.freeze({
    id: 'battery_drain',
    anchor: 'The laptop battery drains very quickly.',
    positive: 'My notebook loses charge fast.',
    negative: 'This pasta sauce tastes sweet and spicy.',
  }),
  Object.freeze({
    id: 'order_tracking',
    anchor: 'I need to track where my order is.',
    positive: 'How can I check my package delivery status?',
    negative: 'The violin concerto was composed in the 1800s.',
  }),
  Object.freeze({
    id: 'account_lockout',
    anchor: 'My account is locked after too many login attempts.',
    positive: 'I cannot sign in because the system temporarily blocked my account.',
    negative: 'Bake the cake at 350 degrees for thirty minutes.',
  }),
  Object.freeze({
    id: 'invoice_request',
    anchor: 'Please send me the invoice for last month.',
    positive: 'Can you provide the billing statement for the previous month?',
    negative: 'The hiking trail follows the river for five miles.',
  }),
  Object.freeze({
    id: 'slow_internet',
    anchor: 'The internet speed is much slower tonight.',
    positive: 'My connection is unusually slow this evening.',
    negative: 'The novel explores themes of memory and loss.',
  }),
]);

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

function resolveEmbeddingSemanticFixtures(runtimeConfig, options = null) {
  const overrides = isPlainObject(options?.embeddingSemantic)
    ? options.embeddingSemantic
    : null;
  const runtimeOverrides = runtimeConfig?.shared?.benchmark?.run?.embeddingSemantic;
  const source = overrides ?? (isPlainObject(runtimeOverrides) ? runtimeOverrides : null);

  const retrievalCases = normalizeRetrievalFixtures(source?.retrievalCases)
    ?? EMBEDDING_SEMANTIC_RETRIEVAL_CASES;
  const pairCases = normalizePairFixtures(source?.pairCases)
    ?? EMBEDDING_SEMANTIC_PAIR_CASES;
  const minRetrievalTop1Acc = Number.isFinite(source?.minRetrievalTop1Acc)
    ? Math.max(0, Math.min(1, Number(source.minRetrievalTop1Acc)))
    : EMBEDDING_SEMANTIC_MIN_RETRIEVAL_TOP1;
  const minPairAcc = Number.isFinite(source?.minPairAcc)
    ? Math.max(0, Math.min(1, Number(source.minPairAcc)))
    : EMBEDDING_SEMANTIC_MIN_PAIR_ACC;
  const pairMargin = Number.isFinite(source?.pairMargin)
    ? Number(source.pairMargin)
    : EMBEDDING_SEMANTIC_PAIR_MARGIN;

  return {
    retrievalCases,
    pairCases,
    minRetrievalTop1Acc,
    minPairAcc,
    pairMargin,
  };
}

function resolveEmbeddingSemanticStyle(pipeline) {
  const manifest = pipeline?.manifest ?? null;
  const style = selectRuleValue('inference', 'config', 'embeddingSemanticStyle', {
    modelId: String(manifest?.modelId ?? '').toLowerCase(),
    presetId: String(manifest?.inference?.presetId ?? '').toLowerCase(),
    manifestModelType: String(
      manifest?.config?.model_type
      ?? manifest?.config?.text_config?.model_type
      ?? ''
    ).toLowerCase(),
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
  return text;
}

export function resolvePrompt(runtimeConfig) {
  const runtimePrompt = runtimeConfig?.inference?.prompt;
  if (typeof runtimePrompt === 'string' && runtimePrompt.trim()) {
    return runtimePrompt.trim();
  }
  return DEFAULT_HARNESS_PROMPT;
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

function buildDefaultGenerationPrompt(templateType) {
  if (templateType === 'qwen') {
    return clonePromptInput(DEFAULT_QWEN_PROMPT);
  }
  if (templateType === 'translategemma') {
    return clonePromptInput(DEFAULT_TRANSLATEGEMMA_PROMPT);
  }
  return DEFAULT_HARNESS_PROMPT;
}

function shouldPreferModelDefaultPrompt(runtimePrompt, templateType) {
  if (templateType !== 'translategemma' && templateType !== 'qwen') {
    return false;
  }
  if (typeof runtimePrompt !== 'string') {
    return false;
  }
  return runtimePrompt.trim() === DEFAULT_RUNTIME_PLACEHOLDER_PROMPT;
}

function assertPromptContract(runtimePrompt, templateType, source = 'runtime.inference.prompt') {
  if (templateType !== 'translategemma') {
    return;
  }
  if (runtimePrompt === undefined || runtimePrompt === null) {
    return;
  }
  if (typeof runtimePrompt === 'string') {
    const trimmed = runtimePrompt.trim();
    if (!trimmed || trimmed === DEFAULT_RUNTIME_PLACEHOLDER_PROMPT) {
      return;
    }
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
    return promptInput.trim() || DEFAULT_HARNESS_PROMPT;
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

function resolveGenerationPromptInput(runtimeConfig, runOverrides = null, source = null) {
  const templateType = resolvePromptTemplateType(source);
  const overridePrompt = runOverrides?.prompt;
  assertPromptContract(overridePrompt, templateType, 'runOverrides.prompt');
  if (typeof overridePrompt === 'string' && overridePrompt.trim()) {
    return overridePrompt.trim();
  }
  if (isStructuredPromptInput(overridePrompt)) {
    return clonePromptInput(overridePrompt);
  }

  const runtimePrompt = runtimeConfig?.inference?.prompt;
  assertPromptContract(runtimePrompt, templateType, 'runtimeConfig.inference.prompt');
  if (shouldPreferModelDefaultPrompt(runtimePrompt, templateType)) {
    return buildDefaultGenerationPrompt(templateType);
  }
  if (typeof runtimePrompt === 'string' && runtimePrompt.trim()) {
    return runtimePrompt.trim();
  }
  if (isStructuredPromptInput(runtimePrompt)) {
    return clonePromptInput(runtimePrompt);
  }

  return buildDefaultGenerationPrompt(templateType);
}

function resolveMaxTokens(runtimeConfig) {
  const runtimeMax = runtimeConfig?.inference?.batching?.maxTokens;
  if (Number.isFinite(runtimeMax)) {
    return Math.max(1, Math.floor(runtimeMax));
  }
  return DEFAULT_HARNESS_MAX_TOKENS;
}

export function resolveBenchmarkRunSettings(runtimeConfig, source = null) {
  const benchConfig = runtimeConfig?.shared?.benchmark?.run || {};
  const runtimeSampling = isPlainObject(runtimeConfig?.inference?.sampling)
    ? runtimeConfig.inference.sampling
    : {};
  const benchSampling = isPlainObject(benchConfig?.sampling)
    ? benchConfig.sampling
    : {};
  const promptInput = typeof benchConfig.customPrompt === 'string' && benchConfig.customPrompt.trim()
    ? benchConfig.customPrompt.trim()
    : resolveGenerationPromptInput(runtimeConfig, null, source);
  const maxTokens = Number.isFinite(benchConfig.maxNewTokens)
    ? Math.max(1, Math.floor(benchConfig.maxNewTokens))
    : resolveMaxTokens(runtimeConfig);

  return {
    warmupRuns: Math.max(0, Math.floor(benchConfig.warmupRuns ?? 0)),
    timedRuns: Math.max(1, Math.floor(benchConfig.timedRuns ?? 1)),
    prompt: promptInput,
    promptLabel: describePromptInput(promptInput),
    maxTokens,
    sampling: {
      ...runtimeSampling,
      ...benchSampling,
    },
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
    const queryEmbedding = await embedStandaloneText(
      pipeline,
      formatEmbeddingSemanticText(testCase.query, 'query', semanticStyle)
    );
    const docEmbeddings = [];
    for (const doc of testCase.docs) {
      docEmbeddings.push(await embedStandaloneText(
        pipeline,
        formatEmbeddingSemanticText(doc, 'document', semanticStyle)
      ));
    }
    const sims = docEmbeddings.map((docEmbedding) => cosineSimilarity(queryEmbedding, docEmbedding));
    const topDoc = top1Index(sims);
    const passed = topDoc === testCase.expectedDoc;
    if (passed) retrievalPassed++;
    retrieval.push({
      id: testCase.id,
      passed,
      expectedDoc: testCase.expectedDoc,
      topDoc,
      sims: sims.map((v) => (Number.isFinite(v) ? Number(v.toFixed(6)) : null)),
    });
  }

  const pairs = [];
  let pairPassed = 0;
  for (const testCase of config.pairCases) {
    const anchor = await embedStandaloneText(
      pipeline,
      formatEmbeddingSemanticText(testCase.anchor, 'query', semanticStyle)
    );
    const positive = await embedStandaloneText(
      pipeline,
      formatEmbeddingSemanticText(testCase.positive, 'query', semanticStyle)
    );
    const negative = await embedStandaloneText(
      pipeline,
      formatEmbeddingSemanticText(testCase.negative, 'query', semanticStyle)
    );
    const simPos = cosineSimilarity(anchor, positive);
    const simNeg = cosineSimilarity(anchor, negative);
    const margin = simPos - simNeg;
    const passed = Number.isFinite(margin) && margin > config.pairMargin;
    if (passed) pairPassed++;
    pairs.push({
      id: testCase.id,
      passed,
      simPos: Number.isFinite(simPos) ? Number(simPos.toFixed(6)) : null,
      simNeg: Number.isFinite(simNeg) ? Number(simNeg.toFixed(6)) : null,
      margin: Number.isFinite(margin) ? Number(margin.toFixed(6)) : null,
    });
  }

  const retrievalTop1Acc = retrieval.length > 0 ? retrievalPassed / retrieval.length : 0;
  const pairAcc = pairs.length > 0 ? pairPassed / pairs.length : 0;
  const passed = retrievalTop1Acc >= config.minRetrievalTop1Acc
    && pairAcc >= config.minPairAcc;
  const failedCaseIds = [
    ...retrieval.filter((item) => !item.passed).map((item) => `retrieval:${item.id}`),
    ...pairs.filter((item) => !item.passed).map((item) => `pair:${item.id}`),
  ];

  return {
    passed,
    style: semanticStyle,
    retrievalTop1Acc,
    pairAcc,
    retrievalPassed,
    retrievalTotal: retrieval.length,
    pairPassed,
    pairTotal: pairs.length,
    minRetrievalTop1Acc: Number(config.minRetrievalTop1Acc.toFixed(4)),
    minPairAcc: Number(config.minPairAcc.toFixed(4)),
    pairMarginThreshold: Number(config.pairMargin.toFixed(4)),
    failedCaseIds,
    retrieval,
    pairs,
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
  const promptInput = resolveGenerationPromptInput(runtimeConfig, runOverrides, pipeline);
  const promptLabel = describePromptInput(promptInput);
  const useChatTemplate = runOverrides?.useChatTemplate
    ?? runtimeConfig?.inference?.chatTemplate?.enabled
    ?? (isStructuredPromptInput(promptInput) ? true : undefined);
  const maxTokens = Number.isFinite(runOverrides?.maxTokens)
    ? Math.max(1, Math.floor(runOverrides.maxTokens))
    : resolveMaxTokens(runtimeConfig);
  const sampling = isPlainObject(runOverrides?.sampling)
    ? runOverrides.sampling
    : (runtimeConfig.inference?.sampling || {});
  const debugProbes = runtimeConfig.shared?.debug?.probes || [];
  const profile = runtimeConfig.shared?.debug?.profiler?.enabled === true;
  const disableCommandBatching = Array.isArray(debugProbes) && debugProbes.length > 0;
  const start = performance.now();

  for await (const tokenText of pipeline.generate(promptInput, {
    maxTokens,
    temperature: sampling.temperature,
    topP: sampling.topP,
    topK: sampling.topK,
    repetitionPenalty: sampling.repetitionPenalty,
    greedyThreshold: sampling.greedyThreshold,
    useChatTemplate,
    profile,
    disableCommandBatching,
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
  const stats = typeof pipeline?.getStats === 'function'
    ? (pipeline.getStats() || {})
    : {};
  const prefillMs = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : 0;
  const ttftMs = Number.isFinite(stats.ttftMs) ? stats.ttftMs : prefillMs;
  const decodeMs = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : 0;
  const prefillTokens = Number.isFinite(stats.prefillTokens) ? stats.prefillTokens : 0;
  const decodeTokens = Number.isFinite(stats.decodeTokens)
    ? stats.decodeTokens
    : Math.max(0, tokens.length - 1);
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
  if (Number.isFinite(stats.decodeSubmitWaitMs)) gpu.decodeSubmitWaitMs = stats.decodeSubmitWaitMs;
  if (Number.isFinite(stats.decodeReadbackWaitMs)) gpu.decodeReadbackWaitMs = stats.decodeReadbackWaitMs;
  const gpuPhase = Object.keys(gpu).length > 0 ? gpu : null;
  const decodeProfileSteps = Array.isArray(stats.decodeProfileSteps)
    ? stats.decodeProfileSteps
    : null;

  return {
    prompt: promptLabel,
    promptInput,
    maxTokens,
    tokens,
    tokenIds,
    tokenDiagnostics: summarizeGenerationTokens(tokenRecords),
    output: tokens.join(''),
    durationMs,
    tokensPerSec,
    phase: {
      totalMs: Number.isFinite(stats.totalTimeMs) ? stats.totalTimeMs : durationMs,
      ttftMs,
      prefillMs,
      decodeMs,
      prefillTokens,
      decodeTokens,
      prefillTokensPerSec,
      prefillTokensPerSecTtft,
      decodeTokensPerSec,
      gpu: gpuPhase,
      decodeProfileSteps,
    },
  };
}

export async function runEmbedding(pipeline, runtimeConfig, runOverrides = null) {
  const prompt = typeof runOverrides?.prompt === 'string' && runOverrides.prompt.trim()
    ? runOverrides.prompt.trim()
    : resolvePrompt(runtimeConfig);
  const start = performance.now();
  const result = await pipeline.embed(prompt);
  const durationMs = Math.max(1, performance.now() - start);
  const tokenCount = Number.isFinite(result?.tokens?.length) ? result.tokens.length : 0;
  const stats = summarizeEmbeddingValues(result?.embedding);
  return {
    prompt,
    tokenCount,
    durationMs,
    ...stats,
  };
}
