

import { initializeInference, parseRuntimeOverridesFromURL } from './test-harness.js';
import { saveReport } from '../storage/reports.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import { initDevice, getKernelCapabilities, getDevice } from '../gpu/device.js';
import { createPipeline } from './pipelines/text.js';
import { parseModelConfigFromManifest } from './pipelines/text/config.js';
import { resolveKernelPathState, activateKernelPathState } from './pipelines/text/model-load.js';
import { openModelStore, loadManifestFromStore } from '../storage/shard-manager.js';
import { parseManifest } from '../storage/rdrr-format.js';
import { computeSampleStats } from '../debug/stats.js';
import {
  setActiveKernelPath,
  getActiveKernelPath,
  getActiveKernelPathSource,
  getActiveKernelPathPolicy,
} from '../config/kernel-path-loader.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { mergeRuntimeValues } from '../config/runtime-merge.js';
import { isPlainObject } from '../utils/plain-object.js';

function parseReportTimestamp(rawTimestamp, label = 'timestamp') {
  if (rawTimestamp == null) {
    return null;
  }

  if (rawTimestamp instanceof Date) {
    const timestamp = rawTimestamp.getTime();
    if (!Number.isFinite(timestamp)) {
      throw new Error(`Invalid ${label}: not a valid Date.`);
    }
    return rawTimestamp.toISOString();
  }

  if (typeof rawTimestamp === 'number') {
    if (!Number.isFinite(rawTimestamp)) {
      throw new Error(`Invalid ${label}: must be a finite epoch timestamp.`);
    }
    return new Date(rawTimestamp).toISOString();
  }

  if (typeof rawTimestamp === 'string') {
    const trimmed = rawTimestamp.trim();
    if (trimmed.length === 0) {
      return null;
    }
    const numericCandidate = Number(trimmed);
    if (Number.isFinite(numericCandidate)) {
      return new Date(numericCandidate).toISOString();
    }
    const parsed = new Date(trimmed);
    if (Number.isNaN(parsed.getTime())) {
      throw new Error(`Invalid ${label}: expected ISO-8601 string or epoch milliseconds.`);
    }
    return parsed.toISOString();
  }

  throw new Error(`Invalid ${label}: expected Date, ISO-8601 string, epoch milliseconds, or nullish.`);
}

function resolveReportTimestamp(rawTimestamp, label, fallbackTimestamp = null) {
  const parsed = parseReportTimestamp(rawTimestamp, label);
  return parsed ?? (fallbackTimestamp == null ? new Date().toISOString() : String(fallbackTimestamp));
}

function resolveRuntime(options) {
  if (options.runtime) return options.runtime;
  if (options.searchParams) return parseRuntimeOverridesFromURL(options.searchParams);
  if (typeof globalThis.location === 'undefined') return parseRuntimeOverridesFromURL(new URLSearchParams());
  return parseRuntimeOverridesFromURL();
}

function normalizePresetPath(value) {
  const trimmed = String(value || '').replace(/^[./]+/, '');
  if (!trimmed) return null;
  return trimmed.endsWith('.json') ? trimmed : `${trimmed}.json`;
}

function resolvePresetBaseUrl() {
  try {
    return new URL('../config/presets/runtime/', import.meta.url).toString().replace(/\/$/, '');
  } catch {
    if (typeof globalThis.location !== 'undefined' && globalThis.location?.href) {
      return new URL('/src/config/presets/runtime/', globalThis.location.href).toString().replace(/\/$/, '');
    }
    return '/src/config/presets/runtime';
  }
}

function cloneRuntimeConfig(runtimeConfig) {
  if (!runtimeConfig) return null;
  if (typeof structuredClone === 'function') {
    return structuredClone(runtimeConfig);
  }
  return JSON.parse(JSON.stringify(runtimeConfig));
}

function snapshotRuntimeState() {
  return {
    runtimeConfig: cloneRuntimeConfig(getRuntimeConfig()),
    activeKernelPath: getActiveKernelPath(),
    activeKernelPathSource: getActiveKernelPathSource(),
    activeKernelPathPolicy: getActiveKernelPathPolicy(),
  };
}

function restoreRuntimeState(snapshot) {
  if (!snapshot) {
    return;
  }
  setRuntimeConfig(snapshot.runtimeConfig);
  setActiveKernelPath(
    snapshot.activeKernelPath,
    snapshot.activeKernelPathSource || 'none',
    snapshot.activeKernelPathPolicy ?? null
  );
}

async function runWithRuntimeIsolationForSuite(run) {
  const snapshot = snapshotRuntimeState();
  try {
    return await run();
  } finally {
    restoreRuntimeState(snapshot);
  }
}

function resolveRuntimeFromConfig(config) {
  if (!config || typeof config !== 'object') return null;
  if (config.runtime && typeof config.runtime === 'object') return config.runtime;
  if (config.shared || config.loading || config.inference || config.emulation) return config;
  return null;
}

function sanitizeReportOutput(output) {
  if (output == null) return null;
  if (typeof output !== 'object') return output;
  if (ArrayBuffer.isView(output)) {
    return {
      type: output.constructor?.name || 'TypedArray',
      length: Number.isFinite(output.length) ? output.length : null,
    };
  }
  if (
    Number.isFinite(output?.width)
    && Number.isFinite(output?.height)
    && ArrayBuffer.isView(output?.pixels)
  ) {
    const { pixels, ...rest } = output;
    return {
      ...rest,
      width: output.width,
      height: output.height,
      pixels: {
        type: pixels.constructor?.name || 'TypedArray',
        length: Number.isFinite(pixels.length) ? pixels.length : null,
      },
    };
  }
  return output;
}

function normalizeExtends(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => String(entry || '').trim()).filter(Boolean);
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed ? [trimmed] : [];
  }
  return [];
}

function normalizeExtendsPath(value) {
  const trimmed = String(value || '').trim();
  if (!trimmed) return null;
  return trimmed.endsWith('.json') ? trimmed : `${trimmed}.json`;
}

function resolveAbsoluteUrl(target, base) {
  try {
    if (base) {
      return new URL(target, base).toString();
    }
    if (typeof globalThis.location !== 'undefined' && globalThis.location?.href) {
      return new URL(target, globalThis.location.href).toString();
    }
    return new URL(target, import.meta.url).toString();
  } catch {
    return target;
  }
}

function isAbsoluteUrl(value) {
  return /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(value);
}

function joinUrl(base, path) {
  if (!base) return path;
  if (isAbsoluteUrl(base)) {
    return new URL(path, base.endsWith('/') ? base : `${base}/`).toString();
  }
  const normalizedBase = base.replace(/\/$/, '');
  const normalizedPath = path.replace(/^\//, '');
  return `${normalizedBase}/${normalizedPath}`;
}

function resolveExtendCandidates(ref, context) {
  const normalized = normalizeExtendsPath(ref);
  if (!normalized) return [];
  if (isAbsoluteUrl(normalized) || normalized.startsWith('/')) {
    return [normalized];
  }
  if (normalized.startsWith('./') || normalized.startsWith('../')) {
    return [resolveAbsoluteUrl(normalized, context.sourceUrl)];
  }
  if (normalized.includes('/')) {
    return [joinUrl(context.presetBaseUrl, normalized)];
  }
  const candidates = [];
  if (context.presetBaseUrl) {
    candidates.push(joinUrl(context.presetBaseUrl, normalized));
    candidates.push(joinUrl(context.presetBaseUrl, `modes/${normalized}`));
  }
  if (context.sourceUrl) {
    const sourceDir = resolveAbsoluteUrl('./', context.sourceUrl);
    candidates.push(resolveAbsoluteUrl(normalized, sourceDir));
  }
  return [...new Set(candidates)];
}

async function fetchRuntimeConfig(url, options = {}) {
  const response = await fetch(url, { signal: options.signal });
  if (!response.ok) {
    const error = new Error(`Failed to load runtime config: ${response.status}`);
    error.code = response.status === 404 ? 'runtime_config_not_found' : 'runtime_config_fetch_failed';
    throw error;
  }
  return response.json();
}

async function resolveRuntimeConfigExtends(config, context) {
  const runtime = resolveRuntimeFromConfig(config);
  if (!runtime) {
    throw new Error('Runtime config is missing runtime fields');
  }

  const extendsRefs = normalizeExtends(config.extends);
  let mergedRuntime = null;
  let mergedConfig = null;

  for (const ref of extendsRefs) {
    const base = await loadRuntimeConfigFromRef(ref, context);
    mergedRuntime = mergedRuntime ? mergeRuntimeValues(mergedRuntime, base.runtime) : base.runtime;
    mergedConfig = mergedConfig ? mergeRuntimeValues(mergedConfig, base.config) : base.config;
  }

  const combinedRuntime = mergedRuntime ? mergeRuntimeValues(mergedRuntime, runtime) : runtime;
  const combinedConfig = mergedConfig ? mergeRuntimeValues(mergedConfig, config) : { ...config };
  const resolved = { ...combinedConfig, runtime: combinedRuntime };
  if (resolved.extends !== undefined) {
    delete resolved.extends;
  }
  return { config: resolved, runtime: combinedRuntime };
}

async function loadRuntimeConfigChain(url, options = {}, stack = []) {
  const presetBaseUrl = options.presetBaseUrl || options.baseUrl || resolvePresetBaseUrl();
  const resolvedUrl = resolveAbsoluteUrl(url);
  if (stack.includes(resolvedUrl)) {
    throw new Error(`Runtime config extends cycle: ${[...stack, resolvedUrl].join(' -> ')}`);
  }
  const config = await fetchRuntimeConfig(resolvedUrl, options);
  return resolveRuntimeConfigExtends(config, {
    ...options,
    sourceUrl: resolvedUrl,
    presetBaseUrl,
    stack: [...stack, resolvedUrl],
  });
}

async function loadRuntimeConfigFromRef(ref, context) {
  const candidates = resolveExtendCandidates(ref, context);
  if (!candidates.length) {
    throw new Error(`Runtime config extends is invalid: ${ref}`);
  }
  let lastError = null;
  for (const candidate of candidates) {
    try {
      return await loadRuntimeConfigChain(candidate, context, context.stack ?? []);
    } catch (error) {
      if (error?.code === 'runtime_config_not_found') {
        lastError = error;
        continue;
      }
      throw error;
    }
  }
  if (lastError) {
    throw lastError;
  }
  throw new Error(`Runtime config extends not found: ${ref}`);
}

export async function loadRuntimeConfigFromUrl(url, options = {}) {
  if (!url) {
    throw new Error('runtime config url is required');
  }
  return loadRuntimeConfigChain(url, options);
}

export async function applyRuntimeConfigFromUrl(url, options = {}) {
  const { runtime } = await loadRuntimeConfigFromUrl(url, options);
  setRuntimeConfig(runtime);
  return runtime;
}

export async function loadRuntimePreset(presetId, options = {}) {
  const baseUrl = options.baseUrl || resolvePresetBaseUrl();
  const normalized = normalizePresetPath(presetId);
  if (!normalized) {
    throw new Error('runtime preset id is required');
  }
  const url = `${baseUrl.replace(/\/$/, '')}/${normalized}`;
  return loadRuntimeConfigFromUrl(url, { ...options, presetBaseUrl: baseUrl });
}

export async function applyRuntimePreset(presetId, options = {}) {
  const { runtime } = await loadRuntimePreset(presetId, options);
  setRuntimeConfig(runtime);
  return runtime;
}

export async function initializeBrowserHarness(options = {}) {
  const { modelUrl, onProgress, log } = options;
  if (!modelUrl) {
    throw new Error('modelUrl is required');
  }

  const runtime = resolveRuntime(options);
  const result = await initializeInference(modelUrl, {
    runtime,
    onProgress,
    log,
  });

  return { ...result, runtime };
}

export async function saveBrowserReport(modelId, report, options = {}) {
  return saveReport(modelId, report, options);
}

export async function runBrowserHarness(options = {}) {
  const harness = await initializeBrowserHarness(options);
  const reportTimestamp = resolveReportTimestamp(options.timestamp, 'runBrowserHarness timestamp');
  const modelId = options.modelId || harness.manifest?.modelId || 'unknown';

  let report = options.report || null;
  if (!report && typeof options.buildReport === 'function') {
    report = await options.buildReport(harness);
  }
  if (!report) {
    report = {
      modelId,
      timestamp: reportTimestamp,
    };
  } else if (!report.timestamp) {
    report = { ...report, timestamp: reportTimestamp };
  }

  const reportInfo = await saveReport(modelId, report, { timestamp: report.timestamp || reportTimestamp });
  return { ...harness, report, reportInfo };
}

function normalizeSuite(value) {
  const suite = String(value || '').trim().toLowerCase();
  if (!suite) return 'inference';
  if (suite === 'benchmark') return 'bench';
  return suite;
}

function buildSuiteSummary(suite, results, startTime) {
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  for (const result of results) {
    if (result.skipped) {
      skipped++;
    } else if (result.passed) {
      passed++;
    } else {
      failed++;
    }
  }
  const duration = Math.max(0, performance.now() - startTime);
  return { suite, passed, failed, skipped, duration, results };
}

function normalizeCacheMode(value) {
  return value === 'cold' || value === 'warm' ? value : 'warm';
}

function normalizeLoadMode(value, hasModelUrl) {
  if (value === 'opfs' || value === 'http' || value === 'memory') {
    return value;
  }
  return hasModelUrl ? 'http' : 'opfs';
}

function toTimingNumber(value, fallback = 0) {
  return formatMetricNumber(value, fallback, 2);
}

function safeToFixed(value, fallback = 0, digits = 2) {
  return formatMetricNumber(value, fallback, digits);
}

function sampleTimingNumber(stats, key, fallback = 0) {
  return formatMetricNumber(stats?.[key], fallback, 2);
}

function formatMetricNumber(value, fallback = 0, digits = 2) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return fallback;
  return Number(numericValue.toFixed(digits));
}

function buildCanonicalTiming(overrides = {}) {
  const cacheMode = normalizeCacheMode(overrides.cacheMode);
  const modelLoadMs = toTimingNumber(overrides.modelLoadMs, 0);
  const prefillMs = toTimingNumber(overrides.prefillMs, 0);
  const decodeMs = toTimingNumber(overrides.decodeMs, 0);
  const decodeMsPerTokenP50 = Number.isFinite(overrides.decodeMsPerTokenP50)
    ? toTimingNumber(overrides.decodeMsPerTokenP50)
    : null;
  const decodeMsPerTokenP95 = Number.isFinite(overrides.decodeMsPerTokenP95)
    ? toTimingNumber(overrides.decodeMsPerTokenP95)
    : null;
  const decodeMsPerTokenP99 = Number.isFinite(overrides.decodeMsPerTokenP99)
    ? toTimingNumber(overrides.decodeMsPerTokenP99)
    : null;
  const decodeTokensPerSec = Number.isFinite(overrides.decodeTokensPerSec)
    ? toTimingNumber(overrides.decodeTokensPerSec)
    : null;
  const prefillTokensPerSec = Number.isFinite(overrides.prefillTokensPerSec)
    ? toTimingNumber(overrides.prefillTokensPerSec)
    : null;
  const totalRunMs = toTimingNumber(
    overrides.totalRunMs,
    toTimingNumber(prefillMs + decodeMs)
  );
  const firstTokenMs = Number.isFinite(overrides.firstTokenMs)
    ? toTimingNumber(overrides.firstTokenMs)
    : null;
  const firstResponseMs = Number.isFinite(overrides.firstResponseMs)
    ? toTimingNumber(overrides.firstResponseMs)
    : toTimingNumber(modelLoadMs + totalRunMs);

  return {
    modelLoadMs,
    firstTokenMs,
    firstResponseMs,
    prefillMs,
    decodeMs,
    decodeMsPerTokenP50,
    decodeMsPerTokenP95,
    decodeMsPerTokenP99,
    decodeTokensPerSec,
    prefillTokensPerSec,
    totalRunMs,
    cacheMode,
    loadMode: overrides.loadMode,
  };
}

async function resolveKernelPathForModel(options = {}) {
  const runtimeConfig = options.runtime?.runtimeConfig ?? getRuntimeConfig();
  let manifest = null;
  let manifestModelId = options.modelId || null;

  if (options.modelId) {
    await openModelStore(options.modelId);
    const manifestText = await loadManifestFromStore();
    if (manifestText) {
      manifest = parseManifest(manifestText);
      manifestModelId = manifest.modelId ?? options.modelId;
    }
  }

  if (!manifest) return null;

  const modelConfig = parseModelConfigFromManifest(manifest, runtimeConfig);
  const kernelPathState = resolveKernelPathState({
    manifest,
    runtimeConfig,
    modelConfig,
  });
  activateKernelPathState(kernelPathState);
  return {
    modelId: manifestModelId,
    kernelPath: kernelPathState.resolvedKernelPath,
    source: kernelPathState.kernelPathSource,
  };
}

async function initializeInferenceFromStorage(modelId, options = {}) {
  const { onProgress } = options;
  if (!modelId) {
    throw new Error('modelId is required');
  }

  if (options.runtime?.runtimeConfig) {
    setRuntimeConfig(options.runtime.runtimeConfig);
  }

  onProgress?.('storage', 0.05, 'Opening model store...');
  await openModelStore(modelId);

  onProgress?.('manifest', 0.1, 'Loading manifest...');
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error('Manifest not found in storage');
  }
  const manifest = parseManifest(manifestText);

  onProgress?.('gpu', 0.2, 'Initializing WebGPU...');
  await initDevice();
  const device = getDevice();
  const capabilities = getKernelCapabilities();

  onProgress?.('pipeline', 0.3, 'Creating pipeline...');
  const pipeline = await createPipeline(manifest, {
    gpu: { device },
    runtime: options.runtime,
    onProgress,
  });

  return { pipeline, manifest, capabilities };
}

async function initializeSuiteModel(options = {}) {
  const loadStart = performance.now();
  const runtime = resolveRuntime(options);
  let harness;
  if (options.modelId && !options.modelUrl) {
    harness = await initializeInferenceFromStorage(options.modelId, { ...options, runtime });
  } else {
    if (!options.modelUrl) {
      throw new Error('modelUrl is required for this suite');
    }
    harness = await initializeInference(options.modelUrl, {
      runtime,
      onProgress: options.onProgress,
      log: options.log,
    });
  }
  const modelLoadMs = Math.max(0, performance.now() - loadStart);
  return { ...harness, modelLoadMs };
}

async function runKernelSuite(options = {}) {
  const startTime = performance.now();
  const { testHarness, initGPU } = await import('../../tests/kernels/browser/test-page.js');
  const { runKernelSuite: runAllKernelTests } = await import('../../tests/kernels/browser/kernel-suite.js');
  await initGPU();

  const previousKernelPath = getActiveKernelPath();
  const previousKernelSource = getActiveKernelPathSource();
  const previousKernelPathPolicy = getActiveKernelPathPolicy();
  if (options.modelId) {
    await resolveKernelPathForModel(options);
  }
  let results = [];
  try {
    results = await runAllKernelTests(testHarness);
  } finally {
    setActiveKernelPath(previousKernelPath, previousKernelSource, previousKernelPathPolicy);
  }

  const summary = buildSuiteSummary('kernels', results, startTime);
  return {
    ...summary,
    deviceInfo: getKernelCapabilities(),
  };
}

const DEFAULT_HARNESS_PROMPT = 'Summarize this input in one sentence.';
const DEFAULT_HARNESS_MAX_TOKENS = 32;
const EMBEDDING_PREVIEW_LENGTH = 16;
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

function resolvePrompt(runtimeConfig) {
  const runtimePrompt = runtimeConfig?.inference?.prompt;
  if (typeof runtimePrompt === 'string' && runtimePrompt.trim()) {
    return runtimePrompt.trim();
  }
  return DEFAULT_HARNESS_PROMPT;
}

function isStructuredPromptInput(value) {
  return Array.isArray(value) || (value != null && typeof value === 'object');
}

function resolveGenerationPromptInput(runtimeConfig, runOverrides = null) {
  const overridePrompt = runOverrides?.prompt;
  if (typeof overridePrompt === 'string' && overridePrompt.trim()) {
    return overridePrompt.trim();
  }
  if (isStructuredPromptInput(overridePrompt)) {
    return overridePrompt;
  }

  const runtimePrompt = runtimeConfig?.inference?.prompt;
  if (typeof runtimePrompt === 'string' && runtimePrompt.trim()) {
    return runtimePrompt.trim();
  }
  if (isStructuredPromptInput(runtimePrompt)) {
    return runtimePrompt;
  }

  return DEFAULT_HARNESS_PROMPT;
}

function resolveMaxTokens(runtimeConfig) {
  const runtimeMax = runtimeConfig?.inference?.batching?.maxTokens;
  if (Number.isFinite(runtimeMax)) {
    return Math.max(1, Math.floor(runtimeMax));
  }
  return DEFAULT_HARNESS_MAX_TOKENS;
}

function resolveBenchmarkRunSettings(runtimeConfig) {
  const benchConfig = runtimeConfig?.shared?.benchmark?.run || {};
  const runtimeSampling = isPlainObject(runtimeConfig?.inference?.sampling)
    ? runtimeConfig.inference.sampling
    : {};
  const benchSampling = isPlainObject(benchConfig?.sampling)
    ? benchConfig.sampling
    : {};
  const prompt = typeof benchConfig.customPrompt === 'string' && benchConfig.customPrompt.trim()
    ? benchConfig.customPrompt.trim()
    : resolvePrompt(runtimeConfig);
  const maxTokens = Number.isFinite(benchConfig.maxNewTokens)
    ? Math.max(1, Math.floor(benchConfig.maxNewTokens))
    : resolveMaxTokens(runtimeConfig);

  return {
    warmupRuns: Math.max(0, Math.floor(benchConfig.warmupRuns ?? 0)),
    timedRuns: Math.max(1, Math.floor(benchConfig.timedRuns ?? 1)),
    prompt,
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

async function runEmbeddingSemanticChecks(pipeline, options = null) {
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

// Matches pad/special tokens that indicate degenerate output: <pad>, <unused123>, <eos>,
// <bos>, <s>, </s>, [PAD], [UNK], [SEP], [CLS], and bare angle-bracket tokens.
const SPECIAL_TOKEN_RE = /^(<pad>|<unused\d*>|<eos>|<bos>|<s>|<\/s>|\[PAD\]|\[UNK\]|\[SEP\]|\[CLS\]|<[^>]{1,32}>)$/i;
const PAD_DOMINANCE_THRESHOLD = 0.5;

function isCoherentOutput(tokens, output) {
  if (tokens.length === 0) return false;
  const specialTokenCount = tokens.filter((t) => SPECIAL_TOKEN_RE.test(String(t).trim())).length;
  if (specialTokenCount / tokens.length >= PAD_DOMINANCE_THRESHOLD) return false;
  const cleanedOutput = String(output || '')
    .replace(/<[^>\n]{1,80}>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return cleanedOutput.length > 0;
}

async function runGeneration(pipeline, runtimeConfig, runOverrides = null) {
  const tokens = [];
  const tokenIds = [];
  const promptInput = resolveGenerationPromptInput(runtimeConfig, runOverrides);
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
    onToken: (tokenId) => {
      tokenIds.push(tokenId);
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
    prompt: promptInput,
    maxTokens,
    tokens,
    tokenIds,
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

async function runEmbedding(pipeline, runtimeConfig, runOverrides = null) {
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

async function runInferenceSuite(options = {}) {
  const startTime = performance.now();
  const harness = await initializeSuiteModel(options);
  const runtimeConfig = getRuntimeConfig();
  const modelType = harness.manifest?.modelType || 'transformer';
  const cacheMode = normalizeCacheMode(options.cacheMode);
  const loadMode = normalizeLoadMode(options.loadMode, !options.modelUrl);
  const safeModelLoadMs = toTimingNumber(harness.modelLoadMs, 0);

  let results;
  let output = null;
  let metrics;

  if (modelType === 'embedding') {
    const run = await runEmbedding(harness.pipeline, runtimeConfig);
    const semantic = await runEmbeddingSemanticChecks(harness.pipeline, options);
    const isValidEmbedding = run.embeddingDim > 0 && run.nonFiniteCount === 0;
    const isSemanticValid = semantic.passed;
    output = {
      mode: 'embedding',
      tokens: run.tokenCount,
      embeddingDim: run.embeddingDim,
      finiteValues: run.finiteCount,
      nonFiniteValues: run.nonFiniteCount,
      finiteRatio: Number((run.finiteRatio ?? 0).toFixed(6)),
      min: run.min == null ? null : Number(run.min.toFixed(6)),
      max: run.max == null ? null : Number(run.max.toFixed(6)),
      maxAbs: run.maxAbs == null ? null : Number(run.maxAbs.toFixed(6)),
      mean: run.mean == null ? null : Number(run.mean.toFixed(6)),
      stdDev: run.stdDev == null ? null : Number(run.stdDev.toFixed(6)),
      l2Norm: run.l2Norm == null ? null : Number(run.l2Norm.toFixed(6)),
      preview: run.preview,
      semantic: {
        passed: isSemanticValid,
        style: semantic.style,
        retrievalTop1Acc: Number(semantic.retrievalTop1Acc.toFixed(4)),
        pairAcc: Number(semantic.pairAcc.toFixed(4)),
        failedCaseIds: semantic.failedCaseIds,
      },
    };
    results = [
      {
        name: 'embedding',
        passed: isValidEmbedding,
        duration: run.durationMs,
        error: isValidEmbedding
          ? undefined
          : (
            run.embeddingDim <= 0
              ? 'No embedding returned'
              : `Embedding contains non-finite values (${run.nonFiniteCount}/${run.embeddingDim})`
          ),
      },
      {
        name: 'embedding-semantic',
        passed: isSemanticValid,
        duration: semantic.durationMs,
        error: isSemanticValid
          ? undefined
          : (
            `Semantic checks below threshold: retrieval=${(semantic.retrievalTop1Acc * 100).toFixed(1)}% `
            + `(min ${(semantic.minRetrievalTop1Acc * 100).toFixed(1)}%), `
            + `pairs=${(semantic.pairAcc * 100).toFixed(1)}% `
            + `(min ${(semantic.minPairAcc * 100).toFixed(1)}%). `
            + (semantic.failedCaseIds.length > 0 ? `Failed: ${semantic.failedCaseIds.join(', ')}` : '')
          ),
      },
    ];
    metrics = {
      prompt: run.prompt,
      embeddingTokens: run.tokenCount,
      embeddingDim: run.embeddingDim,
      finiteValues: run.finiteCount,
      finiteRatio: Number((run.finiteRatio ?? 0).toFixed(6)),
      nonFiniteValues: run.nonFiniteCount,
      embeddingMin: run.min == null ? null : Number(run.min.toFixed(6)),
      embeddingMax: run.max == null ? null : Number(run.max.toFixed(6)),
      embeddingMaxAbs: run.maxAbs == null ? null : Number(run.maxAbs.toFixed(6)),
      embeddingMean: run.mean == null ? null : Number(run.mean.toFixed(6)),
      embeddingStdDev: run.stdDev == null ? null : Number(run.stdDev.toFixed(6)),
      embeddingL2Norm: run.l2Norm == null ? null : Number(run.l2Norm.toFixed(6)),
      embeddingMs: Number(run.durationMs.toFixed(2)),
      semanticPassed: isSemanticValid,
      semanticDurationMs: Number(semantic.durationMs.toFixed(2)),
      semanticRetrievalTop1Acc: Number(semantic.retrievalTop1Acc.toFixed(4)),
      semanticPairAcc: Number(semantic.pairAcc.toFixed(4)),
      semanticRetrievalPassed: semantic.retrievalPassed,
      semanticRetrievalTotal: semantic.retrievalTotal,
      semanticPairPassed: semantic.pairPassed,
      semanticPairTotal: semantic.pairTotal,
      semanticMinRetrievalTop1Acc: Number(semantic.minRetrievalTop1Acc.toFixed(4)),
      semanticMinPairAcc: Number(semantic.minPairAcc.toFixed(4)),
      semanticPairMarginThreshold: Number(semantic.pairMarginThreshold.toFixed(4)),
      semanticStyle: semantic.style,
      semanticFailedCases: semantic.failedCaseIds,
      semanticDetails: {
        retrieval: semantic.retrieval,
        pairs: semantic.pairs,
      },
      modelLoadMs: safeModelLoadMs,
      endToEndMs: safeToFixed(safeModelLoadMs + run.durationMs),
      embeddingPreview: run.preview,
    };
  } else {
    const run = await runGeneration(harness.pipeline, runtimeConfig);
    const coherent = isCoherentOutput(run.tokens, run.output);
    results = [
      {
        name: 'generation',
        passed: run.tokens.length > 0 && coherent,
        duration: run.durationMs,
        error: run.tokens.length === 0
          ? 'No tokens generated'
          : (!coherent ? 'Output dominated by padding or special tokens' : undefined),
      },
    ];
    output = run.output;
    metrics = {
      prompt: run.prompt,
      maxTokens: run.maxTokens,
      tokensGenerated: run.tokens.length,
      tokensPerSec: safeToFixed(run.tokensPerSec),
      totalRunMs: safeToFixed(run.phase.totalMs),
      firstTokenMs: safeToFixed(run.phase.ttftMs),
      firstResponseMs: safeToFixed(safeModelLoadMs + run.phase.ttftMs),
      prefillMs: safeToFixed(run.phase.prefillMs),
      decodeMs: safeToFixed(run.phase.decodeMs),
      prefillTokens: Math.round(run.phase.prefillTokens),
      decodeTokens: Math.round(run.phase.decodeTokens),
      prefillTokensPerSec: safeToFixed(run.phase.prefillTokensPerSec),
      prefillTokensPerSecTtft: safeToFixed(run.phase.prefillTokensPerSecTtft),
      decodeTokensPerSec: safeToFixed(run.phase.decodeTokensPerSec),
      modelLoadMs: safeModelLoadMs,
      gpu: run.phase.gpu,
      decodeProfileSteps: run.phase.decodeProfileSteps,
    };
  }

  const memoryStats = typeof harness.pipeline?.getMemoryStats === 'function'
    ? harness.pipeline.getMemoryStats()
    : null;
  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const summary = buildSuiteSummary(options.suiteName || 'inference', results, startTime);
  const timing = buildCanonicalTiming({
    modelLoadMs: safeModelLoadMs,
    firstTokenMs: metrics.firstTokenMs ?? null,
    firstResponseMs: Number.isFinite(metrics.firstTokenMs)
      ? safeModelLoadMs + metrics.firstTokenMs
      : null,
    prefillMs: metrics.prefillMs ?? 0,
    decodeMs: metrics.decodeMs ?? 0,
    decodeMsPerTokenP50: metrics.decodeMsPerTokenP50 ?? null,
    decodeMsPerTokenP95: metrics.decodeMsPerTokenP95 ?? null,
    decodeMsPerTokenP99: metrics.decodeMsPerTokenP99 ?? null,
    totalRunMs: metrics.totalRunMs ?? metrics.decodeMs ?? 0,
    decodeTokensPerSec: metrics.decodeTokensPerSec,
    prefillTokensPerSec: metrics.prefillTokensPerSec,
    cacheMode,
    loadMode,
  });
  return {
    ...summary,
    modelId: options.modelId || harness.manifest?.modelId || 'unknown',
    cacheMode,
    loadMode,
    env: {
      library: 'doppler',
      runtime: 'browser',
      device: 'webgpu',
      browserUserAgent: typeof navigator !== 'undefined' ? (navigator.userAgent || null) : null,
      browserPlatform: typeof navigator !== 'undefined' ? (navigator.platform || null) : null,
      browserLanguage: typeof navigator !== 'undefined' ? (navigator.language || null) : null,
      browserVendor: typeof navigator !== 'undefined' ? (navigator.vendor || null) : null,
    },
    timing,
    output,
    metrics,
    memoryStats,
    deviceInfo: getKernelCapabilities(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

async function runBenchSuite(options = {}) {
  const startTime = performance.now();
  const runtimeConfig = getRuntimeConfig();
  const benchRun = resolveBenchmarkRunSettings(runtimeConfig);
  const warmupRuns = benchRun.warmupRuns;
  const timedRuns = benchRun.timedRuns;
  const cacheMode = normalizeCacheMode(options.cacheMode);
  const loadMode = normalizeLoadMode(options.loadMode, !options.modelUrl);

  const harness = await initializeSuiteModel(options);
  const modelType = harness.manifest?.modelType || 'transformer';
  const safeModelLoadMs = toTimingNumber(harness.modelLoadMs, 0);

  let results;
  let metrics;
  let output = null;
  let timing;

  if (modelType === 'embedding') {
    const durations = [];
    const timedDurations = [];
    const embeddingDims = [];
    const embeddingTokenCounts = [];
    const embeddingNorms = [];
    let firstTimedEmbeddingMs = null;
    let invalidRuns = 0;
    let totalNonFiniteValues = 0;
    for (let i = 0; i < warmupRuns + timedRuns; i++) {
      harness.pipeline.reset?.();
      const run = await runEmbedding(harness.pipeline, runtimeConfig, benchRun);
      if (i >= warmupRuns) {
        timedDurations.push(run.durationMs);
        if (firstTimedEmbeddingMs == null) {
          firstTimedEmbeddingMs = run.durationMs;
        }
        totalNonFiniteValues += run.nonFiniteCount;
        if (Number.isFinite(run.tokenCount)) {
          embeddingTokenCounts.push(run.tokenCount);
        }
        if (Number.isFinite(run.l2Norm)) {
          embeddingNorms.push(run.l2Norm);
        }
        if (run.embeddingDim > 0 && run.nonFiniteCount === 0) {
          durations.push(run.durationMs);
          embeddingDims.push(run.embeddingDim);
        } else {
          invalidRuns++;
        }
      }
    }

    const embeddingMsStats = computeSampleStats(durations);
    const timedEmbeddingMsStats = computeSampleStats(timedDurations);
    const embeddingDimStats = computeSampleStats(embeddingDims);
    const embeddingTokensStats = computeSampleStats(embeddingTokenCounts);
    const embeddingNormStats = computeSampleStats(embeddingNorms);
    const avgMs = embeddingMsStats.mean;

    results = [
      {
        name: 'benchmark-embedding',
        passed: durations.length > 0 && invalidRuns === 0,
        duration: durations.reduce((sum, value) => sum + value, 0),
        error: durations.length > 0
          ? (
            invalidRuns === 0
              ? undefined
              : `Invalid embedding runs: ${invalidRuns} (non-finite values observed)`
          )
          : 'No valid embedding benchmark runs completed',
      },
    ];

    metrics = {
      warmupRuns,
      timedRuns,
      validRuns: durations.length,
      invalidRuns,
      invalidRatePct: Number((timedRuns > 0 ? (invalidRuns / timedRuns) * 100 : 0).toFixed(2)),
      prompt: benchRun.prompt,
      embeddingDim: Math.round(embeddingDims.reduce((a, b) => a + b, 0) / (embeddingDims.length || 1)),
      nonFiniteValues: totalNonFiniteValues,
      firstTimedEmbeddingMs: Number((firstTimedEmbeddingMs ?? 0).toFixed(2)),
      minEmbeddingMs: Number(embeddingMsStats.min.toFixed(2)),
      medianEmbeddingMs: Number(embeddingMsStats.median.toFixed(2)),
      p95EmbeddingMs: Number(embeddingMsStats.p95.toFixed(2)),
      p99EmbeddingMs: Number(embeddingMsStats.p99.toFixed(2)),
      maxEmbeddingMs: Number(embeddingMsStats.max.toFixed(2)),
      stdDevEmbeddingMs: Number(embeddingMsStats.stdDev.toFixed(2)),
      ci95EmbeddingMs: Number(embeddingMsStats.ci95.toFixed(2)),
      avgEmbeddingMs: Number(avgMs.toFixed(2)),
      avgEmbeddingsPerSec: Number((avgMs > 0 ? (1000 / avgMs) : 0).toFixed(2)),
      avgEmbeddingTokens: Number(embeddingTokensStats.mean.toFixed(2)),
      avgEmbeddingL2Norm: Number(embeddingNormStats.mean.toFixed(4)),
      modelLoadMs: safeModelLoadMs,
      latency: {
        timedEmbeddingMs: timedEmbeddingMsStats,
        embeddingMs: embeddingMsStats,
      },
      dimensions: {
        embedding: embeddingDimStats,
      },
      embedding: {
        tokens: embeddingTokensStats,
        l2Norm: embeddingNormStats,
      },
    };

    const timedStats = computeSampleStats(durations);
    timing = buildCanonicalTiming({
      modelLoadMs: safeModelLoadMs,
      firstTokenMs: null,
      firstResponseMs: Number.isFinite(firstTimedEmbeddingMs)
        ? safeModelLoadMs + firstTimedEmbeddingMs
        : null,
      prefillMs: null,
      decodeMs: null,
      totalRunMs: timedStats.median,
      cacheMode,
      loadMode,
    });
  } else {
    const tokensPerSec = [];
    const durations = [];
    const tokensGenerated = [];
    const decodeMsPerToken = [];
    const ttftMs = [];
    const prefillMs = [];
    const decodeMs = [];
    const prefillTokens = [];
    const decodeTokens = [];
    const decodeTokensPerSec = [];
    const prefillTokensPerSec = [];
    const prefillTokensPerSecTtft = [];
    const gpuPrefillMs = [];
    const gpuDecodeMs = [];
    const gpuDecodeRecordMs = [];
    const gpuDecodeSubmitWaitMs = [];
    const gpuDecodeReadbackWaitMs = [];

    let generatedText = null;
    for (let i = 0; i < warmupRuns + timedRuns; i++) {
      harness.pipeline.reset?.();
      const run = await runGeneration(harness.pipeline, runtimeConfig, benchRun);
      if (i === warmupRuns + timedRuns - 1) {
        generatedText = run?.output ?? null;
      }
      if (i >= warmupRuns) {
        const phase = run?.phase ?? {};
        const phaseTokens = Array.isArray(run?.tokens) ? run.tokens : [];
        const phaseGpu = phase.gpu;
        tokensPerSec.push(run?.tokensPerSec);
        durations.push(run?.durationMs);
        tokensGenerated.push(phaseTokens.length);
        ttftMs.push(phase.ttftMs);
        prefillMs.push(phase.prefillMs);
        decodeMs.push(phase.decodeMs);
        prefillTokens.push(phase.prefillTokens);
        decodeTokens.push(phase.decodeTokens);
        decodeTokensPerSec.push(phase.decodeTokensPerSec);
        prefillTokensPerSec.push(phase.prefillTokensPerSec);
        prefillTokensPerSecTtft.push(phase.prefillTokensPerSecTtft);
        if (phase.decodeMs > 0 && phase.decodeTokens > 0) {
          decodeMsPerToken.push(phase.decodeMs / phase.decodeTokens);
        }
        if (Number.isFinite(phaseGpu?.prefillMs)) gpuPrefillMs.push(phaseGpu.prefillMs);
        if (Number.isFinite(phaseGpu?.decodeMs)) gpuDecodeMs.push(phaseGpu.decodeMs);
        if (Number.isFinite(phaseGpu?.decodeRecordMs)) gpuDecodeRecordMs.push(phaseGpu.decodeRecordMs);
        if (Number.isFinite(phaseGpu?.decodeSubmitWaitMs)) gpuDecodeSubmitWaitMs.push(phaseGpu.decodeSubmitWaitMs);
        if (Number.isFinite(phaseGpu?.decodeReadbackWaitMs)) gpuDecodeReadbackWaitMs.push(phaseGpu.decodeReadbackWaitMs);
      }
    }

    const totalMsStats = computeSampleStats(durations);
    const tokensPerSecStats = computeSampleStats(tokensPerSec);
    const decodeTokensPerSecStats = computeSampleStats(decodeTokensPerSec);
    const prefillTokensPerSecStats = computeSampleStats(prefillTokensPerSec);
    const prefillTokensPerSecTtftStats = computeSampleStats(prefillTokensPerSecTtft);
    const decodeMsPerTokenStats = computeSampleStats(decodeMsPerToken);
    const ttftMsStats = computeSampleStats(ttftMs);
    const prefillMsStats = computeSampleStats(prefillMs);
    const decodeMsStats = computeSampleStats(decodeMs);
    const tokensGeneratedStats = computeSampleStats(tokensGenerated);
    const prefillTokensStats = computeSampleStats(prefillTokens);
    const decodeTokensStats = computeSampleStats(decodeTokens);
    const gpuPhaseStats = gpuPrefillMs.length > 0 || gpuDecodeMs.length > 0 || gpuDecodeRecordMs.length > 0
      || gpuDecodeSubmitWaitMs.length > 0 || gpuDecodeReadbackWaitMs.length > 0
      ? {
          prefillMs: computeSampleStats(gpuPrefillMs),
          decodeMs: computeSampleStats(gpuDecodeMs),
          decodeRecordMs: computeSampleStats(gpuDecodeRecordMs),
          decodeSubmitWaitMs: computeSampleStats(gpuDecodeSubmitWaitMs),
          decodeReadbackWaitMs: computeSampleStats(gpuDecodeReadbackWaitMs),
        }
      : null;

    results = [
      {
        name: 'benchmark',
        passed: tokensPerSec.length > 0,
        duration: durations.reduce((sum, value) => sum + value, 0),
        error: tokensPerSec.length > 0 ? undefined : 'No benchmark runs completed',
      },
    ];

    const normalizedFirstTokenMs = sampleTimingNumber(ttftMsStats, 'median', null);

    metrics = {
      warmupRuns,
      timedRuns,
      prompt: benchRun.prompt,
      maxTokens: benchRun.maxTokens,
      decodeTokensPerSec: sampleTimingNumber(decodeTokensPerSecStats, 'median'),
      avgTokensGenerated: Math.round(tokensGeneratedStats.mean),
      avgPrefillTokens: Math.round(prefillTokensStats.mean),
      avgDecodeTokens: Math.round(decodeTokensStats.mean),
      medianPrefillTokensPerSec: sampleTimingNumber(prefillTokensPerSecStats, 'median'),
      avgPrefillTokensPerSec: sampleTimingNumber(prefillTokensPerSecStats, 'mean'),
      medianPrefillTokensPerSecTtft: sampleTimingNumber(prefillTokensPerSecTtftStats, 'median'),
      avgPrefillTokensPerSecTtft: sampleTimingNumber(prefillTokensPerSecTtftStats, 'mean'),
      avgDecodeTokensPerSec: sampleTimingNumber(decodeTokensPerSecStats, 'mean'),
      firstTokenMs: normalizedFirstTokenMs,
      firstResponseMs: safeToFixed(safeModelLoadMs + normalizedFirstTokenMs, null),
      prefillMs: sampleTimingNumber(prefillMsStats, 'median'),
      decodeMs: sampleTimingNumber(decodeMsStats, 'median'),
      totalRunMs: sampleTimingNumber(totalMsStats, 'median'),
      decodeMsPerTokenP50: sampleTimingNumber(decodeMsPerTokenStats, 'median'),
      decodeMsPerTokenP95: sampleTimingNumber(decodeMsPerTokenStats, 'p95'),
      decodeMsPerTokenP99: sampleTimingNumber(decodeMsPerTokenStats, 'p99'),
      avgPrefillMs: sampleTimingNumber(prefillMsStats, 'mean'),
      modelLoadMs: safeModelLoadMs,
      throughput: {
        tokensPerSec: tokensPerSecStats,
        prefillTokensPerSec: prefillTokensPerSecStats,
        prefillTokensPerSecTtft: prefillTokensPerSecTtftStats,
        decodeTokensPerSec: decodeTokensPerSecStats,
      },
      latency: {
        totalMs: totalMsStats,
        prefillMs: prefillMsStats,
        decodeMs: decodeMsStats,
        firstTokenMs: ttftMsStats,
      },
      tokens: {
        generated: tokensGeneratedStats,
        prefill: prefillTokensStats,
        decode: decodeTokensStats,
      },
      gpu: gpuPhaseStats,
      generatedText,
    };

    timing = buildCanonicalTiming({
      modelLoadMs: safeModelLoadMs,
      firstTokenMs: normalizedFirstTokenMs,
      firstResponseMs: Number.isFinite(normalizedFirstTokenMs)
        ? safeModelLoadMs + normalizedFirstTokenMs
        : null,
      prefillMs: prefillMsStats?.median ?? null,
      decodeMs: decodeMsStats?.median ?? null,
      decodeMsPerTokenP50: decodeMsPerTokenStats?.median ?? null,
      decodeMsPerTokenP95: decodeMsPerTokenStats?.p95 ?? null,
      decodeMsPerTokenP99: decodeMsPerTokenStats?.p99 ?? null,
      totalRunMs: totalMsStats.median,
      decodeTokensPerSec: decodeTokensPerSecStats?.median,
      prefillTokensPerSec: prefillTokensPerSecStats?.median,
      cacheMode,
      loadMode,
    });
  }

  const memoryStats = typeof harness.pipeline?.getMemoryStats === 'function'
    ? harness.pipeline.getMemoryStats()
    : null;

  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const summary = buildSuiteSummary('bench', results, startTime);
  return {
    ...summary,
    modelId: options.modelId || harness.manifest?.modelId || 'unknown',
    cacheMode,
    loadMode,
    env: {
      library: 'doppler',
      runtime: 'browser',
      device: 'webgpu',
      browserUserAgent: typeof navigator !== 'undefined' ? (navigator.userAgent || null) : null,
      browserPlatform: typeof navigator !== 'undefined' ? (navigator.platform || null) : null,
      browserLanguage: typeof navigator !== 'undefined' ? (navigator.language || null) : null,
      browserVendor: typeof navigator !== 'undefined' ? (navigator.vendor || null) : null,
    },
    timing,
    output,
    metrics,
    memoryStats,
    deviceInfo: getKernelCapabilities(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

async function runDiffusionSuite(options = {}) {
  const startTime = performance.now();
  const runtimeConfig = getRuntimeConfig();
  const captureOutput = options.captureOutput === true;
  const benchConfig = runtimeConfig.shared?.benchmark?.run || {};
  const warmupRuns = Math.max(0, Math.floor(benchConfig.warmupRuns ?? 0));
  const timedRuns = Math.max(1, Math.floor(benchConfig.timedRuns ?? 1));

  const diffusionConfig = runtimeConfig.inference?.diffusion;
  if (!diffusionConfig) {
    throw new Error('runtime.inference.diffusion must be set for diffusion harness runs.');
  }
  const scheduler = diffusionConfig.scheduler;
  const latent = diffusionConfig.latent;
  const prompt = resolvePrompt(runtimeConfig);
  const negativePrompt = diffusionConfig.negativePrompt ?? '';

  const width = Math.floor(latent?.width);
  const height = Math.floor(latent?.height);
  const steps = Math.floor(scheduler?.numSteps);
  const guidanceScale = scheduler?.guidanceScale;

  if (!Number.isFinite(width) || width <= 0) {
    throw new Error('runtime.inference.diffusion.latent.width must be set for diffusion harness runs.');
  }
  if (!Number.isFinite(height) || height <= 0) {
    throw new Error('runtime.inference.diffusion.latent.height must be set for diffusion harness runs.');
  }
  if (!Number.isFinite(steps) || steps <= 0) {
    throw new Error('runtime.inference.diffusion.scheduler.numSteps must be set for diffusion harness runs.');
  }
  if (!Number.isFinite(guidanceScale) || guidanceScale <= 0) {
    throw new Error('runtime.inference.diffusion.scheduler.guidanceScale must be set for diffusion harness runs.');
  }

  const harness = await initializeSuiteModel(options);
  const totalMs = [];
  const prefillMs = [];
  const denoiseMs = [];
  const vaeMs = [];
  const prefillTokens = [];
  const decodeTokens = [];
  const gpuTotalMs = [];
  const gpuPrefillMs = [];
  const gpuDenoiseMs = [];
  const gpuVaeMs = [];
  let output = null;

  for (let i = 0; i < warmupRuns + timedRuns; i++) {
    harness.pipeline.reset?.();
    const result = await harness.pipeline.generate({
      prompt,
      negativePrompt,
      steps,
      guidanceScale,
      width,
      height,
    });
    if (captureOutput && i === warmupRuns + timedRuns - 1) {
      output = result;
    }

    if (i < warmupRuns) continue;

    const stats = harness.pipeline.getStats?.() ?? {};
    if (Number.isFinite(stats.totalTimeMs)) totalMs.push(stats.totalTimeMs);
    if (Number.isFinite(stats.prefillTimeMs)) prefillMs.push(stats.prefillTimeMs);
    if (Number.isFinite(stats.decodeTimeMs)) denoiseMs.push(stats.decodeTimeMs);
    if (Number.isFinite(stats.vaeTimeMs)) vaeMs.push(stats.vaeTimeMs);
    if (Number.isFinite(stats.prefillTokens)) prefillTokens.push(stats.prefillTokens);
    if (Number.isFinite(stats.decodeTokens)) decodeTokens.push(stats.decodeTokens);

    const gpu = stats.gpu ?? null;
    if (gpu?.available) {
      if (Number.isFinite(gpu.totalMs)) gpuTotalMs.push(gpu.totalMs);
      if (Number.isFinite(gpu.prefillMs)) gpuPrefillMs.push(gpu.prefillMs);
      if (Number.isFinite(gpu.denoiseMs)) gpuDenoiseMs.push(gpu.denoiseMs);
      if (Number.isFinite(gpu.vaeMs)) gpuVaeMs.push(gpu.vaeMs);
    }
  }

  const memoryStats = typeof harness.pipeline?.getMemoryStats === 'function'
    ? harness.pipeline.getMemoryStats()
    : null;

  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const results = [
    {
      name: 'diffusion',
      passed: totalMs.length > 0,
      duration: totalMs.reduce((sum, value) => sum + value, 0),
      error: totalMs.length > 0 ? undefined : 'No diffusion runs completed',
    },
  ];

  const summary = buildSuiteSummary('diffusion', results, startTime);
  const cpuStats = {
    totalMs: computeSampleStats(totalMs),
    prefillMs: computeSampleStats(prefillMs),
    denoiseMs: computeSampleStats(denoiseMs),
    vaeMs: computeSampleStats(vaeMs),
  };
  const gpuStats = gpuTotalMs.length > 0
    ? {
        available: true,
        totalMs: computeSampleStats(gpuTotalMs),
        prefillMs: computeSampleStats(gpuPrefillMs),
        denoiseMs: computeSampleStats(gpuDenoiseMs),
        vaeMs: computeSampleStats(gpuVaeMs),
      }
    : { available: false };

  const avgPrefillTokens = prefillTokens.length
    ? Math.round(prefillTokens.reduce((a, b) => a + b, 0) / prefillTokens.length)
    : 0;
  const avgDecodeTokens = decodeTokens.length
    ? Math.round(decodeTokens.reduce((a, b) => a + b, 0) / decodeTokens.length)
    : 0;

  return {
    ...summary,
    modelId: options.modelId || harness.manifest?.modelId || 'unknown',
    output,
    metrics: {
      warmupRuns,
      timedRuns,
      width,
      height,
      steps,
      guidanceScale,
      prompt,
      avgPrefillTokens,
      avgDecodeTokens,
      cpu: cpuStats,
      gpu: gpuStats,
    },
    memoryStats,
    deviceInfo: getKernelCapabilities(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

async function runEnergySuite(options = {}) {
  const startTime = performance.now();
  const harness = await initializeSuiteModel(options);
  if (harness.manifest?.modelType !== 'energy') {
    throw new Error('Energy suite requires an energy model manifest.');
  }

  const result = await harness.pipeline.generate();
  const stats = harness.pipeline.getStats?.() ?? {};

  const memoryStats = typeof harness.pipeline?.getMemoryStats === 'function'
    ? harness.pipeline.getMemoryStats()
    : null;

  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const results = [
    {
      name: 'energy',
      passed: Number.isFinite(result.energy ?? NaN),
      duration: result.totalTimeMs ?? Math.max(0, performance.now() - startTime),
      error: Number.isFinite(result.energy ?? NaN) ? undefined : 'Energy did not converge',
    },
  ];

  const summary = buildSuiteSummary('energy', results, startTime);
  return {
    ...summary,
    modelId: options.modelId || harness.manifest?.modelId || 'unknown',
    metrics: {
      steps: result.steps,
      energy: result.energy ?? null,
      dtype: result.dtype,
      shape: result.shape,
      totalTimeMs: result.totalTimeMs ?? null,
      energyHistory: result.energyHistory ?? [],
      stateStats: result.stateStats ?? null,
      readbackCount: stats.readbackCount ?? null,
    },
    memoryStats,
    deviceInfo: getKernelCapabilities(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

export async function runBrowserSuite(options = {}) {
  return runWithRuntimeIsolationForSuite(async () => {
    const suiteTimestamp = resolveReportTimestamp(options.timestamp, 'runBrowserSuite timestamp');
    const suite = normalizeSuite(options.suite);
    let suiteResult;
    if (suite === 'kernels') {
      suiteResult = await runKernelSuite(options);
    } else if (suite === 'bench') {
      suiteResult = await runBenchSuite(options);
    } else if (suite === 'diffusion') {
      suiteResult = await runDiffusionSuite(options);
    } else if (suite === 'energy') {
      suiteResult = await runEnergySuite(options);
    } else if (suite === 'debug') {
      suiteResult = await runInferenceSuite({ ...options, suiteName: 'debug' });
    } else {
      suiteResult = await runInferenceSuite({ ...options, suiteName: 'inference' });
    }

    const modelId = suiteResult.modelId || options.modelId || options.modelUrl || suite;
    const reportOutput = sanitizeReportOutput(suiteResult.output);
    const report = {
      suite,
      modelId,
      runtimePreset: options.runtimePreset ?? null,
      deviceInfo: suiteResult.deviceInfo ?? null,
      results: suiteResult.results,
      durationMs: suiteResult.duration,
      timestamp: suiteTimestamp,
      metrics: suiteResult.metrics ?? null,
      output: reportOutput,
      memory: suiteResult.memoryStats ?? null,
      ...options.report,
    };
    if (!report.timestamp) {
      report.timestamp = suiteTimestamp;
    }
    const reportInfo = await saveReport(modelId, report, { timestamp: report.timestamp });
    return { ...suiteResult, report, reportInfo };
  });
}

function normalizeManifest(manifest) {
  if (!manifest || typeof manifest !== 'object') {
    throw new Error('Harness manifest must be an object.');
  }
  const runs = Array.isArray(manifest.runs) ? manifest.runs : [];
  if (!runs.length) {
    throw new Error('Harness manifest must include at least one run.');
  }
  return {
    defaults: manifest.defaults ?? {},
    runs,
    reportModelId: manifest.reportModelId ?? manifest.id ?? 'manifest',
    report: manifest.report ?? null,
  };
}

function mergeRunDefaults(defaults, run) {
  return {
    ...defaults,
    ...run,
    runtimePreset: run.runtimePreset ?? defaults.runtimePreset ?? null,
    runtimeConfigUrl: run.runtimeConfigUrl ?? defaults.runtimeConfigUrl ?? null,
    runtimeConfig: run.runtimeConfig ?? defaults.runtimeConfig ?? null,
    suite: run.suite ?? defaults.suite ?? 'inference',
  };
}

async function applyRuntimeForRun(run, options) {
  if (run.runtimeConfig) {
    const runtime = resolveRuntimeFromConfig(run.runtimeConfig);
    if (!runtime) {
      throw new Error('runtimeConfig is missing runtime fields');
    }
    setRuntimeConfig(runtime);
    return;
  }
  if (run.runtimeConfigUrl) {
    await applyRuntimeConfigFromUrl(run.runtimeConfigUrl, options);
    return;
  }
  if (run.runtimePreset) {
    await applyRuntimePreset(run.runtimePreset, options);
  }
}

function summarizeManifestRuns(results) {
  let passedRuns = 0;
  let failedRuns = 0;
  let durationMs = 0;
  for (const result of results) {
    const failures = (result.results || []).filter((entry) => !entry.passed && !entry.skipped);
    if (failures.length > 0) {
      failedRuns += 1;
    } else {
      passedRuns += 1;
    }
    durationMs += result.duration || 0;
  }
  return {
    totalRuns: results.length,
    passedRuns,
    failedRuns,
    durationMs,
  };
}

export async function runBrowserManifest(manifest, options = {}) {
  const normalized = normalizeManifest(manifest);
  const results = [];
  const manifestTimestamp = resolveReportTimestamp(options.timestamp, 'runBrowserManifest timestamp');
  const baseRuntimeConfig = cloneRuntimeConfig(getRuntimeConfig());
  const baseKernelPath = getActiveKernelPath();
  const baseKernelPathSource = getActiveKernelPathSource();
  const baseKernelPathPolicy = getActiveKernelPathPolicy();

  for (let i = 0; i < normalized.runs.length; i++) {
    const run = mergeRunDefaults(normalized.defaults, normalized.runs[i] || {});
    try {
      setRuntimeConfig(baseRuntimeConfig);
      setActiveKernelPath(baseKernelPath, baseKernelPathSource, baseKernelPathPolicy);
      await applyRuntimeForRun(run, options);
      const runTimestamp = resolveReportTimestamp(
        run.timestamp,
        `runBrowserManifest run[${i}] timestamp`,
        manifestTimestamp
      );
      const result = await runBrowserSuite({ ...run, timestamp: runTimestamp });
      results.push({
        ...result,
        label: run.label ?? `${run.suite || 'inference'}:${result.modelId || 'unknown'}`,
      });
      options.onProgress?.({
        index: i + 1,
        total: normalized.runs.length,
        label: run.label ?? result.modelId ?? run.suite ?? 'run',
      });
    } finally {
      setRuntimeConfig(baseRuntimeConfig);
      setActiveKernelPath(baseKernelPath, baseKernelPathSource, baseKernelPathPolicy);
    }
  }

  const summary = summarizeManifestRuns(results);
  const report = {
    timestamp: manifestTimestamp,
    summary,
    runs: results.map((result) => ({
      label: result.label,
      suite: result.suite,
      modelId: result.modelId,
      results: result.results,
      metrics: result.metrics ?? null,
      output: typeof result.output === 'string' ? result.output : null,
      reportInfo: result.reportInfo ?? null,
    })),
    manifest: normalized.report ?? null,
  };

  const reportInfo = options.saveReport === false
    ? null
    : await saveReport(normalized.reportModelId, report, { timestamp: options.timestamp });

  return { results, summary, report, reportInfo };
}
